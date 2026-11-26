#!/usr/bin/env python3
"""Bounded, offline models and structural source contracts, not Rust execution.

Run with python3 -B repro-kit/test_bounded_transpose_owners.py. Canonical
geometry is metadata only. Byte fixtures are deliberately tiny. No subprocess,
environment mutation, build, native test, model, cache, server or GPU is used.
Passing this suite does not establish Rust/native parity or READY/quality/rates.
Historical E82 full-file byte-identity pins have been replaced by structural
source-contract checks against current HEAD so unrelated fixes do not drift.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import struct
import unittest


ROOT = Path(__file__).resolve().parents[1]
ALIGN = 256
CAP = 64 * 1024 * 1024
MAX = (1 << 64) - 1
HELPERS = "crates/spark-model/src/layers/moe/helpers_a.rs"
BOUNDED = "crates/spark-model/src/layers/moe/bounded_transpose.rs"
SETUP = "crates/spark-model/src/factory/m2_setup.rs"
MOD = "crates/spark-model/src/layers/moe/mod.rs"


def checked_add(a, b):
    if not 0 <= a <= MAX or not 0 <= b <= MAX or a > MAX - b:
        raise ValueError("addition overflow")
    return a + b


def checked_mul(a, b):
    if not 0 <= a <= MAX or not 0 <= b <= MAX or (b and a > MAX // b):
        raise ValueError("multiplication overflow")
    return a * b


def aligned(value):
    return checked_add(value, ALIGN - 1) & ~(ALIGN - 1)


@dataclass(frozen=True)
class Owner:
    first: int
    experts: int
    size: int


@dataclass(frozen=True)
class Layout:
    experts: int
    packed: int
    scale: int
    scale_offset: int
    stride: int
    per_owner: int
    count: int
    requested: int
    payload: int
    cap: int

    @classmethod
    def build(cls, experts, rows, columns, cap=CAP):
        if experts <= 0 or rows <= 0 or columns <= 0 or columns % 16:
            raise ValueError("invalid geometry")
        if not ALIGN < cap <= CAP:
            raise ValueError("invalid cap")
        padded = checked_add(rows, 63) & ~63
        packed = checked_mul(padded, columns // 2)
        scale = checked_mul(padded, columns // 16)
        offset = aligned(packed)
        stride = aligned(checked_add(offset, scale))
        per_owner = (cap - ALIGN) // stride
        if not per_owner:
            raise ValueError("pair exceeds cap")
        count = checked_add(experts, per_owner - 1) // per_owner
        requested = checked_add(checked_mul(experts, stride), checked_mul(count, ALIGN))
        payload = checked_mul(experts, checked_add(packed, scale))
        return cls(experts, packed, scale, offset, stride, per_owner, count,
                   requested, payload, cap)

    def owner(self, index):
        if not 0 <= index < self.count:
            raise ValueError("owner out of bounds")
        first = checked_mul(index, self.per_owner)
        count = min(self.experts - first, self.per_owner)
        size = checked_add(ALIGN, checked_mul(count, self.stride))
        if size > self.cap:
            raise ValueError("owner exceeds cap")
        return Owner(first, count, size)

    def offsets(self, owner, expert):
        if not owner.first <= expert < checked_add(owner.first, owner.experts):
            raise ValueError("wrong owner")
        weight = checked_add(ALIGN, checked_mul(expert - owner.first, self.stride))
        scale = checked_add(weight, self.scale_offset)
        if (weight < ALIGN or weight % ALIGN or scale % ALIGN
                or checked_add(weight, self.packed) > scale
                or checked_add(scale, self.scale) > owner.size):
            raise ValueError("invalid component bounds")
        return weight, scale


CANONICAL = dict(qwen4=True, layers=48, experts=512, hidden=2560,
                 intermediate=640, ep_world=1, ep_rank=0, keep_originals=False)


def selector(value, **changes):
    config = dict(CANONICAL, **changes)
    if value is None or value == b"0":
        return False
    if value != b"1":
        raise ValueError("selector must be exactly 0 or 1")
    if config != CANONICAL:
        raise ValueError("noncanonical geometry or mode")
    return True


def layer_selector(value, **changes):
    config = dict(CANONICAL, prefix="target", full_interval=0,
                  full_attention_only=False, no_vision=False, no_capture=False)
    config.update(changes)
    if value == b"1" and config == dict(CANONICAL, layers=1, prefix="mtp",
            full_interval=1, full_attention_only=True, no_vision=True, no_capture=True):
        return False
    geometry = {key: config[key] for key in CANONICAL}
    return selector(value, **geometry)


class Arena:
    """A tiny simulated device, never a real allocator or driver."""

    def __init__(self):
        self.owners = []
        self.storage = {}
        self.next_base = 0x100000
        self.calls = 0

    def allocate(self, size, fail=False, returned=None, store=True):
        if not ALIGN < size <= CAP:
            raise ValueError("invalid request")
        self.calls += 1
        if fail:
            raise RuntimeError("allocation failed")
        base = self.next_base if returned is None else returned
        self.owners.append((base, size))
        if base == 0 or base % ALIGN:
            raise ValueError("invalid successful allocation")
        end = checked_add(base, size)
        for other, length in self.owners[:-1]:
            if not (end <= other or checked_add(other, length) <= base):
                raise ValueError("owner overlap")
        self.next_base = aligned(end)
        if store:
            self.storage[base] = bytearray([0xA5]) * size
        return base


@dataclass(frozen=True)
class Source:
    packed: bytes
    scales: bytes
    scalar: bytes
    input_scale: int
    present: bool = True


@dataclass(frozen=True)
class View:
    weight: int
    scale: int
    scalar: bytes
    input_scale: int


def transpose_bytes(source, rows, columns):
    if len(source) != rows * columns:
        raise ValueError("source length mismatch")
    padded = (rows + 63) // 64 * 64
    return bytes(source[row * columns + column] if row < rows else 0
                 for column in range(columns) for row in range(padded))


def fixture(experts=7, rows=3, columns=16, seed=0):
    return [Source(bytes((seed + expert * 37 + i) % 256 for i in range(rows * columns // 2)),
                   bytes((seed + expert * 19 + i + 1) % 256 for i in range(rows * columns // 16)),
                   struct.pack("<I", 0x3F800000 + seed * 1024 + expert * 97),
                   0x9000 + seed * 0x100000 + expert * 256)
            for expert in range(experts)]


def bounded_fixture(sources, rows, columns, cap, arena=None, fail_expert=None):
    arena = arena or Arena()
    layout = Layout.build(len(sources), rows, columns, cap)
    if not all(source.present for source in sources):
        raise ValueError("incomplete source membership")
    views = []
    for index in range(layout.count):
        owner = layout.owner(index)
        base = arena.allocate(owner.size)
        for expert in range(owner.first, owner.first + owner.experts):
            if expert == fail_expert:
                raise RuntimeError("transpose failed")
            if expert != len(views):
                raise ValueError("expert order changed")
            weight, scale = layout.offsets(owner, expert)
            source = sources[expert]
            packed = transpose_bytes(source.packed, rows, columns // 2)
            scales = transpose_bytes(source.scales, rows, columns // 16)
            arena.storage[base][weight:weight + len(packed)] = packed
            arena.storage[base][scale:scale + len(scales)] = scales
            views.append(View(base + weight, base + scale, source.scalar, source.input_scale))
    if len(views) != len(sources):
        raise ValueError("incomplete destination membership")
    return layout, arena, views


def table_bytes(views):
    return (b"".join(struct.pack("<Q", view.weight) for view in views),
            b"".join(struct.pack("<Q", view.scale) for view in views),
            b"".join(view.scalar for view in views))


class GroupModel:
    """Model E82 source-owner tombstones, backend quarantine and alias order."""

    def __init__(self):
        self.states = {"gate": "live", "up": "live", "down": "live"}
        self.backend = dict(self.states)
        self.aliases = {name: [0x1100, 0x1200] for name in self.states}
        self.tombstones = set()
        self.events = []

    def release(self, name, fail=False, pending=False):
        if self.states[name] != "live":
            raise ValueError("source owner is not live")
        self.states[name] = "quarantined"
        self.tombstones.add(name)
        self.events.append(("source-quarantine", name))
        if self.backend[name] != "live":
            raise ValueError("backend owner is not live")
        self.backend[name] = "quarantined"
        self.events.append(("backend-quarantine", name))
        if pending:
            raise RuntimeError("pending DMA")
        self.events.append(("driver-free", name))
        if fail:
            raise RuntimeError("driver failed")
        del self.backend[name]
        self.states[name] = "released"

    def phase(self, names, fail=None, pending=None):
        self.events.append(("synchronize", tuple(names)))
        for name in names:
            self.release(name, fail=name == fail, pending=name == pending)
            self.aliases[name] = [0, 0]
            self.events.append(("retire-aliases", name))

    def ordinary_free(self, name):
        if name in self.backend:
            raise ValueError("grouped storage cannot use ordinary free")


def source(path):
    return (ROOT / path).read_text(encoding="utf-8")


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def rust_block(text, marker):
    """Extract a balanced Rust source block, not a Rust parser/type checker."""
    start = text.index(marker)
    opening = text.index("{", start)
    depth = 0
    quoted = False
    escaped = False
    line_comment = False
    comment_depth = 0
    index = opening
    while index < len(text):
        char = text[index]
        pair = text[index:index + 2]
        if line_comment:
            if char == "\n":
                line_comment = False
        elif comment_depth:
            if pair == "/*":
                comment_depth += 1
                index += 1
            elif pair == "*/":
                comment_depth -= 1
                index += 1
        elif quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif pair == "//":
            line_comment = True
            index += 1
        elif pair == "/*":
            comment_depth = 1
            index += 1
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
        index += 1
    raise ValueError("unterminated Rust source block")


def phase_block(text, phase):
    position = text.index("Phase " + phase + ": free ")
    start = text.rfind("        if !keep_originals {", 0, position)
    if start < 0:
        raise ValueError("missing unified retirement gate")
    return rust_block(text[start:], "        if !keep_originals {")


def compact(text):
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", text))


def bounded_contract_errors(text):
    code = compact(text)
    required = [
        'pub(crate)constSELECTOR:&str="ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS";',
        'pub(crate)constOWNER_CAP:usize=64*1024*1024;',
        'Noneifvalue.is_none()=>Ok(false)', 'Some("0")=>Ok(false)',
        'Some("1")=>Ok(true)', '!enabled||canonical_qwen4_routed_slab_selected(config,keep_originals)',
        'a.checked_add(b)', 'a.checked_mul(b)', '(cap-ALIGN)/pair_stride',
        'add(ALIGN,mul(experts,self.pair_stride)?)?',
        'weight>=ALIGN&&weight%ALIGN==0&&scale%ALIGN==0',
        'add(scale,self.scale_bytes)?<=owner.bytes',
        'base.0.checked_add(u64::try_from(bytes)?)',
        'end<=other.base.0||other_end<=base.0',
        'expert==transposed.len()', 'transposed.len()==experts.len()',
        'layout.requested_bytes-layout.payload_bytes',
        'guard_bytes.checked_sub(LEGACY_GUARD_BYTES)',
        'owner_count==expected_count', 'managed=false', 'base_unexposed=true',
        'down_allocation_after_phase_b=true',
    ]
    errors = [fragment for fragment in required if fragment not in code]
    allocator = compact(rust_block(text, "fn allocate_owner("))
    retained = allocator.find("owners.push(RetainedOwner{base,bytes});")
    validate = allocator.find("ensure!(!base.is_null()")
    if not 0 <= retained < validate:
        errors.append("successful destination not retained before validation")
    for forbidden in ("gpu.alloc_managed(", "gpu.free(", "gpu.free_grouped_backing(",
                      "implDropforRetainedOwner", "claim_grouped_backing(", "unsafe{"):
        if forbidden in code:
            errors.append(forbidden)
    return errors


# Boundary files must remain decoupled from bounded-transpose internals.
ISOLATED_BOUNDARY_FILES = (
    "crates/spark-model/src/weight_loader/qwen35.rs",
    "crates/spark-model/src/factory/build.rs",
    "crates/spark-model/src/factory/qwen4_prefill_reserve.rs",
    "crates/spark-model/src/layers/qwen4_qsa/prefill_fp8.rs",
    "crates/spark-model/src/weight_map/quantized.rs",
    "crates/spark-runtime/src/kv_cache/paged_impl.rs",
    "crates/spark-runtime/src/scratch_init.rs",
    "crates/spark-runtime/src/startup_ledger.rs",
    "crates/spark-runtime/src/cuda_backend/gpu_impl.rs",
    "crates/spark-runtime/src/cuda_backend/allocation_lifetime.rs",
    "crates/spark-runtime/src/weights/routed_backing.rs",
)
BUDGET_AND_FLOOR_FUNCTIONS = (
    "fn retained_transpose_budget(",
    "fn ensure_unified_pass_above_floor(",
    "fn ensure_retained_pass_above_floor(",
    "fn unified_pass_telemetry_decision(",
    "fn transpose_pass_telemetry_decision(",
    "fn transpose_pass_floor_decision(",
)
REVIEWED_CONSUMERS = (
    "kernels/gb10/common/moe_shared_expert_fused_t.cu",
    "kernels/gb10/qwen3-next-80b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu",
    "kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_grouped_gemm.cu",
    "kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_worklist.cuh",
    "kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_orig_i640_compact_prefill_gemm.cuh",
)


class ArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.layout = Layout.build(512, 640, 2560)

    def test_components(self):
        self.assertEqual((self.layout.packed, self.layout.scale), (819200, 102400))

    def test_pair_stride(self):
        self.assertEqual(self.layout.stride, 921600)

    def test_full_owner(self):
        self.assertEqual(self.layout.owner(0), Owner(0, 72, 66355456))

    def test_tail_owner(self):
        self.assertEqual(self.layout.owner(7), Owner(504, 8, 7373056))

    def test_owner_counts(self):
        self.assertEqual((self.layout.count, self.layout.count * 3,
                          self.layout.count * 3 * 48), (8, 24, 1152))

    def test_seven_full_and_one_tail(self):
        self.assertEqual([self.layout.owner(i).experts for i in range(8)], [72] * 7 + [8])

    def test_payload_total(self):
        self.assertEqual(self.layout.payload * 3 * 48, 67947724800)

    def test_requested_total(self):
        self.assertEqual(self.layout.requested * 3 * 48, 67948019712)

    def test_legacy_requested_total(self):
        self.assertEqual((self.layout.payload * 3 + 2 * ALIGN) * 48, 67947749376)

    def test_admission_delta(self):
        delta = self.layout.requested * 3 - self.layout.payload * 3 - 512
        self.assertEqual((delta, delta * 48), (5632, 270336))

    def test_requested_is_sum_of_owners(self):
        self.assertEqual(sum(self.layout.owner(i).size for i in range(8)), self.layout.requested)

    def test_down_geometry_matches_bytes(self):
        self.assertEqual(Layout.build(512, 2560, 640), self.layout)

    def test_all_512_membership_order_and_uniqueness(self):
        members = [expert for index in range(8) for expert in range(
            self.layout.owner(index).first,
            self.layout.owner(index).first + self.layout.owner(index).experts)]
        self.assertEqual(members, list(range(512)))
        self.assertEqual(len(set(members)), 512)

    def test_all_512_offsets_bounds_alignment_and_unexposed_base(self):
        for expert in range(512):
            with self.subTest(expert=expert):
                owner = self.layout.owner(expert // 72)
                weight, scale = self.layout.offsets(owner, expert)
                self.assertGreaterEqual(weight, ALIGN)
                self.assertEqual((weight % ALIGN, scale % ALIGN), (0, 0))
                self.assertLessEqual(weight + self.layout.packed, scale)
                self.assertLessEqual(scale + self.layout.scale, owner.size)
                self.assertLessEqual(owner.size, CAP)

    def test_adjacent_experts_do_not_overlap(self):
        for index in range(8):
            owner = self.layout.owner(index)
            for expert in range(owner.first, owner.first + owner.experts - 1):
                _, scale = self.layout.offsets(owner, expert)
                following, _ = self.layout.offsets(owner, expert + 1)
                self.assertLessEqual(scale + self.layout.scale, following)

    def test_owner_indices_rejected(self):
        for index in (-1, 8, MAX):
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.layout.owner(index)

    def test_cross_owner_membership_rejected(self):
        for expert in (-1, 72, 512):
            with self.subTest(expert=expert), self.assertRaises(ValueError):
                self.layout.offsets(self.layout.owner(0), expert)

    def test_empty_or_bad_geometry_rejected(self):
        for geometry in ((0, 640, 2560), (512, 0, 2560), (512, 640, 0),
                         (512, 640, 2559), (-1, 640, 2560)):
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                Layout.build(*geometry)

    def test_invalid_caps_rejected(self):
        for cap in (0, ALIGN, CAP + 1):
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                Layout.build(512, 640, 2560, cap)

    def test_pair_must_fit_cap(self):
        with self.assertRaises(ValueError):
            Layout.build(512, 640, 2560, 921600)

    def test_one_exact_pair_owner(self):
        layout = Layout.build(2, 640, 2560, 921856)
        self.assertEqual((layout.per_owner, layout.count, layout.owner(0).size), (1, 2, 921856))

    def test_checked_add_mul_and_align_overflow(self):
        for operation in (lambda: checked_add(MAX, 1), lambda: checked_mul(MAX, 2),
                          lambda: aligned(MAX), lambda: checked_add(-1, 0)):
            with self.assertRaises(ValueError):
                operation()

    def test_layout_overflow_models(self):
        for args in ((1, MAX, 16), (1, 64, MAX - 15), (MAX, 640, 2560)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Layout.build(*args)

    def test_exact_checked_boundary(self):
        self.assertEqual(checked_add(MAX - 1, 1), MAX)
        self.assertEqual(checked_mul(MAX, 1), MAX)
        self.assertEqual(checked_mul(MAX, 0), 0)


class SelectorTests(unittest.TestCase):
    def test_exact_native_mtp_layer_is_outside_target_flag_but_not_factory(self):
        mtp = dict(layers=1, prefix="mtp", full_interval=1,
                   full_attention_only=True, no_vision=True, no_capture=True)
        self.assertFalse(layer_selector(b"1", **mtp))
        with self.assertRaises(ValueError):
            selector(b"1", layers=1)
        for key, value in dict(prefix="other", layers=2, full_interval=2,
                full_attention_only=False, no_vision=False, no_capture=False,
                experts=511, keep_originals=True).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                layer_selector(b"1", **dict(mtp, **{key: value}))
        self.assertTrue(layer_selector(b"1"))

    def test_unset_default_off(self):
        self.assertFalse(selector(None))

    def test_exact_zero_off(self):
        self.assertFalse(selector(b"0"))

    def test_exact_one_on(self):
        self.assertTrue(selector(b"1"))

    def test_invalid_text_and_nonunicode_fail_closed(self):
        for value in (b"", b"true", b"false", b"01", b" 1", b"1\n", b"2", b"\xff"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                selector(value)

    def test_each_wrong_geometry_or_mode_rejected(self):
        for key, value in dict(qwen4=False, layers=47, experts=511, hidden=2559,
                               intermediate=639, ep_world=2, ep_rank=1,
                               keep_originals=True).items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                selector(b"1", **{key: value})

    def test_off_does_not_change_other_model_behavior(self):
        self.assertFalse(selector(None, qwen4=False, experts=1, keep_originals=True))
        self.assertFalse(selector(b"0", qwen4=False, experts=1, keep_originals=True))


class ByteAndTableTests(unittest.TestCase):
    def setUp(self):
        self.sources = fixture()
        self.cap = ALIGN + 2 * 768
        self.layout, self.arena, self.views = bounded_fixture(self.sources, 3, 16, self.cap)

    def test_tiny_fixture_spans_multiple_owners_and_tail(self):
        self.assertEqual([size for _, size in self.arena.owners], [1792] * 3 + [1024])

    def test_packed_and_scale_bytes_match_scalar_reference(self):
        for expert, view in enumerate(self.views):
            owner = self.layout.owner(expert // 2)
            base = self.arena.owners[expert // 2][0]
            weight, scale = self.layout.offsets(owner, expert)
            for original, columns, offset in ((self.sources[expert].packed, 8, weight),
                                               (self.sources[expert].scales, 1, scale)):
                reference = bytearray(columns * 64)
                for row in range(3):
                    for column in range(columns):
                        reference[column * 64 + row] = original[row * columns + column]
                self.assertEqual(self.arena.storage[base][offset:offset + len(reference)], reference)
            self.assertEqual(view.weight, base + weight)

    def test_guards_and_alignment_padding_not_written(self):
        for index, (base, size) in enumerate(self.arena.owners):
            owner = self.layout.owner(index)
            written = set()
            for expert in range(owner.first, owner.first + owner.experts):
                weight, scale = self.layout.offsets(owner, expert)
                written.update(range(weight, weight + self.layout.packed))
                written.update(range(scale, scale + self.layout.scale))
            self.assertTrue(all(value == 0xA5 for offset, value in enumerate(self.arena.storage[base])
                                if offset not in written))
            self.assertEqual(len(self.arena.storage[base]), size)

    def test_every_padded_row_is_zero(self):
        for source in self.sources:
            for data, columns in ((source.packed, 8), (source.scales, 1)):
                transposed = transpose_bytes(data, 3, columns)
                for column in range(columns):
                    self.assertEqual(transposed[column * 64 + 3:(column + 1) * 64], bytes(61))

    def test_scalars_and_input_scale_identity(self):
        self.assertEqual([(v.scalar, v.input_scale) for v in self.views],
                         [(s.scalar, s.input_scale) for s in self.sources])

    def test_scalar_bits_not_float_reformatted(self):
        for scalar in (bytes.fromhex("00000080"), bytes.fromhex("0100c07f"), bytes.fromhex("0000807f")):
            sources = [Source(s.packed, s.scales, scalar, s.input_scale) for s in self.sources]
            _, _, views = bounded_fixture(sources, 3, 16, self.cap)
            self.assertEqual(table_bytes(views)[2], scalar * len(sources))

    def test_pointer_tables_preserve_index_order_and_little_endian(self):
        packed, scales, scalars = table_bytes(self.views)
        self.assertEqual(struct.unpack("<7Q", packed), tuple(view.weight for view in self.views))
        self.assertEqual(struct.unpack("<7Q", scales), tuple(view.scale for view in self.views))
        self.assertEqual(scalars, b"".join(source.scalar for source in self.sources))
        self.assertEqual((len(packed), len(scales), len(scalars)), (56, 56, 28))

    def test_pointer_tables_never_expose_owner_base(self):
        bases = {base for base, _ in self.arena.owners}
        self.assertFalse(bases.intersection(view.weight for view in self.views))
        self.assertFalse(bases.intersection(view.scale for view in self.views))

    def test_missing_source_rejected_before_allocation(self):
        sources = list(self.sources)
        source = sources[2]
        sources[2] = Source(source.packed, source.scales, source.scalar, source.input_scale, False)
        arena = Arena()
        with self.assertRaises(ValueError):
            bounded_fixture(sources, 3, 16, self.cap, arena)
        self.assertEqual((arena.calls, arena.owners), (0, []))

    def test_transpose_failure_retains_successful_destination_owners(self):
        arena = Arena()
        with self.assertRaises(RuntimeError):
            bounded_fixture(self.sources, 3, 16, self.cap, arena, fail_expert=2)
        self.assertEqual(len(arena.owners), 2)
        self.assertEqual(len(arena.storage), 2)

    def test_bad_source_byte_length_rejected(self):
        with self.assertRaises(ValueError):
            transpose_bytes(b"123", 2, 2)

    def test_all_512_byte_table_scalar_and_owner_boundary_parity(self):
        sources = fixture(experts=512)
        layout, arena, views = bounded_fixture(sources, 3, 16, ALIGN + 72 * 768)
        self.assertEqual([layout.owner(i).experts for i in range(layout.count)], [72] * 7 + [8])
        self.assertEqual(len(views), 512)
        self.assertLess(sum(size for _, size in arena.owners), 400000)
        for expert, (original, view) in enumerate(zip(sources, views)):
            with self.subTest(expert=expert):
                owner = layout.owner(expert // 72)
                base = arena.owners[expert // 72][0]
                weight, scale = layout.offsets(owner, expert)
                self.assertEqual((view.weight, view.scale), (base + weight, base + scale))
                for data, columns, offset in ((original.packed, 8, weight),
                                               (original.scales, 1, scale)):
                    reference = bytearray(columns * 64)
                    for row in range(3):
                        for column in range(columns):
                            reference[column * 64 + row] = data[row * columns + column]
                    self.assertEqual(arena.storage[base][offset:offset + len(reference)], reference)
                self.assertEqual((view.scalar, view.input_scale), (original.scalar, original.input_scale))
        packed, scales, scalars = table_bytes(views)
        self.assertEqual(struct.unpack("<512Q", packed), tuple(view.weight for view in views))
        self.assertEqual(struct.unpack("<512Q", scales), tuple(view.scale for view in views))
        self.assertEqual(scalars, b"".join(original.scalar for original in sources))
        for expert in (0, 71, 72, 503, 504, 511):
            self.assertEqual(views[expert].weight, arena.owners[expert // 72][0]
                             + ALIGN + (expert % 72) * layout.stride)

    def test_all_512_gate_up_down_tiny_byte_table_scalar_parity(self):
        for name, rows, columns, seed in (("gate", 16, 64, 17),
                                          ("up", 16, 64, 43),
                                          ("down", 64, 16, 91)):
            sources = fixture(experts=512, rows=rows, columns=columns, seed=seed)
            shape = Layout.build(512, rows, columns)
            layout, arena, views = bounded_fixture(
                sources, rows, columns, ALIGN + 72 * shape.stride)
            with self.subTest(projection=name):
                self.assertEqual([layout.owner(i).experts for i in range(layout.count)], [72] * 7 + [8])
                self.assertLess(sum(size for _, size in arena.owners), 1200000)
            padded = (rows + 63) // 64 * 64
            for expert, (original, view) in enumerate(zip(sources, views)):
                with self.subTest(projection=name, expert=expert):
                    owner = layout.owner(expert // 72)
                    base = arena.owners[expert // 72][0]
                    weight, scale = layout.offsets(owner, expert)
                    self.assertEqual((view.weight, view.scale), (base + weight, base + scale))
                    for data, width, offset in ((original.packed, columns // 2, weight),
                                                (original.scales, columns // 16, scale)):
                        reference = bytearray(width * padded)
                        for row in range(rows):
                            for column in range(width):
                                reference[column * padded + row] = data[row * width + column]
                        self.assertEqual(arena.storage[base][offset:offset + len(reference)], reference)
                    self.assertEqual((view.scalar, view.input_scale), (original.scalar, original.input_scale))
            packed, scales, scalars = table_bytes(views)
            self.assertEqual(struct.unpack("<512Q", packed), tuple(view.weight for view in views))
            self.assertEqual(struct.unpack("<512Q", scales), tuple(view.scale for view in views))
            self.assertEqual(scalars, b"".join(original.scalar for original in sources))
            for expert in (0, 71, 72, 503, 504, 511):
                self.assertEqual(views[expert].weight, arena.owners[expert // 72][0]
                                 + ALIGN + (expert % 72) * layout.stride)


class RetentionTests(unittest.TestCase):
    def test_failed_allocation_does_not_install_nonexistent_owner(self):
        arena = Arena()
        arena.allocate(1024)
        with self.assertRaises(RuntimeError):
            arena.allocate(1024, fail=True)
        self.assertEqual(len(arena.owners), 1)

    def test_null_success_is_retained_before_validation(self):
        arena = Arena()
        with self.assertRaises(ValueError):
            arena.allocate(1024, returned=0)
        self.assertEqual(arena.owners, [(0, 1024)])

    def test_unaligned_success_is_retained_before_validation(self):
        arena = Arena()
        with self.assertRaises(ValueError):
            arena.allocate(1024, returned=0x100001)
        self.assertEqual(arena.owners, [(0x100001, 1024)])

    def test_overlapping_success_is_retained_before_validation(self):
        arena = Arena()
        arena.allocate(1024)
        with self.assertRaises(ValueError):
            arena.allocate(1024, returned=0x100100)
        self.assertEqual(len(arena.owners), 2)

    def test_adjacent_owners_are_permitted(self):
        arena = Arena()
        arena.allocate(1024, returned=0x100000)
        arena.allocate(1024, returned=0x100400)
        self.assertEqual(len(arena.owners), 2)

    def test_address_overflow_success_is_retained(self):
        arena = Arena()
        base = MAX & ~(ALIGN - 1)
        with self.assertRaises(ValueError):
            arena.allocate(1024, returned=base)
        self.assertEqual(arena.owners, [(base, 1024)])

    def test_invalid_request_never_calls_allocator(self):
        arena = Arena()
        for size in (0, ALIGN, CAP + 1):
            with self.assertRaises(ValueError):
                arena.allocate(size)
        self.assertEqual(arena.calls, 0)


class GroupRetirementTests(unittest.TestCase):
    def test_successful_phase_b_syncs_then_retires_gate_up_only(self):
        model = GroupModel()
        model.phase(["gate", "up"])
        self.assertEqual(model.events[0], ("synchronize", ("gate", "up")))
        self.assertEqual(model.aliases, {"gate": [0, 0], "up": [0, 0], "down": [0x1100, 0x1200]})
        self.assertEqual(model.states, {"gate": "released", "up": "released", "down": "live"})

    def test_phase_d_retires_down_only_after_sync(self):
        model = GroupModel()
        model.phase(["down"])
        self.assertEqual(model.events[0], ("synchronize", ("down",)))
        self.assertEqual(model.aliases["down"], [0, 0])
        self.assertEqual(model.states["gate"], "live")

    def test_driver_failure_quarantines_before_driver_and_keeps_aliases(self):
        model = GroupModel()
        with self.assertRaises(RuntimeError):
            model.phase(["gate", "up"], fail="gate")
        self.assertEqual(model.events[1:4], [("source-quarantine", "gate"),
                         ("backend-quarantine", "gate"), ("driver-free", "gate")])
        self.assertEqual((model.states["gate"], model.backend["gate"]), ("quarantined", "quarantined"))
        self.assertEqual(model.aliases["gate"], [0x1100, 0x1200])
        self.assertIn("gate", model.tombstones)

    def test_failed_free_cannot_be_retried_or_ordinary_freed(self):
        model = GroupModel()
        with self.assertRaises(RuntimeError):
            model.phase(["down"], fail="down")
        with self.assertRaises(ValueError):
            model.release("down")
        with self.assertRaises(ValueError):
            model.ordinary_free("down")
        self.assertEqual(model.events.count(("driver-free", "down")), 1)

    def test_pending_dma_quarantines_without_driver_call(self):
        model = GroupModel()
        with self.assertRaises(RuntimeError):
            model.phase(["down"], pending="down")
        self.assertNotIn(("driver-free", "down"), model.events)
        self.assertEqual(model.backend["down"], "quarantined")

    def test_gate_success_up_failure_does_not_rewind_or_touch_down(self):
        model = GroupModel()
        with self.assertRaises(RuntimeError):
            model.phase(["gate", "up"], fail="up")
        self.assertEqual(model.aliases["gate"], [0, 0])
        self.assertEqual(model.aliases["up"], [0x1100, 0x1200])
        self.assertEqual(model.states["down"], "live")
        self.assertNotIn("gate", model.backend)


class SourceContractTests(unittest.TestCase):
    def test_native_mtp_exclusion_is_exact_and_only_at_layer_scope(self):
        text = source(BOUNDED)
        block = compact(rust_block(text, "fn layer_mode("))
        for fragment in [
            'enabled&&!keep_originals&&config.weight_prefix=="mtp"',
            'config.num_hidden_layers==1&&config.full_attention_interval==1',
            'config.layer_types.as_slice()==[atlas_core::config::LayerType::FullAttention]',
            'config.vision.is_none()&&config.dflash_capture_layers.is_empty()',
            'target_geometry.num_hidden_layers=48;',
            'canonical_qwen4_routed_slab_selected(&target_geometry,false)',
            'validate_mode(enabled,config,keep_originals)',
        ]:
            self.assertIn(fragment, block)
        self.assertEqual(block.count("returnOk(false);"), 1)
        factory = compact(rust_block(text, "pub(crate) fn selected("))
        self.assertIn("letselection=owner_cap::selection()?;", factory)
        self.assertIn("validate_mode(selection.enabled,", factory)
        self.assertNotIn("layer_mode(", factory)
        inner = compact(rust_block(source(HELPERS), "pub(super) fn transpose_for_prefill_unified_inner("))
        self.assertIn("bounded_transpose::selected_for_layer(config,keep_originals)?", inner)

    def test_phase_b_synchronizes_then_releases_and_nulls_gate_up(self):
        block = compact(phase_block(source(HELPERS), "B"))
        sync = block.index("gpu.synchronize(gpu.default_stream())?;")
        gate = block.index("groups.release_projection(RoutedProjection::Gate,gpu)?;")
        up = block.index("groups.release_projection(RoutedProjection::Up,gpu)?;")
        self.assertLess(sync, gate)
        self.assertLess(gate, up)
        self.assertIn("expert.gate_proj.weight=DevicePtr::NULL;", block)
        self.assertIn("expert.up_proj.weight=DevicePtr::NULL;", block)

    def test_phase_d_synchronizes_then_releases_and_nulls_down(self):
        block = compact(phase_block(source(HELPERS), "D"))
        sync = block.index("gpu.synchronize(gpu.default_stream())?;")
        down = block.index("groups.release_projection(RoutedProjection::Down,gpu)?;")
        self.assertLess(sync, down)
        self.assertIn("expert.down_proj.weight=DevicePtr::NULL;", block)

    def test_unchanged_bf16_kv_context_scratch_floors_and_backends(self):
        for path in ISOLATED_BOUNDARY_FILES:
            with self.subTest(path=path):
                full = ROOT / path
                self.assertTrue(full.is_file(), f"missing boundary file {path}")
                self.assertNotIn("bounded_transpose", full.read_text(encoding="utf-8"))

    def test_original_transient_budget_and_floor_functions_unchanged(self):
        text = source(SETUP)
        for marker in BUDGET_AND_FLOOR_FUNCTIONS:
            with self.subTest(marker=marker):
                body = rust_block(text, marker)
                self.assertNotIn("bounded_transpose", body)
        budget = compact(rust_block(text, "fn retained_transpose_budget("))
        self.assertIn("unified_layer_bytes:checked_transpose_sum(&[full_layer,host_transient,512])?", budget)

    def test_pointer_table_builder_exact_e82(self):
        self.assertEqual(digest(rust_block(source(MOD), "fn build_ptr_table_from_qw(")),
                         "7286a0d8eb061d8610db21d0bd6cff05ed49c5a4610b76f5eb08c052bc3b1cf4")

    def test_reviewed_pointer_table_consumers_use_per_expert_tables(self):
        for path in REVIEWED_CONSUMERS:
            with self.subTest(path=path):
                full = ROOT / path
                self.assertTrue(full.is_file(), f"missing consumer {path}")
                content = full.read_text(encoding="utf-8")
                self.assertNotIn("bounded_transpose", content)
                self.assertTrue(
                    "ptrs" in content or "worklist" in content,
                    f"expected per-expert pointer table references in {path}",
                )

    def test_alignment_geometry_and_legacy_guard_constants(self):
        text = compact(source(HELPERS))
        for name, value in (("ROUTED_TRANSPOSE_SLAB_ALIGNMENT", 256),
                             ("QWEN4_ROUTED_SLAB_LAYERS", 48),
                             ("QWEN4_ROUTED_SLAB_EXPERTS", 512),
                             ("QWEN4_ROUTED_SLAB_HIDDEN", 2560),
                             ("QWEN4_ROUTED_SLAB_INTER", 640)):
            self.assertIn("const" + name + ":usize=" + str(value) + ";", text)
        self.assertIn("constLEGACY_GUARD_BYTES:usize=512;", compact(source(BOUNDED)))

    def test_reentry_with_partial_owners_rejected_before_selector(self):
        body = rust_block(source(HELPERS), "fn transpose_for_prefill_unified_inner(")
        before_selector = compact(body[:body.index("let bounded =")])
        self.assertIn("anyhow::ensure!(self.routed_bounded_transpose_owners.is_empty(),", before_selector)

    def test_canonical_geometry_predicate_exact_e82(self):
        self.assertEqual(digest(rust_block(source(HELPERS), "fn canonical_qwen4_routed_slab_selected(")),
                         "c90d7255faddb8f4db2fe2551b872cccadf0f0758ab0ee5e3af74a4cf836937a")

    def test_bounded_source_contract(self):
        self.assertEqual(bounded_contract_errors(source(BOUNDED)), [])

    def test_contract_rejects_default_on_mutation(self):
        text = source(BOUNDED).replace("None if value.is_none() => Ok(false)",
                                       "None if value.is_none() => Ok(true)")
        self.assertTrue(bounded_contract_errors(text))

    def test_contract_rejects_managed_fallback_mutation(self):
        self.assertTrue(bounded_contract_errors(source(BOUNDED).replace("gpu.alloc(bytes)",
                                                                                   "gpu.alloc_managed(bytes)")))

    def test_contract_rejects_exposed_base_mutation(self):
        text = source(BOUNDED).replace("add(ALIGN, mul(expert - owner.first_expert, self.pair_stride)?)?",
                                       "mul(expert - owner.first_expert, self.pair_stride)?")
        # Source condition also has to demand the guard, not merely advertise it.
        text = text.replace("weight >= ALIGN", "weight >= 0")
        self.assertTrue(bounded_contract_errors(text))

    def test_contract_rejects_missing_overlap_check(self):
        text = source(BOUNDED).replace("end <= other.base.0 || other_end <= base.0", "true")
        self.assertTrue(bounded_contract_errors(text))

    def test_contract_rejects_lost_retention(self):
        text = source(BOUNDED).replace("owners.push(RetainedOwner { base, bytes });", "")
        self.assertTrue(bounded_contract_errors(text))

    def test_contract_rejects_unchecked_arithmetic(self):
        text = source(BOUNDED).replace("a.checked_add(b)", "Some(a + b)")
        self.assertTrue(bounded_contract_errors(text))

    def test_legacy_selection_remains_when_bounded_off(self):
        text = compact(source(HELPERS))
        self.assertIn("ifbounded.is_none()&&canonical_qwen4_routed_slab_selected(config,keep_originals)", text)
        self.assertEqual(text.count("bounded_transpose::transpose_projection("), 3)

    def test_bounded_down_allocation_remains_after_phase_b_before_d(self):
        text = source(HELPERS)
        phase_b = text.index("Phase B: free ")
        phase_c = text.index("Phase C: transpose down routed experts")
        phase_d = text.index("Phase D: free ")
        down = text.index("|expert| &expert.down_proj,", phase_c)
        self.assertLess(phase_b, phase_c)
        self.assertLess(phase_c, down)
        self.assertLess(down, phase_d)
        self.assertIn("bounded_transpose::transpose_projection(", text[phase_c:phase_d])

    def test_admission_changes_only_unified_layer_guard(self):
        block = compact(rust_block(source(SETUP), "    if let Some(owner_cap) = bounded {"))
        self.assertEqual(block, compact("""    if let Some(owner_cap) = bounded {
            retained_budget.unified_layer_bytes = checked_transpose_sum(&[
                retained_budget.unified_layer_bytes,
                crate::layers::moe::bounded_transpose::admission_guard_delta(config, owner_cap)?,
            ])?;
        }"""))

    def test_factory_production_diff_is_only_selector_and_guard_admission(self):
        prod = source(SETUP).split("#[cfg(test)]", 1)[0]
        fn = compact(rust_block(prod, "pub(super) fn maybe_run_minimax_m2_moe_transpose("))
        selector = "letbounded=crate::layers::moe::bounded_transpose::selected(config,!unified_layout||hybrid_layout)?;"
        early_return = 'ifconfig.model_type!="minimax_m2"&&!config.is_qwen4_exp(){returnOk(());}'
        guard = (
            "ifletSome(owner_cap)=bounded{"
            "retained_budget.unified_layer_bytes=checked_transpose_sum(&["
            "retained_budget.unified_layer_bytes,"
            "crate::layers::moe::bounded_transpose::admission_guard_delta(config,owner_cap)?,"
            "])?;}"
        )
        self.assertEqual(fn.count(selector), 1)
        self.assertEqual(fn.count(early_return), 1)
        self.assertEqual(fn.count(guard), 1)
        self.assertLess(fn.index(selector), fn.index(early_return))
        self.assertLess(fn.index(early_return), fn.index(guard))
        self.assertIn("ensure_unified_pass_above_floor(gpu,retained_budget.unified_layer_bytes)?;", fn)
        self.assertIn("letresult=layer.transpose_moe_for_prefill_unified(gpu,config);", fn)

    def test_selector_checked_before_any_model_type_early_return(self):
        body = compact(source(SETUP))
        self.assertLess(
            body.index("letbounded=crate::layers::moe::bounded_transpose::selected("),
            body.index('ifconfig.model_type!="minimax_m2"'),
        )
        self.assertIn("!unified_layout||hybrid_layout", body)

    def test_retirement_counterfactual_is_detected_when_sync_removed(self):
        block = phase_block(source(HELPERS), "B")
        changed = compact(block.replace("gpu.synchronize(gpu.default_stream())?;", ""))
        self.assertNotIn("gpu.synchronize(gpu.default_stream())?;", changed)

    def test_source_scanner_handles_braces_in_comments_and_strings(self):
        text = 'fn demo() { /* } /* { */ } */ let s = "}"; // }\n if true { } } trailing'
        self.assertEqual(rust_block(text, "fn demo("), text.removesuffix(" trailing"))


CAP_FILE = "crates/spark-model/src/layers/moe/bounded_transpose/owner_cap.rs"
TARGET = 'target: "atlas_bounded_transpose_allocation"'


class OwnerCapGeometryTests(unittest.TestCase):
    def test_both_caps_preserve_exact_payload_and_complete_membership(self):
        for mib, per, count, requested, guard in (
            (16, 18, 29, 1_415_599_872, 21_760),
            (64, 72, 8, 1_415_583_744, 5_632),
        ):
            layout = Layout.build(512, 640, 2560, mib * 1024 * 1024)
            with self.subTest(mib=mib):
                self.assertEqual((layout.per_owner, layout.count), (per, count))
                self.assertEqual(layout.payload * 3, 1_415_577_600)
                self.assertEqual(layout.requested * 3, requested)
                self.assertEqual((layout.requested - layout.payload) * 3 - 512, guard)
                seen = []
                for index in range(layout.count):
                    owner = layout.owner(index)
                    self.assertLessEqual(owner.size, layout.cap)
                    for expert in range(owner.first, owner.first + owner.experts):
                        weight, scale = layout.offsets(owner, expert)
                        self.assertGreaterEqual(weight, 256)
                        self.assertEqual((weight % 256, scale % 256), (0, 0))
                        seen.append(expert)
                self.assertEqual(seen, list(range(512)))

    def test_model_totals_and_extra_requested_guard_bytes(self):
        small = Layout.build(512, 640, 2560, 16 * 1024 * 1024)
        default = Layout.build(512, 640, 2560)
        self.assertEqual((small.count * 3 * 48, default.count * 3 * 48), (4176, 1152))
        self.assertEqual(small.requested * 3 * 48, 67_948_793_856)
        self.assertEqual((small.requested - default.requested) * 3 * 48, 774_144)
        self.assertEqual(small.owner(0).size, 16_589_056)
        self.assertEqual(small.owner(28).size, 7_373_056)

    def test_small_bytes_and_scalar_bits_survive_all_three_projection_partitions(self):
        for name, rows, columns, seed in (("gate", 16, 64, 17),
                                         ("up", 16, 64, 43),
                                         ("down", 64, 16, 91)):
            sources = fixture(experts=512, rows=rows, columns=columns, seed=seed)
            stride = Layout.build(512, rows, columns).stride
            results = []
            for per in (18, 72):
                layout, arena, views = bounded_fixture(
                    sources, rows, columns, 256 + per * stride)
                payloads = []
                for expert, view in enumerate(views):
                    owner = layout.owner(expert // per)
                    backing = arena.owners[expert // per][0]
                    weight, scale = layout.offsets(owner, expert)
                    self.assertEqual(view.weight, backing + weight)
                    self.assertEqual(view.scale, backing + scale)
                    payloads.append((bytes(arena.storage[backing][weight:weight + layout.packed]),
                                     bytes(arena.storage[backing][scale:scale + layout.scale]),
                                     view.scalar, view.input_scale))
                results.append(payloads)
            with self.subTest(projection=name):
                self.assertEqual(results[0], results[1])


class OwnerCapSourceTests(unittest.TestCase):
    def test_parser_accepts_only_exact_caps_and_requires_enabled_mode(self):
        text = source(CAP_FILE)
        parser = compact(rust_block(text, "fn parse("))
        for required in (
            'letenabled=requested(enabled)?;',
            'None=>OwnerCap::DEFAULT,',
            'ensure!(enabled,"{CAP_SELECTOR}requires{SELECTOR}=1");',
            'Some("16")=>OwnerCap(16*1024*1024),',
            'Some("64")=>OwnerCap::DEFAULT,',
            '_=>anyhow::bail!("{CAP_SELECTOR}mustbeexactly16or64")',
        ):
            self.assertIn(required, parser)
        self.assertIn('"ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB"', text)
        self.assertIn("fn non_unicode_cap_is_rejected_without_environment_mutation", text)
        self.assertNotIn("parse::<", parser)

    def test_selection_and_failed_selection_are_captured_once(self):
        text = source(CAP_FILE)
        body = compact(rust_block(text, "pub(super) fn selection("))
        self.assertIn("OnceLock<std::result::Result<Selection,String>>", body)
        self.assertEqual(body.count("SELECTION.get_or_init("), 1)
        self.assertIn(".map_err(|error|error.to_string())", body)
        self.assertIn('Err(error)=>anyhow::bail!("{error}")', body)
        self.assertEqual(text.count("std::env::var_os("), 2)
        for path in (BOUNDED, HELPERS, SETUP):
            self.assertNotIn("std::env::var_os(SELECTOR)", source(path))

    def test_same_captured_cap_reaches_allocations_guard_and_receipt(self):
        bounded = source(BOUNDED)
        for marker in ("pub(crate) fn selected(", "pub(super) fn selected_for_layer("):
            selection = compact(rust_block(bounded, marker))
            self.assertIn("letselection=owner_cap::selection()?;", selection)
            self.assertIn(".then_some(selection.cap)", selection)
        allocation = compact(rust_block(bounded, "pub(super) fn transpose_projection("))
        self.assertIn("cap.bytes(),projection_label)", allocation)
        self.assertNotIn("OWNER_CAP", allocation)
        canonical = compact(rust_block(bounded, "fn canonical_layouts("))
        self.assertEqual(canonical.count("cap.bytes()"), 2)
        for marker in ("pub(crate) fn admission_guard_delta(", "pub(super) fn record_ready("):
            self.assertIn("canonical_layouts(config,cap)?", compact(rust_block(bounded, marker)))
        ready = compact(rust_block(bounded, "pub(super) fn record_ready("))
        self.assertIn("owner_cap_bytes=cap.bytes()", ready)
        helpers = compact(rust_block(source(HELPERS),
                                              "pub(super) fn transpose_for_prefill_unified_inner("))
        for projection in ("gate", "up", "down"):
            self.assertIn(f'owner_cap,"{projection}",', helpers)
        self.assertRegex(helpers, r"self\.routed_bounded_transpose_owners\.len\(\),owner_cap,?\)\?")
        guard = compact(rust_block(source(SETUP),
                                             "    if let Some(owner_cap) = bounded {"))
        self.assertIn("admission_guard_delta(config,owner_cap)?", guard)

    def test_trace_calls_original_allocator_once_and_returns_original_result(self):
        text = source(BOUNDED)
        trace = compact(rust_block(text, "fn traced_allocation("))
        self.assertIn("allocate:implFnOnce()->Result<DevicePtr>", trace)
        self.assertEqual(trace.count("allocate()"), 1)
        self.assertIn("letresult=allocate();", trace)
        self.assertTrue(trace.endswith("result}"))
        self.assertNotIn("map_err", trace)
        self.assertNotIn("context(", trace)
        self.assertIn('event="allocation_begin"', trace)
        self.assertIn('Ok(_)=>tracing::debug!', trace)
        self.assertIn('Err(_)=>tracing::debug!', trace)
        projection = compact(rust_block(text, "fn transpose_projection_with_cap("))
        self.assertEqual(projection.count("gpu.alloc(bytes)"), 1)
        self.assertIn("traced_allocation(||gpu.alloc(bytes))", projection)
        self.assertNotIn("alloc_managed", projection)
        self.assertNotIn("gpu.free", projection)
        self.assertIn("fn traced_failure_is_not_wrapped_retried_or_installed_as_an_owner", source(CAP_FILE))

    def test_tracing_is_debug_only_and_exposes_no_pointer_or_path_fields(self):
        for name in (BOUNDED, SETUP):
            for line in source(name).splitlines():
                if TARGET in line:
                    self.assertTrue("tracing::debug!(" in line or "tracing::debug_span!(" in line
                                    or line.strip().startswith("target:"), line)
        projection = rust_block(source(BOUNDED), "fn transpose_projection_with_cap(")
        span = projection[projection.index("let span ="):projection.index("let _entered =")]
        self.assertIn('backend_operation = "GpuBackend::alloc"', span)
        self.assertIn("projection_owner_ordinal = index + 1", span)
        for forbidden in ("base =", "ptr =", "path =", "weight =", "owner.base"):
            self.assertNotIn(forbidden, span)
        setup = compact(rust_block(source(SETUP),
                                             "        for (layer_index, layer) in layers.iter_mut().enumerate() {"))
        self.assertEqual(setup.count("layer.transpose_moe_for_prefill_unified(gpu,config)"), 1)
        self.assertEqual(setup.count("ensure_unified_pass_above_floor("), 1)
        self.assertIn("letresult=layer.transpose_moe_for_prefill_unified(gpu,config);", setup)
        self.assertTrue(setup.endswith("result?;}"))
        for event in ("layer_begin", "layer_end", "layer_error"):
            self.assertIn(f'event="{event}"', setup)

    def test_new_rust_files_remain_under_source_size_limit(self):
        for name in (BOUNDED, CAP_FILE):
            self.assertLessEqual(len(source(name).splitlines()), 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
