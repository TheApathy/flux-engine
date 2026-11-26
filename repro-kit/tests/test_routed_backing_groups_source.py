"""Bounded source contracts and independent layout reference, not Rust execution."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ALIGN = 256
LIMIT = 64 * 1024 * 1024
PAYLOAD = 67_947_724_800


def reference_plain_tp1_scope(tp_world_size=1, tp_rank=0, speculative=False,
                              dflash=False, self_speculative=False,
                              ngram_speculative=False, disable_speculation=False):
    """Admission reference: Plain, native MTP (`speculative`), or DFlash
    (`dflash`); an execution override does not widen load eligibility to
    self- or n-gram speculation."""
    return (tp_world_size == 1 and tp_rank == 0
            and not self_speculative and not ngram_speculative)


def reference_eager(value, grouped):
    """Eager sub-flag reference: unset/0/1 only, and 1 requires grouping."""
    if value is None or value == "0":
        return False
    if value == "1":
        return True if grouped else "error"
    return "error"


def reference_layout(shard_for):
    """Stream geometry without allocating tensor payloads or a model inventory."""
    current = {}
    sizes = []
    count = 0
    payload = 0
    for layer in range(48):
        for projection in range(3):
            for expert in range(512):
                for scale, size in ((False, 819_200), (True, 102_400)):
                    key = (shard_for(layer, projection, expert, scale), layer, projection)
                    group = current.get(key)
                    if group is None or sizes[group] + size > LIMIT:
                        group = len(sizes)
                        sizes.append(ALIGN)
                        current[key] = group
                    sizes[group] += size
                    count += 1
                    payload += size
    return sizes, count, payload


class RoutedBackingSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = (ROOT / "crates/spark-runtime/src/fast_weights/mod.rs").read_text()
        cls.plan = (ROOT / "crates/spark-runtime/src/fast_weights/routed_groups.rs").read_text()
        cls.owner = (ROOT / "crates/spark-runtime/src/weights/routed_backing.rs").read_text()
        cls.store = (ROOT / "crates/spark-runtime/src/weights.rs").read_text()
        cls.gpu = (ROOT / "crates/spark-runtime/src/gpu.rs").read_text()
        cls.life = (ROOT / "crates/spark-runtime/src/cuda_backend/allocation_lifetime.rs").read_text()
        cls.cuda = (ROOT / "crates/spark-runtime/src/cuda_backend/gpu_impl.rs").read_text()
        cls.moe = (ROOT / "crates/spark-model/src/layers/moe/helpers_a.rs").read_text()
        cls.attach = (ROOT / "crates/spark-model/src/weight_loader/qwen35/load_layers.rs").read_text()
        cls.serve = (ROOT / "crates/spark-server/src/main_modules/serve_phases/weights.rs").read_text()

    def test_reference_canonical_payload_cardinality_and_guard_cost(self):
        sizes, components, payload = reference_layout(lambda *_: 0)
        self.assertEqual(components, 147_456)
        self.assertEqual(payload, PAYLOAD)
        self.assertEqual(len(sizes), 1152)
        self.assertTrue(all(ALIGN < size <= LIMIT for size in sizes))
        self.assertEqual(sum(sizes) - payload, 294_912)
        self.assertNotEqual(len(sizes), 144)

    def test_reference_shard_boundaries_change_exact_group_count(self):
        whole, _, payload = reference_layout(lambda *_: 0)
        split, _, split_payload = reference_layout(lambda _l, _p, expert, _s: expert // 64)
        tiny, _, tiny_payload = reference_layout(lambda _l, _p, expert, scale: (expert, scale))
        self.assertEqual(payload, split_payload)
        self.assertEqual(payload, tiny_payload)
        self.assertEqual(len(whole), 1152)
        self.assertEqual(len(split), 1152)
        self.assertEqual(len(tiny), 147_456)
        self.assertEqual(sum(tiny) - payload, 147_456 * ALIGN)

    def test_default_is_off_and_environment_is_strict(self):
        self.assertEqual(self.loader.count("grouped_routed_originals: false"), 2)
        for fragment in ('Some("0") => Ok(false)', 'Some("1") => Ok(true)',
                         "None if value.is_none() => Ok(false)", "must be exactly 0 or 1"):
            self.assertIn(fragment, self.plan)
        self.assertIn("value.and_then(std::ffi::OsStr::to_str)", self.plan)

    def test_eager_default_is_off_and_environment_is_strict(self):
        self.assertEqual(self.loader.count("grouped_routed_originals_eager: false"), 2)
        parser = self.plan.split("pub fn routed_grouping_eager_requested(", 1)[1].split("#[derive(Clone)]", 1)[0]
        parser_ns = "".join(parser.split())
        for fragment in ('None if value.is_none() => false', 'Some("0") => false', 'Some("1") => true',
                         "must be exactly 0 or 1", "ensure!(!eager || grouped"):
            self.assertIn("".join(fragment.split()), parser_ns)
        self.assertIn('"ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER"', self.plan)
        for value, grouped, expected in ((None, False, False), (None, True, False), ("0", False, False),
                                         ("0", True, False), ("1", True, True), ("1", False, "error"),
                                         ("true", True, "error"), (" 1", True, "error"), ("", True, "error")):
            self.assertEqual(reference_eager(value, grouped), expected, (value, grouped))

    def test_server_parses_eager_after_grouping_and_passes_it_to_the_loader(self):
        serve_ws = " ".join(self.serve.split())
        grouping = serve_ws.index("let grouped_routed_originals = {")
        eager = serve_ws.index("let grouped_routed_originals_eager = spark_runtime::fast_weights::routed_grouping_eager_requested(")
        store = serve_ws.index("let store = if use_fast_load {")
        self.assertLess(grouping, eager)
        self.assertLess(eager, store)
        self.assertIn("grouped_routed_originals, )?;", serve_ws)
        self.assertIn("loader.grouped_routed_originals_eager = grouped_routed_originals_eager;", serve_ws)

    def test_eager_bases_follow_preflight_and_precede_the_first_shard_read(self):
        load = self.loader.split("fn load(", 1)[1]
        plan = load.index("RoutedBackingLoad::plan(")
        preflight_refusal = load.index("if peak.saturating_add(oom_reserve_bytes) > free {")
        audit = load.index("let mut allocation_audit = FastLoadAllocationAudit::default();")
        eager = load.index("if self.grouped_routed_originals_eager {")
        first_shard = load.index("for (i, shard_path) in shard_files.iter().enumerate() {")
        self.assertLess(plan, preflight_refusal)
        self.assertLess(preflight_refusal, eager)
        self.assertLess(audit, eager)
        self.assertLess(eager, first_shard)
        block = load[eager:first_shard]
        self.assertIn("load.allocate_all(gpu)?", block)
        self.assertIn("allocation_audit.record(bytes, false);", block)
        self.assertEqual(block.count("tracing::info!"), 1)
        self.assertIn("bases = sizes.len(), bytes = sizes.iter().sum::<usize>()", block)
        # Admission is unchanged: the peak term still adds only planned guard bytes.
        self.assertEqual(load.count("RoutedBackingLoad::overhead_bytes"), 1)
        self.assertNotIn("allocate_all", load[:preflight_refusal])

    def test_eager_materializes_every_planned_base_and_first_views_reuse_them(self):
        allocate_all = self.plan.split("pub(super) fn allocate_all(", 1)[1].split("\n    }\n", 1)[0]
        self.assertIn("(0..self.groups.len()).map(|group_index|self.materialize(group_index,gpu)).collect()", "".join(allocate_all.split()))
        destination = self.plan.split("pub(super) fn destination(", 1)[1].split("fn materialize(", 1)[0]
        self.assertIn("let allocated = if needs_base { Some(self.materialize(group_index, gpu)?) } else { None };", " ".join(destination.split()))
        self.assertIn("(view.offset, group.base.is_none())", destination)
        materialize = self.plan.split("fn materialize(", 1)[1]
        self.assertIn('ensure!(group.base.is_none(), "grouped backing already allocated");', " ".join(materialize.split()))

    def test_server_gate_requires_target_geometry_fast_ep1(self):
        gate = " ".join(self.serve.split("let grouped_routed_originals =", 1)[1].split("let store =", 1)[0].split())
        for fragment in ("use_fast_load && config.is_qwen4_exp()", "num_hidden_layers == 48",
                         "num_experts == 512", "hidden_size == 2560", "moe_intermediate_size == 640",
                         "ep_rank == 0 && ep_size == 1", "config.ep_world_size == 1"):
            self.assertIn(fragment, gate)
        self.assertEqual(self.serve.count("loader.grouped_routed_originals ="), 1)

    def test_server_enabled_guard_requires_resolved_tp1_and_every_plain_flag(self):
        gate = self.serve.split("let grouped_routed_originals =", 1)[1].split("let store =", 1)[0]
        condition = " ".join(gate.split("anyhow::ensure!(", 1)[1].split(',\n', 1)[0].split())
        for fragment in ("&& config.tp_world_size == 1", "&& config.tp_rank == 0",
                         "&& !args.self_speculative && !args.ngram_speculative"):
            self.assertIn(fragment, condition)
        # Native MTP and DFlash are admitted: neither --speculative nor --dflash is refused.
        self.assertNotIn("args.speculative", condition)
        self.assertNotIn("args.dflash", condition)
        self.assertNotIn("||", condition)
        self.assertNotIn("args.tp_", condition)
        self.assertNotIn("ATLAS_DISABLE_SPECULATION", gate)
        self.assertIn("TP1/EP1 and Plain, native MTP, or DFlash", gate)

    def test_reference_rejects_tp_world_sizes_and_nonzero_resolved_ranks(self):
        self.assertTrue(reference_plain_tp1_scope())
        for width, rank in ((0, 0), (2, 0), (2, 1), (128, 0), (1, 1), (1, 128)):
            with self.subTest(width=width, rank=rank):
                self.assertFalse(reference_plain_tp1_scope(tp_world_size=width, tp_rank=rank))

    def test_reference_admits_native_mtp_and_rejects_every_other_drafter_combination(self):
        self.assertTrue(reference_plain_tp1_scope(speculative=True))
        self.assertTrue(reference_plain_tp1_scope(dflash=True))
        flags = ("speculative", "dflash", "self_speculative", "ngram_speculative")
        for mask in range(1, 16):
            values = {name: bool(mask & (1 << bit)) for bit, name in enumerate(flags)}
            with self.subTest(mask=mask):
                expected = (mask & ~0b0011) == 0  # native MTP and/or DFlash only
                self.assertEqual(reference_plain_tp1_scope(**values), expected)

    def test_disable_speculation_override_cannot_admit_loaded_serial_modes(self):
        for name in ("self_speculative", "ngram_speculative"):
            for disabled in (False, True):
                with self.subTest(mode=name, disable_speculation=disabled):
                    self.assertFalse(reference_plain_tp1_scope(**{
                        name: True, "disable_speculation": disabled}))
        for name in ("speculative", "dflash"):
            for disabled in (False, True):
                with self.subTest(mode=name, disable_speculation=disabled):
                    self.assertTrue(reference_plain_tp1_scope(**{
                        name: True, "disable_speculation": disabled}))

    def test_complete_inventory_preflight_precedes_gpu_calls_and_extra_copies(self):
        load = self.loader.split("impl WeightLoader for FastSafetensorsLoader", 1)[1]
        self.assertLess(load.index("RoutedBackingLoad::plan("), load.index("gpu.free_memory()?"))
        self.assertIn("slots.len() == COMPONENTS && payload == PAYLOAD_BYTES", self.plan)
        self.assertIn("slots.insert((layer, expert, projection, scale))", self.plan)
        self.assertIn("extra weights must not replace grouped target originals", self.plan)
        self.assertIn("index.is_none_or", self.plan)
        self.assertIn("!skip(&meta.name)", self.plan)

    def test_plan_enforces_narrow_dtype_shape_name_and_file_membership(self):
        for fragment in ("layer < LAYERS && expert < EXPERTS", "meta.shape == shape && meta.dtype == dtype",
                         "meta.len == shape[0] * shape[1]", "WeightDtype::FP8E4M3", "WeightDtype::UInt8",
                         "noncanonical grouped routed name", "end <= file_bytes",
                         "group.shard == shard", "view.abs_offset == meta.abs_offset"):
            self.assertIn(fragment, self.plan)
        self.assertIn('const PREFIX: &str = "model.language_model.layers."', self.plan)

    def test_every_group_is_shard_layer_projection_scoped_and_bounded(self):
        for fragment in ("let key = (shard.clone(), layer, projection)", "end <= GROUP_LIMIT",
                         "group.bytes <= GROUP_LIMIT", "bytes: ALIGN", "const GROUP_LIMIT: usize = 64 * 1024 * 1024"):
            self.assertIn(fragment, self.plan)
        self.assertIn("guard_and_alignment_bytes = requested_bytes - payload", self.plan)
        self.assertIn("group_count = groups.len()", self.plan)
        self.assertIn("RoutedBackingLoad::overhead_bytes", self.loader)

    def test_group_claim_and_registration_precede_any_copy_and_alias_publication(self):
        destination = self.plan.split("pub(super) fn destination(", 1)[1].split("fn materialize(", 1)[0]
        self.assertLess(destination.index("self.materialize(group_index, gpu)?"), destination.index("Ok(Some("))
        materialize = self.plan.split("fn materialize(", 1)[1].split("pub(super) fn allocate_all(", 1)[0]
        # Register before claim (L1): alloc, register, claim, then publish the base.
        order = ["gpu.alloc(bytes)?", ".register(prefix, projection, base, bytes, members)",
                 "gpu.claim_grouped_backing(base)", "self.groups[group_index].base = Some(base)"]
        positions = [materialize.index(fragment) for fragment in order]
        self.assertEqual(positions, sorted(positions))
        copy = self.loader.split("let grouped_destination =", 1)[1].split("// Drop also", 1)[0]
        self.assertLess(copy.index("copy_into_owned_range(src, base, offset)?"), copy.index("out.insert("))
        self.assertIn("No managed fallback for grouped owners", materialize)
        self.assertNotIn("alloc_managed", materialize)

    def test_refused_registration_or_claim_leaves_no_orphan(self):
        materialize = self.plan.split("fn materialize(", 1)[1].split("pub(super) fn allocate_all(", 1)[0]
        register_failure = materialize.split(".register(", 1)[1].split("gpu.claim_grouped_backing", 1)[0]
        self.assertIn("return Err(release_unclaimed(gpu, base, error));", register_failure)
        claim_failure = materialize.split("gpu.claim_grouped_backing(base)", 1)[1].split("self.groups[group_index].base", 1)[0]
        # bcb1 contract: unregister (a failed unregister propagates and leaves
        # the registered base unfreed), then free the unclaimed base.
        self.assertLess(claim_failure.index("registry.unregister_unclaimed(base)?"),
                        claim_failure.index("return Err(release_unclaimed(gpu, base, error));"))
        release = self.plan.split("fn release_unclaimed(", 1)[1].split("impl Drop for RoutedBackingLoad", 1)[0]
        self.assertIn("gpu.free(base)", release)
        unregister = self.owner.split("pub(crate) fn unregister_unclaimed(", 1)[1].split("pub(crate) fn validate_store(", 1)[0]
        self.assertIn("last.base == base && *last.state.lock() == State::Live", unregister)
        self.assertIn("self.groups.pop()", unregister)
        self.assertIn("self.names.remove(&member.name)", unregister)
        self.assertNotIn("gpu", unregister)

    def test_registered_views_are_unexposed_aligned_disjoint_checked_members(self):
        for fragment in ("base.0 % 256 == 0", "previous_end = 256usize", "member.offset % 256 == 0",
                         "member.offset >= previous_end", "previous_end <= bytes", "overlapping grouped owners",
                         "duplicate grouped member", "grouped member shape/byte mismatch",
                         "grouped member crosses owner layer/projection scope"):
            self.assertIn(fragment, self.owner)

    def test_backend_defaults_are_unsupported_not_permissive(self):
        capability = self.gpu.split("fn supports_grouped_backing(", 1)[1].split("fn claim_grouped_backing(", 1)[0]
        self.assertIn("false", capability)
        loading = self.loader.split("let mut routed_load =", 1)[1]
        self.assertLess(loading.index("gpu.supports_grouped_backing()"), loading.index("RoutedBackingLoad::plan("))
        claim = self.gpu.split("fn claim_grouped_backing(", 1)[1].split("fn free_grouped_backing(", 1)[0]
        free = self.gpu.split("fn free_grouped_backing(", 1)[1].split("/// Free every", 1)[0]
        self.assertIn("anyhow::bail!", claim)
        self.assertIn("anyhow::bail!", free)
        self.assertNotIn("self.free(", free)

    def test_backend_marks_quarantine_before_driver_under_same_mutex(self):
        free = self.cuda.split("fn free_grouped_backing(", 1)[1].split("fn free_all_allocations(", 1)[0]
        self.assertLess(free.index("self.live_allocations.lock()"), free.index("prepare_grouped_free"))
        self.assertLess(free.index("prepare_grouped_free"), free.index("cuMemFree_v2"))
        self.assertLess(free.index("cuMemFree_v2"), free.index("complete_grouped_free"))
        self.assertNotIn("drop(live)", free)
        self.assertNotIn("take_for_free", free)
        prepare = self.life.split("pub(super) fn prepare_grouped_free", 1)[1].split("pub(super) fn complete_grouped_free", 1)[0]
        self.assertLess(prepare.index("allocation.quarantined = true"), prepare.index("allocation.pending_h2d == 0"))

    def test_ordinary_and_bulk_frees_and_future_retains_cannot_bypass_quarantine(self):
        self.assertIn("!allocation.grouped && !allocation.quarantined", self.life)
        self.assertIn("non-owning grouped view cannot be freed individually", self.life)
        bulk = self.life.split("pub(super) fn take_releasable", 1)[1].split("#[cfg(test)]", 1)[0]
        self.assertIn("!allocation.grouped && !allocation.quarantined", bulk)
        retain = self.life.split("pub(super) fn retain(", 1)[1].split("pub(super) fn release(", 1)[0]
        self.assertIn('ensure!(!self.quarantined, "grouped allocation is quarantined")', retain)

    def test_group_release_uses_only_explicit_base_api_and_installs_tombstones_first(self):
        self.assertIn("gpu.free_grouped_backing(base)", self.owner)
        self.assertNotIn("gpu.free(", self.owner)
        release = self.owner.split("fn release_projection_with(", 1)[1]
        release_ns = "".join(release.split())
        self.assertLess(release_ns.index("*state=State::Quarantined"), release_ns.index("free(group.base)?"))
        self.assertLess(release_ns.index("self.released.lock().extend"), release_ns.index("free(group.base)?"))
        self.assertLess(release_ns.index("free(group.base)?"), release_ns.index("*state=State::Released"))
        self.assertIn("*state == State::Live", release)

    def test_store_rejects_view_consumption_and_shares_group_tombstones(self):
        reserve = self.store.split("fn reserve_consumed_alias(", 1)[1].split("fn rollback_consumed", 1)[0]
        self.assertLess(reserve.index("!self.routed_groups.contains(name)"), reserve.index("released.insert"))
        self.assertIn("released: routed_groups.released.clone()", self.store)
        self.assertIn("other_groups.is_empty()", self.store)

    def test_model_validates_every_projection_before_attachment(self):
        attach = self.moe.split("pub fn attach_routed_original_groups", 1)[1].split("/// Pick", 1)[0]
        for role in ("Gate", "Up", "Down"):
            self.assertIn("RoutedProjection::" + role, attach)
        self.assertLess(attach.index("groups.validate_projection"), attach.index("self.routed_original_groups = Some(groups)"))
        self.assertLess(self.attach.index("attach_routed_original_groups(groups)?"), self.attach.index("moe_layer.transpose_for_prefill"))

    def test_grouped_release_is_quiescent_and_separates_phase_b_from_d(self):
        phase_b = self.moe.split("// ── Phase B:", 1)[1].split("// ── Phase C:", 1)[0]
        phase_d = self.moe.split("// ── Phase D:", 1)[1].split("if let Some(layout)", 1)[0]
        for fragment in ("gpu.synchronize(gpu.default_stream())?", "RoutedProjection::Gate", "RoutedProjection::Up"):
            self.assertIn(fragment, phase_b)
        self.assertNotIn("RoutedProjection::Down", phase_b)
        self.assertIn("gpu.synchronize(gpu.default_stream())?", phase_d)
        self.assertIn("RoutedProjection::Down", phase_d)
        self.assertNotIn("RoutedProjection::Gate", phase_d)
        self.assertIn("self.weights.shared_expert", phase_b)
        self.assertIn("gpu.free(self.weights.shared_expert", phase_d)

    def test_drop_retains_owner_records_and_never_invokes_gpu_free(self):
        for source, marker in ((self.owner, "impl Drop for RoutedGroupRegistry"),
                               (self.plan, "impl Drop for RoutedBackingLoad")):
            block = source.split(marker, 1)[1].split("\n}\n", 1)[0]
            self.assertIn("std::mem::forget", block)
            self.assertNotIn("gpu.free", block)

    def test_native_regression_definitions_exist_but_are_not_executed(self):
        fixtures = {
            "crates/spark-runtime/src/cuda_backend/allocation_lifetime_grouped_api_tests.rs": (
                "real_api_driver_failure_then_ordinary_bulk_or_grouped_retry_never_calls_driver_again",
                "real_api_pending_lease_prevents_driver_and_stays_quarantined_after_completion",
                "real_api_driver_panic_keeps_backend_quarantine_through_poisoned_mutex",
                "real_model_owner_tombstones_views_and_uses_only_grouped_backend_api"),
            "crates/spark-runtime/src/weights/routed_backing_tests.rs": (
                "strict_fake_rejects_interior_unknown_and_double_free",
                "driver_free_failure_or_pending_h2d_quarantines_without_retry",
                "backend_panic_keeps_tombstones_and_quarantine_installed",
                "consumed_group_handle_cannot_free_legitimately_reused_address"),
            "crates/spark-runtime/src/fast_weights/routed_groups_tests.rs": (
                "complete_fixture_has_bounded_phase_scoped_groups_and_exact_payload_and_guards",
                "global_inventory_rejects_missing_filtered_duplicate_and_malformed_pairs_before_allocation",
                "bounded_layout_never_crosses_shards_even_for_one_projection",
                "eager_flag_accepts_only_unset_zero_or_one_and_requires_grouping",
                "refused_registration_frees_the_unclaimed_base_and_publishes_nothing",
                "refused_claim_unregisters_then_frees_and_keeps_the_claim_error",
                "allocate_all_materializes_every_planned_base_once_and_first_views_reuse_them",
                "lazy_first_view_materializes_exactly_one_base",
                "strict_owner_mock_rejects_unknown_interior_and_double_frees_and_keeps_failed_bases",
                "ordinary_free_error_in_rollback_keeps_base_live_and_propagates",
                "partial_eager_prefix_failure_retains_claimed_prefix_and_frees_only_the_refused_base",
                "panic_mid_claim_retains_the_registered_owner_and_frees_nothing",
                "materialize_twice_is_refused_without_a_second_allocation",
                "claim_refusal_is_transactional_and_a_rollback_free_error_is_reported"),
        }
        eager_owners = (ROOT / "crates/spark-runtime/src/weights/routed_backing_tests.rs").read_text()
        for name in ("unregister_unclaimed_pops_only_the_latest_live_owner_with_that_base",
                     "unregister_unclaimed_refuses_a_released_latest_owner"):
            self.assertIn("fn " + name, eager_owners)
        mock = (ROOT / "crates/spark-runtime/src/fast_weights/routed_groups_tests.rs").read_text()
        self.assertIn("impl GpuBackend for OwnershipMock", mock)
        strict = mock.split("impl GpuBackend for StrictOwnerMock", 1)[1]
        for fragment in ("strict mock rejects an interior free", "strict mock rejects a double free",
                         "strict mock rejects an unknown free", "injected ordinary free failure",
                         "claimed base requires the grouped free API", "injected claim panic"):
            self.assertIn(fragment, strict)
        self.assertIn("fn tiny_load(count: usize) -> RoutedBackingLoad", mock)
        for method in ("fn alloc(", "fn alloc_managed(", "fn free(", "fn copy_h2d(", "fn copy_d2h(",
                       "fn copy_d2d(", "fn launch(", "fn synchronize(", "fn default_stream(",
                       "fn kernel(", "fn memset(", "fn memset_async(", "fn total_memory(",
                       "fn free_memory(", "fn supports_grouped_backing(", "fn claim_grouped_backing("):
            self.assertIn(method, mock.split("impl GpuBackend for OwnershipMock", 1)[1])
        for path, names in fixtures.items():
            source = (ROOT / path).read_text()
            for name in names:
                self.assertIn("fn " + name, source)
        api = (ROOT / next(iter(fixtures))).read_text()
        self.assertIn("impl GpuBackend for StrictApi", api)
        self.assertIn("let gpu: &dyn GpuBackend", api)
        self.assertIn("prepare_grouped_free(&mut live, base.0)?", api)


if __name__ == "__main__":
    unittest.main()
