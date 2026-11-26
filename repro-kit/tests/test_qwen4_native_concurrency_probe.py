#!/usr/bin/env python3

import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "repro-kit" / "qwen4_native_concurrency_probe.py"
SPEC = importlib.util.spec_from_file_location(
    "qwen4_native_concurrency_probe", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class NativeConcurrencyProbeTests(unittest.TestCase):
    class FakeResponse:
        status = 200

        def __init__(self, lines: list[str]):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            return iter((line + "\n").encode() for line in self.lines)

    @staticmethod
    def state_line(session: str = "00000000000000aa", output_tokens: int = 64) -> str:
        seq_len = 100 + output_tokens - 1
        return (
            "QWEN4_NATIVE_MTP_REQUEST_RECEIPT schema=1 "
            "mode=unloaded-target-only scheduler_path=ordinary-target-only "
            f"proposer_loaded=false session_hash={session} seq_len={seq_len} "
            f"token_count={seq_len} output_tokens={output_tokens} "
            "gdn_layers=36 attention_layers=12 logical_kv_blocks=12 "
            f"kv_token_positions={seq_len * 12} compared_buffers=49 "
            "compared_bytes=123456 "
            "state_scope=live-gdn-ple-plus-logical-bf16-kv "
            "checkpoints_excluded=true fed_tokens_sha256="
            + "c" * 64
            + " target_state_sha256="
            + "d" * 64
        )

    @staticmethod
    def result(*, finish_reason: str = "length", token_ids: list[int] | None = None):
        ids = [7, 11] if token_ids is None else token_ids
        return probe.RequestResult(
            label="row-00",
            token_ids=ids,
            usage={
                "prompt_tokens": 41,
                "completion_tokens": len(ids),
                "total_tokens": 41 + len(ids),
                "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
                "completion_tokens_details": {
                    "reasoning_tokens": 0,
                    "audio_tokens": 0,
                    "accepted_prediction_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "time_to_first_token_ms": 1.0,
                "response_token/s": 2.0,
            },
            finish_reason=finish_reason,
            started_unix=1.0,
            finished_unix=2.0,
        )

    def test_cases_are_distinct_stable_and_short(self) -> None:
        first = probe.make_cases(8)
        second = probe.make_cases(8)
        self.assertEqual(first, second)
        self.assertEqual(len({case.prompt_sha256 for case in first}), 8)
        self.assertEqual(
            [case.label for case in first], [f"row-{i:02d}" for i in range(8)]
        )
        self.assertLessEqual(probe.WAVE_ADMISSION_STAGGER_S * 7, 0.035)

    def test_full_frame_evidence_floor_cannot_be_weakened(self) -> None:
        argv = [
            "qwen4_native_concurrency_probe.py",
            "--base-url",
            "http://127.0.0.1:18899",
            "--model",
            "qwen38",
            "--width",
            "2",
            "--selector-profile",
            "native-reference",
            "--evidence-tier",
            "full",
            "--server-log",
            "server.log",
            "--receipt",
            "receipt.json",
            "--binary",
            "flux-server",
            "--binary-sha256",
            "1" * 64,
            "--manifest",
            "launch.json",
            "--manifest-sha256",
            "2" * 64,
            "--minimum-full-frames",
            "7",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch("sys.stderr"):
            with self.assertRaisesRegex(SystemExit, "2"):
                probe.parse_args()

    def test_native_frame_binds_slots_sessions_indices_and_sha(self) -> None:
        text = "\n".join(
            [
                "QWEN4_NATIVE_BATCH_FRAME rows=2 sessions=00000000000000aa,00000000000000bb slots=7,2 pre_step_seq_lens=41,73 active_only=true target_only=true qsa=false graph=false",
                "LOGITS_FNV session=00000000000000aa idx=1 fnv=1111111111111111 sha256="
                + "a" * 64,
                "LOGITS_FNV session=00000000000000bb idx=1 fnv=2222222222222222 sha256="
                + "b" * 64,
            ]
        )
        parsed = probe.parse_log(text)
        self.assertEqual(len(parsed.frames), 1)
        frame = parsed.frames[0]
        self.assertEqual(frame.route, "native")
        self.assertEqual(frame.slots, [7, 2])
        self.assertEqual(frame.pre_step_seq_lens, [41, 73])
        self.assertEqual(
            [row.session for row in frame.logits],
            ["00000000000000aa", "00000000000000bb"],
        )

    def test_serial_qsa_frame_is_distinct(self) -> None:
        text = "\n".join(
            [
                "QWEN4_SERIAL_BATCH_FRAME rows=2 sessions=00000000000000aa,00000000000000bb slots=0,3 pre_step_seq_lens=2048,3072 selector_on=true qsa_required=true graph=false",
                "LOGITS_FNV session=00000000000000aa idx=4 fnv=1111111111111111 sha256="
                + "a" * 64,
                "LOGITS_FNV session=00000000000000bb idx=4 fnv=2222222222222222 sha256="
                + "b" * 64,
            ]
        )
        frame = probe.parse_log(text).frames[0]
        self.assertEqual(frame.route, "serial")
        self.assertTrue(frame.selector_on)
        self.assertTrue(frame.qsa_required)

    def test_incomplete_or_duplicate_receipts_fail_closed(self) -> None:
        incomplete = "\n".join(
            [
                "QWEN4_NATIVE_BATCH_FRAME rows=2 sessions=00000000000000aa,00000000000000bb slots=0,1 pre_step_seq_lens=7,9 active_only=true target_only=true qsa=false graph=false",
                "LOGITS_FNV session=00000000000000aa idx=1 fnv=1111111111111111 sha256="
                + "a" * 64,
            ]
        )
        with self.assertRaises(probe.ProbeError):
            probe.parse_log(incomplete)

        duplicate = "\n".join(
            [
                "LOGITS_FNV session=00000000000000aa idx=1 fnv=1111111111111111 sha256="
                + "a" * 64,
                "LOGITS_FNV session=00000000000000aa idx=1 fnv=1111111111111111 sha256="
                + "a" * 64,
            ]
        )
        with self.assertRaises(probe.ProbeError):
            probe.parse_log(duplicate)

    def test_frame_session_order_must_match_following_logits_rows(self) -> None:
        text = "\n".join(
            [
                "QWEN4_NATIVE_BATCH_FRAME rows=2 sessions=00000000000000aa,00000000000000bb slots=0,1 pre_step_seq_lens=7,9 active_only=true target_only=true qsa=false graph=false",
                "LOGITS_FNV session=00000000000000bb idx=1 fnv=1111111111111111 sha256="
                + "a" * 64,
                "LOGITS_FNV session=00000000000000aa idx=1 fnv=2222222222222222 sha256="
                + "b" * 64,
            ]
        )
        with self.assertRaisesRegex(probe.ProbeError, "session order"):
            probe.parse_log(text)

    def test_target_state_receipt_is_complete_and_unique(self) -> None:
        parsed = probe.parse_log(self.state_line())
        state = parsed.state_by_session["00000000000000aa"]
        probe.validate_state_receipt("C1", state, 64, 100)
        self.assertEqual(state.gdn_layers, 36)
        self.assertEqual(state.attention_layers, 12)
        self.assertEqual(state.kv_token_positions, state.seq_len * 12)

        with self.assertRaisesRegex(probe.ProbeError, "duplicate target-state"):
            probe.parse_log(self.state_line() + "\n" + self.state_line())

    def test_normal_terminal_keeps_last_sampled_output_pending(self) -> None:
        for output_tokens in (1, 3, 64):
            with self.subTest(output_tokens=output_tokens):
                state = next(
                    iter(
                        probe.parse_log(
                            self.state_line(output_tokens=output_tokens)
                        ).state_by_session.values()
                    )
                )
                self.assertEqual(state.seq_len, 100 + output_tokens - 1)
                probe.validate_state_receipt("normal", state, output_tokens, 100)
                for offset in (-1, 1):
                    wrong = dataclasses.replace(
                        state,
                        seq_len=state.seq_len + offset,
                        token_count=state.token_count + offset,
                        kv_token_positions=(state.seq_len + offset) * 12,
                    )
                    with self.assertRaisesRegex(probe.ProbeError, "token census"):
                        probe.validate_state_receipt(
                            "normal", wrong, output_tokens, 100
                        )

    def test_cancelled_geometry_binds_every_computed_row_to_fed_extent(self) -> None:
        state = next(
            iter(
                probe.parse_log(
                    self.state_line(output_tokens=8)
                ).state_by_session.values()
            )
        )
        rows = {
            index: probe.LogitRow(state.session, index, "a" * 16, f"{index:064x}")
            for index in range(1, 9)
        }
        pending = {index: rows[index] for index in range(1, 8)}
        self.assertEqual(
            probe.validate_cancelled_state_receipt("cancelled", state, 100, pending),
            "pending-output",
        )
        computed = dataclasses.replace(
            state, seq_len=108, token_count=108, kv_token_positions=108 * 12
        )
        self.assertEqual(
            probe.validate_cancelled_state_receipt("cancelled", computed, 100, rows),
            "computed-unemitted",
        )
        for receipt, raw in ((state, rows), (computed, pending)):
            with self.subTest(receipt=receipt.seq_len, row_count=len(raw)):
                with self.assertRaisesRegex(probe.ProbeError, "token census"):
                    probe.validate_cancelled_state_receipt(
                        "cancelled", receipt, 100, raw
                    )
        for indices in ([1, 2, 4, 5, 6, 7, 8], list(range(1, 10))):
            raw = {
                index: probe.LogitRow(state.session, index, "a" * 16, f"{index:064x}")
                for index in indices
            }
            with self.assertRaisesRegex(probe.ProbeError, "prefix is incomplete"):
                probe.validate_cancelled_state_receipt("cancelled", computed, 100, raw)

        frame = probe.Frame(
            "native", 1, [state.session], [7], [107], None, None, [rows[8]], 10
        )
        probe.validate_cancelled_frame_extent("cancelled", [frame], computed, rows)
        for wrong in (
            dataclasses.replace(frame, pre_step_seq_lens=[108]),
            dataclasses.replace(frame, logits=[rows[7]]),
        ):
            with self.assertRaisesRegex(
                probe.ProbeError, "does not bind target-fed extent"
            ):
                probe.validate_cancelled_frame_extent(
                    "cancelled", [wrong], computed, rows
                )

    def test_cancelled_helper_preserves_common_mode_and_coverage_guards(self) -> None:
        state = next(
            iter(
                probe.parse_log(
                    self.state_line(output_tokens=8)
                ).state_by_session.values()
            )
        )
        rows = {
            index: probe.LogitRow(state.session, index, "a" * 16, "b" * 64)
            for index in range(1, 8)
        }
        for wrong in (
            dataclasses.replace(state, mode="engaged-native-mtp", proposer_loaded=True),
            dataclasses.replace(state, compared_bytes=0),
            dataclasses.replace(state, kv_token_positions=1),
        ):
            with self.assertRaises(probe.ProbeError):
                probe.validate_cancelled_state_receipt("cancelled", wrong, 100, rows)

    def test_production_faithful_wave_uses_pre_forward_not_post_forward_lengths(
        self,
    ) -> None:
        cases = probe.make_cases(2)
        sessions = {case.label: f"{index + 1:016x}" for index, case in enumerate(cases)}
        lines = []
        for index in (1, 2):
            lines.append(
                "QWEN4_NATIVE_BATCH_FRAME rows=2 sessions="
                + ",".join(sessions.values())
                + f" slots=0,1 pre_step_seq_lens={99 + index},{99 + index} "
                "active_only=true target_only=true qsa=false graph=false"
            )
            for session in sessions.values():
                lines.append(
                    f"LOGITS_FNV session={session} idx={index} "
                    f"fnv={index:016x} sha256={index:064x}"
                )
        lines.extend(self.state_line(session, 3) for session in sessions.values())
        parsed = probe.parse_log("\n".join(lines))
        results = []
        for case in cases:
            result = dataclasses.replace(
                self.result(token_ids=[7, 11, 13]), label=case.label
            )
            result.usage["prompt_tokens"] = 100
            result.usage["total_tokens"] = 103
            results.append(result)

        baseline_states = {
            case.label: parsed.state_by_session[sessions[case.label]] for case in cases
        }

        def validate(observed):
            return probe.validate_wave(
                "forward",
                "native",
                2,
                cases,
                results,
                observed,
                {result.label: result for result in results},
                sessions,
                {
                    case.label: probe.row_digest_map(
                        parsed.rows_by_session[sessions[case.label]]
                    )
                    for case in cases
                },
                baseline_states,
                2,
            )

        result = validate(parsed)
        self.assertEqual(result["frames"][0]["pre_step_seq_lens"], [100, 100])
        for offset in (-1, 1):
            wrong = dataclasses.replace(
                parsed,
                frames=[
                    dataclasses.replace(
                        frame,
                        pre_step_seq_lens=[
                            value + offset for value in frame.pre_step_seq_lens
                        ],
                    )
                    for frame in parsed.frames
                ],
            )
            with self.assertRaisesRegex(probe.ProbeError, "sequence length differs"):
                validate(wrong)

    def test_strict_json_rejects_duplicates_and_nonfinite_values(self) -> None:
        with self.assertRaisesRegex(probe.ProbeError, "duplicate key"):
            probe.json_loads_strict('{"a": 1, "a": 2}', "fixture")
        with self.assertRaisesRegex(probe.ProbeError, "non-finite"):
            probe.json_loads_strict('{"a": NaN}', "fixture")
        with self.assertRaisesRegex(probe.ProbeError, "non-finite"):
            probe.json_loads_strict('{"a": 1e309}', "fixture")

    def test_every_malformed_evidence_marker_fails_closed(self) -> None:
        malformed = (
            "QWEN4_NATIVE_BATCH_FRAME rows=2 sessions=00000000000000aa",
            "LOGITS_FNV",
            "LOGITS_FNV session=00000000000000aa idx=1 fnv=1111111111111111 sha256="
            + "a" * 63
            + "z",
            self.state_line() + " trailing-garbage",
            "QWEN4_NATIVE_BATCH_ENGAGED rows=2 active_only=true target_only=true",
            "ENGAGED ATLAS_ATTN_QKV_EXACT_STRIDED multi_seq M=2",
            "QWEN4_NATIVE_BATCH_ENGAGED QWEN4_NATIVE_BATCH_ENGAGED rows=2 "
            "active_only=true target_only=true qsa=false",
            "QWEN4_NATIVE_MTP_REQUEST_RECEIPT_FAILED reason=hash",
        )
        for line in malformed:
            with self.subTest(line=line[:50]):
                with self.assertRaisesRegex(probe.ProbeError, "malformed|multiple"):
                    probe.validate_evidence_line_syntax(line)

    def test_r15_source_closure_uses_full_validator_and_restores_release(self) -> None:
        original = probe.r9_provenance.RELEASE_VERSION
        with mock.patch.object(
            probe.r9_provenance, "configure_release"
        ) as configure, mock.patch.object(
            probe.r9_provenance, "validate_qualification_provenance"
        ) as validate:
            provenance = {"release_version": probe.RELEASE_VERSION}
            probe.validate_r15_source_closure(provenance)
        self.assertEqual(
            configure.call_args_list,
            [mock.call(probe.RELEASE_VERSION), mock.call(original)],
        )
        validate.assert_called_once_with(provenance)

    def test_launch_example_contains_exact_r15_source_closure(self) -> None:
        path = (
            ROOT
            / "repro-kit"
            / "examples"
            / "qwen4-native-concurrency-launch.example.json"
        )
        value = probe.json_loads_strict(path.read_bytes(), "launch example")
        self.assertEqual(value["schema"], probe.LAUNCH_SCHEMA)
        self.assertEqual(set(value["provenance"]), probe.PROVENANCE_KEYS)
        self.assertEqual(value["provenance"]["release_version"], probe.RELEASE_VERSION)

    def test_public_checkpoint_id_grammar_rejects_path_like_values(self) -> None:
        self.assertIsNotNone(
            probe.PUBLIC_MODEL_ID_RE.fullmatch(
                "hf://JMNI-Labs/Qwen3.8-Flash-Next-NVFP4-QAD5500-Hybrid"
            )
        )
        for checkpoint_id in (
            "/home/private/checkpoint",
            "home/private",
            "home/user/private/checkpoint",
            "../private",
            "owner/model",
            "owner/model/extra",
            "owner\\model",
            "owner model/name",
        ):
            with self.subTest(checkpoint_id=checkpoint_id):
                self.assertIsNone(probe.PUBLIC_MODEL_ID_RE.fullmatch(checkpoint_id))

    def test_public_model_alias_grammar_rejects_path_like_values(self) -> None:
        self.assertEqual(
            probe.public_model_name("qwen38-qad5500", "model"), "qwen38-qad5500"
        )
        for model in (
            "/home/private/model",
            "home/private",
            "../model",
            "owner/model",
            r"C:\\Users\\private\\model",
        ):
            with self.subTest(model=model):
                with self.assertRaisesRegex(probe.ProbeError, "path-free"):
                    probe.public_model_name(model, "model")

    def test_path_like_model_is_rejected_before_dry_run_can_emit_it(self) -> None:
        argv = [
            "qwen4_native_concurrency_probe.py",
            "--base-url",
            "http://127.0.0.1:18899",
            "--model",
            "/home/private/private-model",
            "--width",
            "2",
            "--selector-profile",
            "native-reference",
            "--evidence-tier",
            "full",
            "--server-log",
            "/private/server.log",
            "--receipt",
            "/private/receipt.json",
            "--binary",
            "/private/flux-server",
            "--binary-sha256",
            "1" * 64,
            "--manifest",
            "/private/launch.json",
            "--manifest-sha256",
            "2" * 64,
            "--dry-run",
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch(
            "sys.stdout", stdout
        ), mock.patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit) as raised:
                probe.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn("/home/private/private-model", stderr.getvalue())
        self.assertIn("path-free public API model alias", stderr.getvalue())

    @staticmethod
    def terminal_usage(completion_tokens: object = 2) -> dict:
        return {
            "prompt_tokens": 41,
            "completion_tokens": completion_tokens,
            "total_tokens": 43,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }

    def run_stream_fixture(self, lines: list[str]) -> probe.RequestResult:
        response = self.FakeResponse(lines)
        with mock.patch.object(probe.urllib.request, "urlopen", return_value=response):
            return probe.stream_request(
                "http://127.0.0.1:18899",
                "model",
                probe.make_cases(1)[0],
                2,
                5.0,
            )

    def valid_sse_lines(self) -> list[str]:
        return [
            "data: "
            + json.dumps({"choices": [{"index": 0, "delta": {"content": "x"}}]}),
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "length",
                            "token_ids": [7, 11],
                        }
                    ]
                }
            ),
            "data: " + json.dumps({"choices": [], "usage": self.terminal_usage()}),
            "data: [DONE]",
        ]

    def test_sse_requires_exact_terminal_sequence_and_usage_arithmetic(self) -> None:
        result = self.run_stream_fixture(self.valid_sse_lines())
        self.assertEqual(result.token_ids, [7, 11])

        hostile = {}
        hostile["missing done"] = self.valid_sse_lines()[:-1]
        hostile["duplicate done"] = self.valid_sse_lines() + ["data: [DONE]"]
        hostile["data after done"] = self.valid_sse_lines() + ['data: {"choices":[]}']
        missing_usage = self.valid_sse_lines()
        del missing_usage[-2]
        hostile["missing usage"] = missing_usage
        early_usage = self.valid_sse_lines()
        early_usage.insert(
            1,
            "data: " + json.dumps({"choices": [], "usage": self.terminal_usage()}),
        )
        hostile["duplicate usage"] = early_usage
        missing_finish = self.valid_sse_lines()
        finish_value = json.loads(missing_finish[1].removeprefix("data: "))
        del finish_value["choices"][0]["finish_reason"]
        missing_finish[1] = "data: " + json.dumps(finish_value)
        hostile["missing finish"] = missing_finish
        missing_ids = self.valid_sse_lines()
        token_value = json.loads(missing_ids[1].removeprefix("data: "))
        del token_value["choices"][0]["token_ids"]
        missing_ids[1] = "data: " + json.dumps(token_value)
        hostile["missing token ids"] = missing_ids
        early_ids = self.valid_sse_lines()
        early_ids[0] = "data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {"content": "x"}, "token_ids": [7, 11]}]}
        )
        hostile["early token ids"] = early_ids
        second_finish = self.valid_sse_lines()
        second_finish.insert(1, second_finish[1])
        hostile["duplicate finish"] = second_finish
        bool_usage = self.valid_sse_lines()
        bool_usage[-2] = "data: " + json.dumps(
            {"choices": [], "usage": self.terminal_usage(True)}
        )
        hostile["bool usage"] = bool_usage
        float_usage = self.valid_sse_lines()
        float_usage[-2] = "data: " + json.dumps(
            {"choices": [], "usage": self.terminal_usage(2.0)}
        )
        hostile["float usage"] = float_usage
        bad_arithmetic = self.valid_sse_lines()
        usage = self.terminal_usage()
        usage["total_tokens"] = 44
        bad_arithmetic[-2] = "data: " + json.dumps({"choices": [], "usage": usage})
        hostile["bad arithmetic"] = bad_arithmetic
        for name, lines in hostile.items():
            with self.subTest(name=name):
                with self.assertRaises(probe.ProbeError):
                    self.run_stream_fixture(lines)

    def test_log_cursor_hash_range_is_byte_exact_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_bytes(b"alpha\n")
            cursor = probe.LogCursor(path)
            end = cursor.mark()
            path.write_bytes(b"alpha\nbeta\n")
            self.assertEqual(
                cursor.read_bytes_from(0, settle_seconds=0.0, end_offset=end),
                b"alpha\n",
            )
            with self.assertRaisesRegex(probe.ProbeError, "invalid UTF-8"):
                path.write_bytes(b"alpha\n\xff")
                cursor.read_from(0, settle_seconds=0.0)
            with self.assertRaisesRegex(probe.ProbeError, "truncated"):
                path.write_bytes(b"x")
                cursor.read_bytes_from(end, settle_seconds=0.0)

    def test_frame_ownership_requires_stable_slots_and_retirement_before_reuse(
        self,
    ) -> None:
        def frame(session: str, slot: int, line: int, seq_len: int) -> probe.Frame:
            return probe.Frame(
                route="native",
                rows=1,
                sessions=[session],
                slots=[slot],
                pre_step_seq_lens=[seq_len],
                selector_on=None,
                qsa_required=None,
                logits=[],
                line_number=line,
            )

        def state(session: str, line: int) -> probe.StateReceipt:
            return probe.StateReceipt(
                session=session,
                schema=1,
                mode="unloaded-target-only",
                scheduler_path="ordinary-target-only",
                proposer_loaded=False,
                seq_len=100,
                token_count=100,
                output_tokens=10,
                gdn_layers=36,
                attention_layers=12,
                logical_kv_blocks=7,
                kv_token_positions=1200,
                compared_buffers=49,
                compared_bytes=1,
                fed_tokens_sha256="a" * 64,
                target_state_sha256="b" * 64,
                line_number=line,
            )

        aa = "00000000000000aa"
        bb = "00000000000000bb"
        with self.assertRaisesRegex(probe.ProbeError, "moved physical slots"):
            probe.validate_frame_ownership(
                "wave",
                [frame(aa, 0, 1, 10), frame(aa, 1, 2, 11)],
                {aa: state(aa, 3)},
                allow_retired_slot_reuse=False,
            )

        slots = probe.validate_frame_ownership(
            "churn",
            [frame(aa, 0, 1, 10), frame(bb, 0, 3, 20)],
            {aa: state(aa, 2), bb: state(bb, 4)},
            allow_retired_slot_reuse=True,
        )
        self.assertEqual(slots, {aa: 0, bb: 0})

        with self.assertRaisesRegex(probe.ProbeError, "changed live owner"):
            probe.validate_frame_ownership(
                "churn",
                [frame(aa, 0, 1, 10), frame(bb, 0, 3, 20)],
                {aa: state(aa, 4), bb: state(bb, 5)},
                allow_retired_slot_reuse=True,
            )

    def test_exact_budget_stop_is_admitted_and_must_match_c1(self) -> None:
        self.assertEqual(probe.EXACT_BUDGET_FINISH_REASONS, {"length", "stop"})
        baseline = self.result(finish_reason="stop")
        observed = self.result(finish_reason="stop")
        observed.usage["time_to_first_token_ms"] = 99.0
        observed.usage["response_token/s"] = 101.0
        probe.validate_request_identity("forward/row-00", observed, baseline)

        with self.assertRaisesRegex(probe.ProbeError, "finish reason differs from C1"):
            probe.validate_request_identity(
                "forward/row-00", self.result(finish_reason="length"), baseline
            )

    def test_c1_identity_includes_nontiming_usage_and_token_ids(self) -> None:
        baseline = self.result()
        wrong_usage = self.result()
        wrong_usage.usage["prompt_tokens"] = 42
        with self.assertRaisesRegex(probe.ProbeError, "usage.prompt_tokens differs"):
            probe.validate_request_identity("forward/row-00", wrong_usage, baseline)

        with self.assertRaisesRegex(probe.ProbeError, "token IDs differ"):
            probe.validate_request_identity(
                "forward/row-00", self.result(token_ids=[7, 12]), baseline
            )

    def test_cancel_signal_fires_after_response_context_closes(self) -> None:
        class FakeResponse:
            status = 200

            def __init__(self):
                self.closed = False

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.closed = True

            def __iter__(self):
                for _ in range(8):
                    yield (
                        "data: "
                        + json.dumps({"choices": [{"delta": {"content": "token"}}]})
                        + "\n"
                    ).encode()

        class CloseCheckingEvent:
            def __init__(self, response):
                self.response = response
                self.was_set = False

            def set(self):
                self.assert_closed()
                self.was_set = True

            def assert_closed(self):
                if not self.response.closed:
                    raise AssertionError("cancellation signalled before response close")

        response = FakeResponse()
        event = CloseCheckingEvent(response)
        with mock.patch.object(probe.urllib.request, "urlopen", return_value=response):
            result = probe.stream_then_cancel(
                "http://127.0.0.1:18899",
                "model",
                probe.make_cases(1)[0],
                64,
                5.0,
                8,
                event,
            )
        self.assertTrue(event.was_set)
        self.assertTrue(response.closed)
        self.assertEqual(result.content_chunks, 8)

    def test_cancel_handoff_requires_exact_boundary_and_closed_order(self) -> None:
        replacement = self.result()
        replacement.started_unix = 10.0
        probe.validate_cancel_handoff(
            probe.CancelResult("row-00", 8, 1.0, 10.0), replacement, 8
        )

        with self.assertRaisesRegex(probe.ProbeError, "exactly 8 content chunks"):
            probe.validate_cancel_handoff(
                probe.CancelResult("row-00", 9, 1.0, 9.0), replacement, 8
            )
        with self.assertRaisesRegex(probe.ProbeError, "started before"):
            probe.validate_cancel_handoff(
                probe.CancelResult("row-00", 8, 1.0, 10.1), replacement, 8
            )


if __name__ == "__main__":
    unittest.main()
