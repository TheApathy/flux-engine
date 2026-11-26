#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""TLA+-style explicit-state temporal model checker for Flux v0.9.17 Scheduler.

Exhaustively explores the reachable state graph (`BFS` deadlock/liveness
checking) of concurrent multi-stream prefill, multi-rank prefix restore
(`agree_restore`), speculative `pending_drafts` decode, Wave-15 GPU `q_row`
splice/truncation invalidation, code-fence thinking runway enforcement, and
mid-stream client SSE disconnect cleanup.

Checks 8 safety invariants (`[]Inv` / `[][Action]_vars`) on EVERY reachable state
and transition plus 1 liveness property (`<>AllStreamsRetired` / deadlock-freedom
on all non-terminal states):
  1. `Inv_NoMixedBatchUnderSpec`: `mixed_forward_batch` never fires when spec is active
     or any active stream holds `pending_drafts`.
  2. `Inv_UniformMultiRankPrefix`: Rank 0 and Rank 1 always agree on restored prefix
     offset before entering collective prefill.
  3. `Inv_NoSlotFreedBeforeStreamDrain`: A disconnected sequence's KV/SSM slot is
     never freed before both producer and consumer streams are synchronized.
  4. `Inv_FenceRunwayRespected`: `in_code_fence` never suppresses `</think>` once
     `emitted >= max_tokens - min_runway`.
  5. `Inv_SingleSeqExhaustionNeverAbortsPeer`: Single-sequence KV/budget exhaustion
     retires only the exhausted stream and never aborts or mutates a concurrent peer.
  6. `Inv_NoPendingQRowClobber`: Slot-isolated `q_row` buffers (`slot * 8 .. +8`)
     are pairwise disjoint across concurrent streams and never overwritten while pending.
  7. `Inv_SpecGreedyFenceRunwayRespected`: Speculative verify steps honor the
     fence-deferred `</think>` runway override and never skip past `max_tokens - min_runway`.
  8. `Inv_SplicedDraftNeverReadsStaleQRow`: Splicing or truncating a pending draft frame
     immediately invalidates `PendingGpuRejectionQ` (`pending_q_row_base == -1`), so a
     mutated draft never reads a stale GPU `q_row` during maximal-coupling verify.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import sys


# Per-stream lifecycle phases
IDLE = "IDLE"
WAIT_PREFIX_AGREE = "WAIT_PREFIX_AGREE"
PREFILLING = "PREFILLING"
ACTIVE = "ACTIVE"
DISCONNECTED_UNDRAINED = "DISCONNECTED_UNDRAINED"
RETIRED = "RETIRED"

MAX_TOKENS = 4
MIN_RUNWAY = 2  # for MAX_TOKENS=4: min(4//2, 256) = 2 -> forced </think> at emitted == 2
QROW_STRIDE = 8  # Wave-15 slot-isolated q_row span: [slot * 8, slot * 8 + 8)


@dataclass(frozen=True)
class StreamState:
    phase: str
    r0_snap: int
    r1_snap: int
    agreed_prefix: int
    pending_drafts: bool
    draft_spliced: bool
    pending_q_row_base: int
    in_think: bool
    in_fence: bool
    emitted: int
    slot_allocated: bool
    streams_drained: bool


@dataclass(frozen=True)
class SystemState:
    spec_enabled: bool
    s0: StreamState
    s1: StreamState
    last_action: str


def can_batch_mixed_step(spec_enabled: bool, prefilling_count: int, active_count: int) -> bool:
    # Exact gate from crates/spark-server/src/scheduler/phase_continue_prefills.rs
    return prefilling_count >= 2 and active_count > 0 and not spec_enabled


def check_invariants(st: SystemState) -> None:
    streams = (st.s0, st.s1)
    # Invariant 1: mixed_forward_batch never executes when spec_enabled or pending_drafts exist
    if st.last_action == "mixed_forward_batch":
        if st.spec_enabled or any(s.pending_drafts for s in streams if s.phase == ACTIVE):
            raise AssertionError(f"Inv_NoMixedBatchUnderSpec violated in {st}")

    # Invariant 6 (state part): Inv_NoPendingQRowClobber — slot-isolated q_row ranges are
    # pairwise disjoint across concurrent streams and strictly tied to active `pending_drafts`.
    if st.s0.pending_q_row_base >= 0 and st.s1.pending_q_row_base >= 0:
        if abs(st.s0.pending_q_row_base - st.s1.pending_q_row_base) < QROW_STRIDE:
            raise AssertionError(f"Inv_NoPendingQRowClobber violated (cross-slot overlap!): {st}")

    for slot_id, s in enumerate(streams):
        # Invariant 2: once in PREFILLING, agreed_prefix <= min(r0_snap, r1_snap)
        if s.phase == PREFILLING:
            if s.agreed_prefix > min(s.r0_snap, s.r1_snap) or s.agreed_prefix < 0:
                raise AssertionError(f"Inv_UniformMultiRankPrefix violated in {st}")

        # Invariant 3: RETIRED slot must be freed, and freeing after disconnect requires streams_drained
        if s.phase == RETIRED and s.slot_allocated:
            raise AssertionError(f"Slot leaked in RETIRED state: {st}")
        if s.phase == DISCONNECTED_UNDRAINED and not s.slot_allocated:
            raise AssertionError(f"Inv_NoSlotFreedBeforeStreamDrain violated (freed before drain!): {st}")

        # Invariant 4: in_think cannot persist past MAX_TOKENS - MIN_RUNWAY
        # (at emitted == MAX_TOKENS - MIN_RUNWAY == 2, next step MUST close in_think even if in_fence!)
        if s.phase == ACTIVE and s.in_think and s.emitted > (MAX_TOKENS - MIN_RUNWAY):
            raise AssertionError(f"Inv_FenceRunwayRespected violated (thinking overran runway!): {st}")

        # Invariant 6 (per-stream ownership part): Inv_NoPendingQRowClobber
        expected_q_base = slot_id * QROW_STRIDE if (s.pending_drafts and not s.draft_spliced) else -1
        if s.pending_q_row_base != expected_q_base or (s.pending_drafts and s.phase != ACTIVE):
            raise AssertionError(f"Inv_NoPendingQRowClobber violated (bad q_row ownership!): {st}")

        # Invariant 7 (state part): Inv_SpecGreedyFenceRunwayRespected
        if st.spec_enabled and s.phase == ACTIVE and s.in_think and s.emitted > (MAX_TOKENS - MIN_RUNWAY):
            raise AssertionError(
                f"Inv_SpecGreedyFenceRunwayRespected violated (spec greedy overran runway!): {st}"
            )

        # Invariant 8 (state part): Inv_SplicedDraftNeverReadsStaleQRow
        if s.draft_spliced and (not s.pending_drafts or s.pending_q_row_base != -1):
            raise AssertionError(
                f"Inv_SplicedDraftNeverReadsStaleQRow violated (spliced draft kept stale q_row!): {st}"
            )


def check_transition_invariants(prev: SystemState, nxt: SystemState) -> None:
    # Invariant 5: Inv_SingleSeqExhaustionNeverAbortsPeer
    if nxt.last_action == "s0:retire_kv_exhausted":
        if nxt.s0.phase != RETIRED or nxt.s0.slot_allocated or nxt.s1 != prev.s1:
            raise AssertionError(
                f"Inv_SingleSeqExhaustionNeverAbortsPeer violated on s0 exhaustion: prev={prev} nxt={nxt}"
            )
    elif nxt.last_action == "s1:retire_kv_exhausted":
        if nxt.s1.phase != RETIRED or nxt.s1.slot_allocated or nxt.s0 != prev.s0:
            raise AssertionError(
                f"Inv_SingleSeqExhaustionNeverAbortsPeer violated on s1 exhaustion: prev={prev} nxt={nxt}"
            )

    # Invariant 6 (transition part): Inv_NoPendingQRowClobber — never re-propose over pending q_rows
    if nxt.last_action == "s0:propose_drafts":
        if (
            prev.s0.pending_q_row_base != -1
            or prev.s0.pending_drafts
            or nxt.s0.draft_spliced
            or nxt.s1.pending_q_row_base != prev.s1.pending_q_row_base
        ):
            raise AssertionError(
                f"Inv_NoPendingQRowClobber violated on s0 propose: prev={prev} nxt={nxt}"
            )
    elif nxt.last_action == "s1:propose_drafts":
        if (
            prev.s1.pending_q_row_base != -1
            or prev.s1.pending_drafts
            or nxt.s1.draft_spliced
            or nxt.s0.pending_q_row_base != prev.s0.pending_q_row_base
        ):
            raise AssertionError(
                f"Inv_NoPendingQRowClobber violated on s1 propose: prev={prev} nxt={nxt}"
            )

    # Invariant 7 & 8 (transition parts):
    # - Inv_SpecGreedyFenceRunwayRespected: at/beyond `MAX_TOKENS - MIN_RUNWAY`, spec verify closes `</think>`.
    # - Inv_SplicedDraftNeverReadsStaleQRow: splice/truncation clears `pending_q_row_base` and
    #   GPU `q_row` verify never executes on a spliced or q_row-invalidated draft.
    for slot_id, tag, prev_s, nxt_s, prev_peer, nxt_peer in (
        (0, "s0:", prev.s0, nxt.s0, prev.s1, nxt.s1),
        (1, "s1:", prev.s1, nxt.s1, prev.s0, nxt.s0),
    ):
        if not nxt.last_action.startswith(tag):
            continue
        act = nxt.last_action[len(tag) :]
        if (
            prev.spec_enabled
            and prev_s.phase == ACTIVE
            and prev_s.in_think
            and prev_s.emitted >= (MAX_TOKENS - MIN_RUNWAY)
            and act.startswith(("spec_greedy_verify", "spec_qrow_verify", "emit_", "inject_think_end"))
            and nxt_s.in_think
        ):
            raise AssertionError(
                f"Inv_SpecGreedyFenceRunwayRespected violated on {nxt.last_action}: prev={prev} nxt={nxt}"
            )
        if act == "splice_drafts":
            if (
                not prev_s.pending_drafts
                or prev_s.draft_spliced
                or not nxt_s.pending_drafts
                or not nxt_s.draft_spliced
                or nxt_s.pending_q_row_base != -1
                or nxt_peer != prev_peer
            ):
                raise AssertionError(
                    f"Inv_SplicedDraftNeverReadsStaleQRow violated on {nxt.last_action}: prev={prev} nxt={nxt}"
                )
        elif act == "truncate_drafts_invalidate_q":
            if (
                nxt_s.pending_drafts
                or nxt_s.draft_spliced
                or nxt_s.pending_q_row_base != -1
                or nxt_peer != prev_peer
            ):
                raise AssertionError(
                    f"Inv_SplicedDraftNeverReadsStaleQRow violated on {nxt.last_action}: prev={prev} nxt={nxt}"
                )
        elif act.startswith("spec_qrow_verify"):
            if (
                not prev_s.pending_drafts
                or prev_s.draft_spliced
                or prev_s.pending_q_row_base != slot_id * QROW_STRIDE
                or nxt_s.pending_drafts
                or nxt_s.draft_spliced
                or nxt_s.pending_q_row_base != -1
            ):
                raise AssertionError(
                    f"Inv_SplicedDraftNeverReadsStaleQRow violated on {nxt.last_action}: prev={prev} nxt={nxt}"
                )


def step_single_stream(s: StreamState, slot_id: int, spec_enabled: bool) -> list[tuple[str, StreamState]]:
    out: list[tuple[str, StreamState]] = []
    if s.phase == IDLE:
        # Admit with divergent multi-rank resident snapshots (e.g., (2, 1) or (1, 0))
        for r0, r1 in ((2, 1), (1, 1), (1, 0)):
            out.append(
                (
                    "admit",
                    StreamState(
                        phase=WAIT_PREFIX_AGREE,
                        r0_snap=r0,
                        r1_snap=r1,
                        agreed_prefix=-1,
                        pending_drafts=False,
                        draft_spliced=False,
                        pending_q_row_base=-1,
                        in_think=True,
                        in_fence=False,
                        emitted=0,
                        slot_allocated=True,
                        streams_drained=False,
                    ),
                )
            )
    elif s.phase == WAIT_PREFIX_AGREE:
        agreed = min(s.r0_snap, s.r1_snap)
        out.append(("agree_restore", StreamState(**{**s.__dict__, "phase": PREFILLING, "agreed_prefix": agreed})))
        out.append(("disconnect_prefill", StreamState(**{**s.__dict__, "phase": DISCONNECTED_UNDRAINED})))
    elif s.phase == PREFILLING:
        out.append(("promote_active", StreamState(**{**s.__dict__, "phase": ACTIVE})))
        out.append(("disconnect_prefill", StreamState(**{**s.__dict__, "phase": DISCONNECTED_UNDRAINED})))
    elif s.phase == ACTIVE:
        if s.emitted >= MAX_TOKENS:
            out.append(
                (
                    "finish_eos",
                    StreamState(
                        **{
                            **s.__dict__,
                            "phase": RETIRED,
                            "pending_drafts": False,
                            "draft_spliced": False,
                            "pending_q_row_base": -1,
                            "slot_allocated": False,
                            "streams_drained": True,
                        }
                    ),
                )
            )
        else:
            # Disconnect mid-decode
            out.append(
                (
                    "disconnect_decode",
                    StreamState(
                        **{
                            **s.__dict__,
                            "phase": DISCONNECTED_UNDRAINED,
                            "pending_drafts": False,
                            "draft_spliced": False,
                            "pending_q_row_base": -1,
                        }
                    ),
                )
            )

            # Single-sequence KV/budget exhaustion retires ONLY this stream, never its peer
            out.append(
                (
                    "retire_kv_exhausted",
                    StreamState(
                        **{
                            **s.__dict__,
                            "phase": RETIRED,
                            "pending_drafts": False,
                            "draft_spliced": False,
                            "pending_q_row_base": -1,
                            "slot_allocated": False,
                            "streams_drained": True,
                        }
                    ),
                )
            )

            # Speculative draft proposal if spec_enabled and not already holding drafts
            if spec_enabled and not s.pending_drafts:
                out.append(
                    (
                        "propose_drafts",
                        StreamState(
                            **{
                                **s.__dict__,
                                "pending_drafts": True,
                                "draft_spliced": False,
                                "pending_q_row_base": slot_id * QROW_STRIDE,
                            }
                        ),
                    )
                )

            # Wave-15 splice & prefix-truncation invalidation of `PendingGpuRejectionQ`
            if spec_enabled and s.pending_drafts and not s.draft_spliced:
                out.append(
                    (
                        "splice_drafts",
                        StreamState(
                            **{
                                **s.__dict__,
                                "draft_spliced": True,
                                "pending_q_row_base": -1,
                            }
                        ),
                    )
                )
                out.append(
                    (
                        "truncate_drafts_invalidate_q",
                        StreamState(
                            **{
                                **s.__dict__,
                                "pending_drafts": False,
                                "draft_spliced": False,
                                "pending_q_row_base": -1,
                            }
                        ),
                    )
                )

            # Decode / verify emit step with fence-deferred </think> + hard runway override
            runway_override = s.emitted >= (MAX_TOKENS - MIN_RUNWAY)

            # Wave-15 GPU q_row maximal-coupling verify: enabled ONLY when PendingGpuRejectionQ is valid (unspliced!)
            if spec_enabled and s.pending_drafts and s.pending_q_row_base >= 0:
                if s.in_think:
                    if not runway_override:
                        out.append(
                            (
                                "spec_qrow_verify",
                                StreamState(
                                    **{
                                        **s.__dict__,
                                        "in_fence": not s.in_fence,
                                        "emitted": s.emitted + 1,
                                        "pending_drafts": False,
                                        "draft_spliced": False,
                                        "pending_q_row_base": -1,
                                    }
                                ),
                            )
                        )
                    if (not s.in_fence) or runway_override:
                        out.append(
                            (
                                "spec_qrow_verify",
                                StreamState(
                                    **{
                                        **s.__dict__,
                                        "in_think": False,
                                        "in_fence": False,
                                        "emitted": s.emitted + 1,
                                        "pending_drafts": False,
                                        "draft_spliced": False,
                                        "pending_q_row_base": -1,
                                    }
                                ),
                            )
                        )
                else:
                    out.append(
                        (
                            "spec_qrow_verify",
                            StreamState(
                                **{
                                    **s.__dict__,
                                    "emitted": min(MAX_TOKENS, s.emitted + 2),
                                    "pending_drafts": False,
                                    "draft_spliced": False,
                                    "pending_q_row_base": -1,
                                }
                            ),
                        )
                    )

            # Speculative greedy multi-token verify step honoring fence-deferred </think> runway
            if spec_enabled and s.pending_drafts:
                if s.in_think:
                    if not runway_override:
                        step_gain = min(2, (MAX_TOKENS - MIN_RUNWAY) - s.emitted)
                        out.append(
                            (
                                "spec_greedy_verify_think",
                                StreamState(
                                    **{
                                        **s.__dict__,
                                        "in_fence": not s.in_fence,
                                        "emitted": s.emitted + step_gain,
                                        "pending_drafts": False,
                                        "draft_spliced": False,
                                        "pending_q_row_base": -1,
                                    }
                                ),
                            )
                        )
                    if (not s.in_fence) or runway_override:
                        out.append(
                            (
                                "spec_greedy_verify_force_think_end",
                                StreamState(
                                    **{
                                        **s.__dict__,
                                        "in_think": False,
                                        "in_fence": False,
                                        "emitted": s.emitted + 1,
                                        "pending_drafts": False,
                                        "draft_spliced": False,
                                        "pending_q_row_base": -1,
                                    }
                                ),
                            )
                        )
                else:
                    out.append(
                        (
                            "spec_greedy_verify_answer",
                            StreamState(
                                **{
                                    **s.__dict__,
                                    "emitted": min(MAX_TOKENS, s.emitted + 2),
                                    "pending_drafts": False,
                                    "draft_spliced": False,
                                    "pending_q_row_base": -1,
                                }
                            ),
                        )
                    )

            if s.in_think:
                # Model may toggle code fence if below runway
                if not runway_override:
                    out.append(
                        (
                            "emit_think_toggle_fence",
                            StreamState(
                                **{
                                    **s.__dict__,
                                    "in_fence": not s.in_fence,
                                    "emitted": s.emitted + 1,
                                    "pending_drafts": False,
                                    "draft_spliced": False,
                                    "pending_q_row_base": -1,
                                }
                            ),
                        )
                    )
                # Forced </think> injection fires when !in_fence or runway_override
                if (not s.in_fence) or runway_override:
                    out.append(
                        (
                            "inject_think_end",
                            StreamState(
                                **{
                                    **s.__dict__,
                                    "in_think": False,
                                    "in_fence": False,
                                    "emitted": s.emitted + 1,
                                    "pending_drafts": False,
                                    "draft_spliced": False,
                                    "pending_q_row_base": -1,
                                }
                            ),
                        )
                    )
            else:
                out.append(
                    (
                        "emit_answer_token",
                        StreamState(
                            **{
                                **s.__dict__,
                                "emitted": s.emitted + 1,
                                "pending_drafts": False,
                                "draft_spliced": False,
                                "pending_q_row_base": -1,
                            }
                        ),
                    )
                )
    elif s.phase == DISCONNECTED_UNDRAINED:
        if not s.streams_drained:
            out.append(("drain_producer_and_default_streams", StreamState(**{**s.__dict__, "streams_drained": True})))
        else:
            out.append(("free_drained_slot", StreamState(**{**s.__dict__, "phase": RETIRED, "slot_allocated": False})))
    return out


def next_states(st: SystemState) -> list[SystemState]:
    succ: list[SystemState] = []
    for act, s0_next in step_single_stream(st.s0, 0, st.spec_enabled):
        succ.append(SystemState(st.spec_enabled, s0_next, st.s1, f"s0:{act}"))
    for act, s1_next in step_single_stream(st.s1, 1, st.spec_enabled):
        succ.append(SystemState(st.spec_enabled, st.s0, s1_next, f"s1:{act}"))

    # Scheduler-level mixed_forward_batch check (with 2 prefilling + >=1 active generalized check)
    pref_cnt = sum(1 for s in (st.s0, st.s1) if s.phase == PREFILLING)
    act_cnt = sum(1 for s in (st.s0, st.s1) if s.phase == ACTIVE)
    # Test both the real 2-stream boundary and virtual 3rd-stream prefill cohort (pref_cnt + 1 >= 2)
    if can_batch_mixed_step(st.spec_enabled, pref_cnt + 1, act_cnt):
        s0_n = (
            StreamState(**{**st.s0.__dict__, "phase": ACTIVE})
            if st.s0.phase == PREFILLING
            else StreamState(**{**st.s0.__dict__, "emitted": min(MAX_TOKENS, st.s0.emitted + 1), "in_think": False})
            if st.s0.phase == ACTIVE
            else st.s0
        )
        s1_n = (
            StreamState(**{**st.s1.__dict__, "phase": ACTIVE})
            if st.s1.phase == PREFILLING
            else StreamState(**{**st.s1.__dict__, "emitted": min(MAX_TOKENS, st.s1.emitted + 1), "in_think": False})
            if st.s1.phase == ACTIVE
            else st.s1
        )
        succ.append(SystemState(st.spec_enabled, s0_n, s1_n, "mixed_forward_batch"))
    return succ


def is_terminal(st: SystemState) -> bool:
    return st.s0.phase == RETIRED and st.s1.phase == RETIRED


def verify_tla_state_space() -> dict[str, int | str | list[str]]:
    idle_stream = StreamState(
        phase=IDLE,
        r0_snap=0,
        r1_snap=0,
        agreed_prefix=0,
        pending_drafts=False,
        draft_spliced=False,
        pending_q_row_base=-1,
        in_think=False,
        in_fence=False,
        emitted=0,
        slot_allocated=False,
        streams_drained=True,
    )
    init_states = [
        SystemState(spec_enabled=False, s0=idle_stream, s1=idle_stream, last_action="init"),
        SystemState(spec_enabled=True, s0=idle_stream, s1=idle_stream, last_action="init"),
    ]

    visited: set[SystemState] = set(init_states)
    queue: deque[SystemState] = deque(init_states)
    transitions = 0
    terminal_states = 0

    while queue:
        cur = queue.popleft()
        check_invariants(cur)
        if is_terminal(cur):
            terminal_states += 1
            continue
        succs = next_states(cur)
        if not succs:
            raise AssertionError(f"Liveness/Deadlock violation: non-terminal state has 0 enabled actions: {cur}")
        transitions += len(succs)
        for nxt in succs:
            check_transition_invariants(cur, nxt)
            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)

    assert terminal_states > 0, "Expected reachable terminal RETIRED states"
    return {
        "status": "PROVED (TLA+ Explicit-State BFS)",
        "reachable_states": len(visited),
        "transitions_checked": transitions,
        "terminal_states": terminal_states,
        "deadlocks": 0,
        "invariants_checked": [
            "Inv_NoMixedBatchUnderSpec",
            "Inv_UniformMultiRankPrefix",
            "Inv_NoSlotFreedBeforeStreamDrain",
            "Inv_FenceRunwayRespected",
            "Inv_SingleSeqExhaustionNeverAbortsPeer",
            "Inv_NoPendingQRowClobber",
            "Inv_SpecGreedyFenceRunwayRespected",
            "Inv_SplicedDraftNeverReadsStaleQRow",
        ],
    }


def main() -> int:
    summary = verify_tla_state_space()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
