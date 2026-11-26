// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn long_in_flight_is_preempted_by_short_pending_above_hysteresis() {
    let cfg = SrptConfig::default();
    let in_flight = [InFlightPrefill {
        prefill_idx: 0,
        remaining_tokens: 131_072 - 4_096,
        chunk_offset: 4_096,
        wait_ticks: 0,
    }];
    let pending = [PendingPrefill {
        pending_idx: 0,
        prompt_tokens: 512,
        wait_ticks: 0,
    }];
    assert_eq!(
        select_prefill_work(cfg, &in_flight, &pending, 1, 0, true),
        SrptSelection::AdmitPending {
            pending_idx: 0,
            prompt_tokens: 512,
        }
    );

    // When no admission slot is free or concurrency cap is full, continue in-flight.
    assert_eq!(
        select_prefill_work(cfg, &in_flight, &pending, 1, 0, false),
        SrptSelection::ContinueInFlight {
            prefill_idx: 0,
            remaining_tokens: 131_072 - 4_096,
            chunk_offset: 4_096,
        }
    );
    let capped_cfg = SrptConfig {
        max_concurrent_prefills: 1,
        ..cfg
    };
    assert_eq!(
        select_prefill_work(capped_cfg, &in_flight, &pending, 1, 0, true),
        SrptSelection::ContinueInFlight {
            prefill_idx: 0,
            remaining_tokens: 131_072 - 4_096,
            chunk_offset: 4_096,
        }
    );
}

#[test]
fn pending_within_hysteresis_continues_in_flight() {
    let cfg = SrptConfig::default(); // hysteresis = 512
    let pending = [PendingPrefill {
        pending_idx: 0,
        prompt_tokens: 512,
        wait_ticks: 0,
    }];

    // 900 <= 512 + 512 (1024): stay with in-flight.
    let near = [InFlightPrefill {
        prefill_idx: 0,
        remaining_tokens: 900,
        chunk_offset: 2_048,
        wait_ticks: 0,
    }];
    assert_eq!(
        select_prefill_work(cfg, &near, &pending, 1, 0, true),
        SrptSelection::ContinueInFlight {
            prefill_idx: 0,
            remaining_tokens: 900,
            chunk_offset: 2_048,
        }
    );

    // Exact boundary 1024 == 512 + 512: still stay with in-flight.
    let boundary = [InFlightPrefill {
        prefill_idx: 0,
        remaining_tokens: 1_024,
        chunk_offset: 2_048,
        wait_ticks: 0,
    }];
    assert_eq!(
        select_prefill_work(cfg, &boundary, &pending, 1, 0, true),
        SrptSelection::ContinueInFlight {
            prefill_idx: 0,
            remaining_tokens: 1_024,
            chunk_offset: 2_048,
        }
    );

    // One token beyond hysteresis (1025 > 1024): preempt for pending.
    let past = [InFlightPrefill {
        prefill_idx: 0,
        remaining_tokens: 1_025,
        chunk_offset: 2_048,
        wait_ticks: 0,
    }];
    assert_eq!(
        select_prefill_work(cfg, &past, &pending, 1, 0, true),
        SrptSelection::AdmitPending {
            pending_idx: 0,
            prompt_tokens: 512,
        }
    );
}

#[test]
fn anti_starvation_aging_prevents_starvation_under_continuous_short_arrivals() {
    let cfg = SrptConfig::default(); // aging = 256/tick, hysteresis = 512
    let mut long_remaining = 131_072usize;
    let mut long_offset = 0usize;
    let mut long_wait = 0u64;
    let mut max_consecutive_wait = 0u64;
    let mut short_completed = 0usize;
    let mut ticks = 0usize;
    let chunk_size = 4_096usize;

    while long_remaining > 0 && ticks < 20_000 {
        ticks += 1;
        let in_flight = [InFlightPrefill {
            prefill_idx: 0,
            remaining_tokens: long_remaining,
            chunk_offset: long_offset,
            wait_ticks: long_wait,
        }];
        // A fresh 512-token prompt arrives every single tick!
        let pending = [PendingPrefill {
            pending_idx: 0,
            prompt_tokens: 512,
            wait_ticks: 0,
        }];
        match select_prefill_work(cfg, &in_flight, &pending, 1, 0, true) {
            SrptSelection::AdmitPending { .. } => {
                short_completed += 1;
                long_wait += 1;
                max_consecutive_wait = max_consecutive_wait.max(long_wait);
            }
            SrptSelection::ContinueInFlight { prefill_idx: 0, .. } => {
                let step = long_remaining.min(chunk_size);
                long_remaining -= step;
                long_offset += step;
                long_wait = 0;
            }
            other => panic!("unexpected selection: {other:?}"),
        }
    }

    assert_eq!(
        long_remaining, 0,
        "128K prefill must finish despite continuous 512-tok arrivals"
    );
    assert!(
        max_consecutive_wait <= 512,
        "aging credit 256 tok/tick bounds consecutive skips of 128K to <= 512; got {max_consecutive_wait}"
    );
    assert!(
        short_completed > 0,
        "short arrivals must still preempt long prefill while aging builds"
    );

    // Also verify pending-queue SrptWaitTracker ages skipped long requests.
    let mut tracker = SrptWaitTracker::new();
    let long_key = 0xDEAD_BEEFu64;
    let mut admitted_long = false;
    for tick in 0..32u64 {
        let short_key = 0x1000 + tick;
        let selected = tracker.select_pending_order(cfg, &[(long_key, 4_096), (short_key, 256)], 1);
        if selected == [0] {
            admitted_long = true;
            break;
        }
        assert_eq!(selected, [1]);
    }
    assert!(
        admitted_long,
        "4096-tok pending request must overtake continuous 256-tok arrivals via aging"
    );
}

#[test]
fn two_hundred_tick_decode_alternation_invariant() {
    let cfg = SrptConfig::default();
    let active_decode_count = 2usize;
    let chunk_size = 1_024usize;
    let mut in_flight: Vec<(usize, usize, u64)> = vec![(65_536, 0, 0)];
    let mut pending: Vec<(usize, u64)> = Vec::new();
    let mut decode_steps = 0usize;
    let mut total_chunks = 0usize;
    let mut max_chunks_per_tick = 0usize;

    for tick in 0..200usize {
        if tick % 5 == 0 {
            pending.push((512, 0));
        }
        let mut chunks_this_tick = 0usize;

        let in_flight_view: Vec<InFlightPrefill> = in_flight
            .iter()
            .enumerate()
            .map(|(i, &(rem, off, wait))| InFlightPrefill {
                prefill_idx: i,
                remaining_tokens: rem,
                chunk_offset: off,
                wait_ticks: wait,
            })
            .collect();
        let pending_view: Vec<PendingPrefill> = pending
            .iter()
            .enumerate()
            .map(|(i, &(toks, wait))| PendingPrefill {
                pending_idx: i,
                prompt_tokens: toks,
                wait_ticks: wait,
            })
            .collect();

        match select_prefill_work(
            cfg,
            &in_flight_view,
            &pending_view,
            active_decode_count,
            chunks_this_tick,
            true,
        ) {
            SrptSelection::AdmitPending { pending_idx, .. } => {
                let (prompt_tokens, _) = pending.remove(pending_idx);
                let step = prompt_tokens.min(chunk_size);
                if prompt_tokens > step {
                    in_flight.push((prompt_tokens - step, step, 0));
                }
                for entry in &mut in_flight {
                    entry.2 += 1;
                }
                chunks_this_tick += 1;
            }
            SrptSelection::ContinueInFlight { prefill_idx, .. } => {
                for (i, entry) in in_flight.iter_mut().enumerate() {
                    if i == prefill_idx {
                        let step = entry.0.min(chunk_size);
                        entry.0 -= step;
                        entry.1 += step;
                        entry.2 = 0;
                    } else {
                        entry.2 += 1;
                    }
                }
                in_flight.retain(|&(rem, _, _)| rem > 0);
                chunks_this_tick += 1;
            }
            SrptSelection::Idle => {}
        }

        // A second prefill attempt in the same tick MUST be blocked while decoding!
        let in_flight_after: Vec<InFlightPrefill> = in_flight
            .iter()
            .enumerate()
            .map(|(i, &(rem, off, wait))| InFlightPrefill {
                prefill_idx: i,
                remaining_tokens: rem,
                chunk_offset: off,
                wait_ticks: wait,
            })
            .collect();
        let pending_after: Vec<PendingPrefill> = pending
            .iter()
            .enumerate()
            .map(|(i, &(toks, wait))| PendingPrefill {
                pending_idx: i,
                prompt_tokens: toks,
                wait_ticks: wait,
            })
            .collect();
        if chunks_this_tick >= 1 {
            assert_eq!(
                select_prefill_work(
                    cfg,
                    &in_flight_after,
                    &pending_after,
                    active_decode_count,
                    chunks_this_tick,
                    true,
                ),
                SrptSelection::Idle,
                "tick {tick}: second prefill chunk must be blocked while decode rows are active"
            );
        }

        max_chunks_per_tick = max_chunks_per_tick.max(chunks_this_tick);
        total_chunks += chunks_this_tick;
        decode_steps += 1;
    }

    assert_eq!(decode_steps, 200);
    assert!(max_chunks_per_tick <= 1);
    assert!(total_chunks > 100);
}

#[test]
fn srpt_disabled_preserves_strict_fifo_parity() {
    assert!(!SrptConfig::parse_enabled(Some("0"), true));
    assert!(!SrptConfig::parse_enabled(Some("0"), false));
    assert!(SrptConfig::parse_enabled(Some("1"), false));
    assert!(SrptConfig::parse_enabled(None, true));
    assert!(!SrptConfig::parse_enabled(None, false));

    let fifo_cfg = SrptConfig {
        enabled: false,
        ..SrptConfig::default()
    };
    let in_flight = [
        InFlightPrefill {
            prefill_idx: 0,
            remaining_tokens: 131_072,
            chunk_offset: 4_096,
            wait_ticks: 0,
        },
        InFlightPrefill {
            prefill_idx: 1,
            remaining_tokens: 256,
            chunk_offset: 256,
            wait_ticks: 0,
        },
    ];
    let pending = [
        PendingPrefill {
            pending_idx: 0,
            prompt_tokens: 65_536,
            wait_ticks: 0,
        },
        PendingPrefill {
            pending_idx: 1,
            prompt_tokens: 128,
            wait_ticks: 0,
        },
    ];

    assert_eq!(
        select_prefill_work(fifo_cfg, &in_flight, &pending, 1, 0, true),
        SrptSelection::ContinueInFlight {
            prefill_idx: 0,
            remaining_tokens: 131_072,
            chunk_offset: 4_096,
        }
    );
    assert_eq!(
        select_prefill_work(fifo_cfg, &[], &pending, 1, 0, true),
        SrptSelection::AdmitPending {
            pending_idx: 0,
            prompt_tokens: 65_536,
        }
    );

    let mut tracker = SrptWaitTracker::new();
    assert_eq!(
        tracker.select_in_flight(fifo_cfg, &[(1, 131_072, 4_096), (2, 256, 256)]),
        Some(0)
    );
    assert_eq!(
        tracker.select_pending_order(fifo_cfg, &[(10, 65_536), (20, 128)], 2),
        vec![0, 1]
    );
}
