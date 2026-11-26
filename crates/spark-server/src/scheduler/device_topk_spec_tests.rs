// SPDX-License-Identifier: AGPL-3.0-only

//! The routed accept walks: a device row source that samples through the
//! sparse route must leave the walk (outcome, committed tokens, last token)
//! exactly as the host walk over full rows leaves it; `Failed` finishes the
//! sequence; `Host` falls back to `fetch_row`.

use super::*;
use crate::scheduler::test_fixtures;
use spark_runtime::sampler::device_topk::{sparse_row_from_candidates, topk_candidates_reference};

type Rx = tokio::sync::oneshot::Receiver<anyhow::Result<crate::api::InferenceResponse>>;

const VOCAB: usize = 64;
const THINK_END: u32 = 5;
const SUPPRESS: [u32; 1] = [2];

fn ctx() -> ThinkSpecCtx<'static> {
    ThinkSpecCtx {
        enabled: true,
        code_fence_token: Some(3),
        reflection_suppress_ids: &SUPPRESS,
        adaptive_sampling: false,
        honor_eos_inside_thinking: Some(false),
    }
}

fn seq(thinking: bool, seed: u64) -> (ActiveSeq, Rx) {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let mut a = test_fixtures::active_seq(ResponseSink::Blocking(Some(tx)), 0);
    a.temperature = 1.0;
    a.top_k = 6;
    a.top_p = 0.9;
    a.seed = Some(seed);
    a.max_output_tokens = 1000;
    a.remaining = 1000;
    a.inside_thinking = thinking;
    a.enable_thinking = thinking;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(4);
    (a, rx)
}

/// Distinct BF16 values (no exact ties: canonical ties are off in tests)
/// with a soft leader so sampling varies with the seed.
fn rows(case: u64) -> Vec<Vec<u8>> {
    (0..3u64)
        .map(|r| {
            (0..VOCAB as u64)
                .map(|i| {
                    let k = (i * 37 + r * 11 + case * 5) % VOCAB as u64;
                    0x3F00u16 + k as u16 * 3
                })
                .flat_map(|b| b.to_le_bytes())
                .collect()
        })
        .collect()
}

/// The device route emulated on the host: route, CPU-spec candidates, sparse
/// row, the shared intervention path.
fn emulated_device_row<'a>(
    rows: &'a [Vec<u8>],
    calls: &'a std::cell::Cell<usize>,
) -> impl FnMut(usize, &mut ActiveSeq) -> DeviceRow + 'a {
    move |i, a| {
        calls.set(calls.get() + 1);
        let TopkRoute::Device {
            m,
            temperature,
            f2: false,
        } = device_topk_route(a, false, false, &SUPPRESS)
        else {
            return DeviceRow::Host;
        };
        let cands = topk_candidates_reference(&rows[i], temperature, m);
        let sparse = sparse_row_from_candidates(VOCAB, &cands).unwrap();
        DeviceRow::Sampled(process_seq_sparse_logits(
            a,
            sparse,
            None,
            a.think_end_token,
            a.tool_call_start_token,
            &SUPPRESS,
            false,
        ))
    }
}

fn serve(rows: &[Vec<u8>]) -> impl FnMut(usize, &mut Vec<u8>) -> bool + '_ {
    move |i, buf| {
        buf.clear();
        buf.extend_from_slice(&rows[i]);
        true
    }
}

fn never_fetch(_: usize, _: &mut Vec<u8>) -> bool {
    panic!("a device-sampled row must not fetch the full row");
}

#[test]
fn routed_thinking_walk_matches_the_host_walk() {
    let calls = std::cell::Cell::new(0usize);
    for case in 0..60u64 {
        let rows = rows(case);
        let drafts = [9u32, 9];
        let verified = [9u32, 9, 9];
        let (mut host, _r1) = seq(true, case);
        let (mut dev, _r2) = seq(true, case);
        let want =
            dflash_thinking_accept(&mut host, &drafts, &verified, &ctx(), serve(&rows), |_| {});
        let got = dflash_thinking_accept_routed(
            &mut dev,
            &drafts,
            &verified,
            &ctx(),
            never_fetch,
            emulated_device_row(&rows, &calls),
            |_| {},
        );
        assert_eq!(
            (got.num_accepted, got.bonus, got.consumed_rows),
            (want.num_accepted, want.bonus, want.consumed_rows),
            "case {case}"
        );
        assert_eq!(dev.output_tokens, host.output_tokens, "case {case}");
        assert_eq!(dev.last_token, host.last_token, "case {case}");
    }
    assert!(
        calls.get() >= 60,
        "the device row source must be consulted ({})",
        calls.get()
    );
}

#[test]
fn routed_content_walk_matches_the_host_walk() {
    let calls = std::cell::Cell::new(0usize);
    for case in 0..60u64 {
        let rows = rows(case);
        let drafts = [9u32, 9];
        let verified = [9u32, 9, 9];
        let (mut host, _r1) = seq(false, case);
        let (mut dev, _r2) = seq(false, case);
        let fmt = VerifyLogitsFormat::Bf16;
        let want = dflash_content_accept(
            &mut host,
            &drafts,
            &verified,
            &ctx(),
            fmt,
            VOCAB,
            serve(&rows),
        );
        let got = dflash_content_accept_routed(
            &mut dev,
            &drafts,
            &verified,
            &ctx(),
            fmt,
            VOCAB,
            never_fetch,
            emulated_device_row(&rows, &calls),
        );
        assert_eq!(
            (got.num_accepted, got.bonus, got.consumed_rows),
            (want.num_accepted, want.bonus, want.consumed_rows),
            "case {case}"
        );
        assert_eq!(dev.output_tokens, host.output_tokens, "case {case}");
    }
    assert!(
        calls.get() >= 60,
        "the device row source must be consulted ({})",
        calls.get()
    );
}

#[test]
fn a_failed_device_row_finishes_the_sequence_without_fetching() {
    let rows = rows(0);
    let (mut a, _r) = seq(true, 1);
    let out = dflash_thinking_accept_routed(
        &mut a,
        &[9],
        &[9, 9],
        &ctx(),
        never_fetch,
        |_, _| DeviceRow::Failed,
        |_| {},
    );
    assert!(a.finished);
    assert_eq!(
        (out.num_accepted, out.consumed_rows, out.bonus),
        (0, 0, None)
    );
    let (mut c, _r) = seq(false, 1);
    let out = dflash_content_accept_routed(
        &mut c,
        &[9],
        &[9, 9],
        &ctx(),
        VerifyLogitsFormat::Bf16,
        VOCAB,
        never_fetch,
        |_, _| DeviceRow::Failed,
    );
    assert!(c.finished);
    assert_eq!((out.num_accepted, out.consumed_rows), (0, 0));
    drop(rows);
}

#[test]
fn a_host_device_row_falls_back_to_fetch_row() {
    let rows = rows(3);
    let (mut host, _r1) = seq(true, 4);
    let (mut dev, _r2) = seq(true, 4);
    let want = dflash_thinking_accept(&mut host, &[9], &[9, 9], &ctx(), serve(&rows), |_| {});
    let got = dflash_thinking_accept_routed(
        &mut dev,
        &[9],
        &[9, 9],
        &ctx(),
        serve(&rows),
        |_, _| DeviceRow::Host,
        |_| {},
    );
    assert_eq!(
        (got.num_accepted, got.bonus),
        (want.num_accepted, want.bonus)
    );
    assert_eq!(dev.output_tokens, host.output_tokens);
}
