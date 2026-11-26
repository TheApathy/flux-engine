// SPDX-License-Identifier: AGPL-3.0-only

//! The all-chunks first prefill must stop at a chunk boundary. Before this,
//! one `start_chunked_prefill` call ran a whole long prompt (a 300K Flash-Next
//! prefill is ~25 min) with no cooperative check, so neither a disconnect nor
//! shutdown retirement could stop it.

use super::test_fixtures::{blocking_request, model};
use super::*;

const PROMPT: usize = 100;
const CHUNK: usize = 10;

fn run(
    complete_all_chunks: bool,
    disconnect_after: usize,
) -> (StartPrefillResult, model::TestModel) {
    let test_model = model::recording();
    *test_model.prefill_chunks.lock().unwrap() = Some(Vec::new());
    let (req, rx) = blocking_request(vec![1; PROMPT]);
    *test_model.disconnect_after_chunks.lock().unwrap() = Some((disconnect_after, Box::new(rx)));
    let result = start_chunked_prefill_with_defer(
        None,
        None,
        None,
        None,
        None,
        &test_model,
        req,
        &[17],
        CHUNK,
        1,
        2,
        &mut None,
        0,
        false,
        complete_all_chunks,
        None,
    )
    .expect("prefill start");
    (result, test_model)
}

fn chunks(test_model: &model::TestModel) -> Vec<(usize, usize)> {
    test_model.prefill_chunks.lock().unwrap().clone().unwrap()
}

#[test]
fn a_disconnect_mid_prompt_stops_the_all_chunks_loop_at_the_next_boundary() {
    let (result, test_model) = run(true, 2);
    assert!(matches!(result, StartPrefillResult::Finished));
    assert_eq!(chunks(&test_model), vec![(0, CHUNK), (CHUNK, CHUNK)]);
    assert_eq!(
        test_model.frees.lock().unwrap().len(),
        1,
        "the sequence is retired"
    );
}

#[test]
fn a_live_request_still_prefills_every_chunk() {
    // Disconnect only after the last chunk: the loop itself never stops early.
    let (result, test_model) = run(true, PROMPT / CHUNK);
    assert!(matches!(result, StartPrefillResult::Finished));
    let ran = chunks(&test_model);
    assert_eq!(ran.len(), PROMPT / CHUNK);
    assert_eq!(ran.last(), Some(&(PROMPT - CHUNK, CHUNK)));
}

#[test]
fn stepwise_prefill_hands_off_at_the_first_chunk() {
    let (result, test_model) = run(false, usize::MAX);
    assert_eq!(chunks(&test_model), vec![(0, CHUNK)]);
    match result {
        StartPrefillResult::InProgress(prefill) => assert_eq!(prefill.chunk_offset, CHUNK),
        _ => panic!("expected an in-progress prefill"),
    }
}
