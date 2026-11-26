// SPDX-License-Identifier: AGPL-3.0-only

// Compiled here for its type and const checks; this crate calls none of
// its methods, and the workspace denies dead_code.
#[allow(dead_code)]
#[path = "../src/model/qwen4_mtp_hidden.rs"]
mod hidden_frames;

fn compact(source: &str) -> String {
    source.chars().filter(|c| !c.is_whitespace()).collect()
}

#[test]
fn production_native_save_validates_then_copies_then_publishes() {
    let source = compact(include_str!("../src/model/trait_impl/speculative.rs"));
    let method = source
        .split("pub(super)fnsave_hidden_for_mtp_sequence_dispatch(")
        .nth(1)
        .unwrap()
        .split("pub(super)fngenerate_speculative_dispatch(")
        .next()
        .unwrap();
    let validate = method.find(".begin_write(seq.slot_idx,seq.session_hash,seq.seq_len,token_idx,self.buffers.sizes().hidden_states,").unwrap();
    let legacy = method
        .find("self.save_hidden_for_mtp_dispatch(token_idx,stream)?")
        .unwrap();
    let copy = method.find("self.gpu.copy_d2d_async(").unwrap();
    let publish = method.find("frames.publish(write)").unwrap();
    assert!(validate < legacy && legacy < copy && copy < publish);
    assert!(method.contains("storage.offset(write.retained.prefix.offset)"));
    assert!(method.contains("write.retained.prefix.bytes,self.gpu.default_stream()"));
}

#[test]
fn native_propose_requires_owner_and_cursor_before_mutating_proposer() {
    let source = compact(include_str!("../src/model/impl_b3.rs"));
    let method = source
        .split("pub(super)fnrun_mtp_propose_inner(")
        .nth(1)
        .unwrap()
        .split("pub(super)fnmtp_recondition_staging_buf(")
        .next()
        .unwrap();
    let take = method
        .find(".take(seq.slot_idx,seq.session_hash,position)")
        .unwrap();
    assert!(
        take < method
            .find("self.k8a_late_replay_if_pending(seq)?")
            .unwrap()
    );
    assert!(take < method.find("self.sync_proposer_rotary(seq)?").unwrap());
    assert!(method.contains("native_hidden.map_or(self.mtp_hidden_save,|(selected,_)|selected)"));
    assert!(method.contains("native_hidden.map(|(_,prefix)|prefix)"));
}

#[test]
fn lifecycle_retires_and_relocates_owned_frames_before_slot_publication() {
    let source = compact(include_str!("../src/model/trait_impl/sequence.rs"));
    let free = source
        // 90ce7ffa2 widened the visibility to pub(in crate::model).
        .split("pub(incrate::model)fnfree_sequence_dispatch(")
        .nth(1)
        .unwrap()
        .split("pub(super)fncompact_sequence_dispatch(")
        .next()
        .unwrap();
    assert!(
        free.find("frames.lock().retire(slot)").unwrap()
            < free.find("self.ssm_pool.release_slot(slot)").unwrap()
    );
    let movement = source
        .split("pub(super)fncompact_sequence_dispatch(")
        .nth(1)
        .unwrap();
    assert!(
        movement
            .find("frames.begin_move(old_slot,new_slot,seq.session_hash)")
            .unwrap()
            < movement.find("ple.copy_slot(").unwrap()
    );
    assert!(
        movement.find("storage.offset(source.offset)").unwrap()
            < movement.find("||self.gpu.synchronize(stream)").unwrap()
    );
    assert!(
        movement.find("||self.gpu.synchronize(stream)").unwrap()
            < movement.find("frames.finish_move(movement)").unwrap()
    );
    assert!(
        movement.find("frames.finish_move(movement)").unwrap()
            < movement.find("seq.slot_idx=new_slot").unwrap()
    );
}

#[test]
fn all_scheduler_native_hidden_saves_are_sequence_aware() {
    for source in [
        include_str!("../../spark-server/src/scheduler/mtp_step.rs"),
        include_str!("../../spark-server/src/scheduler/verify_k2_step.rs"),
        include_str!("../../spark-server/src/scheduler/verify_k4_step.rs"),
        include_str!("../../spark-server/src/scheduler/verify_csk_step_k2.rs"),
        include_str!("../../spark-server/src/scheduler/verify_dflash_step.rs"),
    ] {
        // Only inline test modules end production: 9e3c69f95 put a #[cfg(test)] #[path] mod
        // near the top of mtp_step.rs.
        let production = source.split("#[cfg(test)]\nmod ").next().unwrap();
        assert!(!production.contains("model.save_hidden_for_mtp("));
        // 78517a820/70370490d route K2/K4 saves through save_hidden_for_active_proposer.
        assert!(
            production.contains("model.save_hidden_for_mtp_sequence(")
                || production.contains("save_hidden_for_active_proposer(")
        );
    }
    let helper = compact(include_str!("../../spark-server/src/scheduler/mtp_step.rs"))
        .split("pub(super)fnsave_hidden_for_active_proposer(")
        .nth(1)
        .unwrap()
        .split("fndispatch_verify(")
        .next()
        .unwrap()
        .to_owned();
    assert!(helper.contains("model.save_hidden_for_mtp_sequence(row,&a.seq,0)"));
    assert!(!helper.contains("model.save_hidden_for_mtp("));
}

#[test]
fn hidden_storage_does_not_remove_native_target_only_guards() {
    let source = compact(include_str!(
        "../src/model/trait_impl/decode_qwen4_native_batch.rs"
    ));
    assert!(source.contains(
        "!self.self_speculative&&(!self.any_proposer()||self.speculation_forced_disabled)"
    ));
    assert!(source.contains("nativeQwen4decodeistarget-only;disablespeculativedecoding"));
    let constructor = compact(include_str!("../src/model/impl_a1.rs"));
    assert!(constructor.contains(
        "NativeMtpHiddenFrames::new(max_batch_size,num_intermediates,mtp_hidden_bytes,)"
    ));
    assert!(constructor.contains("gpu.alloc(frames.allocation_bytes())?"));
}
