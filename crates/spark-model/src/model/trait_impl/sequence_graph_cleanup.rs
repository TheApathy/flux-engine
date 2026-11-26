// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use spark_runtime::gpu::GraphHandle;

use crate::layers::qwen3_attention::Qwen4K5DeviceAttentionTopology;

pub(super) type VerifyKgammaGraphKey = (usize, usize, u32, Option<Qwen4K5DeviceAttentionTopology>);

pub(super) fn verify_kgamma_keys_for_slots(
    cache: &HashMap<VerifyKgammaGraphKey, GraphHandle>,
    slots: &[usize],
) -> Vec<VerifyKgammaGraphKey> {
    cache
        .keys()
        .copied()
        .filter(|key| slots.contains(&key.0))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQUENCE: &str = include_str!("sequence.rs");

    #[test]
    fn selects_every_shape_for_only_the_released_slot() {
        let cache = HashMap::from([
            ((2, 4, 0, None), GraphHandle(10)),
            ((2, 4, 1, None), GraphHandle(11)),
            ((2, 17, 0, None), GraphHandle(12)),
            ((3, 4, 0, None), GraphHandle(13)),
        ]);

        let mut keys = verify_kgamma_keys_for_slots(&cache, &[2]);
        keys.sort_unstable_by_key(|key| (key.0, key.1, key.2));
        assert_eq!(
            keys,
            vec![(2, 4, 0, None), (2, 4, 1, None), (2, 17, 0, None)]
        );
    }

    #[test]
    fn selects_both_slots_affected_by_compaction() {
        let cache = HashMap::from([
            ((2, 4, 0, None), GraphHandle(10)),
            ((3, 4, 1, None), GraphHandle(11)),
            ((4, 17, 0, None), GraphHandle(12)),
        ]);

        let mut keys = verify_kgamma_keys_for_slots(&cache, &[2, 3]);
        keys.sort_unstable_by_key(|key| (key.0, key.1, key.2));
        assert_eq!(keys, vec![(2, 4, 0, None), (3, 4, 1, None)]);
    }

    #[test]
    fn free_and_compaction_both_invalidate_slot_bound_graphs() {
        assert_eq!(SEQUENCE.matches("verify_kgamma_keys_for_slots").count(), 2);
        assert!(SEQUENCE.contains("free_sequence: destroy_graph(verify_kgamma_graph"));
        assert!(SEQUENCE.contains("compact_sequence: destroy_graph(verify_kgamma_graph"));
        assert!(SEQUENCE.contains("for stale_slot in [old_slot, new_slot]"));
        assert!(SEQUENCE.contains("slots.contains(&old_slot) || slots.contains(&new_slot)"));
    }

    #[test]
    fn free_waits_for_secondary_commit_before_zeroing_the_slot() {
        let free = SEQUENCE
            .split_once("fn free_sequence_dispatch")
            .expect("free_sequence_dispatch must exist")
            .1;
        let wait = free
            .find("self.sync_secondary_dispatch(),")
            .expect("free must wait for the secondary commit");
        let zero = free
            .find("self.ssm_pool.zero_slot")
            .expect("free must zero the released slot");
        let release = free
            .find("self.ssm_pool.release_slot")
            .expect("free must release the slot");
        assert!(wait < zero);
        assert!(zero < release);
    }
}
