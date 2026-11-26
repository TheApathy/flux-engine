// SPDX-License-Identifier: AGPL-3.0-only

//! Tree-aware GDN/conv1d dispatch predicate for DDTree verification.
//!
//! Determines whether a [`TreePayload`] diverges from a flat chain and
//! therefore requires the parent-aware `gated_delta_rule_tree` CUDA kernel
//! (`kernels/gb10/common/gated_delta_rule_tree.cu`) instead of the flat
//! `gated_delta_rule_wy_k` verify path.

use super::ddtree::TreePayload;

/// Whether the given payload requires the tree-aware kernel.
/// Returns `false` for `None` payloads and for trivially flat chains
/// (where `parent_indices` is `[-1, 0, 1, ..., n-2]`).
pub fn requires_tree_kernel(payload: Option<&TreePayload>) -> bool {
    let Some(p) = payload else { return false };
    if p.is_empty() {
        return false;
    }
    // Flat chain check: parent_indices[0] == -1 and each subsequent ==
    // the previous compact index.
    if p.parent_indices.first().copied() != Some(-1) {
        return true; // non-flat root layout
    }
    for (i, &parent) in p.parent_indices.iter().enumerate().skip(1) {
        let expected = (i - 1) as i32;
        if parent != expected {
            return true; // diverges from flat chain
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::ddtree::TreePayload;
    use super::*;

    fn payload(tokens: &[u32], parents: &[i32]) -> TreePayload {
        TreePayload {
            tree_token_ids: tokens.to_vec(),
            parent_indices: parents.to_vec(),
        }
    }

    #[test]
    fn none_payload_is_flat() {
        assert!(!requires_tree_kernel(None));
    }

    #[test]
    fn empty_payload_is_flat() {
        let p = payload(&[], &[]);
        assert!(!requires_tree_kernel(Some(&p)));
    }

    #[test]
    fn flat_chain_payload_is_flat() {
        // tokens [10,20,30], parents [-1, 0, 1] = flat chain.
        let p = payload(&[10, 20, 30], &[-1, 0, 1]);
        assert!(!requires_tree_kernel(Some(&p)));
    }

    #[test]
    fn diverging_payload_requires_tree() {
        // tokens [10, 11], parents [-1, -1] = two root-children siblings.
        let p = payload(&[10, 11], &[-1, -1]);
        assert!(requires_tree_kernel(Some(&p)));
    }

    #[test]
    fn deep_diverging_tree_requires_tree() {
        // 0: child of bonus root, 1: child of 0, 2: child of 0 (sibling of 1).
        let p = payload(&[10, 11, 12], &[-1, 0, 0]);
        assert!(requires_tree_kernel(Some(&p)));
    }

    #[test]
    fn two_node_flat_chain_does_not_require_tree() {
        let flat = payload(&[10, 20], &[-1, 0]);
        assert!(!requires_tree_kernel(Some(&flat)));
    }
}
