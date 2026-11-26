// SPDX-License-Identifier: AGPL-3.0-only

//! Cargo regression importing the actual production SSM planner, including
//! its real anyhow/atlas_core dependencies. This is not a standalone std-only
//! test. No GPU execution is required, but native-linked compilation is gated.
#![allow(dead_code)]

#[path = "../src/model/qwen4_mtp_hidden.rs"]
mod hidden_frames;
#[path = "../src/model/ssm_pool_geometry.rs"]
mod ssm_geometry;

const ROW_BYTES: usize = 4 * 2_560 * 2;

fn retain_row_31_with_spare_capacity(rows: usize) {
    assert_eq!(rows, 33, "production capacity includes the spare row");
    let mut pool = hidden_frames::NativeMtpHiddenFrames::new(2, rows, ROW_BYTES).unwrap();
    assert_eq!(pool.allocation_bytes(), 2 * rows * ROW_BYTES);
    let mut retained = vec![0xFFu8; pool.allocation_bytes()];
    let mut target = vec![0u8; 32 * ROW_BYTES];
    for slot in 0..2 {
        for row in 0..32 {
            target[row * ROW_BYTES..(row + 1) * ROW_BYTES].fill((slot * 64 + row) as u8);
        }
        let write = pool
            .begin_write(slot, 100 + slot as u64, 9_000 + slot, 31, target.len())
            .unwrap();
        let prefix = write.retained.prefix;
        assert_eq!(prefix.bytes, target.len());
        retained[prefix.offset..prefix.offset + prefix.bytes].copy_from_slice(&target);
        pool.publish(write).unwrap();
        target.fill(0xEE);
    }
    for slot in (0..2).rev() {
        let read = pool.take(slot, 100 + slot as u64, 9_000 + slot).unwrap();
        assert_eq!(read.selected.bytes, ROW_BYTES);
        assert_eq!(read.selected.offset, (slot * rows + 31) * ROW_BYTES);
        for row in 0..32 {
            let offset = read.prefix.offset + row * ROW_BYTES;
            assert!(
                retained[offset..offset + ROW_BYTES]
                    .iter()
                    .all(|&value| value == (slot * 64 + row) as u8)
            );
        }
        let spare = (slot * rows + 32) * ROW_BYTES;
        assert!(
            retained[spare..spare + ROW_BYTES]
                .iter()
                .all(|&value| value == 0xFF)
        );
    }
    // A spare retained row does not enlarge the actual source arena. Refuse
    // to read it without erasing the valid handoff that preceded this call.
    let write = pool.begin_write(0, 100, 9_000, 31, target.len()).unwrap();
    pool.publish(write).unwrap();
    assert!(pool.begin_write(0, 100, 9_000, 32, target.len()).is_err());
    assert!(pool.take(0, 100, 9_000).is_ok());
}

#[test]
fn production_31_draft_geometry_accepts_spare_and_retains_row_31_bytes() {
    let geometry = ssm_geometry::checked_ssm_speculative_geometry(true, false, 31, None).unwrap();
    retain_row_31_with_spare_capacity(geometry.num_intermediates);
}

#[test]
fn production_uncapped_tree_geometry_accepts_capacity_32_plus_spare() {
    const CHILD: &str = "ATLAS_MTP_HIDDEN_TREE_GEOMETRY_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg("production_uncapped_tree_geometry_accepts_capacity_32_plus_spare")
            .arg("--nocapture")
            .env(CHILD, "1")
            .env("ATLAS_DDTREE_UNCAP", "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    assert_eq!(std::env::var("ATLAS_DDTREE_UNCAP").unwrap(), "1");
    let geometry = ssm_geometry::checked_ssm_speculative_geometry(true, true, 4, Some(32)).unwrap();
    assert_eq!(geometry.dflash_verify_width, 5);
    assert_eq!(geometry.ddtree_capacity, 32);
    retain_row_31_with_spare_capacity(geometry.num_intermediates);
}
