// SPDX-License-Identifier: AGPL-3.0-only

//! `spark display-carveout`: lend the GB10 display carveout to a server.
//!
//! The firmware reserves 2 GiB for the display (`DISPLAY_FRM`) that Linux
//! never sees. The NVIDIA driver manages it as its scanout heap, and with no
//! monitor attached the display holds only a few MiB of it. This launcher
//! verifies `/dev/nvidia0` is GB10 (`device`), allocates the rest from that
//! heap and exports it as an RM object fd (`rm`), which needs no privilege,
//! then removes CAP_SYS_ADMIN (if a container granted it) from itself and its
//! bounding set and execs the command with the fd inherited and named in
//! `spark_runtime::gpu::carveout::FD_ENV`. `spark serve` then puts KV pools
//! there (`factory::build::kv_carveout`). Without the fd it serves as before.
//!
//! One process per host holds the carveout. `--lock` names a file on the
//! host that every launcher shares; the command inherits the held lock, so
//! it lasts exactly as long as the process that maps the memory. (The heap
//! itself never hands one range out twice; the lock makes a second launcher
//! say why it got nothing.) If the lock is held or the driver cannot give
//! the memory, the command runs without the carveout and the reason is
//! logged. No driver version is required: `rm` checks what the driver gives.

use std::io::Error;
use std::os::fd::IntoRawFd;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::carveout::{FD_ENV, SIZE_ENV};

#[cfg(target_os = "linux")]
pub(super) mod device;
#[cfg(target_os = "linux")]
mod rm;

#[derive(clap::Args, Debug, Clone, PartialEq, Eq)]
pub struct DisplayCarveoutArgs {
    /// Lock file shared by every launcher on this host (bind-mount the same
    /// host path into each container).
    #[arg(long)]
    pub lock: PathBuf,
    /// The command to run with the carveout, usually `flux-server serve ...`.
    #[arg(last = true, required = true)]
    pub command: Vec<String>,
}

/// Exports the carveout if it can, then execs the command. Returns only on
/// failure to exec or to drop CAP_SYS_ADMIN.
pub fn run(args: DisplayCarveoutArgs) -> Result<()> {
    let mut cmd = Command::new(&args.command[0]);
    cmd.args(&args.command[1..])
        .env_remove(FD_ENV)
        .env_remove(SIZE_ENV);
    match lend(&args) {
        Ok((fd, size)) => {
            tracing::info!(
                "display carveout: {} MiB exported as fd {fd}, lock {}",
                size >> 20,
                args.lock.display()
            );
            cmd.env(FD_ENV, fd.to_string())
                .env(SIZE_ENV, size.to_string());
        }
        Err(e) => tracing::warn!("display carveout not lent: {e:#}; running without it"),
    }
    drop_sys_admin().context("dropping CAP_SYS_ADMIN before exec")?;
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        bail!("exec {}: {err}", args.command[0]);
    }
    #[cfg(not(unix))]
    {
        let _ = cmd;
        bail!("the display carveout launcher requires Unix exec");
    }
}

/// Takes the host lock (kept open across exec), then exports.
fn lend(args: &DisplayCarveoutArgs) -> Result<(i32, u64)> {
    let lock = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&args.lock)
        .with_context(|| format!("open {}", args.lock.display()))?
        .into_raw_fd();
    // Inherited by the command, which then holds the lock until it exits.
    // SAFETY: `lock` is a valid owned fd just extracted via `into_raw_fd`; clearing
    // `FD_CLOEXEC` (`F_SETFD, 0`) passes a single `c_int` flag as `fcntl` expects.
    unsafe { libc::fcntl(lock, libc::F_SETFD, 0) };
    // SAFETY: `lock` is a valid open file descriptor; `LOCK_EX | LOCK_NB` never blocks.
    if unsafe { libc::flock(lock, libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        // SAFETY: `lock` is owned exclusively here and not used again after closing.
        unsafe { libc::close(lock) };
        bail!(
            "{} is held: another process has the carveout",
            args.lock.display()
        );
    }
    export().inspect_err(|_| {
        // SAFETY: `export()` failed, so `lock` will not be inherited and must be closed here.
        unsafe {
            libc::close(lock);
        }
    })
}

#[cfg(target_os = "linux")]
fn export() -> Result<(i32, u64)> {
    device::check_nvidia0()?;
    let e = rm::export_display_frm()?;
    tracing::info!(
        "display carveout: driver {}, {} of {} MiB of DISPLAY_FRM free (the display holds the rest)",
        e.driver,
        e.size >> 20,
        e.region >> 20
    );
    Ok((e.fd, e.size))
}

#[cfg(not(target_os = "linux"))]
fn export() -> Result<(i32, u64)> {
    bail!("the display carveout is a Linux GB10 feature")
}

/// Removes `CAP_SYS_ADMIN` from the bounding set and this thread's sets, so
/// the command (root in a container) can never create RM objects over
/// physical memory. Where `CAP_SYS_ADMIN` is not in the permitted set, this
/// irreversibly sets `PR_SET_NO_NEW_PRIVS` instead so exec'd binaries cannot
/// gain privileges via setuid or file capabilities (the export itself does not
/// need `CAP_SYS_ADMIN`).
#[cfg(target_os = "linux")]
fn drop_sys_admin() -> Result<()> {
    const CAP_SYS_ADMIN: u32 = 21;
    #[repr(C)]
    struct Header {
        version: u32,
        pid: i32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct Sets {
        effective: u32,
        permitted: u32,
        inheritable: u32,
    }
    const LINUX_CAPABILITY_VERSION_3: u32 = 0x2008_0522;
    let mut header = Header {
        version: LINUX_CAPABILITY_VERSION_3,
        pid: 0,
    };
    let mut sets = [Sets::default(); 2];
    // SAFETY: `header` and `sets` match the Linux v3 `__user_cap_header_struct` and
    // two-element `__user_cap_data_struct` ABI required by `SYS_capget`.
    if unsafe { libc::syscall(libc::SYS_capget, &mut header, sets.as_mut_ptr()) } != 0 {
        bail!("capget: {}", Error::last_os_error());
    }
    if sets[0].permitted & (1u32 << CAP_SYS_ADMIN) == 0 {
        // SAFETY: `PR_SET_NO_NEW_PRIVS` with arg2=1 and zeroed trailing args is thread-safe
        // and requires no capabilities.
        if unsafe { libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) } != 0 {
            bail!("PR_SET_NO_NEW_PRIVS: {}", Error::last_os_error());
        }
        return Ok(());
    }
    // SAFETY: `PR_CAPBSET_READ` and `PR_CAPBSET_DROP` take only integer arguments
    // (the capability bit index in arg2 and zeros in the trailing args) and perform
    // no pointer dereferences; if `CAP_SETPCAP` is absent, the kernel returns EPERM.
    if unsafe { libc::prctl(libc::PR_CAPBSET_READ, CAP_SYS_ADMIN, 0, 0, 0) } == 1
        && unsafe { libc::prctl(libc::PR_CAPBSET_DROP, CAP_SYS_ADMIN, 0, 0, 0) } != 0
    {
        bail!("PR_CAPBSET_DROP: {}", Error::last_os_error());
    }
    let bit = !(1u32 << CAP_SYS_ADMIN);
    sets[0].effective &= bit;
    sets[0].permitted &= bit;
    sets[0].inheritable &= bit;
    // SAFETY: `header` and `sets` hold valid v3 capability sets read by `SYS_capget` above.
    if unsafe { libc::syscall(libc::SYS_capset, &mut header, sets.as_ptr()) } != 0 {
        bail!("capset: {}", Error::last_os_error());
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn drop_sys_admin() -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_held_lock_lends_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("carveout.lock");
        let holder = std::fs::File::create(&path).unwrap();
        use std::os::fd::AsRawFd;
        // SAFETY: `holder` is an open file whose descriptor stays valid for the duration of this test.
        assert_eq!(unsafe { libc::flock(holder.as_raw_fd(), libc::LOCK_EX) }, 0);
        let args = DisplayCarveoutArgs {
            lock: path,
            command: vec!["true".into()],
        };
        let err = lend(&args).unwrap_err().to_string();
        assert!(err.contains("is held"), "{err}");
    }

    #[test]
    fn linux_export_checks_gb10_pci_identity_before_rm_allocation() {
        let src = include_str!("display_carveout.rs");
        let export_fn = src
            .split_once("fn export() -> Result<(i32, u64)> {")
            .expect("linux export fn")
            .1
            .split_once("\n}\n")
            .expect("end of linux export fn")
            .0;
        let check = export_fn
            .find("device::check_nvidia0()?")
            .expect("GB10 PCI check must be called");
        let rm = export_fn
            .find("rm::export_display_frm()?")
            .expect("RM scanout export must be called");
        assert!(check < rm, "GB10 PCI identity check must precede RM export");
    }
}
