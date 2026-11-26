// SPDX-License-Identifier: AGPL-3.0-only

//! JPEG decode through libjpeg-turbo, loaded at runtime.
//!
//! Pillow decodes JPEG with libjpeg-turbo (3.0.3 bundled in Pillow 10.4). The
//! `image` crate's zune-jpeg disagrees with it on ~28% of RGB values, by up to
//! 12 levels, on our fixtures, so a V4.1 image would reach the model as
//! different pixels than under the Python engine. The system libjpeg-turbo
//! (Ubuntu `libturbojpeg`, 2.1.5) decodes those fixtures BIT-IDENTICALLY to
//! Pillow, measured through ctypes and again by `vision::tests`.
//!
//! The library is dlopen'd (`libturbojpeg.so.0`): no build-time dependency.
//! If it is absent, the caller falls back to zune-jpeg with a warning.
//! libjpeg-turbo is BSD-3-Clause / IJG / zlib, which is AGPL-compatible.

use std::ffi::{c_int, c_uchar, c_ulong, c_void};
use std::sync::OnceLock;

use anyhow::{Result, bail, ensure};
use image::RgbImage;
use libloading::Library;

use super::vision::MAX_IMAGE_PIXELS;

type InitFn = unsafe extern "C" fn() -> *mut c_void;
type HeaderFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    c_ulong,
    *mut c_int,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;
type DecompressFn = unsafe extern "C" fn(
    *mut c_void,
    *const c_uchar,
    c_ulong,
    *mut c_uchar,
    c_int,
    c_int,
    c_int,
    c_int,
    c_int,
) -> c_int;
type DestroyFn = unsafe extern "C" fn(*mut c_void) -> c_int;

const TJPF_RGB: c_int = 0;

struct TurboJpeg {
    _lib: Library,
    init: InitFn,
    header: HeaderFn,
    decompress: DecompressFn,
    destroy: DestroyFn,
}

struct TjHandle {
    ptr: *mut c_void,
    destroy: DestroyFn,
}

impl Drop for TjHandle {
    fn drop(&mut self) {
        // SAFETY: `self.ptr` is a non-null handle created by `tjInitDecompress` and
        // destroyed exactly once here; `self.destroy` comes from the `'static`
        // `TurboJpeg` instance kept alive by `LIB`.
        unsafe {
            (self.destroy)(self.ptr);
        }
    }
}

fn load(path: &str) -> Option<TurboJpeg> {
    // SAFETY:
    // - `Library::new(path)` runs the shared library's `DT_INIT`/`DT_INIT_ARRAY`
    //   constructors; `path` is either the default system soname `"libturbojpeg.so.0"`
    //   or an operator-supplied `FLUX_TURBOJPEG` / `ATLAS_TURBOJPEG` path.
    // - The four resolved symbols belong to the stable TurboJPEG 2.x/3.x C ABI
    //   (`tjInitDecompress`, `tjDecompressHeader3`, `tjDecompress2`, `tjDestroy`).
    // - `load` is called exclusively by `library()`, which stores `TurboJpeg` in a
    //   `'static OnceLock` so `_lib` remains mapped for the lifetime of the process
    //   and the copied raw function pointers never dangle.
    unsafe {
        let lib = Library::new(path).ok()?;
        let init = *lib.get::<InitFn>(b"tjInitDecompress\0").ok()?;
        let header = *lib.get::<HeaderFn>(b"tjDecompressHeader3\0").ok()?;
        let decompress = *lib.get::<DecompressFn>(b"tjDecompress2\0").ok()?;
        let destroy = *lib.get::<DestroyFn>(b"tjDestroy\0").ok()?;
        Some(TurboJpeg {
            _lib: lib,
            init,
            header,
            decompress,
            destroy,
        })
    }
}

fn library() -> Option<&'static TurboJpeg> {
    static LIB: OnceLock<Option<TurboJpeg>> = OnceLock::new();
    LIB.get_or_init(|| {
        let path = std::env::var("FLUX_TURBOJPEG")
            .ok()
            .or_else(|| std::env::var("ATLAS_TURBOJPEG").ok())
            .unwrap_or_else(|| "libturbojpeg.so.0".into());
        let lib = load(&path);
        match &lib {
            Some(tj) => {
                // Which file actually got mapped, and whether it is the 3.x ABI:
                // parity was verified against 2.1.5 (Ubuntu) == Pillow's 3.0.3,
                // so a different build must be visible in the startup log.
                let file = std::fs::read_to_string("/proc/self/maps")
                    .ok()
                    .and_then(|m| {
                        m.lines().find(|l| l.contains("turbojpeg")).map(|l| {
                            l.split_whitespace().last().unwrap_or_default().to_string()
                        })
                    })
                    .unwrap_or_else(|| "?".into());
                // SAFETY: symbol lookup only; the pointer is never called.
                let v3 = unsafe { tj._lib.get::<InitFn>(b"tj3Init\0").is_ok() };
                tracing::info!(
                    "deepseek_v41 JPEG decoder: libjpeg-turbo {file} ({} ABI); pixel parity with \
                     Pillow verified on 2.1.5",
                    if v3 { "3.x" } else { "2.x" }
                );
            }
            None => tracing::warn!(
                "libturbojpeg ({path}) not found: DeepSeek-V4.1 JPEG input falls back to zune-jpeg, \
                 whose pixels differ from Pillow (up to ~28% of RGB values / 12 levels). \
                 Install libturbojpeg or set FLUX_TURBOJPEG to restore parity."
            ),
        }
        lib
    })
    .as_ref()
}

/// Also forces the load, so the decoder choice is logged at startup.
pub fn available() -> bool {
    library().is_some()
}

/// Decode a JPEG to RGB with libjpeg-turbo's defaults (ISLOW IDCT, fancy
/// upsampling), as Pillow does. `Ok(None)` when the library is unavailable.
pub fn decode_rgb(raw: &[u8]) -> Result<Option<RgbImage>> {
    let Some(tj) = library() else { return Ok(None) };
    let raw_len =
        c_ulong::try_from(raw.len()).map_err(|_| anyhow::anyhow!("invalid PNG/JPEG image"))?;
    // SAFETY: `tjInitDecompress` takes no arguments and returns either null or a
    // valid TurboJPEG handle owned by `handle` until `Drop`.
    let ptr = unsafe { (tj.init)() };
    if ptr.is_null() {
        bail!("tjInitDecompress failed");
    }
    let handle = TjHandle {
        ptr,
        destroy: tj.destroy,
    };
    let (mut w, mut hgt, mut sub, mut cs) = (0, 0, 0, 0);
    // SAFETY: `handle.ptr` is a live TurboJPEG handle, `raw` is valid for
    // `raw_len` bytes, and all four output pointers point to valid stack `c_int`s.
    let rc = unsafe {
        (tj.header)(
            handle.ptr,
            raw.as_ptr(),
            raw_len,
            &mut w,
            &mut hgt,
            &mut sub,
            &mut cs,
        )
    };
    if rc != 0 || w <= 0 || hgt <= 0 {
        bail!("invalid PNG/JPEG image");
    }
    let pixels = (w as u64)
        .checked_mul(hgt as u64)
        .ok_or_else(|| anyhow::anyhow!("image exceeds pixel limit {MAX_IMAGE_PIXELS}"))?;
    ensure!(
        pixels <= MAX_IMAGE_PIXELS,
        "image exceeds pixel limit {MAX_IMAGE_PIXELS}"
    );
    let buf_len = usize::try_from(pixels)
        .ok()
        .and_then(|p| p.checked_mul(3))
        .ok_or_else(|| anyhow::anyhow!("image exceeds pixel limit {MAX_IMAGE_PIXELS}"))?;
    let mut buf = vec![0u8; buf_len];
    // SAFETY: `handle.ptr` is live, `raw` is valid for `raw_len` bytes, and `buf`
    // has length `w * hgt * 3` bytes matching `TJPF_RGB` with pitch 0.
    let rc = unsafe {
        (tj.decompress)(
            handle.ptr,
            raw.as_ptr(),
            raw_len,
            buf.as_mut_ptr(),
            w,
            0,
            hgt,
            TJPF_RGB,
            0,
        )
    };
    if rc != 0 {
        bail!("invalid PNG/JPEG image");
    }
    Ok(RgbImage::from_raw(w as u32, hgt as u32, buf))
}
