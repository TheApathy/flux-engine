# R8 complete-source closure

This directory closes the source gap recorded by the R8 qualification receipt.
It is part of the source release for tag `v0.9.13-r8`, whose lineage revision
is `f22041b5dd7617737885ff3abe2d92d76a45c62f`.

The release tree now tracks all 17 formerly ignored Rust inputs and all three
linked FI-GDN objects. The objects are build conveniences, not substitutes for
source: `gdn-aot/` contains their preferred generator sources, the SM121a
FlashInfer DSL modification, a portable rebuild script, and a pinned toolchain.

FlashInfer 0.6.14 is pinned by the wheel SHA-256
`d124369346a3d48eac67e31c42f7a3c813bcc0abc10e2e36db413b7b3dfd97df`.
Its 2,472 build-input files (21,703,707 bytes) are independently pinned by
`flashinfer-0.6.14-build-inputs.manifest.tsv`, whose SHA-256 is
`72bec1a0aa3b6b3807ee43291d2b939e659959bc39a0f494146ff7fdd0a1bb46`.
The source bundle must include the matching wheel or its exactly verified
extracted snapshot; a manifest alone is not the source payload.

The upstream wheel's `delta_rule_sm120.py` is preserved under
`gdn-aot/upstream/`. The objects were generated with the minimal (4-line / 3-hunk) SM121a
adaptation preserved under `gdn-aot/flashinfer-0.6.14-modified/`: explicit
CUDA stream typing and `sm_121a` code generation. The NF8 variant is generated
from that adapted file by `make_variant.py`.

The BF16 and FP16 exporter copies differ from the historical captured files only
in replacing a machine-local output directory with `FLUX_GDN_AOT_OUT`.
Original captured exporter hashes are:

- BF16: `c746d28adc1151da36725e35df8493fc8de0d6ddcf16bc6e151dceb9b1707262`
- FP16: `611bd5b0711c7cd6143d4c816b3ddd8da5547b03b9b105daeebff974c8800688`

No checkpoint weights, access tokens, credentials, home-directory paths, or
container-registry credentials belong in the source bundle.

## Rebuild

```text
gdn-aot/rebuild-aot.sh PYTHON FLASHINFER_WHEEL OUTPUT_DIR
```

Use a CUDA-capable SM121a host and the versions in `gdn-aot/toolchain.lock`.
The script verifies the wheel hash, overlays the recorded SM121a source change,
regenerates and checks the NF8 source, and invokes all three exporters.

The final source archive must be generated from the final tagged release commit,
not from the lineage commit or an intermediate closure commit. It must include
the whole Git source tree plus the pinned FlashInfer source payload and license
texts. Generate binary/source linkage only after rebuilding the release binary.
