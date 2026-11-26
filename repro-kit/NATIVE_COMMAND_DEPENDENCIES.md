# Native Compiler Dependency Records

Set `ATLAS_NATIVE_TRACE_POLICY=flux.native-command/1` to request dependency
production and per-output records for the native NVCC and C++ build jobs. With
the variable unset, the existing compiler flags and tool discovery are retained.
Any other set value is refused. This source feature does not qualify a build.

The enabled build requires explicit canonical absolute `ATLAS_NVCC` and
`ATLAS_CXX` executable paths and an existing canonical absolute
`CARGO_TARGET_DIR`. Each crate derives the checkout root from its canonical
`CARGO_MANIFEST_DIR` at `<checkout>/crates/<crate>`. `FLASHINFER_HOME` is used
only for a source outside the checkout. The path-remap source directory is
not used to infer the checkout root.

## Actual Compiler Work

The Atlas kernel PTX jobs, storage predictor PTX jobs, four model NVCC bridge
jobs, and the host C++ ABI shim add `-MD -MF <output>.d` to their actual existing
compiler command. Archives and pre-existing AOT objects do not receive traces
from this mechanism. Existing optimization, architecture, math and remap flags
are retained.

Trace mode bypasses the Atlas kernel build script's PTX freshness cache so
every selected kernel reaches the real compilation and trace wrapper. Unset
trace policy retains that cache. Cargo itself may still reuse a whole build
script result, so a release capture must use its admitted fresh target tree
and complete independently classified output inventory. An abandoned trace
claim or old capture must not be promoted into fresh build evidence.

Each output has separate identities, including its original extension:

- `kernel.ptx.d` is the real compiler-generated dependency file.
- `kernel.ptx.native-command` is the successful actual command record.
- `kernel.ptx.native-job.json` is a portable per-output inventory sidecar.
- `kernel.ptx.native-trace.lock` exclusively claims that output while compiling.

The build removes previous dependency/command/job artifacts after claiming an
output and before invoking the compiler. It emits no success record for a
failed compiler invocation. After successful compilation, the dependency file
must be regular, nonempty and at most 8 MiB, and the output must be regular.
The command record is limited to 1 MiB. A failed or interrupted build is not
recoverable by relabeling old records; the build resource owner must resolve
any abandoned exclusive claim before a fresh compiler invocation.

## Record Formats

The command record contains UTF-8 NUL-terminated fields, including a terminal
NUL after the last argument:

```text
flux.native-command/1
actual Command.get_program()
absolute source path
absolute output path
absolute depfile path
0
actual Command.get_args(), one field per argument
```

The per-output JSON sidecar has exactly these fields:

```json
{
  "schema": "flux.native-job/1",
  "output": "release/build/example/out/kernel.ptx",
  "source": {"root": "source", "path": "kernels/example/kernel.cu"},
  "depfile": "release/build/example/out/kernel.ptx.d",
  "tool_role": "nvcc",
  "command_record": "release/build/example/out/kernel.ptx.native-command"
}
```

The other allowed source root is `flashinfer`, and the other tool role is
`cxx`. Output, depfile and command-record paths are relative to the target
root. The sidecar itself must be `<output>.native-job.json` and belong to the
sealed target artifact inventory.

## Producer Integration

`native_job_inventory.fold_native_jobs` takes target paths from the sealed
artifact inventory, an independently classified list of Atlas generated
`.ptx`/`.o` outputs, and a reader for the same sealed files. It verifies exact
sidecar schemas, per-output identities, source-root references, pinned tools,
actual command bindings and exact output coverage. It returns the map for
`manifest.native.jobs`.

The folding step does not create dependency files, classify output roles,
capture system headers, or infer expected coverage. The release consumer must
independently check real dependency rule LHS/RHS, source and FlashInfer closure,
the explicitly sealed CUDA/system header bundle, compiler hashes and artifact
hashes. AOT provenance and public-path eligibility remain separate gates.

The only accepted dependency spelling in an actual command is one `-MD` and
one `-MF` followed by the exact declared depfile. Output uses one `-o` followed
by the exact declared output. Other `-M*`, `--generate-dependenc*`,
`--generate-nonsystem-dependenc*`, `--dependency-*`, joined `-o*` and
`--output-file*` arguments are refused. Response files (`@...`, `-optf*`,
`--options-file*`) are refused because their options are absent from the
command record. Comma, equals and whitespace components are checked as well, so forwarding
an override through compiler/preprocessor options does not bypass admission.
Canonical options must also be at the top level, not operands of a separate
forwarding option. Argument inventories are bounded to 4096 fields and 8192
UTF8 bytes per nonempty field, in addition to the complete record byte limit.
These prefixes are reserved in trace mode. Normal architecture, math and
path-remap arguments retain their existing treatment.

The Python source/mock tests do not execute Rust, NVCC, a host compiler, a model
or a container. Rust unit tests call the real trace lifecycle and Nvidia cache
admission with inert commands and explicitly synthetic files, but are source
only until a separately admitted Rust test compilation executes them. Their
synthetic artifacts can never serve as compiler evidence. A guarded real
compiler run, header capture, complete producer
export and independent consumer validation remain required before any build
or public-path qualification claim.
