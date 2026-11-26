"""Export an FP16-dtype twin of the init_state GDN prefill AOT artifact.

Same kernel and init_state as export_init.py, but q/k/v/o are float16, so the
kernel's internal operand casts (state-as-MMA-operand, V - S K, new V, the
WY inverse) keep 10 mantissa bits instead of bf16's 7. Prefix and output dir
differ so it can be linked next to the bf16 object.

Original docstring (export_init.py):

The currently-integrated artifact was exported with init_state=None, so the
kernel always starts from a zero state and the bridge must refuse a nonzero
engine state (only the first chunk of a chunked prefill can use it). Passing a
real `init_state` flips `needs_init_state` to True and makes the kernel read the
incoming recurrent state, so every chunk can use it.

Emits the header + object into aot_init/ and prints the new argument arity.
"""

import os
import subprocess

import torch
from cutlass.base_dsl.compiler import DumpDir  # noqa: F401
import cutlass.cute as cute

from flashinfer.gdn_kernels import chunk_gated_delta_rule_sm120
from flashinfer.gdn_kernels.delta_rule_dsl import delta_rule_sm120 as mod

T, HQ, HV, D = 8192, 16, 48, 128
OUT = os.environ.get("FLUX_GDN_AOT_OUT", "aot_fp16")
os.makedirs(OUT, exist_ok=True)

captured = {}
_orig = mod.cached_compile


def spy(*args, **kwargs):
    ko = kwargs["compile_options"]
    kwargs["compile_options"] = tuple(ko) + (cute.KeepCUBIN(True),)
    fn = _orig(*args, **kwargs)
    captured["fn"] = fn
    return fn


mod.cached_compile = spy

torch.manual_seed(0)
q = torch.nn.functional.normalize(
    torch.randn(T, HQ, D, device="cuda"), dim=-1).half()
k = torch.nn.functional.normalize(
    torch.randn(T, HQ, D, device="cuda"), dim=-1).half()
v = torch.randn(T, HV, D, device="cuda", dtype=torch.float16)
alpha = torch.full((T, HV), 0.99, device="cuda", dtype=torch.float32)
beta = torch.full((T, HV), 0.5, device="cuda", dtype=torch.float32)
cu = torch.tensor([0, T], device="cuda", dtype=torch.int64)
o = torch.zeros(T, HV, D, device="cuda", dtype=torch.float16)
st = torch.zeros(1, HV, D, D, device="cuda", dtype=torch.float32)

# NONZERO initial state: proves the flag flips and that the value is consumed.
init_state = (torch.randn(1, HV, D, D, device="cuda",
                          dtype=torch.float32) * 0.1)

chunk_gated_delta_rule_sm120(o, st, q, k, v, init_state, alpha, beta, cu,
                             1.0 / D ** 0.5)
torch.cuda.synchronize()
print("in-process run with init_state:")
print("  finite:", bool(torch.isfinite(o.float()).all().item()),
      " o_absmax:", round(float(o.float().abs().max().item()), 6),
      " st_absmax:", round(float(st.abs().max().item()), 6))

# Sanity: a zero init_state must give a DIFFERENT (and this one a nonzero) result
# than the zero-state run, i.e. the input is genuinely read.
init_zero = torch.zeros(1, HV, D, D, device="cuda", dtype=torch.float32)
o2 = torch.zeros(T, HV, D, device="cuda", dtype=torch.float16)
st2 = torch.zeros(1, HV, D, D, device="cuda", dtype=torch.float32)
chunk_gated_delta_rule_sm120(o2, st2, q, k, v, init_zero, alpha, beta, cu,
                             1.0 / D ** 0.5)
torch.cuda.synchronize()
same = bool(torch.equal(o, o2))
print("  nonzero-init output == zero-init output:", same,
      "(MUST be False)")
print("  zero-init o_absmax:", round(float(o2.float().abs().max().item()), 6))

fn = captured["fn"]
PREFIX = "atlas_fi_gdn_sm121a_prefill_f16"

print("\n=== c_header_arguments ===")
cha = fn.c_header_arguments
print("  error_msg:", (cha.error_msg or "None")[:400])
print("  arguments:", cha.arguments)
print("  packed_args:", cha.packed_args)

print("\n=== export_to_c ===")
fn.export_to_c(OUT, PREFIX)
print("  export_to_c OK (dir:", OUT, ")")

print("\n=== OUT dir ===")
for root, _d, files in os.walk(OUT):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"  {f}  {os.path.getsize(p)}B")

print("\n=== generated structs / wrapper signature ===")
hdr = os.path.join(OUT, f"{PREFIX}.h")
txt = open(hdr).read()
import re
print("  init_state struct present:",
      "Tensor_g_init_state_t" in txt)
for line in txt.splitlines():
    if "init_state" in line or "void *args[" in line or "static inline int32_t cute_dsl" in line:
        print("   |", line.strip()[:220])
