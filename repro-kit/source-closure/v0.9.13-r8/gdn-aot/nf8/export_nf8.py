"""AOT-export the norm-fp8 FI GDN variant (FP16 q/k/v/o, init_state on).

    NF8_LDST_REGS=.. NF8_NORM_WARPS=.. python export_nf8.py <variant dir> <out dir>

Same capture as ../exporters/export_fp16.py (spy on cached_compile, KeepCUBIN,
export_to_c), prefix atlas_fi_gdn_sm121a_prefill_f16_nf8. Prints the C
argument list so the bridge can be checked against it.
"""

import math
import os
import sys

import torch

sys.path.insert(0, sys.argv[1])
import cutlass.cute as cute  # noqa: E402
import delta_rule_sm120_nf8 as mod  # noqa: E402

OUT = sys.argv[2]
PREFIX = "atlas_fi_gdn_sm121a_prefill_f16_nf8"
os.makedirs(OUT, exist_ok=True)

captured = {}
_orig = mod.cached_compile


def spy(*args, **kwargs):
    kwargs["compile_options"] = tuple(kwargs["compile_options"]) + (cute.KeepCUBIN(True),)
    fn = _orig(*args, **kwargs)
    captured["fn"] = fn
    return fn


mod.cached_compile = spy

T, HQ, HV, D = 8192, 16, 48, 128
CONV, VD = 2 * HQ * D + HV * D, HV * D
torch.manual_seed(0)
dev = "cuda"
q = torch.nn.functional.normalize(torch.randn(T, HQ, D, device=dev), dim=-1).half()
k = torch.nn.functional.normalize(torch.randn(T, HQ, D, device=dev), dim=-1).half()
v = torch.randn(T, HV, D, device=dev, dtype=torch.float16)
alpha = torch.full((T, HV), 0.99, device=dev)
beta = torch.full((T, HV), 0.5, device=dev)
cu = torch.tensor([0, T], device=dev, dtype=torch.int64)
o = torch.zeros(T, HV, D, device=dev, dtype=torch.float16)
st = torch.zeros(1, HV, D, D, device=dev)
init = torch.randn(1, HV, D, D, device=dev) * 0.1
zfull = torch.randn(T, CONV + VD, device=dev).bfloat16()
w = torch.ones(D, device=dev, dtype=torch.bfloat16)
out8 = torch.zeros(T, VD, device=dev, dtype=torch.uint8)
sc = torch.zeros(1, device=dev)
mod.delta_rule_prefill_dsl_nf8(
    o, st, q, k, v, init, alpha, beta, cu, 1.0 / math.sqrt(D),
    z=zfull[:, CONV:], normw=w, out8=out8, scale_out=sc, norm_eps=1e-6,
)
torch.cuda.synchronize()
print("run ok: out8 nonzero", int((out8 != 0).sum().item()), "scale", sc.item())

fn = captured["fn"]
cha = fn.c_header_arguments
print("error_msg:", (cha.error_msg or "None")[:400])
print("arguments:", cha.arguments)
fn.export_to_c(OUT, PREFIX)
for f in sorted(os.listdir(OUT)):
    print(f, os.path.getsize(os.path.join(OUT, f)))
