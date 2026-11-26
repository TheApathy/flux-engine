// SPDX-License-Identifier: AGPL-3.0-only

// The Flash-Next GDN has the same runtime-shaped 128-wide recurrence used by
// the Qwen3.8 kernel family.  Reuse that complete module so prefill, decode,
// multi-row verification, register-resident nosnap replay, and state snapshots
// (including 3-CTA/SM lazyfinal_multi with fused retain_gates) are exported.
#include "../../qwen3.8-27b/nvfp4/gated_delta_rule.cu"
