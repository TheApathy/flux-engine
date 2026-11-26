# Plain QSA graph admission

This patch is source-only until the exact resulting binary passes native and
model checks. It does not establish an accuracy fix, performance improvement,
startup success or container readiness.

Plain decode's QSA indexer runs even before position 2048 for a request that
will need sparse attention. Without `ATLAS_QWEN4_QSA_GRAPH_SAFE=1`, capturing
that route freezes host sequence-length and visible-group kernel arguments.
The graph-safe indexer uses device metadata, but the dense/sparse attention
topology is still chosen on the host. A pre-2048 graph must not be reused
after that transition.

For Qwen4, the new pure policy admits capture only when QSA is not required,
or when graph-safe QSA is enabled and the host sequence length is at least
2048. Both ordinary and PLE-segmented Plain graph decisions use this policy.
The suffix entry independently refuses an unsafe request before any GPU
work. Disallowed captures use the existing eager path rather than rejecting
the request. This does not enable graph-safe QSA by default or change K5
verification's separate graph policy.

The scheduler also sets the request's QSA reachability hint for other model
families. The policy explicitly checks the target model family so those
unrelated models retain their existing graph admission.

Six Rust regression definitions cover non-Qwen4 and non-QSA behavior, host-metadata
refusal, the dense/sparse boundary and long lengths. The bounded Python
source-wiring suite checks both admission sites and the defensive check,
including removal, wrong-metadata and late-check mutants. Python checks are
not a substitute for compiling or executing the Rust tests.

Runtime validation must compare exact greedy tokens and committed state in
fresh processes with graphs disabled/enabled, graph-safe QSA unset/enabled,
and lengths around 2047/2048/2049 and at longer contexts. PLE and non-PLE
paths both need coverage. No output drift, truncation or stale cache may be
counted as a performance result. Native MTP, true 262K retrieval, quality
benchmarks, native concurrency and exact-image serving remain independent
release gates.
