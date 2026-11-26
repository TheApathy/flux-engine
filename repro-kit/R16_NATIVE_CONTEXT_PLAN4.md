# Native Context Plan 4

The shipped qualification plan requires both native-context gates:

| Required gate | Prefill chunk | Recipe |
| --- | --- | --- |
| `native-262k` | 16384 | `recipes/flux-qwen4-native-262k-bf16.env` |
| `native-262k-w4096-owner16` | 4096 | `recipes/flux-qwen4-native-262k-w4096-bf16-owner16.env` |

The second gate is additional. It cannot substitute for the W16384 control.
Both use BF16 KV, native capacity 262144, exact retrieval at 258000, 261888 and
262080 prompt tokens, the 262143 boundary, and rejection of 262144 prompt
tokens plus one output token. W4096 additionally checks the owner16 census and
the 64-chunk post-rejection canary.

Plan `flux-r15-container-suite-plan/4` adds
`native_context_contract: {schema, gates}`. Its schema is
`flux-native-context-recipe-contract/1`; both exact recipe paths, hashes and
chunk values are mandatory. The profile subsystem retains
`flux-performance-profile-contract/2` and its sealed geometry recipes.

The standalone suite and outer wrapper both require all twelve gate jobs.
The suite receipt is `flux-r15-container-suite-receipt/2`. Individual gate
receipt and raw validator envelopes remain their existing version 1 formats.
Older suite plans are refused by this release revision.

Each native driver reads the actual launching suite's sealed plan descriptor.
It selects its own recipe from that contract and hashes the recipe read from
the immutable container. The default image recipe does not replace either
native recipe. The live broker checks the gate, recipe, both WIDE aliases,
actual scheduler chunk, BF16 KV and QSA, then emits a native-context process
proof. The proof names the gate, chunk and complete contract hash.

At cleanup, each driver rechecks the original default image plan and recipe,
then independently reloads its selected native recipe from the sealed parent
plan and immutable image. Both snapshots must match their own initial values.
The selected seven-field native plan is distinct from the four-field default
image plan; their differing recipes are expected and are not compared to one
another.

The wrapper independently binds both native jobs to their exact driver,
lifecycle helper, endpoint helper, recipe and source inventory. The W4096 job
also binds the owner16 engagement helper. The independent validator itself is
hash checked before execution and must appear in the proven source inventory.
The wrapper validates the raw owner16 census, live controls, process proof,
retrievals, boundary/rejection results and canary rather than accepting a
summary PASS.

`examples/r15-container-suite-plan.example.json` shows the new format. It is a
source template. Existing startup-readiness, tool-api-streaming-blocking and
cancellation-slot-reuse job entries still require actual reviewed driver
paths and pins. The example is not a deployable plan or a release receipt.
Before execution, rebuild the complete source inventory and all affected
suite, driver, validator and image pins. The geometry packet's executable pin
must also be refreshed for the actual release binary.

Source tests and mocked orchestration do not qualify runtime accuracy, actual
262144-token inference, performance or a container. Every required live gate
must pass on the exact final image before promotion.
