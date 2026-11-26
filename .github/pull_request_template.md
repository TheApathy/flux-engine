<!--
Thanks for contributing to Flux Engine! A few reminders:

- Read CONTRIBUTING.md if you haven't yet.
- CI runs cargo fmt, cargo clippy -Dwarnings, the SPDX-header check, typos,
  and cargo-deny (via the security workflow). Please run `cargo fmt --all`
  and `bash scripts/check.sh` locally before pushing.
- If you add a new source file, it needs `// SPDX-License-Identifier: AGPL-3.0-only`
  on line 1.
-->

## Summary

<!-- 1-3 sentences: what changes and why. Reference the driving issue if any. -->

Closes #

## Test plan

<!-- Concrete checklist of how you verified this. Logs/benchmarks welcome. -->

- [ ] `cargo fmt --all -- --check`
- [ ] `ATLAS_SKIP_BUILD=1 cargo clippy --workspace --tests -- -D warnings`
- [ ] `bash scripts/check.sh`
- [ ] Tested against a real model / hardware if the change affects runtime behaviour
- [ ] Added or updated tests where applicable

## Notes for reviewers

<!-- Design rationale, trade-offs, follow-ups you deferred, things you want a second opinion on. -->

## AI-Assisted Contribution Disclosure

<!-- Optional transparency note for reviewers (modeled on vLLM / llama.cpp / mistral.rs). -->

- [ ] No AI tools were used, **or** any AI-assisted code/docs in this PR were fully reviewed, understood, and tested by me.

## License & Contribution Agreement

- [ ] I have read and agree to the [Contributing Guidelines](../CONTRIBUTING.md) and [AGPL-3.0 License](../LICENSE).
