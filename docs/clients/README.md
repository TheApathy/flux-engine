<!-- SPDX-License-Identifier: AGPL-3.0-only -->

# Drop-In Coding-Agent Client Configs (`docs/clients/`)

Ready-to-copy client configurations pre-wired for a local Flux Engine server on
loopback (`http://127.0.0.1:8888`; change port to `8899` when using the
containerized `flux-autodetect` launcher).

| Client | Template | Target Location |
|---|---|---|
| **Claude Code** | [`claude-code.json`](claude-code.json) | `~/.claude/settings.json` |
| **Codex CLI** | [`codex.toml`](codex.toml) | `~/.codex/config.toml` |
| **OpenCode** | [`opencode.json`](opencode.json) | `./opencode.json` or `~/.config/opencode/opencode.json` |

Set `export FLUX_API_KEY="..."` to match `--api-key` if authentication is enabled
(or leave empty for loopback-only local sessions without `--api-key`). See
[`../CLAUDE_CODE.md`](../CLAUDE_CODE.md) and [`../API.md`](../API.md) for full details.
