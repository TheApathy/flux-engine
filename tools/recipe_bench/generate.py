#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Emit env (KEY=VAL lines) or argv (one token per line) for a built-in recipe
YAML, mirroring crates/spark-server/src/recipe/schema.rs: `defaults:` maps to
`--flag value`, with renames and presence-only booleans; `atlas.env` maps
straight to environment variables.

Rationale: hand-transcribing recipe env blocks across shell launchers easily
drops required selectors or diverges on CLI flag hyphenation. This generator
is the single source of truth for translating recipe YAML into `env` and `argv`.

usage:
  generate.py <recipe.yaml> <model_dir> <model_name> env
  generate.py <recipe.yaml> <model_dir> <model_name> argv [key=value overrides...]
    (override value 'DEL' removes that default)
"""
import sys
import yaml


# Recipe keys whose CLI flag name differs from `key.replace('_', '-')`.
RENAMES = {"max_model_len": "max-seq-len", "tensor_parallel": "tp-size", "host": "bind"}

# Flags that take no value — presence alone means true (clap `ArgAction::SetTrue`
# in `crates/spark-server/src/cli.rs` / `recipe/schema.rs`). Checked AFTER
# hyphenation/renaming, so members here must already be hyphenated
# ('qwen4-qsa', not 'qwen4_qsa').
PRESENCE = {
    "adaptive-sampling",
    "allow-special-tokens-in-content",
    "check-kernels",
    "dflash",
    "disable-confidence-early-stop",
    "disable-loop-watchdog",
    "disable-simhash-watchdog",
    "disable-thinking",
    "disable-thinking-loop-watchdog",
    "enable-prefix-caching",
    "experimental-dflash-probabilistic",
    "high-speed-swap",
    "ngram-speculative",
    "no-fast-load",
    "no-tui",
    "profile",
    "qwen4-qsa",
    "require-auth",
    "self-speculative",
    "speculative",
}


def build_env(recipe: dict) -> list[str]:
    env = recipe.get("atlas", {}).get("env", {})
    return [f"{k}={'' if v is None else v}" for k, v in env.items()]


def build_argv(recipe: dict, model_dir: str, model_name: str, overrides: list[str]) -> list[str]:
    defaults = {
        k: ("true" if v is True else "false" if v is False else str(v))
        for k, v in recipe["defaults"].items()
    }
    for o in overrides:
        k, v = o.split("=", 1)
        if v == "DEL":
            defaults.pop(k, None)
        else:
            defaults[k] = v

    out = ["serve", "--model-from-path", model_dir, "--model-name", model_name, "--no-tui"]
    for k, v in defaults.items():
        flag = RENAMES.get(k, k.replace("_", "-"))
        if flag in PRESENCE:
            if v == "true":
                out.append("--" + flag)
            continue
        out += ["--" + flag, v]
    return out


def main() -> None:
    recipe_path, model_dir, model_name, mode, *overrides = sys.argv[1:]
    recipe = yaml.safe_load(open(recipe_path))
    if mode == "env":
        print("\n".join(build_env(recipe)))
    elif mode == "argv":
        print("\n".join(build_argv(recipe, model_dir, model_name, overrides)))
    else:
        raise SystemExit(f"mode must be 'env' or 'argv', got {mode!r}")


if __name__ == "__main__":
    main()
