#!/usr/bin/env bash
# ==============================================================================
# Flux Engine — Environment & Operating Mode Bridge (FLUX_* <-> ATLAS_*)
#
# Purpose:
#   1. Resolves high-level Flux operating modes (--mode / FLUX_MODE) into their
#      certified recipe files under recipes/.
#   2. Translates any FLUX_* environment variable (e.g., FLUX_QWEN4_PREFILL_WIDE=16384)
#      into the underlying engine's ATLAS_* runtime selector (ATLAS_QWEN4_PREFILL_WIDE=16384)
#      while preserving caller-supplied FLUX_* overrides over recipe defaults.
#   3. Enables 100% clean FLUX_* configuration in user environments, containers, and
#      orchestrators while maintaining binary compatibility with spark-server.
#
# Usage:
#   # 1. Source into current shell:
#   source scripts/flux_env_bridge.sh --mode max-prefill
#
#   # 2. Inspect translated environment for a mode:
#   bash scripts/flux_env_bridge.sh --mode interactive-w4096 --print-env
#
#   # 3. Execute a command with translated environment:
#   bash scripts/flux_env_bridge.sh --mode k5-decode -- env
# ==============================================================================

# Detect whether the script is being sourced or executed directly
_flux_is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _flux_is_sourced=1
fi

_flux_bridge_usage() {
    cat <<'EOF'
Usage: flux_env_bridge.sh [OPTIONS] [-- COMMAND [ARGS...]]

Options:
  -m, --mode <MODE>     Select Flux Engine operating mode (or set FLUX_MODE=<MODE>):
                          - max-prefill       (v9 w16384: 6,082.2 tok/s @ 16K + 57.4 tok/s K=5 decode)
                          - interactive-w4096 (v9 w4096:  5,276.3 tok/s @ 8K  + 12.6 GiB free floor)
                          - fast-prefill      (pure prefill velocity, GDN_NORM_FP8=3 Ultra tier)
                          - k5-decode         (dedicated 57.4 tok/s peak LOSSLESS 10/10 K=5 MTP decode)
                          - long-context-136k (certified 136,584-token long-context 5/5 PASS recipe)
                          - exact-reference   (unfused/reference W4A16 bit-exact 0-ULP auditing baseline)
                          - 256k              (native 262K BF16 KV long-context recipe)
                          - c8                (8-way native concurrency recipe)
                          - nextgen           (nextgen max prefill + decode balanced profile)
                          - 1m-nvfp4          (1M-token NVFP4 KV cache profile)
                          - c128              (high-concurrency C=128 serving profile)
                          - 2x-spark          (2-node TP=2 disaggregated cluster profile)
                          - 3x-spark          (3-node PP=3 disaggregated cluster profile)
                          - 4x-spark          (4-node cluster profile)
  -f, --env-file <FILE> Load an explicit .env recipe file instead of a named mode.
  --list-modes          List all available operating modes and their recipe paths.
  --print-env           Print all active FLUX_* and ATLAS_* environment variables after translation.
  --print-argv          Print the paired CLI argv flags for the selected mode (substituting MODELS_ROOT).
  -q, --quiet           Suppress informational banner messages on stderr.
  -h, --help            Show this help message.
EOF
}

_flux_bridge_main() {
local _flux_script_dir _flux_repo_root _flux_recipes_dir
local _flux_selected_mode _flux_explicit_env_file
local _flux_print_env _flux_print_argv _flux_list_modes _flux_quiet
local -a _flux_exec_cmd=()
local _flux_recipe_file _flux_argv_file _flux_canonical_mode
local _k _v _line _trimmed _var_name _var_val _override_key _suffix _flux_peer _models_root

_flux_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_flux_repo_root="$(cd "${_flux_script_dir}/.." && pwd)"
_flux_recipes_dir="${_flux_repo_root}/recipes"

_flux_selected_mode="${FLUX_MODE:-}"
_flux_explicit_env_file=""
_flux_print_env=0
_flux_print_argv=0
_flux_list_modes=0
_flux_quiet="${FLUX_QUIET:-0}"
_flux_exec_cmd=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            if [[ $# -lt 2 ]]; then
                echo "[flux_env_bridge] ERROR: Missing argument for $1" >&2
                return 2
            fi
            _flux_selected_mode="$2"
            shift 2
            ;;
        -f|--env-file)
            if [[ $# -lt 2 ]]; then
                echo "[flux_env_bridge] ERROR: Missing argument for $1" >&2
                return 2
            fi
            _flux_explicit_env_file="$2"
            shift 2
            ;;
        --list-modes)
            _flux_list_modes=1
            shift
            ;;
        --print-env)
            _flux_print_env=1
            shift
            ;;
        --print-argv)
            _flux_print_argv=1
            shift
            ;;
        -q|--quiet)
            _flux_quiet=1
            shift
            ;;
        -h|--help)
            _flux_bridge_usage
            return 0
            ;;
        --)
            shift
            _flux_exec_cmd=("$@")
            break
            ;;
        *)
            echo "[flux_env_bridge] ERROR: Unknown option: $1" >&2
            _flux_bridge_usage >&2
            return 2
            ;;
    esac
done

if [[ "${_flux_list_modes}" -eq 1 ]]; then
    cat <<'EOF'
Available Flux Engine Operating Modes (recipes/):
   1. max-prefill         -> recipes/flux-v9-max-prefill.env            (argv: recipes/flux-serve.argv)
   2. interactive-w4096   -> recipes/flux-v9-w4096-interactive.env      (argv: recipes/flux-serve-w4096.argv)
   3. fast-prefill        -> recipes/flux-fast-prefill.env              (argv: recipes/flux-serve.argv)
   4. k5-decode           -> recipes/flux-k5-lossless-decode.env        (argv: recipes/flux-serve.argv)
   5. long-context-136k   -> recipes/flux-estonia-136k.env              (argv: recipes/flux-serve.argv)
   6. exact-reference     -> recipes/flux-exact-reference.env           (argv: recipes/flux-serve.argv)
   7. 256k                -> recipes/flux-qwen4-native-262k-bf16.env    (argv: recipes/flux-serve.argv)
   8. c8                  -> recipes/flux-qwen4-native-concurrency-c8.env (argv: recipes/flux-serve-w4096.argv)
   9. nextgen             -> recipes/flux-nextgen-max-prefill-decode.env (argv: recipes/flux-serve.argv)
  10. 1m-nvfp4            -> recipes/flux-nextgen-1m-nvfp4.env          (argv: recipes/flux-serve.argv)
  11. c128                -> recipes/flux-nextgen-c128-concurrency.env  (argv: recipes/flux-serve-w4096.argv)
  12. 2x-spark            -> recipes/2x-spark-tp2-disagg.env            (argv: recipes/flux-serve.argv)
  13. 3x-spark            -> recipes/3x-spark-pp3-disagg.env            (argv: recipes/flux-serve.argv)
  14. 4x-spark            -> recipes/4x-spark-cluster.env               (argv: recipes/flux-serve.argv)
EOF
    return 0
fi

# Resolve mode alias to canonical mode name, recipe .env path, and paired .argv path
_flux_recipe_file=""
_flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
_flux_canonical_mode=""

if [[ -n "${_flux_explicit_env_file}" ]]; then
    _flux_recipe_file="${_flux_explicit_env_file}"
    _flux_canonical_mode="custom"
elif [[ -n "${_flux_selected_mode}" ]]; then
    case "${_flux_selected_mode}" in
        max-prefill|v9|v9-max-prefill|w16384|serve-v9)
            _flux_canonical_mode="max-prefill"
            _flux_recipe_file="${_flux_recipes_dir}/flux-v9-max-prefill.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        interactive-w4096|w4096|v9-w4096|interactive|serve-v9-w4096)
            _flux_canonical_mode="interactive-w4096"
            _flux_recipe_file="${_flux_recipes_dir}/flux-v9-w4096-interactive.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve-w4096.argv"
            ;;
        fast-prefill|fast|unified-fast)
            _flux_canonical_mode="fast-prefill"
            _flux_recipe_file="${_flux_recipes_dir}/flux-fast-prefill.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        k5-decode|k5|k5-lossless-decode|decode|k5all)
            _flux_canonical_mode="k5-decode"
            _flux_recipe_file="${_flux_recipes_dir}/flux-k5-lossless-decode.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        long-context-136k|estonia|estonia-136k|136k)
            _flux_canonical_mode="long-context-136k"
            _flux_recipe_file="${_flux_recipes_dir}/flux-estonia-136k.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        exact-reference|exact|reference|combB)
            _flux_canonical_mode="exact-reference"
            _flux_recipe_file="${_flux_recipes_dir}/flux-exact-reference.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        256k|262k|native-262k|native262k)
            _flux_canonical_mode="256k"
            _flux_recipe_file="${_flux_recipes_dir}/flux-qwen4-native-262k-bf16.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        c8|concurrency|native-concurrency)
            _flux_canonical_mode="c8"
            _flux_recipe_file="${_flux_recipes_dir}/flux-qwen4-native-concurrency-c8.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve-w4096.argv"
            ;;
        nextgen|nextgen-max-prefill|v1.0|v1.0.0|1.0|1.0.0|v1.0.1|1.0.1|v1.0.2|1.0.2|v1.0.3|1.0.3|v0.9.16|9.16)
            _flux_canonical_mode="nextgen"
            _flux_recipe_file="${_flux_recipes_dir}/flux-nextgen-max-prefill-decode.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        1m-nvfp4|nvfp4-1m)
            _flux_canonical_mode="1m-nvfp4"
            _flux_recipe_file="${_flux_recipes_dir}/flux-nextgen-1m-nvfp4.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        c128|c64|c128-concurrency)
            _flux_canonical_mode="c128"
            _flux_recipe_file="${_flux_recipes_dir}/flux-nextgen-c128-concurrency.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve-w4096.argv"
            ;;
        2x-spark|tp2|1p1d)
            _flux_canonical_mode="2x-spark"
            _flux_recipe_file="${_flux_recipes_dir}/2x-spark-tp2-disagg.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        3x-spark|pp3|1p2d)
            _flux_canonical_mode="3x-spark"
            _flux_recipe_file="${_flux_recipes_dir}/3x-spark-pp3-disagg.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        4x-spark|tp4|ep4|4x-cluster)
            _flux_canonical_mode="4x-spark"
            _flux_recipe_file="${_flux_recipes_dir}/4x-spark-cluster.env"
            _flux_argv_file="${_flux_recipes_dir}/flux-serve.argv"
            ;;
        *)
            echo "[flux_env_bridge] ERROR: Unknown Flux operating mode '${_flux_selected_mode}'." >&2
            echo "  Valid modes: max-prefill, nextgen, interactive-w4096, fast-prefill, k5-decode, long-context-136k, exact-reference, 256k, 1m-nvfp4, c8, c128, 2x-spark, 3x-spark, 4x-spark" >&2
            return 2
            ;;
    esac
fi

# Snapshot caller-supplied FLUX_* variables so explicit user overrides win over recipe defaults
declare -A _flux_caller_overrides=()
while IFS='=' read -r _k _v; do
    if [[ "${_k}" == FLUX_* && "${_k}" != "FLUX_MODE" ]]; then
        _flux_caller_overrides["${_k}"]="${_v}"
    fi
done < <(env)

# Load the selected recipe file (if specified)
if [[ -n "${_flux_recipe_file}" ]]; then
    if [[ ! -f "${_flux_recipe_file}" ]]; then
        echo "[flux_env_bridge] ERROR: Recipe file not found: ${_flux_recipe_file}" >&2
        return 1
    fi
    while IFS= read -r _line || [[ -n "${_line}" ]]; do
        # Strip leading/trailing whitespace
        _trimmed="${_line#"${_line%%[![:space:]]*}"}"
        _trimmed="${_trimmed%"${_trimmed##*[![:space:]]}"}"
        [[ -z "${_trimmed}" || "${_trimmed}" == \#* ]] && continue
        # Strip optional leading 'export '
        if [[ "${_trimmed}" == export[[:space:]]* ]]; then
            _trimmed="${_trimmed#export }"
            _trimmed="${_trimmed#"${_trimmed%%[![:space:]]*}"}"
        fi
        if [[ "${_trimmed}" =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
            _var_name="${BASH_REMATCH[1]}"
            _var_val="${BASH_REMATCH[2]}"
            export "${_var_name}=${_var_val}"
        fi
    done < "${_flux_recipe_file}"
fi

# Default O_DIRECT routed-group fast loader and C=1 2-slab transpose layout (matching flux-autodetect)
export ATLAS_FAST_START="${ATLAS_FAST_START:-0}"
export ATLAS_FAST_LOAD_ROUTED_GROUPS="${ATLAS_FAST_LOAD_ROUTED_GROUPS:-1}"
export ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER="${ATLAS_FAST_LOAD_ROUTED_GROUPS_EAGER:-1}"
export ATLAS_FAST_LOAD_DIRECT_IO_TENSOR_CAP="${ATLAS_FAST_LOAD_DIRECT_IO_TENSOR_CAP:-20000}"
export ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE="${ATLAS_MEMAVAILABLE_FLOOR_OVERRIDE:-1}"
if [[ -z "${_flux_caller_overrides[FLUX_QWEN4_BOUNDED_TRANSPOSE_OWNERS]+x}" && -z "${ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS+x}" && -z "${FLUX_QWEN4_BOUNDED_TRANSPOSE_OWNERS+x}" ]]; then
    export ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNERS=0
    export FLUX_QWEN4_BOUNDED_TRANSPOSE_OWNERS=0
    unset ATLAS_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB
    unset FLUX_QWEN4_BOUNDED_TRANSPOSE_OWNER_CAP_MIB
fi
if [[ "${FLUX_CONCURRENCY:-1}" != "1" ]]; then
    if [[ -z "${_flux_caller_overrides[FLUX_QWEN4_PREFILL_QSA_FP8]+x}" ]]; then
        export ATLAS_QWEN4_PREFILL_QSA_FP8=0
        export FLUX_QWEN4_PREFILL_QSA_FP8=0
    fi
    if [[ -z "${_flux_caller_overrides[FLUX_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE]+x}" ]]; then
        export ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=0
        export FLUX_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE=0
    fi
    if [[ -z "${_flux_caller_overrides[FLUX_QWEN4_VERIFY_GRAPH]+x}" ]]; then
        export ATLAS_QWEN4_VERIFY_GRAPH=0
        export FLUX_QWEN4_VERIFY_GRAPH=0
    fi
    if [[ -z "${_flux_caller_overrides[FLUX_SCHED_BURST_GATHER_MS]+x}" ]]; then
        export ATLAS_SCHED_BURST_GATHER_MS=50
        export FLUX_SCHED_BURST_GATHER_MS=50
    fi
fi

# Re-apply caller-supplied FLUX_* overrides so user overrides take highest precedence
for _override_key in "${!_flux_caller_overrides[@]}"; do
    export "${_override_key}=${_flux_caller_overrides[${_override_key}]}"
done

if [[ -n "${_flux_canonical_mode}" ]]; then
    export FLUX_MODE="${_flux_canonical_mode}"
    export FLUX_ACTIVE_ARGV_FILE="${_flux_argv_file}"
fi

# Meta-variables that belong solely to the Flux wrapper/launcher layer and should not become ATLAS_*
# (mirrors _FLUX_META_ENV_KEYS in repro-kit/flux-autodetect and META_KEYS in crates/spark-server/src/main.rs;
# intentionally leaves FLUX_DISABLE_SPECULATION translatable to ATLAS_DISABLE_SPECULATION).
_flux_is_meta_key() {
    case "$1" in
        FLUX_MODE|FLUX_QUIET|FLUX_MODELS_ROOT|FLUX_ATLAS_ROOT|FLUX_ACTIVE_ARGV_FILE|FLUX_PORT|FLUX_HOST|FLUX_MAX_SEQ_LEN|FLUX_MAX_PREFILL_TOKENS|FLUX_GPU_MEMORY_UTILIZATION|FLUX_KV_CACHE_DTYPE|FLUX_KV_POOL_TOKENS|FLUX_SPEC_MODE|FLUX_DRAFT_MODEL|FLUX_MTP_PATH|FLUX_MTP_VOCAB|FLUX_GAMMA|FLUX_DRAFTER_PATH|FLUX_THINKING|FLUX_THINKING_BUDGET|FLUX_RECIPE|FLUX_RECIPE_FILE|FLUX_EXTRA_ARGS|FLUX_BIN|FLUX_DRY_RUN|FLUX_CONCURRENCY|FLUX_TOPOLOGY|FLUX_TARGET|FLUX_MODEL|FLUX_MODEL_ID|FLUX_MODEL_NAME|FLUX_OPT_OUT|FLUX_LOAD_DEBUG|FLUX_LOAD_DEBUG_FILE|FLUX_DEBUG|FLUX_DEBUG_FILE|FLUX_DEBUG_MAX_MB|FLUX_DEBUG_FILTER|FLUX_CLUSTER_WORLD_SIZE|FLUX_TENSOR_PARALLEL_SIZE|FLUX_EXPERT_PARALLEL_SIZE|FLUX_PIPELINE_PARALLEL_SIZE|FLUX_TP_SIZE|FLUX_EP_SIZE|FLUX_PP_SIZE|FLUX_WORLD_SIZE|FLUX_NODE_RANK|FLUX_MASTER_ADDR|FLUX_MASTER_PORT|FLUX_CPU_AFFINITY)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Translate every FLUX_<SUFFIX> -> ATLAS_<SUFFIX> (including shorthand aliases)
while IFS='=' read -r _k _v; do
    if [[ "${_k}" == FLUX_* ]] && ! _flux_is_meta_key "${_k}"; then
        _suffix="${_k#FLUX_}"
        # Handle convenient shorthand aliases
        case "${_suffix}" in
            QWEN4_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB)
                export "ATLAS_QWEN4_PREFILL_FAST_PROJ_FP8_WEIGHT_CACHE_RESERVE_GB=${_v}"
                ;;
            PREFILL_WIDE)
                export "ATLAS_QWEN4_PREFILL_WIDE=${_v}"
                ;;
        esac
        export "ATLAS_${_suffix}=${_v}"
    fi
done < <(env)

# Also mirror active ATLAS_<SUFFIX> -> FLUX_<SUFFIX> if not already set so inspection tools see both
while IFS='=' read -r _k _v; do
    if [[ "${_k}" == ATLAS_* ]]; then
        _suffix="${_k#ATLAS_}"
        _flux_peer="FLUX_${_suffix}"
        if [[ -z "${!_flux_peer+x}" ]]; then
            export "${_flux_peer}=${_v}"
        fi
    fi
done < <(env)

if [[ "${_flux_quiet}" -ne 1 && "${_flux_print_env}" -eq 0 && "${_flux_print_argv}" -eq 0 ]]; then
    if [[ -n "${_flux_canonical_mode}" ]]; then
        echo "[flux_env_bridge] Activated FLUX_MODE=${_flux_canonical_mode} (FLUX_* <-> ATLAS_* synchronized)" >&2
    else
        echo "[flux_env_bridge] Synchronized FLUX_* <-> ATLAS_* environment selectors" >&2
    fi
fi

if [[ "${_flux_print_argv}" -eq 1 ]]; then
    _models_root="${FLUX_MODELS_ROOT:-${MODELS_ROOT:-<MODELS_ROOT>}}"
    sed "s|<MODELS_ROOT>|${_models_root}|g" "${_flux_argv_file}"
    return 0
fi

if [[ "${_flux_print_env}" -eq 1 ]]; then
    env | grep -E '^(FLUX_|ATLAS_|RUST_LOG=)' | sort
    return 0
fi

if [[ "${#_flux_exec_cmd[@]}" -gt 0 ]]; then
    exec "${_flux_exec_cmd[@]}"
fi

return 0
}

_flux_bridge_main "$@"
_flux_rc=$?
if [[ "${_flux_is_sourced}" -eq 1 ]]; then
    return "${_flux_rc}"
else
    exit "${_flux_rc}"
fi
