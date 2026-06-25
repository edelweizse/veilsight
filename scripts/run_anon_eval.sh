#!/usr/bin/env bash
# run_anon_eval.sh — Orchestrate anonymizer eval runs across 6 config variants.
#
# Usage:
#   bash scripts/run_anon_eval.sh [--dry-run] [--sequences <seq1,seq2,...>]
#
# Options:
#   --dry-run     Print what would run without executing
#   --sequences   Comma-separated sequence list (default: P1L_S3_C1,P1L_S3_C2)
#   --skip-build  Pass --skip-build to run_chokepoint_eval (default: always passed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ——— CLI args ———
DRY_RUN=false
SEQUENCES_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --sequences)
            SEQUENCES_ARG="--sequences $(echo "$2" | tr ',' ' ')"
            shift 2 ;;
        --skip-build) shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

VARIANTS=(
    "pixelate|weak|10|31|pixelate_weak"
    "pixelate|medium|25|31|pixelate_medium"
    "pixelate|huge|50|31|pixelate_huge"
    "blur|weak|50|15|blur_weak"
    "blur|medium|50|31|blur_medium"
    "blur|huge|50|51|blur_huge"
)

EVAL_SCRIPT="$REPO_ROOT/scripts/run_chokepoint_eval.py"
EVAL_CONFIG_TEMPLATE="$REPO_ROOT/configs/chokepoint_eval.yaml"
TMPDIR="/tmp/veilsight-anon-eval-$$"
mkdir -p "$TMPDIR"

# ——— Dry-run ———
if $DRY_RUN; then
    echo "DRY RUN: would execute the following 6 eval runs"
    echo "================================================"
    for variant in "${VARIANTS[@]}"; do
        IFS='|' read -r method intensity divisor kernel label <<< "$variant"
        out_dir="$REPO_ROOT/results/veilsight/chokepoint_anon_$label"
        config_file="configs/eval_anon_$label.yaml"
        echo "  [$method/$intensity]"
        echo "    config:     $config_file"
        echo "    output_dir: $out_dir"
        echo ""
    done
    rm -rf "$TMPDIR"
    exit 0
fi

# ——— Execution ———
echo "=== Anonymizer Eval Runner ==="
echo "Repo:   $REPO_ROOT"
echo "Temp:   $TMPDIR"
echo ""

FAILED=()
TOTAL=${#VARIANTS[@]}
CURRENT=0

for variant in "${VARIANTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    IFS='|' read -r method intensity divisor kernel label <<< "$variant"

    config_file="$REPO_ROOT/configs/eval_anon_$label.yaml"
    out_dir="$REPO_ROOT/results/veilsight/chokepoint_anon_$label"
    eval_config="$TMPDIR/eval_${label}.yaml"

    echo "=== [$CURRENT/$TOTAL] $method / $intensity ==="
    echo "  Config:   eval_anon_$label.yaml"
    echo "  Output:   results/veilsight/chokepoint_anon_$label"

    # Generate temp eval config with correct base_config and output_root
    python3 -c "
import yaml

with open('$EVAL_CONFIG_TEMPLATE') as f:
    cfg = yaml.safe_load(f)

cfg['paths']['base_config'] = 'configs/eval_anon_$label.yaml'
cfg['paths']['output_root'] = 'results/veilsight/chokepoint_anon_$label'

with open('$eval_config', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" || {
        echo "  FAIL: could not generate eval config"
        FAILED+=("$label")
        continue
    }

    CMD_ARGS=(
        python3 "$EVAL_SCRIPT"
        --config "$eval_config"
        --skip-build
    )
    [[ -n "$SEQUENCES_ARG" ]] && CMD_ARGS+=($SEQUENCES_ARG)

    echo "  Cmd:     ${CMD_ARGS[*]}"

    set +e
    "${CMD_ARGS[@]}"
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "  FAIL: run_chokepoint_eval exited with code $EXIT_CODE"
        FAILED+=("$label")
    else
        echo "  OK: completed"
    fi
    echo ""
done

# ——— Summary ———
echo "========================================"
echo "Runner summary:"
echo "  Total:   $TOTAL"
echo "  Success: $((TOTAL - ${#FAILED[@]}))"
echo "  Failed:  ${#FAILED[@]}"

if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed runs: ${FAILED[*]}"
    rm -rf "$TMPDIR"
    exit 1
fi

echo "  All runs successful."
rm -rf "$TMPDIR"
