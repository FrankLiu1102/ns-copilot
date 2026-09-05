#!/usr/bin/env bash
# ==============================================================================
# NS-Copilot Reproducibility Runner
#
# Usage (auto mode — default):
#   ./run_test.sh --domain wm --dataset dandi_000006 --count 2
#   ./run_test.sh --domain ad --dataset ds004504 --count 10
#   ./run_test.sh --domain pd --dataset ds004584 --count 5
#
# Usage (single model):
#   ./run_test.sh --domain wm --dataset dandi_000006 --model POYO --count 3
#
# Arguments:
#   --domain   : wm | ad | pd                          (required)
#   --dataset  : dandi_000006 | ds004504 | ds004584     (required)
#   --count    : number of runs with random seeds       (required)
#   --model    : specific model to run (omit for auto mode)
#   --seeds    : comma-separated seeds (e.g., 42,123)   (optional, overrides count)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Defaults ----
DOMAIN=""
DATASET=""
MODEL=""
COUNT=""
SEEDS=""

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain)  DOMAIN="$(echo "$2" | tr '[:upper:]' '[:lower:]')"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --model)   MODEL="$2"; shift 2 ;;
        --count)   COUNT="$2"; shift 2 ;;
        --seeds)   SEEDS="$2"; shift 2 ;;
        -h|--help)
            head -17 "$0" | tail -15
            exit 0
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            exit 1
            ;;
    esac
done

# ---- Validate required arguments ----
if [[ -z "$DOMAIN" || -z "$DATASET" ]]; then
    echo "Error: --domain and --dataset are required."
    echo "Usage: ./run_test.sh --domain wm --dataset dandi_000006 --count 2"
    exit 1
fi

if [[ -z "$COUNT" && -z "$SEEDS" ]]; then
    echo "Error: --count or --seeds is required."
    exit 1
fi

# ---- Validate domain ----
if [[ "$DOMAIN" != "wm" && "$DOMAIN" != "ad" && "$DOMAIN" != "pd" ]]; then
    echo "Error: --domain must be one of: wm, ad, pd"
    exit 1
fi

# ---- Validate domain + dataset ----
case "$DOMAIN" in
    wm)
        if [[ "$DATASET" != "dandi_000006" ]]; then
            echo "Error: domain 'wm' only supports dataset 'dandi_000006'"; exit 1
        fi ;;
    ad)
        if [[ "$DATASET" != "ds004504" ]]; then
            echo "Error: domain 'ad' only supports dataset 'ds004504'"; exit 1
        fi ;;
    pd)
        if [[ "$DATASET" != "ds004584" ]]; then
            echo "Error: domain 'pd' only supports dataset 'ds004584'"; exit 1
        fi ;;
esac

# ---- Resolve prompt file ----
PROMPT_DIR="$SCRIPT_DIR/prompt"
case "$DOMAIN" in
    wm)
        DATASET_UPPER="DANDI_000006"
        if [[ -n "$MODEL" ]]; then
            PROMPT_FILE="$PROMPT_DIR/WM/WM_${DATASET_UPPER}_${MODEL}_prompt.md"
        else
            PROMPT_FILE="$PROMPT_DIR/WM/WM_${DATASET_UPPER}_auto_prompt.md"
        fi
        ;;
    ad)
        if [[ -n "$MODEL" ]]; then
            PROMPT_FILE="$PROMPT_DIR/AD/AD_${DATASET}_${MODEL}_prompt.md"
        else
            PROMPT_FILE="$PROMPT_DIR/AD/AD_${DATASET}_auto_prompt.md"
        fi
        ;;
    pd)
        if [[ -n "$MODEL" ]]; then
            PROMPT_FILE="$PROMPT_DIR/PD/PD_${DATASET}_${MODEL}_prompt.md"
        else
            PROMPT_FILE="$PROMPT_DIR/PD/PD_${DATASET}_auto_prompt.md"
        fi
        ;;
esac

if [[ ! -f "$PROMPT_FILE" ]]; then
    echo "Error: Prompt file not found: $PROMPT_FILE"
    exit 1
fi

# ---- Resolve dataset path (container) ----
case "$DATASET" in
    dandi_000006) DATASET_PATH_CONTAINER="/app/dataset/Working_Memory/dandi_000006" ;;
    ds004504)     DATASET_PATH_CONTAINER="/app/dataset/AD/ds004504" ;;
    ds004584)     DATASET_PATH_CONTAINER="/app/dataset/Parkinson/ds004584" ;;
esac

# ---- Check Docker container is running ----
CONTAINER_NAME="neuro-copilot-web"
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Error: Docker container '$CONTAINER_NAME' is not running."
    echo "Please start it first: docker-compose up -d --build"
    exit 1
fi

# ---- Build batch command ----
BATCH_ARGS="--prompt-file /app/prompt/$(basename "$(dirname "$PROMPT_FILE")")/$(basename "$PROMPT_FILE")"
BATCH_ARGS="$BATCH_ARGS --dataset $DATASET_PATH_CONTAINER"

if [[ -n "$SEEDS" ]]; then
    BATCH_ARGS="$BATCH_ARGS --seeds $SEEDS"
else
    BATCH_ARGS="$BATCH_ARGS --n $COUNT"
fi

# ---- Print configuration ----
MODE="auto"
[[ -n "$MODEL" ]] && MODE="single ($MODEL)"
echo "============================================================"
echo "  NS-Copilot Reproducibility Runner"
echo "============================================================"
echo "  Domain:      $DOMAIN"
echo "  Dataset:     $DATASET"
echo "  Mode:        $MODE"
echo "  Runs:        ${COUNT:-$(echo "$SEEDS" | tr ',' '\n' | wc -l | tr -d ' ')}"
[[ -n "$SEEDS" ]] && echo "  Seeds:       $SEEDS"
echo "  Prompt:      $PROMPT_FILE"
echo "============================================================"
echo ""

# ---- Execute ----
# `docker exec` does not inherit the host environment, so an optional
# AUTO_TARGET_VALUE has to be forwarded explicitly.
TARGET_ENV=()
if [[ -n "${AUTO_TARGET_VALUE:-}" ]]; then
    TARGET_ENV+=(-e "AUTO_TARGET_VALUE=$AUTO_TARGET_VALUE")
fi

exec docker exec ${TARGET_ENV[@]+"${TARGET_ENV[@]}"} "$CONTAINER_NAME" python /app/run_batch.py $BATCH_ARGS
