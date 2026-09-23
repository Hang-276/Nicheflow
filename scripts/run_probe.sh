#!/usr/bin/env bash
# Run or resume the NicheFlow prerequisite probe.
#
#   bash scripts/run_probe.sh                      # new run
#   bash scripts/run_probe.sh runs/<run_id>        # resume that run
#
# The model is loaded from the local Hugging Face cache; no network access is
# needed once Qwen/Qwen2.5-7B-Instruct-AWQ has been downloaded.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_DIR="${1:-}"
export HF_HUB_OFFLINE=1          # inference is local; never fetch during a run
export TOKENIZERS_PARALLELISM=false

if [[ -n "$RUN_DIR" ]]; then
  echo "resuming $RUN_DIR"
  exec python -u scripts/run_probe.py --run-dir "$RUN_DIR" --max-seconds 3600
fi

exec python -u scripts/run_probe.py --warmup --max-seconds 3600
