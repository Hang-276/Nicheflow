#!/usr/bin/env bash
# Exactly one user-authorized 30-round stage. No retries or terminal evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir=runs/main_v051_local_flash_r30
config=configs/main_math_v051_local_flash.json
mkdir -p runs
if ! mkdir "$run_dir"; then
  echo 'Run directory already exists; inspect it without restarting or overwriting.'
  exit 4
fi
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$run_dir/launched_at.txt"
set +e
PYTHONUNBUFFERED=1 bash scripts/nicheflow.sh run --mode main --config "$config" --rounds 30 --run-dir "$run_dir" > "$run_dir/console.log" 2>&1
run_exit=$?
set -e
printf '%s\n' "$run_exit" > "$run_dir/exit_code.txt"
echo "30-round stage exited with code $run_exit; checkpoint and evidence retained. No automatic continuation."
exit "$run_exit"
