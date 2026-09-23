#!/usr/bin/env bash
# Frozen monitor entry. No retries, environment installation or automatic resume.
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir=runs/smoke_v030
mkdir -p runs
if ! mkdir "$run_dir"; then
  echo "Run directory already exists. Read its evidence; do not restart or overwrite."
  exit 4
fi
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$run_dir/launched_at.txt"
set +e
PYTHONUNBUFFERED=1 bash scripts/nicheflow.sh run --mode full-smoke \
  --config configs/smoke.json --run-dir "$run_dir" 2>&1 | tee "$run_dir/console.log"
run_exit=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_exit" > "$run_dir/exit_code.txt"
echo "Smoke ended with exit code $run_exit. Read summary.json; do not auto-retry."
exit "$run_exit"
