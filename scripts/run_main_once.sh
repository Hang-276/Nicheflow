#!/usr/bin/env bash
# One authorized main-method round. No retries, installations or budget changes.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -gt 2 ]]; then echo 'Usage: run_main_once.sh [quick|full] [rounds]'; exit 2; fi
profile=${1:-quick}
rounds=${2:-1}
if [[ ! "$rounds" =~ ^[1-9][0-9]*$ ]]; then echo "rounds must be a positive integer"; exit 2; fi
case "$profile" in
  quick) config=configs/main_math.json ;;
  full) config=configs/main_math_full.json ;;
  *) echo 'Usage: run_main_once.sh [quick|full] [rounds]'; exit 2 ;;
esac
run_dir="runs/main_v044_${profile}_r${rounds}"
mkdir -p runs
if ! mkdir "$run_dir"; then
  echo "Run directory already exists. Inspect its evidence; do not restart or overwrite."
  exit 4
fi
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$run_dir/launched_at.txt"
set +e
PYTHONUNBUFFERED=1 bash scripts/nicheflow.sh run --mode main \
  --config "$config" --rounds "$rounds" --run-dir "$run_dir" 2>&1 | tee "$run_dir/console.log"
run_exit=${PIPESTATUS[0]}
set -e
printf '%s\n' "$run_exit" > "$run_dir/exit_code.txt"
echo "Main-method round ended with exit code $run_exit. Read evidence; do not auto-retry or start formal training."
exit "$run_exit"
