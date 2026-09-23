#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 1
name=fixed_validation_v060
mkdir -p runs
if [ -e "runs/$name" ]; then
  echo 'Output already exists; explicit audited recovery required.' >&2
  exit 2
fi
date -Iseconds > "runs/$name.launched_at.txt"
export PYTHONUNBUFFERED=1
.venv/bin/python scripts/run_fixed_validation.py \
  --config configs/fixed_validation_v060.json \
  --out-dir "runs/$name" --execute > "runs/$name.console.log" 2>&1
status=$?
printf '%s\n' "$status" > "runs/$name.exit_code.txt"
exit "$status"
