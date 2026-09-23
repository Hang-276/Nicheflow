#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/NicheFlow_Server_Handoff/NicheFlow_Evaluation_v052
mkdir -p runs
if [ -e runs/comparison_v052_before_after ]; then
  echo 'Comparison directory already exists; refusing overwrite or automatic retry.'
  exit 2
fi
date -u '+%Y-%m-%dT%H:%M:%SZ' > runs/comparison_v052.launched_at.txt
.venv/bin/python -u scripts/compare_research_checkpoints.py \
  --execute \
  --source /root/autodl-tmp/NicheFlow_Server_Handoff/NicheFlow_Revision_v051/runs/main_v051_local_flash_r30 \
  --config configs/main_math_v051_local_flash_full_eval.json \
  --compatibility-manifest configs/checkpoint_compat_v051_to_v052.json \
  --out-dir runs/comparison_v052_before_after \
  > runs/comparison_v052.console.log 2>&1
result=$?
printf '%s\n' "$result" > runs/comparison_v052.exit_code.txt
exit "$result"
