#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/NicheFlow_Server_Handoff/NicheFlow_Evaluation_v052 || exit 1
run_dir=runs/comparison_v052_recovered
if [ -e "$run_dir" ] || [ -e runs/comparison_v052_recovered.exit_code.txt ]; then
  echo 'Recovery already has output; refusing implicit restart.'
  exit 2
fi
.venv/bin/python scripts/recover_v052_comparison.py \
  --source runs/comparison_v052_before_after --out-dir "$run_dir" \
  --config configs/main_math_v051_local_flash_full_eval.json --execute \
  > runs/comparison_v052_recovered.console.log 2>&1
result=$?
printf '%s\n' "$result" > runs/comparison_v052_recovered.exit_code.txt
.venv/bin/python scripts/render_v052_comparison.py \
  --run "$run_dir" --exit-code-file runs/comparison_v052_recovered.exit_code.txt \
  > runs/comparison_v052_recovered.report.log 2>&1
report_result=$?
printf '%s\n' "$report_result" > runs/comparison_v052_recovered.report_exit_code.txt
exit "$result"
