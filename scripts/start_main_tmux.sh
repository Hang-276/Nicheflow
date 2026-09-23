#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -gt 2 ]]; then echo 'Usage: start_main_tmux.sh [quick|full] [rounds]'; exit 2; fi
profile=${1:-quick}
rounds=${2:-1}
if [[ ! "$rounds" =~ ^[1-9][0-9]*$ ]]; then echo "rounds must be a positive integer"; exit 2; fi
case "$profile" in
  quick) config=configs/main_math.json ;;
  full) config=configs/main_math_full.json ;;
  *) echo 'Usage: start_main_tmux.sh [quick|full] [rounds]'; exit 2 ;;
esac
session="nicheflow-main-v044-${profile}-r${rounds}"
run_dir="runs/main_v044_${profile}_r${rounds}"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists. Inspect it; do not start another run."
  exit 4
fi
if [[ -e "$run_dir" ]]; then
  echo "Run directory already exists. Inspect it; no automatic restart."
  exit 4
fi
while IFS= read -r other; do
  [[ "$other" == nicheflow-main-* ]] || continue
  if tmux list-panes -t "$other" -F '#{pane_dead}' | grep -q '^0$'; then
    echo "Another main experiment is active: $other. Inspect it; do not start or replace it."
    exit 4
  fi
done < <(tmux list-sessions -F '#{session_name}' 2>/dev/null || true)
# Zero inference: fail before consuming the unique run directory or opening tmux.
bash scripts/nicheflow.sh main-plan --config "$config" --rounds "$rounds"
tmux new-session -d -s "$session" -c "$PWD"
tmux set-option -w -t "$session:0" remain-on-exit on
tmux send-keys -t "$session:0.0" "exec bash scripts/run_main_once.sh $profile $rounds" C-m
echo "Started $session for exactly $rounds rounds. No automatic restarts or formal runs are configured."
