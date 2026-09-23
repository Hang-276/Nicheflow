#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
session=nicheflow-smoke-v030
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Session already exists. Inspect it; do not start another smoke."
  exit 4
fi
if [[ -e runs/smoke_v030 ]]; then
  echo "Run directory already exists. Inspect it; no automatic restart."
  exit 4
fi
tmux new-session -d -s "$session" -c "$PWD"
tmux set-option -w -t "$session:0" remain-on-exit on
tmux send-keys -t "$session:0.0" 'exec bash scripts/run_smoke_once.sh' C-m
echo "Started $session. No automatic restarts are configured."
