#!/usr/bin/env bash
# User-run credential entry only; does not contact the API or start an experiment.
set +x
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -t 0 ]]; then
  echo "Run interactively in your SSH terminal; never pass a key as a command argument." >&2
  exit 3
fi
if [[ -e /root/.config/nicheflow/credentials.env ]]; then
  echo "Credential file already exists. Preserved without reading or overwriting it."
  exit 0
fi
umask 077
read -r -s -p 'DeepSeek API key (hidden; saved only on this server): ' nicheflow_api_key
printf '\n' >&2
if [[ -z "$nicheflow_api_key" ]]; then
  echo "No key supplied; nothing saved." >&2
  exit 3
fi
printf '%s' "$nicheflow_api_key" | /root/miniconda3/bin/python -c '
import os, pathlib, sys
key = sys.stdin.read().strip()
if not key or "\n" in key or "\r" in key:
    raise SystemExit("Invalid empty or multiline credential; nothing saved.")
p = pathlib.Path("/root/.config/nicheflow")
p.mkdir(mode=0o700, parents=True, exist_ok=True)
p.chmod(0o700)
fd = os.open(p / "credentials.env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as f:
    f.write("DEEPSEEK_API_KEY=" + key + "\n")
    f.flush()
    os.fsync(f.fileno())
print("Credential saved with mode 0600. No API request or experiment started.")'
unset nicheflow_api_key
