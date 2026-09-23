#!/usr/bin/env python3
"""One-shot detached pilot then offline scoring; no retry or extra inference."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'NicheFlow_ModelSelection_v070'
exit_code=1
try:
    exit_code=subprocess.call([sys.executable,str(ROOT/'scripts/qwen38_screen_v071.py'),
                               '--base',str(BASE),'--stage','run'],cwd=ROOT)
    if exit_code==0:
        exit_code=subprocess.call([sys.executable,str(ROOT/'scripts/score_qwen38_v071.py'),
                                   '--base',str(BASE)],cwd=ROOT)
finally:
    p=ROOT/'exit.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps({'exit_code':exit_code,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                              'scoring_is_provisional':True,'automatic_retries':0},indent=2)+'\n')
    tmp.replace(p)
sys.exit(exit_code)
