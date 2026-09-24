#!/usr/bin/env python3
"""Run each authorized bounded search once; never auto-resume interrupted runs."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
jobs=[('search-seeds',['--mode','search-seeds'])]
for seed,order in [(2026092401,['elite','mixed']),(2026092402,['mixed','elite'])]:
    jobs.extend((f'search_{strategy}_{seed}',['--mode','search','--strategy',strategy,'--seed',str(seed)]) for strategy in order)
for name,args in jobs:
    path=ROOT/'runs/prescale_v073'/name/'state.sqlite3'
    if path.exists():
        db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
        done=db.execute("SELECT 1 FROM artifacts WHERE key='complete'").fetchone();db.close()
        if done:continue
        print(json.dumps({'status':'manual_reconciliation_required','run':name,'message':'Existing incomplete run; inspect pause, requests and writer. Do not retry unknown or quota-rejected calls automatically.'}));raise SystemExit(2)
    result=subprocess.run(['bash',str(ROOT/'scripts/launch_v073.sh'),*args],cwd=ROOT)
    if result.returncode:raise SystemExit(result.returncode)
raise SystemExit(subprocess.run([sys.executable,str(ROOT/'scripts/analyze_v073.py')],cwd=ROOT).returncode)
