#!/usr/bin/env python3
"""Local scoring subprocess; stdin contains private labels, never sent to models."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from nicheflow.datasets import Task
from nicheflow.v072.scoring import score,selftest
if '--selftest' in sys.argv:
    print(json.dumps(selftest(),ensure_ascii=False))
else:
    record=json.load(sys.stdin)
    print(json.dumps(score(Task(**record['task']),record['response']),ensure_ascii=False))
