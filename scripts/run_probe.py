#!/usr/bin/env python
"""Run the frozen NicheFlow prerequisite probe.

Usage:
    python scripts/run_probe.py                     # new run under runs/
    python scripts/run_probe.py --run-dir runs/<id> # resume that run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nicheflow_probe import report  # noqa: E402
from nicheflow_probe.backends import TransformersChatBackend  # noqa: E402
from nicheflow_probe.runner import (  # noqa: E402
    Runner, build_plan, file_sha256, load_tasks,
)

WARMUP_PROBLEMS = ["Compute 17 + 25.", "Compute 91 - 46."]


def warm_up(backend, config, log_path: Path) -> list[dict]:
    """At most two short calls on independent problems; never part of the results.

    The allowance is spent once: an existing warmup.json is never overwritten, so
    resuming a run cannot add warm-up calls to the ones already recorded.
    """
    if log_path.exists():
        return json.loads(log_path.read_text())["calls"]
    records = []
    for index, problem in enumerate(WARMUP_PROBLEMS):
        payload = [
            {"role": "system", "content": "You solve mathematics problems."},
            {"role": "user", "content": problem + " Answer with a single number."},
        ]
        result = backend.generate(payload, seed=config["generation_seed"] + index,
                                  max_new_tokens=64, temperature=config["temperature"],
                                  top_p=config["top_p"])
        records.append({"problem": problem, "index": index, "status": result["status"],
                        "text": result["text"], "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "finish_reason": result["finish_reason"],
                        "error": result.get("error")})
    log_path.write_text(json.dumps({"note": "warm-up only; excluded from all probe results",
                                    "calls": records}, indent=2, ensure_ascii=False))
    return records


def record_energy(run_dir: Path, load_seconds: float, wall_seconds: float,
                  budget_seconds: float, new_slots: int) -> dict:
    """Append this invocation as a segment; a resume never overwrites earlier cost.

    An invocation that executes no new slot (pure report regeneration) is not
    counted as formal experiment time.
    """
    path = run_dir / "energy.json"
    payload = json.loads(path.read_text()) if path.exists() else {"segments": []}
    if "segments" not in payload:  # migrate a pre-segment energy.json without losing it
        payload = {"segments": [{
            "started_utc": None,
            "wall_seconds": payload.get("formal_experiment_wall_seconds"),
            "model_load_seconds": payload.get("model_load_seconds"),
            "budget_seconds": payload.get("formal_budget_seconds"),
            "new_slots_executed": None,
            "note": "recorded before segment accounting was added",
        }]}
    if new_slots > 0 or not payload["segments"]:
        payload["segments"].append({
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(time.time() - wall_seconds)),
            "wall_seconds": wall_seconds,
            "model_load_seconds": load_seconds,
            "budget_seconds": budget_seconds,
            "new_slots_executed": new_slots,
        })
    payload["formal_experiment_wall_seconds"] = sum(s["wall_seconds"] for s in payload["segments"])
    payload["note"] = ("wall clock only; GPU utilisation was not sampled, so no GPU-hours "
                       "are claimed")
    path.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--stages", default="ABC")
    parser.add_argument("--max-seconds", type=float, default=3600.0,
                        help="wall-clock budget for the formal experiment")
    parser.add_argument("--warmup", action="store_true", help="run up to 2 warm-up calls first")
    parser.add_argument("--backend", default="transformers", choices=["transformers", "replay"])
    args = parser.parse_args()

    config = json.loads((ROOT / "configs" / "probe.json").read_text())
    manifest = json.loads((ROOT / "data" / "manifest.json").read_text())
    tasks_path = ROOT / "data" / "tasks.jsonl"
    tasks_sha256 = file_sha256(tasks_path)
    if tasks_sha256 != manifest["tasks_sha256"]:
        print(f"data SHA256 mismatch: {tasks_sha256} != {manifest['tasks_sha256']}")
        return 2
    tasks = load_tasks(tasks_path)
    if [task["id"] for task in tasks] != manifest["task_ids"]:
        print("task order differs from manifest")
        return 2
    repeat_ids = manifest["repeat_ids"]
    assert len(repeat_ids) == 4 and len(set(repeat_ids)) == 4, "expected 4 distinct repeat ids"
    assert set(repeat_ids) <= {task["id"] for task in tasks}, "repeat ids must be known tasks"
    print(f"preflight ok: 12 unique tasks, 4 repeat ids, all gold answers parseable, "
          f"tasks_sha256={tasks_sha256[:16]}...")

    plan = build_plan(tasks, repeat_ids)
    planned_calls = sum(len(entry["node_ids"]) for entry in plan)
    print(f"plan: {len(plan)} slots, {planned_calls} planned model calls "
          f"(A={sum(1 for e in plan if e['stage'] == 'A')}, "
          f"B={sum(1 for e in plan if e['stage'] == 'B')}, "
          f"C={sum(1 for e in plan if e['stage'] == 'C')})")

    run_id = Path(args.run_dir).name if args.run_dir else time.strftime("probe_%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(args.run_dir) if args.run_dir else ROOT / "runs" / run_id

    if args.backend == "replay":
        from nicheflow_probe.backends import ReplayBackend
        backend = ReplayBackend()
        environment = {"backend": "replay (test only)", "model": {"requested_model_id": "replay"},
                       "gpu": {"name": "none", "total_vram_bytes": 0, "cuda_runtime": "n/a"}}
        load_seconds = 0.0
    else:
        model_id = config["model_id"]
        print(f"loading {model_id} ...")
        backend = TransformersChatBackend(model_id, max_context_tokens=config["max_context_tokens"])
        load_seconds = backend.load_seconds
        environment = backend.environment()
        print(f"loaded in {load_seconds:.1f}s; "
              f"AWQ layers={environment['model']['num_awq_linear_layers']}; "
              f"kernels={environment['awq_kernels']}")

    runner = Runner(run_dir, backend, tasks, config, plan, tasks_sha256, run_id)
    run_config = runner.initialise()
    if not (run_dir / "environment.json").exists():
        (run_dir / "environment.json").write_text(
            json.dumps(environment, indent=2, ensure_ascii=False))

    if args.warmup:
        records = warm_up(backend, config, run_dir / "warmup.json")
        print("warm-up:", [(r["status"], r["output_tokens"]) for r in records])

    deadline = time.time() + args.max_seconds
    started = time.time()
    completed_before = len(runner._completed)
    first_record = {"t": None}

    def progress(record):
        if first_record["t"] is None:
            first_record["t"] = time.time()
        done = len(runner._completed)
        print(f"[{done}/{len(plan)}] {record['execution_id']} "
              f"status={record['status']} correct={record['correct']} "
              f"answer={record['extracted_answer']!r} "
              f"tokens={record['total_output_tokens']} "
              f"{record['total_node_seconds']:.1f}s", flush=True)

    for stage in args.stages:
        remaining = max(0.0, deadline - time.time())
        result = runner.run(stages=(stage,), max_seconds=remaining, progress=progress)
        if stage == "A":
            summary = report.build_summary(run_dir, tasks, plan)
            report.write_matrix(run_dir, tasks, plan)
            report.write_summary(run_dir, summary)
            report.write_interim(run_dir, summary, tasks, plan)
            print("interim_report.md written after stage A")
        if result["stopped_for"]:
            print(f"stopped after stage {stage}: {result['stopped_for']}")

    record_energy(run_dir, load_seconds, time.time() - started, args.max_seconds,
                  len(runner._completed) - completed_before)
    summary = report.build_summary(run_dir, tasks, plan)
    report.write_matrix(run_dir, tasks, plan)
    report.write_summary(run_dir, summary)
    report.write_final(run_dir, summary, tasks, environment, run_config)
    print(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
