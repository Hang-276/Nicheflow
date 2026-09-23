"""Frozen execution plan, per-node seeds, append-only logs and resumable runs.

The plan is written to disk before any model call, so no slot can be selected
after seeing the scores.  Every model call is appended to ``calls.jsonl`` and
every finished slot to ``executions.jsonl`` immediately, which makes a run
resumable without recomputing or double-counting anything.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

from .evaluation import score
from .workflows import WORKFLOWS, messages

GENERATION_SEED = 73129
STAGE_A_TASKS = 6
MAX_RETRIES_TOTAL = 8          # extra model calls allowed for transient errors
MAX_CONSECUTIVE_FAILURES = 3
TERMINAL_STATUSES = ("ok", "failed")


def node_seed(task_id: str, workflow_id: str, repeat_index: int, node_id: str) -> int:
    key = f"{GENERATION_SEED}|{task_id}|{workflow_id}|{repeat_index}|{node_id}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31)


def canonical_hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tasks(path: Path) -> list[dict]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for task in tasks:
        score(r"\boxed{0}", task["gold"])  # raises if the reference is unparseable
    return tasks


def build_plan(tasks: list[dict], repeat_ids: list[str], workflows=WORKFLOWS) -> list[dict]:
    """Full slot list: stage A, stage B, then stage C repeats. Order fixed here."""
    by_id = {task["id"]: task for task in tasks}
    plan: list[dict] = []
    stages = [
        ("A", tasks[:STAGE_A_TASKS], 0, "main run, first 6 tasks"),
        ("B", tasks[STAGE_A_TASKS:], 0, "main run, remaining 6 tasks"),
        ("C", [by_id[task_id] for task_id in repeat_ids], 1, "extra repeat of 4 fixed tasks"),
    ]
    order = 0
    for stage, stage_tasks, repeat_index, note in stages:
        for task in stage_tasks:
            rng = random.Random(f"{GENERATION_SEED}|order|{task['id']}|{repeat_index}")
            workflow_ids = [workflow.id for workflow in workflows]
            rng.shuffle(workflow_ids)
            for workflow_id in workflow_ids:
                workflow = next(w for w in workflows if w.id == workflow_id)
                plan.append({
                    "order": order,
                    "execution_id": f"{task['id']}|{workflow_id}|r{repeat_index}",
                    "stage": stage,
                    "stage_note": note,
                    "task_id": task["id"],
                    "workflow_id": workflow_id,
                    "repeat_index": repeat_index,
                    "node_ids": [node.id for node in workflow.nodes],
                    "node_seeds": {node.id: node_seed(task["id"], workflow_id, repeat_index, node.id)
                                   for node in workflow.nodes},
                })
                order += 1
    return plan


class Runner:
    def __init__(self, run_dir: Path, backend, tasks: list[dict], config: dict,
                 plan: list[dict], tasks_sha256: str, run_id: str):
        self.run_dir = Path(run_dir)
        self.backend = backend
        self.tasks = {task["id"]: task for task in tasks}
        self.config = config
        self.plan = plan
        self.tasks_sha256 = tasks_sha256
        self.run_id = run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.run_dir / "calls.jsonl"
        self.executions_path = self.run_dir / "executions.jsonl"
        self.retries_used = 0
        self.consecutive_failures = 0
        self._completed = self._load_completed()

    # ------------------------------------------------------------------ setup
    def workflow_fingerprint(self) -> dict:
        return {workflow.id: canonical_hash(workflow.definition()) for workflow in WORKFLOWS}

    def run_config(self) -> dict:
        return {
            "run_id": self.run_id,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frozen_config": self.config,
            "tasks_sha256": self.tasks_sha256,
            "task_ids": [task["id"] for task in self.tasks.values()],
            "repeat_ids": [entry["task_id"] for entry in self.plan if entry["repeat_index"] == 1][::3],
            "workflow_definitions": {w.id: w.definition() for w in WORKFLOWS},
            "workflow_hashes": self.workflow_fingerprint(),
            "plan": self.plan,
            "plan_hash": canonical_hash(self.plan),
            "seed_rule": "seed = int(sha256(f'{GENERATION_SEED}|task_id|workflow_id|repeat_index|node_id').hexdigest()[:8],16) % 2**31",
            "totals": {
                "planned_slots": len(self.plan),
                "planned_model_calls": sum(len(entry["node_ids"]) for entry in self.plan),
                "max_extra_retry_calls": MAX_RETRIES_TOTAL,
            },
        }

    def initialise(self) -> dict:
        """Create or validate ``run_config.json``; refuse to mix experiments."""
        path = self.run_dir / "run_config.json"
        fresh = self.run_config()
        if path.exists():
            existing = json.loads(path.read_text())
            for key in ("tasks_sha256", "plan_hash"):
                if existing.get(key) != fresh[key]:
                    raise RuntimeError(
                        f"refusing to resume: {key} differs from the stored run config"
                    )
            for key, value in fresh["workflow_hashes"].items():
                if existing.get("workflow_hashes", {}).get(key) != value:
                    raise RuntimeError(f"refusing to resume: workflow {key} changed")
            return existing
        path.write_text(json.dumps(fresh, indent=2, ensure_ascii=False))
        return fresh

    def _load_completed(self) -> set[str]:
        done = set()
        if self.executions_path.exists():
            for line in self.executions_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("status") in TERMINAL_STATUSES:
                    done.add(record["execution_id"])
        return done

    # ------------------------------------------------------------------ logging
    def _append(self, path: Path, record: dict) -> None:
        with open(path, "a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    # --------------------------------------------------------------------- run
    def run(self, stages=("A", "B", "C"), max_seconds: float | None = None,
            progress=None) -> dict:
        started = time.time()
        stopped_for = None
        for entry in self.plan:
            if entry["stage"] not in stages or entry["execution_id"] in self._completed:
                continue
            if max_seconds is not None and time.time() - started > max_seconds:
                stopped_for = "wall_clock_limit"
                break
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                stopped_for = "three_consecutive_execution_failures"
                break
            record = self._run_execution(entry)
            self._append(self.executions_path, record)
            self._completed.add(record["execution_id"])
            self.consecutive_failures = (
                self.consecutive_failures + 1 if record["status"] != "ok" else 0
            )
            if progress:
                progress(record)
        return {
            "run_id": self.run_id,
            "completed_slots": len(self._completed),
            "planned_slots": len(self.plan),
            "stopped_for": stopped_for,
            "wall_seconds": time.time() - started,
            "retry_calls_used": self.retries_used,
        }

    def _run_execution(self, entry: dict) -> dict:
        task = self.tasks[entry["task_id"]]
        workflow = next(w for w in WORKFLOWS if w.id == entry["workflow_id"])
        outputs: dict[str, str] = {}
        call_ids: list[str] = []
        totals = {"input": 0, "output": 0, "elapsed": 0.0, "retries": 0}
        status, error, truncated = "ok", None, False
        started = time.time()

        for node in workflow.nodes:
            attempt = 0
            while True:
                call_id = f"{entry['execution_id']}|{node.id}|a{attempt}"
                payload = messages(node, task["question"], outputs)
                result = self.backend.generate(
                    payload, seed=entry["node_seeds"][node.id],
                    max_new_tokens=self.config["max_new_tokens"],
                    temperature=self.config["temperature"], top_p=self.config["top_p"],
                )
                self._append(self.calls_path, {
                    "call_id": call_id,
                    "execution_id": entry["execution_id"],
                    "task_id": entry["task_id"],
                    "workflow_id": entry["workflow_id"],
                    "repeat_index": entry["repeat_index"],
                    "stage": entry["stage"],
                    "node_id": node.id,
                    "operator": node.operator,
                    "attempt": attempt,
                    "seed": entry["node_seeds"][node.id],
                    "messages": payload,
                    "raw_output": result["text"],
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "finish_reason": result["finish_reason"],
                    "context_limit": result.get("context_limit", False),
                    "status": result["status"],
                    "error": result.get("error"),
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                })
                call_ids.append(call_id)
                totals["input"] += result["input_tokens"] or 0
                totals["output"] += result["output_tokens"] or 0
                totals["elapsed"] += result["elapsed_seconds"] or 0.0

                if result["status"] == "ok":
                    break
                transient = result.get("transient", False)
                if attempt == 0 and transient and self.retries_used < MAX_RETRIES_TOTAL:
                    attempt = 1
                    self.retries_used += 1
                    totals["retries"] += 1
                    time.sleep(1.0)
                    continue
                status, error = "failed", result.get("error")
                break

            if status == "failed":
                break
            outputs[node.id] = result["text"]
            if node is workflow.nodes[-1]:
                truncated = str(result["finish_reason"]).startswith("length")

        final_output = outputs.get(workflow.nodes[-1].id) if status == "ok" else None
        scored = {"answer": None, "parsed": False, "correct": False}
        if final_output is not None:
            scored = score(final_output, task["gold"])
        return {
            "execution_id": entry["execution_id"],
            "task_id": entry["task_id"],
            "workflow_id": entry["workflow_id"],
            "repeat_index": entry["repeat_index"],
            "stage": entry["stage"],
            "call_ids": call_ids,
            "planned_model_calls": len(entry["node_ids"]),
            "model_calls": len(call_ids),
            "retries": totals["retries"],
            "node_seeds": entry["node_seeds"],
            "final_output": final_output,
            "extracted_answer": scored["answer"],
            "parsed": scored["parsed"],
            "correct": scored["correct"],
            "truncated": truncated,
            "total_input_tokens": totals["input"],
            "total_output_tokens": totals["output"],
            "total_node_seconds": totals["elapsed"],
            "wall_seconds": time.time() - started,
            "status": status,
            "error": error,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
