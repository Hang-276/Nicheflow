#!/usr/bin/env python3
"""Freeze a development/calibration pack from existing source snapshots, no downloads.

This is an explicit learning-data subset; it is not the whole MATH benchmark.
Evaluation scope remains deferred at the user's request.
"""
from dataclasses import replace
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.datasets import normalize, load_tasks
from nicheflow.spec import file_hash, digest


def main():
    output = ROOT / "data/main_math_learning_v1"
    if output.exists():
        raise SystemExit("Existing pack preserved; no automatic re-selection/overwrite.")
    seed = 20260920
    excluded_questions = {digest(t.model_input()["question"]) for t in load_tasks(ROOT / "data/tasks.jsonl")}
    parts = {"development": [], "calibration": []}
    sources, exclusions = [], []
    for path in sorted((ROOT / "data/raw").glob("*.json")):
        subject = path.stem.rsplit("_", 1)[0]
        rows = json.loads(path.read_text())["rows"]
        choices = []
        for row in rows:
            raw = row["row"]
            if row.get("truncated_cells") or "[asy]" in raw["problem"] or digest(raw["problem"]) in excluded_questions:
                exclusions.append({"id": f"{subject}/train/{row['row_idx']}", "reason": "diagram/truncated/previously_used_question"})
                continue
            task = normalize("math", {**raw, "id": f"{subject}/{row['row_idx']}"}, row["row_idx"])
            choices.append(task)
        random.Random(f"{seed}/{subject}").shuffle(choices)
        if len(choices) < 10:
            raise ValueError("not enough eligible rows")
        parts["development"].extend(choices[:2])
        parts["calibration"].extend(replace(t, role="calibration") for t in choices[2:10])
        sources.append({"path": str(path.relative_to(ROOT)), "sha256": file_hash(path), "subject": subject,
                        "available_rows": len(rows), "eligible_rows": len(choices)})
    seen = set()
    for role, tasks in parts.items():
        random.Random(f"{seed}/order/{role}").shuffle(tasks)
        for task in tasks:
            h = digest(task.model_input())
            if h in seen:
                raise ValueError("duplicate public task; selection must be reviewed before any run")
            seen.add(h)
    output.mkdir()
    entries = {}
    for role, tasks in parts.items():
        path = output / f"{role}.jsonl"
        path.write_text("".join(json.dumps(t.record(), ensure_ascii=False) + "\n" for t in tasks))
        entries[role] = {"path": path.name, "sha256": file_hash(path), "count": len(tasks), "task_ids": [t.id for t in tasks]}
    manifest = {"schema": "nicheflow_data_pack_v1", "dataset": "math", "repo": "EleutherAI/hendrycks_math",
        "source_revision_at_original_download": json.loads((ROOT / "data/manifest.json").read_text())["source_revision_at_download"],
        "source_access": "previously archived viewer responses; bytes pinned by SHA256, not a new revision-pinned download",
        "scope": "fixed learning subset of archived first 100 train rows in three subjects; not all MATH",
        "sampling_seed": seed, "per_subject_development": 2, "per_subject_calibration": 8,
        "order_policy": "seeded shuffle across subjects within each partition",
        "selection_before_new_model_outputs": True, "answer_type_filter": None,
        "source_files": sources, "exclusions": exclusions, "partitions": entries,
        "evaluation": {"status": "deferred", "reason": "user will decide full test set or frozen subset later"}}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "counts": {k: len(v) for k,v in parts.items()},
                      "manifest_sha256": file_hash(output / "manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
