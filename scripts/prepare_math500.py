#!/usr/bin/env python3
"""Prepare the full frozen protocol from pinned local source downloads, no model calls.

Requires PyArrow only during preparation. Execution uses JSONL and needs no new library.
"""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.math_protocol import (SUBJECTS, TEXT_PROTOCOL, MATH500_REVISION, MATH_TRAIN_REVISION,
                                    MATH500_SHA256, math500_tasks, learning_tasks, distribution)
from nicheflow.spec import file_hash, IntegrityError


def main():
    import pyarrow
    import pyarrow.parquet as pq
    source, target = ROOT / "data/source_math_v041", ROOT / "data/main_math500_v1"
    if target.exists():
        raise IntegrityError("existing pack is immutable; refusing overwrite")
    downloads = json.loads((source / "downloads.json").read_text())
    for record in downloads:
        path = ROOT / record["path"]
        if file_hash(path) != record["sha256"]:
            raise IntegrityError("source download hash mismatch")
        revision = MATH500_REVISION if path.suffix == ".jsonl" else MATH_TRAIN_REVISION
        if f"/resolve/{revision}/" not in record["url"]:
            raise IntegrityError("source revision is not pinned")
    if file_hash(source / "math500.jsonl") != MATH500_SHA256:
        raise IntegrityError("published MATH-500 bytes changed")
    evaluation = math500_tasks([json.loads(line) for line in (source / "math500.jsonl").read_text().splitlines()])
    training = {s: pq.read_table(source / f"{s}.parquet").to_pylist() for s in SUBJECTS}
    if sum(map(len, training.values())) != 7500:
        raise IntegrityError("official training source must contain 7500 rows")
    old = [t.question for t in load_tasks(ROOT / "data/tasks.jsonl")]
    old += [t.question for role in ("development", "calibration")
            for t in load_tasks(ROOT / f"data/main_math_learning_v1/{role}.jsonl")]
    parts, excluded = learning_tasks(training, evaluation, old)
    parts["evaluation"] = evaluation
    target.mkdir()
    entries = {}
    for role, tasks in parts.items():
        path = target / f"{role}.jsonl"
        path.write_text("".join(json.dumps(t.record(), ensure_ascii=False) + "\n" for t in tasks))
        entries[role] = {"path": path.name, "count": len(tasks), "sha256": file_hash(path),
                         "task_ids": [t.id for t in tasks], "distribution": distribution(tasks)}
    manifest = {"schema": "nicheflow_data_pack_v1", "protocol": "main_math500_v1",
        "input_protocol": TEXT_PROTOCOL, "sampling_seed": 20260920,
        "training_source_count": 7500, "development_per_subject_level": 1, "calibration_per_subject_level": 2,
        "learning_size_basis": "engineering choice: all 35 subject/difficulty strata; not all MATH train",
        "learning_order": "seeded shuffle within each partition", "evaluation_order": "published source order",
        "evaluation_scope": "all 500 official MATH-500 rows, zero exclusions", "evaluation_exclusions": [],
        "answer_type_filter": None, "asy_policy": "preserve all source code verbatim as text; no rendering or execution",
        "selection_before_model_calls": True, "exclusions_from_learning_pool": excluded,
        "source_files": downloads, "preparation_pyarrow_version": pyarrow.__version__, "partitions": entries}
    path = target / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"manifest": str(path), "sha256": file_hash(path),
                      "counts": {r: len(ts) for r, ts in parts.items()},
                      "distribution": {r: distribution(ts) for r, ts in parts.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
