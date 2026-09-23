"""Pinned MATH text protocol: unchanged problems, including inline Asymptote."""
from collections import Counter
from dataclasses import replace
import random
import re
import unicodedata
from .datasets import Task, normalize
from .spec import IntegrityError, digest

SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus")
TEXT_PROTOCOL = "math_verbatim_with_inline_asymptote_v1"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MATH_TRAIN_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH500_SHA256 = "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132"
QUICK35_SEED = 20260920
QUICK35_METHOD = "one_per_subject_level_sha256_rank_v1"


def question_hash(question):
    return digest(re.sub(r"\s+", " ", unicodedata.normalize("NFKC", question)).strip())


def math_text_task(row, index, *, split, role, subject):
    if subject not in SUBJECTS:
        raise ValueError("unknown MATH subject")
    task = normalize("math", row, index, split=split, role=role)
    level = str(row["level"]).removeprefix("Level ")
    if level not in {"1", "2", "3", "4", "5"}:
        raise ValueError("unknown MATH difficulty")
    gold = row.get("answer", task.gold)
    if not isinstance(gold, str) or not gold.strip():
        raise ValueError("MATH reference must be a nonempty string")
    from .vendor.math_equivalence import is_equiv
    if not is_equiv(gold, task.gold):
        raise IntegrityError("explicit answer disagrees with boxed reference solution")
    if task.attachments not in {(), ("embedded_asy_requires_renderer",)}:
        raise IntegrityError("only self-contained inline Asymptote is supported")
    # This is an explicit text-benchmark protocol, not a rendered-image adapter.
    # Preserve the full source in the question; never execute or delete it.
    return replace(task, gold=gold, attachments=(), private={**task.private,
        "subject": subject, "level": int(level), "source_id": row.get("unique_id", row.get("id", index)),
        "input_protocol": TEXT_PROTOCOL, "inline_asymptote": "[asy]" in task.question})


def math500_tasks(rows):
    if len(rows) != 500:
        raise IntegrityError("MATH-500 must contain exactly 500 source rows")
    tasks, seen, questions = [], set(), set()
    for row in rows:
        original = row["unique_id"]
        pieces = original.split("/")
        if len(pieces) != 3 or pieces[0] != "test":
            raise IntegrityError("MATH-500 requires original test identities")
        if original in seen or question_hash(row["problem"]) in questions:
            raise IntegrityError("duplicate MATH-500 query")
        seen.add(original); questions.add(question_hash(row["problem"]))
        tasks.append(math_text_task({**row, "id": "/".join(pieces[1:])}, len(tasks),
                                   split="test", role="evaluation", subject=pieces[1]))
    if {t.private["subject"] for t in tasks} != set(SUBJECTS):
        raise IntegrityError("MATH-500 subject coverage changed")
    return tasks  # Published order is retained; no sample selection or filtering.


def learning_tasks(rows_by_subject, evaluation, old_questions, seed=20260920,
                   development_per_stratum=1, calibration_per_stratum=2):
    """Frozen stratified sampling; default preserves the historical 35/70 split."""
    if any(type(n) is not int or n < 1 for n in (development_per_stratum, calibration_per_stratum)):
        raise ValueError('positive integer per-stratum counts required')
    if set(rows_by_subject) != set(SUBJECTS):
        raise IntegrityError("all seven training subjects are required")
    forbidden = {question_hash(t.question) for t in evaluation} | {question_hash(q) for q in old_questions}
    seen, strata, exclusions = set(), {}, []
    from nicheflow_probe.evaluation import extract_boxed
    for subject in SUBJECTS:
        for index, row in enumerate(rows_by_subject[subject]):
            key, q = f"{subject}/{index}", question_hash(row["problem"])
            reason = None
            if q in forbidden:
                reason = "held_out_or_previous_probe_overlap"
            elif q in seen:
                reason = "duplicate_training_question"
            elif str(row["level"]).removeprefix("Level ") not in {"1", "2", "3", "4", "5"}:
                reason = "source_difficulty_unspecified"
            elif not (extract_boxed(row["solution"]) or "").strip():
                reason = "reference_missing_or_empty_braced_box"
            if reason:
                exclusions.append({"source_id": key, "reason": reason})
                continue
            seen.add(q)
            task = math_text_task({**row, "id": key}, index, split="train", role="development", subject=subject)
            strata.setdefault((subject, task.private["level"]), []).append(task)
    parts = {"development": [], "calibration": []}
    for subject in SUBJECTS:
        for level in range(1, 6):
            choices = strata[subject, level]
            random.Random(f"{seed}/{subject}/{level}").shuffle(choices)
            if len(choices) < development_per_stratum + calibration_per_stratum:
                raise IntegrityError("insufficient training stratum")
            parts["development"].extend(choices[:development_per_stratum])
            parts["calibration"].extend(replace(t, role="calibration") for t in choices[development_per_stratum:development_per_stratum + calibration_per_stratum])
    for role, tasks in parts.items():
        random.Random(f"{seed}/order/{role}").shuffle(tasks)
    return parts, exclusions


def distribution(tasks):
    return {"subjects": dict(sorted(Counter(t.private["subject"] for t in tasks).items())),
            "levels": dict(sorted(Counter(str(t.private["level"]) for t in tasks).items())),
            "inline_asymptote": sum(t.private["inline_asymptote"] for t in tasks)}


def quick35_indices(tasks):
    """Select without answers/results; retain original full-evaluation positions."""
    if len(tasks) != 500 or len({t.id for t in tasks}) != 500:
        raise IntegrityError("quick35 requires the complete unique MATH-500 parent")
    chosen = []
    for subject in SUBJECTS:
        for level in range(1, 6):
            candidates = [i for i, t in enumerate(tasks)
                          if t.private["subject"] == subject and t.private["level"] == level]
            if not candidates:
                raise IntegrityError("quick35 requires all 35 subject/difficulty strata")
            chosen.append(min(candidates, key=lambda i: (digest([QUICK35_SEED, tasks[i].id]), tasks[i].id)))
    return sorted(chosen)
