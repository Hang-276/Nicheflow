"""Benchmark-specific scoring; missing official integrations never become wrong answers."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import re
from .spec import EnvironmentBlocked
from .sandbox import extract_code


def _vendor(name):
    path = Path(__file__).parent / "vendor" / (name + ".py")
    if not path.exists():
        raise EnvironmentBlocked(f"missing pinned official evaluator {name}")
    spec = importlib.util.spec_from_file_location("nicheflow.vendor." + name, path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise EnvironmentBlocked(f"official evaluator dependency missing: {exc.name}") from exc
    return module


def evaluate(task, output, sandbox=None, official_adapters=None):
    from nicheflow_probe.evaluation import score, extract_boxed, rational
    kind = task.evaluator
    if kind == "rational":
        s = score(output, task.gold)
        return {"status": "ok", "quality": float(s["correct"]), "metrics": s, "evaluator": kind}
    if kind in {"math_official", "gsm8k"}:
        answer = extract_boxed(output)
        if kind == "math_official":
            correct = answer is not None and _vendor("math_equivalence").is_equiv(answer, task.gold)
        else:
            target = rational(task.gold)
            if target is None:
                raise ValueError("unsupported GSM8K reference")
            correct = answer is not None and rational(answer) == target
        return {"status": "ok", "quality": float(correct), "evaluator": kind,
                "metrics": {"correct": bool(correct), "parsed": answer is not None, "answer": answer}}
    if kind == "choice":
        answers = re.findall(r"(?im)^\s*ANSWER:\s*([A-J])\s*[.)]?\s*$", output)
        answer = answers[-1] if answers else None
        if answer is not None and ord(answer) - 65 >= len(task.options):
            answer = None
        correct = answer == task.gold
        return {"status": "ok", "quality": float(correct), "evaluator": kind,
                "metrics": {"correct": correct, "parsed": answer is not None, "answer": answer}}
    if kind == "python_tests":
        if sandbox is None:
            raise EnvironmentBlocked("code sandbox not configured")
        code = extract_code(output)
        private = task.private or {}
        if task.dataset == "humaneval" and not re.search(r"\bdef\s+" + re.escape(private["entry_point"]) + r"\s*\(", code):
            code = private["prompt"] + code
        result = sandbox.run(code, private["tests"], private.get("setup", ""))
        return {"status": "ok", "quality": float(result["passed"]), "evaluator": kind, "metrics": result}
    if kind == "hotpot_official":
        evaluator = _vendor("hotpot_evaluate")
        try:
            pred = json.loads(output)
            answer, sp = pred["answer"], pred["sp"]
            if not isinstance(answer, str) or not isinstance(sp, list) or any(not isinstance(x, list) or len(x) != 2 or not isinstance(x[0], str) or type(x[1]) is not int for x in sp):
                raise ValueError("invalid Hotpot answer/support format")
        except (ValueError, KeyError, TypeError):
            return {"status": "ok", "quality": 0., "evaluator": kind, "metrics": {"parsed": False}}
        metrics = dict.fromkeys(["em", "f1", "prec", "recall", "sp_em", "sp_f1", "sp_prec", "sp_recall"], 0.)
        em, p, r = evaluator.update_answer(metrics, answer, task.gold)
        sem, sprec, sr = evaluator.update_sp(metrics, sp, task.private["supporting_facts"])
        jp, jr = p * sprec, r * sr
        metrics.update(joint_em=em * sem, joint_f1=2 * jp * jr / (jp + jr) if jp + jr else 0., parsed=True)
        return {"status": "ok", "quality": float(em), "evaluator": kind,
                "quality_metric": "answer_em_smoke_only", "metrics": metrics}
    if kind == "drop_official":
        evaluator = _vendor("drop_evaluate")
        try:
            answers = json.loads(output)["answers"]
            if not isinstance(answers, list) or not answers or not all(isinstance(a, str) for a in answers):
                raise ValueError("invalid DROP spans")
        except (ValueError, KeyError, TypeError):
            return {"status": "ok", "quality": 0., "evaluator": kind, "metrics": {"parsed": False}}
        em, f1 = evaluator.get_metrics(answers, task.gold)
        return {"status": "ok", "quality": float(em), "evaluator": kind, "metrics": {"em": em, "f1": f1, "parsed": True}}
    if kind == "gaia_official":
        if not isinstance(task.gold, str) or not task.gold.strip():
            raise EnvironmentBlocked("GAIA reference answer is not public/available")
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            correct = _vendor("gaia_evaluate").question_scorer(output, task.gold)
        return {"status": "ok", "quality": float(correct), "evaluator": kind, "metrics": {"correct": bool(correct)}}
    if kind == "travelplanner_official":
        adapter = (official_adapters or {}).get(kind)
        if adapter is None:
            raise EnvironmentBlocked(f"{kind}: official evaluation environment not provisioned")
        return adapter(task, output)
    raise ValueError(f"unknown evaluator: {kind}")
