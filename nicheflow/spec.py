"""Never turn missing research definitions into implicit algorithm defaults."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

DOCUMENT_SHA256 = "8fae9e00ad672773c63c7e7f382b8f5775d4d7933c9a0a94e622c5f0540766f9"


class SpecGap(RuntimeError):
    pass


class EnvironmentBlocked(RuntimeError):
    pass


class BudgetStop(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


class ExecutionStop(RuntimeError):
    pass


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


GAPS = {
    "search_initialization": "3.1.6 initializes n_i=0 but divides by n_i in UCB; a cold-start rule and feedback attribution need to be fixed explicitly",
    "cost_semantics": "3.2.1–3.2.2 call the second head Cost while maximizing its UCB; the reward transformation is not defined",
    "curvature_observations": "3.3.2–3.3.3 do not define a common set utility f and a measured pair f(i|S), f(i|empty) from Vendi/HV feedback",
    "scheduler_allocation": "3.3.7 says allocate based on h(gamma), without an executable h/context -> beta or budget-block rule",
}

def source_check(root: Path) -> dict:
    path = root / "NicheFlow_3.4_技术方案.docx"
    actual = file_hash(path) if path.exists() else None
    return {"path": str(path), "expected": DOCUMENT_SHA256, "actual": actual,
            "ok": actual == DOCUMENT_SHA256}


def full_smoke_readiness(config=None) -> dict:
    from .policy import PROFILE, validate_config
    try:
        if config is None:
            raise ValueError("a frozen runtime configuration is required")
        validate_config(config)
    except (ValueError, KeyError, TypeError) as exc:
        return {"ready": False, "status": "configuration_blocked", "gaps": {"configuration": str(exc)},
                "profile": PROFILE, "engineering_completions": GAPS}
    return {"ready": True, "status": "ready_with_documented_engineering_completions", "gaps": {},
            "profile": PROFILE, "engineering_completions": GAPS,
            "exact_source_reproduction": False, "real_smoke_passed": False,
            "message": "Runnable implementation with explicitly authorized assumptions; readiness is not an experiment result."}
