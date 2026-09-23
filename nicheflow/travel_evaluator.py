"""Bridge to a pinned, separately provisioned official TravelPlanner checkout."""
import json
from pathlib import Path
import subprocess
import sys
from .spec import EnvironmentBlocked


class TravelPlannerEvaluator:
    def __init__(self, checkout, revision, python=sys.executable, timeout=60):
        self.checkout, self.revision = Path(checkout).resolve(), revision
        self.python, self.timeout = python, timeout

    def readiness(self):
        required = ["evaluation/commonsense_constraint.py", "evaluation/hard_constraint.py", "database", "tools"]
        if any(not (self.checkout / p).exists() for p in required):
            return {"ready": False, "reason": "official TravelPlanner checkout/database incomplete"}
        r = subprocess.run(["git", "-C", str(self.checkout), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return {"ready": r.returncode == 0 and r.stdout.strip() == self.revision,
                "reason": "official checkout commit must match the frozen evaluation protocol"}

    def __call__(self, task, output):
        status = self.readiness()
        if not status["ready"]:
            raise EnvironmentBlocked(status["reason"])
        if task.split == "test":
            raise EnvironmentBlocked("TravelPlanner official test evaluation requires its submission protocol")
        try:
            plan = json.loads(output)["plan"]
            if not isinstance(plan, list):
                raise ValueError("plan must be a list")
        except (ValueError, TypeError, KeyError):
            return {"status": "ok", "quality": 0., "evaluator": "travelplanner_official", "metrics": {"parsed": False}}
        # The official constraint functions evaluate a provided query and parsed plan.
        # No full-dataset force_redownload, hidden-test access, or eval(model_text).
        worker = Path(__file__).with_name("travel_worker.py")
        payload = {"query": task.private["official_row"], "plan": plan, "checkout": str(self.checkout)}
        p = subprocess.run([self.python, str(worker)], input=json.dumps(payload), capture_output=True,
                           text=True, timeout=self.timeout, cwd=self.checkout / "evaluation")
        if p.returncode:
            raise EnvironmentBlocked("official TravelPlanner evaluator failed; inspect its isolated environment")
        result = json.loads(p.stdout)
        result["revision"] = self.revision
        return result
