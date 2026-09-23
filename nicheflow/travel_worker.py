"""Official constraint evaluation worker; intentionally invoked as a separate process."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys


def main():
    payload = json.load(sys.stdin)
    root = Path(payload["checkout"])
    sys.path[:0] = [str(root / "evaluation"), str(root)]
    with contextlib.redirect_stdout(io.StringIO()):
        from commonsense_constraint import evaluation as commonsense_eval
        from hard_constraint import evaluation as hard_eval
        query, plan = payload["query"], payload["plan"]
        if isinstance(query.get("local_constraint"), str):
            query["local_constraint"] = ast.literal_eval(query["local_constraint"])
        common = commonsense_eval(query, plan) if plan else None
        hard = hard_eval(query, plan) if common and common["is_not_absent"][0] and common["is_valid_information_in_sandbox"][0] else None
        passed = bool(common) and hard is not None and all(v[0] is None or bool(v[0]) for v in common.values()) and all(v[0] is None or bool(v[0]) for v in hard.values())
    print(json.dumps({"status": "ok", "quality": float(passed), "evaluator": "travelplanner_official",
                      "metrics": {"parsed": True, "commonsense": common, "hard": hard, "final_pass": passed}}))


if __name__ == "__main__":
    main()
