"""Read-only audit and derived reports from the immutable event journal."""
import json
from pathlib import Path
from .ledger import Journal, atomic_json
from .spec import IntegrityError


def report(directory):
    directory = Path(directory)
    if not (directory / "config.json").exists():
        if (directory / "events.jsonl").exists():
            raise IntegrityError("journal exists without frozen run configuration")
        if (directory / "preflight.json").exists():
            return {"mode": "full-smoke", "run_started": False,
                    "real_adaptive_smoke_pass": False, **json.loads((directory / "preflight.json").read_text())}
        return {"status": "run_not_created", "run_started": False, "run_dir": str(directory),
                "calls_attempted": None, "real_adaptive_smoke_pass": False,
                "message": "No run evidence exists here; no call count can be inferred."}
    config = json.loads((directory / "config.json").read_text())
    if config["config"]["mode"] == "main":
        from .main_reporting import main_report
        return main_report(directory)
    events = Journal.read(directory / "events.jsonl")
    starts = {e["id"]: e for e in events if e["kind"] == "call_started"}
    results = {e["id"]: e for e in events if e["kind"] == "call_finished"}
    if not set(results) <= set(starts) or len(starts) > config["max_calls"]:
        raise IntegrityError("call accounting invariant failed")
    mode = config["config"]["mode"]
    completion = next((e["payload"] for e in reversed(events) if e["kind"] == "run_finished"), None)
    coverage = (completion or {}).get("coverage", {})
    real_pass = bool(mode == "full-smoke" and completion and completion["status"] == "full_smoke_pass"
                     and completion.get("synthetic") is False and starts
                     and coverage.get("cycles_completed", 0) >= 4
                     and all(coverage.get(key) is True for key in ["candidate_generation", "combinatorial_selection",
                         "candidate_evaluation", "candidate_archive_decision", "new_elite_route_feedback", "budget_actions_executed"]))
    summary = {"mode": mode, "status": completion["status"] if completion else "incomplete",
               "calls_attempted": len(starts), "calls_finished": len(results), "max_calls": config["max_calls"],
               "unknown_calls": sorted(set(starts) - set(results)),
               "execution_count": sum(e["kind"] == "execution" for e in events),
               "input_tokens": sum(e["payload"].get("input_tokens") or 0 for e in results.values()),
               "output_tokens": sum(e["payload"].get("output_tokens") or 0 for e in results.values()),
               "unknown_token_calls": sum(e["payload"].get("output_tokens") is None for e in results.values()),
               "hash_chain_valid": True, "real_adaptive_smoke_pass": real_pass,
               "elapsed_seconds": max((e["time"] for e in events), default=config["started"]) - config["started"],
               "completion": completion}
    policy = config["config"].get("settings", {}).get("policy", {})
    if "reference_usd_per_gpu_hour" in policy:
        summary["reference_cost_usd"] = sum(e["payload"].get("elapsed_seconds", 0) for e in results.values()) * policy["reference_usd_per_gpu_hour"] / 3600
        summary["reference_cost_is_actual_bill"] = False
        summary["exact_source_reproduction"] = False
    if summary["unknown_calls"]:
        summary["status"] = "needs_manual_recovery"
        summary["real_adaptive_smoke_pass"] = False
    atomic_json(directory / "summary.json", summary)
    mappings = {"calls.jsonl": "call_finished", "executions.jsonl": "execution", "generation_events.jsonl": "generation",
                "workflow_versions.jsonl": "workflow", "archive_events.jsonl": "archive", "router_events.jsonl": "router",
                "scheduler_events.jsonl": "scheduler", "feedback_events.jsonl": "feedback",
                "operation_plans.jsonl": "operation_plan", "curvature_observations.jsonl": "curvature_observation",
                "archive_rebins.jsonl": "archive_rebin"}
    for filename, kind in mappings.items():
        # These are projections. events.jsonl is the untouched authoritative record.
        rows = [{"id": e["id"], **e["payload"]} for e in events if e["kind"] == kind]
        (directory / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    (directory / "report.md").write_text("# NicheFlow run report\n\n" +
        f"Status: **{summary['status']}**\n\nMode: `{mode}`. Real adaptive smoke passed: **{summary['real_adaptive_smoke_pass']}**.\n\n" +
        f"Calls attempted: {len(starts)}/{config['max_calls']}; executions: {summary['execution_count']}.\n\n" +
        "All projections derive from events.jsonl. Fixture feedback is synthetic; component probes do not validate adaptive search or scheduling.\n\n" +
        "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")
    return summary
