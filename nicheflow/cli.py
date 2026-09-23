"""Bounded run modes. Full smoke never falls back to an invented policy."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from .spec import BudgetStop, EnvironmentBlocked, ExecutionStop, IntegrityError, SpecGap, file_hash, source_check, full_smoke_readiness
from .datasets import REGISTRY, load_tasks, fetch_hf
from .ledger import Journal, atomic_json
from .reporting import report


def code_identity(root):
    paths = sorted(list((root / "nicheflow").rglob("*.py")) + list((root / "nicheflow_probe").glob("*.py")))
    return {str(p.relative_to(root)): file_hash(p) for p in paths}


def doctor(root):
    from .sandbox import PythonSandbox
    checks = {"numpy": importlib.util.find_spec("numpy") is not None,
              "datasets": importlib.util.find_spec("datasets") is not None,
              "ujson": importlib.util.find_spec("ujson") is not None,
              "scipy": importlib.util.find_spec("scipy") is not None}
    vendor_manifest = root / "nicheflow/vendor/manifest.json"
    vendor = {}
    if vendor_manifest.exists():
        for name, item in json.loads(vendor_manifest.read_text()).items():
            p = root / "nicheflow/vendor" / (name + ".py")
            vendor[name] = {"hash_ok": p.exists() and file_hash(p) == item.get("local_sha256", item["sha256"]),
                            "revision": item["revision"]}
    return {"source": source_check(root), "dependencies": checks, "vendor": vendor,
            "sandbox": PythonSandbox().readiness(), "full_smoke": full_smoke_readiness(json.loads((root / "configs/smoke.json").read_text())),
            "datasets": {name: {**spec, "state": "registered", "real_smoke": "not_tested"} for name, spec in REGISTRY.items()}}


def run(args, root):
    if args.mode == "main":
        from .main_entry import run_main
        result, code = run_main(args, root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return code
    if args.rounds is not None:
        raise ValueError("--rounds is only valid for main mode")
    from .fixtures import run_fixture
    from .runtime import GraphExecutor
    from .graph import probe_graphs, WorkflowGraph
    config = json.loads(Path(args.config).read_text())
    if args.mode == "full-smoke":
        result = full_smoke_readiness(config)
        if not result["ready"]:
            directory = Path(args.run_dir)
            if any((directory / name).exists() for name in ("config.json", "events.jsonl", "preflight.json")):
                raise IntegrityError("existing run directory; use status instead of writing new preflight evidence")
            directory.mkdir(parents=True, exist_ok=True)
            atomic_json(directory / "preflight.json", {**result, "calls_attempted": 0,
                        "model_loaded": False, "run_started": False, "config_path": str(args.config),
                        "settings": config, "code": code_identity(root)})
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 2
    tasks_path = root / config["tasks"]
    tasks = load_tasks(tasks_path)
    if file_hash(tasks_path) != config["tasks_sha256"]:
        raise IntegrityError("task hash mismatch")
    for task in tasks:
        task.require_learning()
    frozen = {"mode": args.mode, "settings": config, "code": code_identity(root), "tasks_sha256": file_hash(tasks_path)}
    source = source_check(root)
    if not source["ok"]:
        raise IntegrityError("authoritative source hash mismatch")
    # Resume completed runs without loading a GPU or emitting new journal events.
    if (Path(args.run_dir) / "config.json").exists():
        with Journal(args.run_dir, frozen, config["max_calls"], config["max_seconds"]) as journal:
            if any(journal.lookup("run_finished", id) for id in ("fixture:done", "component:done", "smoke:done")):
                print(json.dumps(report(args.run_dir), ensure_ascii=False, indent=2))
                return 0
            audit = journal.audit()
            if audit["unknown_calls"]:
                raise IntegrityError("unknown previous call outcome; manual review needed")
            journal.reserve(0)
    if args.mode == "fixture":
        with Journal(args.run_dir, frozen, config["max_calls"], config["max_seconds"]) as journal:
            run_fixture(journal)
        print(json.dumps(report(args.run_dir), ensure_ascii=False, indent=2))
        return 0
    from .backend_process import LocalWorker
    model_path = Path(config["model_path"])
    if not model_path.is_dir():
        raise EnvironmentBlocked("local model snapshot absent; no automatic download")
    backend = LocalWorker(str(model_path), config["max_context_tokens"])
    try:
        with Journal(args.run_dir, frozen, config["max_calls"], config["max_seconds"]) as journal:
            env_path = Path(args.run_dir) / "environment.json"
            if not env_path.exists():
                atomic_json(env_path, backend.environment)
            else:
                old_environment = json.loads(env_path.read_text())
                if old_environment["model"] != backend.environment["model"]:
                    raise IntegrityError("loaded model identity changed; resume refused")
            backend.call_timeout = min(180, max(1, journal.remaining_seconds()))
            executor = GraphExecutor(journal, {"local": backend}, config["decode"])
            if args.mode == "full-smoke":
                from .policy import MinimalPolicy
                completion = MinimalPolicy(executor, config, tasks, synthetic=getattr(backend, "is_synthetic", False)).driver.run()
                print(json.dumps(report(args.run_dir), ensure_ascii=False, indent=2))
                return {"budget_stopped": 5, "environment_blocked": 3, "execution_stopped": 6,
                        "spec_blocked": 2, "implementation_failure": 1}.get(completion["status"], 0)
            # Component probe is deliberately small. It is not a scheduled adaptive experiment.
            result = executor.execute(probe_graphs()[0], tasks[0], "component:health") if args.mode == "component-probe" else {"status": "not_repeated"}
            completed = {"health": result["status"], "candidate": "not_attempted", "candidate_evaluation": "not_attempted"}
            if result["status"] in {"ok", "not_repeated"}:
                backend.call_timeout = min(180, max(1, journal.remaining_seconds()))
                generation = executor.generate_candidate(probe_graphs()[:1] if args.mode == "generation-probe" else probe_graphs(),
                    "component:proposal", max_nodes=config.get("generation_max_nodes", 12))
                completed["candidate"] = generation["status"]
                if generation["status"] == "valid":
                    candidate = WorkflowGraph.from_dict(generation["graph"])
                    try:
                        journal.reserve(candidate.model_calls)
                        backend.call_timeout = min(180, max(1, journal.remaining_seconds()))
                        evaluated = executor.execute(candidate, tasks[0], "component:candidate")
                        completed["candidate_evaluation"] = evaluated["status"]
                    except (BudgetStop, EnvironmentBlocked) as exc:
                        completed["candidate_evaluation"] = str(exc)
            journal.append("run_finished", "component:done", {"status": "component_probe_finished", "coverage": completed,
                            "full_smoke": False, "gaps": full_smoke_readiness()["gaps"]})
    finally:
        backend.close()
    print(json.dumps(report(args.run_dir), ensure_ascii=False, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    status = sub.add_parser("status")
    status.add_argument("run_dir")
    execute = sub.add_parser("run")
    execute.add_argument("--mode", choices=["fixture", "component-probe", "generation-probe", "full-smoke", "main"], required=True)
    execute.add_argument("--config", required=True)
    execute.add_argument("--run-dir", required=True)
    execute.add_argument("--rounds", type=int, help="main mode only; sole short/full-run override")
    execute.add_argument("--stop-after-round", type=int, help="v2: stop cleanly at this checkpoint, without terminal evaluation")
    execute.add_argument("--resume-from", help="v2: clean source stage directory; use a new output directory")
    execute.add_argument("--compatibility-manifest", help="explicit exact source/target code migration manifest")
    main_plan = sub.add_parser("main-plan")
    main_plan.add_argument("--config", required=True)
    main_plan.add_argument("--rounds", type=int)
    fetch = sub.add_parser("prepare-data")
    fetch.add_argument("dataset", choices=list(REGISTRY))
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--count", type=int, required=True)
    fetch.add_argument("--dataset-config")
    fetch.add_argument("--split")
    args = p.parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(args.root), ensure_ascii=False, indent=2))
        elif args.command == "status":
            print(json.dumps(report(args.run_dir), ensure_ascii=False, indent=2))
        elif args.command == "prepare-data":
            print(json.dumps(fetch_hf(args.dataset, args.output, count=args.count, revision=args.revision,
                                     config=args.dataset_config, split=args.split), ensure_ascii=False, indent=2))
        elif args.command == "main-plan":
            from .main_entry import plan
            result = plan(args.config, args.root, args.rounds)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready_for_configured_run"] else 3
        else:
            return run(args, args.root)
        return 0
    except (SpecGap, EnvironmentBlocked, ExecutionStop, IntegrityError, BudgetStop, ValueError) as exc:
        print(json.dumps({"status": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return {SpecGap: 2, EnvironmentBlocked: 3, IntegrityError: 4, BudgetStop: 5, ExecutionStop: 6}.get(type(exc), 1)


if __name__ == "__main__":
    raise SystemExit(main())
