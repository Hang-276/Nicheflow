"""Read-only planning and explicit execution entry for the main method."""
from pathlib import Path
import importlib.util
import json
from .ledger import atomic_json
from .main_budget import ResourceJournal
from .main_config import load_protocol, derive_limits, model_selection_problems
from .main_driver import MainDriver
from .main_models import ModelPool, model_readiness
from .main_policy import MainPolicy, main_seeds
from .main_reporting import main_report
from .runtime import GraphExecutor
from .sandbox import PythonSandbox
from .spec import EnvironmentBlocked, IntegrityError, source_check


def completion_exit_code(status):
    return {"main_run_complete": 0, "learning_complete_evaluation_pending": 0, "stage_paused": 0,
            "main_completed_with_execution_errors": 6, "budget_stopped": 5,
            "environment_blocked": 3, "execution_stopped": 6, "spec_blocked": 2}.get(status, 1)


def prepare(path, root, rounds=None, check_environment=True):
    config, partitions, provenance = load_protocol(path, root, rounds)
    if not source_check(root)["ok"]:
        raise IntegrityError("authoritative Word source changed")
    from .research_policy import research_seeds, REVISION
    seeds = research_seeds(config) if config.get('research_revision') == REVISION else main_seeds(config)
    problems = model_selection_problems(config)
    limits = None if problems else derive_limits(config, partitions, seeds)
    sandbox, adapters = None, {}
    if check_environment:
        problems.extend(p for p in model_readiness(config) if p not in problems)
        if config.get('research_revision') == REVISION:
            from .semantic import verify_encoder
            try:
                verify_encoder(config['semantic'])
            except (EnvironmentBlocked, IntegrityError) as exc:
                problems.append(str(exc))
    all_tasks = [t for part in partitions.values() for t in part]
    if any(t.attachments for t in all_tasks):
        problems.append("attachment tasks need an explicit provisioned input adapter; no silent text-only conversion")
    if set(config["allowed_operators"]) & {"Code", "Test"} or any(t.evaluator == "python_tests" for t in all_tasks):
        sandbox = PythonSandbox(config.get("sandbox_image"), config.get("sandbox_timeout_seconds", 10))
        if check_environment and not sandbox.readiness()["ready"]:
            problems.append("code operators/benchmarks require the pinned isolated execution environment")
    evaluators = {t.evaluator for t in all_tasks}
    for evaluator, library in [("drop_official", "scipy"), ("hotpot_official", "ujson")]:
        if evaluator in evaluators and importlib.util.find_spec(library) is None:
            problems.append(f"{evaluator} requires {library}")
    if "travelplanner_official" in evaluators:
        from .travel_evaluator import TravelPlannerEvaluator
        spec = config.get("travelplanner")
        if spec:
            adapter = TravelPlannerEvaluator(**spec)
            adapters["travelplanner_official"] = adapter
            if check_environment and not adapter.readiness()["ready"]:
                problems.append("TravelPlanner official environment is not ready")
        else:
            problems.append("TravelPlanner official environment must be configured")
    return config, partitions, provenance, limits, sandbox, adapters, problems


def plan(path, root, rounds=None):
    config, parts, provenance, limits, _, _, problems = prepare(path, root, rounds)
    return {"profile": config["profile"], "ready_for_configured_learning": not problems,
            "ready_for_configured_run": not problems,
            "evaluation_status": config["evaluation"]["status"], "rounds": config["rounds"],
            "evaluation_benchmark": config["evaluation"].get("benchmark"),
            "counts": {k: len(v) for k, v in parts.items()}, "problems": problems, "limits": limits,
            "protocol_hash_excluding_rounds": provenance["protocol_hash_excluding_rounds"],
            "model_calls": 0, "model_loaded": False}


def run_main(args, root, backend_factory=None):
    from .cli import code_identity
    backend_factory = backend_factory or ModelPool
    # Completed runs are inspected without checking credentials or loading a model.
    config, parts, provenance, limits, sandbox, adapters, _ = prepare(args.config, root, args.rounds, check_environment=False)
    if limits is None:
        raise EnvironmentBlocked('; '.join(model_selection_problems(config)))
    synthetic = bool(getattr(backend_factory, "is_synthetic", False))
    code = code_identity(root)
    source, stop_after = None, getattr(args, 'stop_after_round', None)
    resume_from = getattr(args, 'resume_from', None)
    revised = config.get('research_revision') == 'feedback_semantic_v2'
    if (resume_from or stop_after is not None) and not revised:
        raise IntegrityError('staged learning is supported only by v2 protocol')
    if revised:
        from .research_state import read_stage_source, stage_limits
        source = read_stage_source(resume_from, config, code, getattr(args, 'compatibility_manifest', None)) if resume_from else None
        if source and source['state']['synthetic'] != synthetic:
            raise IntegrityError('cannot mix real and synthetic learning stages')
        start = source['round'] if source else 0
        stop_after = config['rounds'] if stop_after is None else stop_after
        if type(stop_after) is not int or not start < stop_after <= config['rounds']:
            raise ValueError('stage end must be greater than source round and within horizon')
        limits = stage_limits(config, limits, start, stop_after)
    frozen = {"mode": "main", "settings": config, "code": code, "provenance": provenance,
              "limits": limits, "synthetic": synthetic}
    if revised:
        frozen['stage'] = {'source': source['evidence'] if source else None, 'stop_after_round': stop_after}
    with ResourceJournal(args.run_dir, frozen, limits["max_calls"], limits["max_seconds"], limits=limits) as journal:
        if journal.lookup("run_finished", "main:done"):
            result = main_report(args.run_dir)
            return result, completion_exit_code(result["status"])
        if journal.audit()["unknown_calls"] or journal.spending()["unknown_cost_calls"]:
            raise IntegrityError("unknown previous call/charge; automatic replay forbidden")
        if not synthetic:
            _, _, _, _, _, _, problems = prepare(args.config, root, args.rounds)
            if problems:
                atomic_json(Path(args.run_dir) / "preflight.json", {"ready": False, "problems": problems,
                            "new_model_calls": 0, "model_loaded": False})
                raise EnvironmentBlocked("; ".join(problems))
        journal.reserve(0)
        pool = backend_factory(config)
        try:
            identity = {name: backend.environment for name, backend in pool.backends.items()}
            if source and any(source['environment'][name]['model'] != value['model'] for name, value in identity.items()):
                raise IntegrityError('model environment changed across learning stages')
            old = journal.lookup("model_environment", "main")
            if old is None:
                journal.append("model_environment", "main", identity)
            else:
                for name, environment in identity.items():
                    if old[name]["model"] != environment["model"]:
                        raise IntegrityError("backend model identity changed on resume")
            atomic_json(Path(args.run_dir) / "environment.json", identity)
            executor = GraphExecutor(journal, pool.backends, config["decode"], sandbox=sandbox, official_adapters=adapters,
                                     workflow_output_policy=config.get("workflow_output_policy", "legacy"))
            executor.length_limit_as_zero = config.get("length_limit_outcome") == "zero_quality_no_retry"
            if revised:
                from .research_policy import ResearchPolicy
                from .research_driver import ResearchDriver
                encoder_factory = getattr(backend_factory, 'semantic_encoder_factory', None) if synthetic else None
                policy = ResearchPolicy(executor, config, parts['development'], synthetic=synthetic,
                                        encoder=encoder_factory(config['semantic']) if encoder_factory else None)
                completion = ResearchDriver(policy, parts, provenance).run(source=source, stop_after=stop_after)
            else:
                policy = MainPolicy(executor, config, parts["development"], synthetic=synthetic)
                completion = MainDriver(policy, parts, provenance).run()
            result = main_report(args.run_dir)
            return result, completion_exit_code(completion["status"])
        finally:
            pool.close()
