"""Frozen main-method protocol. Round count is the only short/full-run override."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import json
from .datasets import load_tasks
from .graph import OPERATORS
from .spec import IntegrityError, digest, file_hash

PROFILE = "main_method_v1"


def model_selection_problems(config):
    return [f"{name}: model selection pending; set model identity and verified tariff before execution"
            for name, profile in config['models'].items()
            if profile.get('configuration_status') == 'pending_selection']


def positive(value, name, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"positive finite {name} required")
    if integer and type(value) is not int:
        raise ValueError(f"integer {name} required")


def validate_main(config):
    from .workflow_contracts import validate_output_policy
    validate_output_policy(config.get("workflow_output_policy", "legacy"))
    if config.get("length_limit_outcome", "stop") not in {"stop", "zero_quality_no_retry"}:
        raise ValueError("unknown length-limit outcome policy")
    if config.get("profile") != PROFILE:
        raise ValueError("main_method_v1 profile required")
    for key in ("rounds", "formal_rounds", "generation_max_nodes", "queries_per_deploy_block"):
        positive(config[key], key, True)
    if config["generation_max_nodes"] > 12:
        raise ValueError("DAG validator supports at most 12 nodes")
    models = config["models"]
    if 'research_revision' in config:
        if config['research_revision'] != 'feedback_semantic_v2':
            raise ValueError('unknown research revision')
        if set(models) not in ({'local', 'strong'}, {'local', 'middle', 'strong'}):
            raise ValueError('v2 requires local and strong roles, with optional middle')
        if models.get('middle', {}).get('model') and models['middle']['model'] == models['strong']['model'] and models['middle']['endpoint'] == models['strong']['endpoint']:
            raise ValueError('middle and strong must not alias the same requested model')
        s = config['semantic']
        positive(s['dimension'], 'semantic projection dimension', True)
        if s['dimension'] > 32 or type(s['projection_seed']) is not int or len(s['manifest_sha256']) != 64:
            raise ValueError('invalid frozen semantic protocol')
    if not models or config["generation_model"] not in models or config["seed_model"] not in models:
        raise ValueError("explicit generation/seed models required")
    objective = config.get('cost_objective', 'api_plus_declared_local_time')
    if objective not in {'api_plus_declared_local_time', 'external_api_spend'}:
        raise ValueError('unknown cost objective')
    for name, m in models.items():
        selection = m.get('configuration_status', 'configured')
        if selection not in {'configured', 'pending_selection'}:
            raise ValueError('unknown model configuration status')
        if selection == 'pending_selection':
            if name != 'middle' or config.get('research_revision') != 'feedback_semantic_v2' or m['kind'] != 'api':
                raise ValueError('only the v2 middle API slot may be pending')
            if m.get('model') or m.get('accepted_response_models') or m.get('pricing'):
                raise ValueError('pending middle slot must not silently select a model or tariff')
        if m["kind"] not in {"local", "api"} or type(m["noncheap"]) is not bool:
            raise ValueError(f"invalid model profile {name}")
        positive(m["call_timeout_seconds"], "call timeout")
        positive(m["max_context_tokens"], "context limit", True)
        if m["kind"] == "local":
            if objective == 'external_api_spend':
                rate = m['accounting_usd_per_gpu_hour']
                if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate != 0:
                    raise ValueError('external API objective requires explicit zero local monetary charge')
            else:
                positive(m["accounting_usd_per_gpu_hour"], "local accounting rate")
            if not m.get("path"):
                raise ValueError("local snapshot path required")
        else:
            if not m["endpoint"].startswith("https://") or not m.get("key_environment"):
                raise ValueError("HTTPS endpoint and credential variable name required")
            if "key" in m or "api_key" in m:
                raise ValueError("credentials must never appear in frozen config")
            if selection == 'pending_selection':
                continue
            if not m.get('model') or not m.get('accepted_response_models'):
                raise ValueError('configured API model and accepted response identities required')
            for price in m["pricing"]["peak"].values():
                positive(price, "API token price")
    if not set(config["allowed_operators"]) <= OPERATORS or not config["allowed_operators"]:
        raise ValueError("known explicit operator list required")
    p = config["policy"]
    for key in ("k", "blocks_per_cycle", "population_threshold", "proxy_hidden", "proxy_steps", "curvature_window"):
        positive(p[key], key, True)
    if p["k"] < 2:
        raise ValueError("main method requires combinatorial selection k >= 2")
    for key in ("ridge", "proxy_learning_rate", "kernel_bandwidth", "search_exploration", "rho0", "alpha",
                "cost_scale_usd", "curvature_confidence_constant", "noise_bound", "parameter_bound"):
        positive(p[key], key)
    for key in ("proxy_fraction", "delta", "broad_depth_ratio"):
        if not 0 < p[key] <= 1:
            raise ValueError(f"invalid {key}")
    if p["delta"] == 1 or p["broad_depth_ratio"] == 1 or p["proxy_learning_rate"] > 1:
        raise ValueError("invalid confidence, topology or optimizer parameter")
    if not p["weight_grid"] or any(len(w) != 2 or any(not math.isfinite(v) or v <= 0 for v in w) for w in p["weight_grid"]):
        raise ValueError("positive quality/cost weight pairs required")
    for key in ("diversity_weight", "epsilon_max"):
        if not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"invalid {key}")
    if config["evaluation"]["status"] not in {"configured", "deferred"}:
        raise ValueError("explicit evaluation status required")
    positive(config["evaluation"]["samples_per_task"], "evaluation samples", True)
    if not config["evaluation"]["pass_k"] or any(type(k) is not int or not 1 <= k <= config["evaluation"]["samples_per_task"] for k in config["evaluation"]["pass_k"]):
        raise ValueError("Pass@k requires n >= k >= 1 samples")
    if config["evaluation"]["selection"] != "frozen_oful_chebyshev":
        raise ValueError("terminal evaluation retains OFUL/Chebyshev with frozen heads, no test-label updates")
    for key in ("seed", "max_new_tokens"):
        if type(config["decode"][key]) is not int or (key == "max_new_tokens" and config["decode"][key] < 1):
            raise ValueError(f"invalid decode {key}")
    positive(config["limits"]["seconds_per_round"], "round wall time")
    positive(config["limits"]["fixed_seconds"], "initialization/evaluation wall time")
    positive(config["limits"]["api_usd_fixed"], "fixed API ceiling")
    positive(config["limits"]["api_usd_per_round"], "per-round API ceiling")
    if config["evaluation"]["status"] == "configured":
        positive(config["limits"]["evaluation_seconds"], "separate evaluation wall time")
        positive(config["limits"]["evaluation_api_usd"], "separate evaluation API ceiling")


def load_protocol(path, root, rounds=None):
    config = json.loads(Path(path).read_text())
    if rounds is not None:
        config["rounds"] = rounds
    validate_main(config)
    pack_path = root / config["data_manifest"]
    pack = json.loads(pack_path.read_text())
    if file_hash(pack_path) != config["data_manifest_sha256"]:
        raise IntegrityError("data manifest changed")
    tasks = {}
    from .math_protocol import question_hash, TEXT_PROTOCOL
    ids, public_hashes, math_questions = set(), set(), set()
    for role in ("development", "calibration", "evaluation"):
        if role == "evaluation" and config["evaluation"]["status"] == "deferred":
            tasks[role] = []
            continue
        entry = pack["partitions"][role]
        source = pack_path.parent / entry["path"]
        if file_hash(source) != entry["sha256"]:
            raise IntegrityError(f"{role} task file changed")
        tasks[role] = load_tasks(source)
        if not tasks[role] or len(tasks[role]) != entry["count"]:
            raise IntegrityError(f"invalid {role} task count")
        for t in tasks[role]:
            if t.role != role or (t.split == "test" and role != "evaluation"):
                raise IntegrityError("dataset role/test-split violation")
            h = digest(t.model_input())
            if t.id in ids or h in public_hashes:
                raise IntegrityError("duplicate query within/across data partitions")
            ids.add(t.id); public_hashes.add(h)
            if t.dataset == "math":
                q = question_hash(t.question)
                if q in math_questions:
                    raise IntegrityError("duplicate MATH question despite changed metadata or formatting")
                math_questions.add(q)
            if role != "evaluation":
                t.require_learning()
    benchmark = config["evaluation"].get("benchmark")
    subset_keys = {"subset_manifest", "subset_manifest_sha256"}
    if subset_keys & config["evaluation"].keys() and benchmark != "math500_quick35":
        raise IntegrityError("subset settings are only valid for the explicit quick35 benchmark")
    evaluation_provenance = {}
    if benchmark in {"math500_full", "math500_quick35"}:
        if config["evaluation"]["status"] != "configured" or len(tasks["evaluation"]) != 500:
            raise IntegrityError("MATH-500 parent cannot be silently deferred or reduced")
        if pack.get("protocol") != "main_math500_v1" or pack.get("evaluation_exclusions") != []:
            raise IntegrityError("full MATH-500 requires the pinned zero-exclusion data protocol")
        if config.get("input_protocol") != TEXT_PROTOCOL or pack.get("input_protocol") != TEXT_PROTOCOL:
            raise IntegrityError("explicit unchanged inline-Asymptote text protocol required")
        for ts in tasks.values():
            if any((t.private or {}).get("input_protocol") != TEXT_PROTOCOL for t in ts):
                raise IntegrityError("task input representation differs from the frozen MATH protocol")
        if benchmark == "math500_quick35":
            from .math_protocol import quick35_indices, QUICK35_METHOD, QUICK35_SEED
            subset_path = root / config["evaluation"]["subset_manifest"]
            if file_hash(subset_path) != config["evaluation"]["subset_manifest_sha256"]:
                raise IntegrityError("quick35 selection manifest changed")
            subset = json.loads(subset_path.read_text())
            indices = quick35_indices(tasks["evaluation"])
            expected = {"benchmark": benchmark, "selection_method": QUICK35_METHOD, "seed": QUICK35_SEED,
                        "parent_evaluation_sha256": pack["partitions"]["evaluation"]["sha256"],
                        "source_indices": indices, "task_ids": [tasks["evaluation"][i].id for i in indices]}
            if subset != expected:
                raise IntegrityError("quick35 must use the frozen result-independent stratified selection")
            tasks["evaluation"] = [tasks["evaluation"][i] for i in indices]
            evaluation_provenance = {"evaluation_subset": subset,
                                     "evaluation_source_indices": indices,
                                     "evaluation_subset_sha256": file_hash(subset_path)}
    # Only operating horizon and its derived engineering caps differ in a short run.
    learning_contract = copy.deepcopy(config)
    learning_contract.pop("rounds")
    return config, tasks, {"protocol_hash_excluding_rounds": digest(learning_contract),
                           "data_manifest": pack, "data_manifest_sha256": file_hash(pack_path),
                           **evaluation_provenance}


def max_call_usd(profile, output_tokens):
    if profile["kind"] == "local":
        return profile["call_timeout_seconds"] * profile["accounting_usd_per_gpu_hour"] / 3600
    p = profile["pricing"]["peak"]
    return (profile["max_context_tokens"] * p["input_per_million_usd"] + output_tokens * p["output_per_million_usd"]) / 1e6


def derive_limits(config, tasks, seeds):
    """Reserve terminal evaluation separately; no round may consume its allowance."""
    if model_selection_problems(config):
        raise ValueError('cannot derive cost envelopes before middle model selection')
    nodes = max(config["generation_max_nodes"], *(g.model_calls for g in seeds))
    search = config["policy"]["k"] + math.ceil(config["policy"]["k"] * config["policy"]["proxy_fraction"]) * nodes * len(tasks["development"])
    deploy = nodes * config["queries_per_deploy_block"]
    block = max(search, deploy)
    bootstrap = sum(g.model_calls for g in seeds) * len(tasks["development"])
    evaluation = len(tasks["evaluation"]) * config["evaluation"]["samples_per_task"] * len(config["policy"]["weight_grid"]) * nodes
    blocks = config["rounds"] * config["policy"]["blocks_per_cycle"]
    call_usd = max(max_call_usd(m, config["decode"]["max_new_tokens"]) for m in config["models"].values())
    learning_seconds = config["limits"]["fixed_seconds"] + config["rounds"] * config["limits"]["seconds_per_round"]
    evaluation_seconds = config["limits"].get("evaluation_seconds", 0) if evaluation else 0
    learning_api = config["limits"]["api_usd_fixed"] + config["rounds"] * config["limits"]["api_usd_per_round"]
    evaluation_api = config["limits"].get("evaluation_api_usd", 0) if evaluation else 0
    return {"bootstrap_calls": bootstrap, "block_calls": block, "evaluation_calls": evaluation,
            "evaluation_workflows": len(tasks["evaluation"]) * config["evaluation"]["samples_per_task"] * len(config["policy"]["weight_grid"]),
            "learning_calls": bootstrap + blocks * block,
            "max_calls": bootstrap + blocks * block + evaluation,
            "max_seconds": learning_seconds + evaluation_seconds,
            "learning_seconds": learning_seconds, "evaluation_seconds": evaluation_seconds,
            "max_call_accounting_usd": call_usd, "block_accounting_usd": block * call_usd,
            "max_accounting_usd": (bootstrap + blocks * block + evaluation) * call_usd,
            "max_api_usd": learning_api + evaluation_api,
            "learning_api_usd": learning_api, "evaluation_api_usd": evaluation_api,
            "budget_meaning": "fixed upper-bound USD-equivalent envelopes; unused allocation is reported, not billed"}
