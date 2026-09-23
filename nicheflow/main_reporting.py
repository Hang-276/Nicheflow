"""Main-method reports derived from receipts and frozen evaluation state."""
from collections import defaultdict
import fcntl
import json
from pathlib import Path
from .ledger import Journal, atomic_json
from .metrics import pass_at_k, hypervolume
from .spec import IntegrityError


def main_report(directory):
    directory = Path(directory)
    frozen = json.loads((directory / "config.json").read_text())
    config = frozen["config"]["settings"]
    events = Journal.read(directory / "events.jsonl")
    kinds = defaultdict(list)
    for e in events:
        kinds[e["kind"]].append(e)
    started = {e["id"] for e in kinds["call_started"]}
    calls = {e["id"]: e["payload"] for e in kinds["call_finished"]}
    if not calls.keys() <= started or len(started) > frozen["max_calls"]:
        raise IntegrityError("call accounting mismatch")
    completion = kinds["run_finished"][-1]["payload"] if kinds["run_finished"] else {}
    known = [c for c in calls.values() if c.get("accounted_usd") is not None]
    call_models = {e['id']: e['payload']['model'] for e in kinds['call_started']}
    local_ids = {name if frozen['config'].get('synthetic', False) else profile['path']
                 for name, profile in config['models'].items() if profile['kind'] == 'local'}
    local_seconds = sum(c.get('elapsed_seconds') or 0 for key, c in calls.items()
                        if call_models[key] in local_ids)
    health = [{"id": e["id"], **e["payload"]} for e in kinds["parameter_health"]]
    condition_values = [h["condition_number"] for e in health for h in e["heads"].values() if h["condition_number"] is not None]
    scored_length = [e["id"] for e in kinds["execution"] if config.get("length_limit_outcome") == "zero_quality_no_retry"
                     and e["payload"]["status"] == "truncated" and e["payload"].get("error") == "length"]
    errors = [e["id"] for e in kinds["execution"] if e["payload"]["status"] != "ok" and e["id"] not in scored_length]
    generated = {e["payload"]["version"] for e in kinds["generation"] if e["payload"]["status"] == "valid"}
    registered = {e["id"] for e in kinds["router_arm"]}
    routed = {e["payload"]["decision"]["arm"] for e in kinds["router"]}
    summary = {"mode": "main", "status": completion.get("status", "incomplete"),
        "synthetic": frozen["config"].get("synthetic", False), "rounds_requested": config["rounds"],
        "synthetic_backend": frozen["config"].get("synthetic", False),
        "synthetic_data": frozen["config"]["provenance"]["data_manifest"].get("synthetic", False),
        "rounds_completed": completion.get('rounds_completed', len(kinds["cycle_finished"])), "calls_attempted": len(started), "calls_finished": len(calls),
        "unknown_calls": sorted(started - calls.keys()), "unknown_cost_calls": [id for id, c in calls.items() if c.get("accounted_usd") is None],
        "input_tokens": sum(c.get("input_tokens") or 0 for c in calls.values()),
        "output_tokens": sum(c.get("output_tokens") or 0 for c in calls.values()),
        "unknown_token_calls": sum(c.get("input_tokens") is None or c.get("output_tokens") is None for c in calls.values()),
        "accounted_usd": sum(c["accounted_usd"] for c in known),
        "api_tariff_usd": sum((c.get("api_charge") or {}).get("usd", 0) for c in calls.values()),
        "local_inference_accounting_usd": sum(c.get("local_accounting_usd") or 0 for c in calls.values()),
        "cost_objective": config.get('cost_objective', 'api_plus_declared_local_time'),
        "local_inference_seconds": local_seconds,
        "local_time_measurement": "sum of measured local call durations, not device energy or total server rental time",
        "actual_invoice_verified": False, "cost_excludes_model_load_idle_and_cpu": True,
        "elapsed_seconds": max((e["time"] for e in events), default=frozen["started"]) - frozen["started"],
        "execution_count": len(kinds["execution"]), "execution_errors": errors, "hash_chain_valid": True,
        "length_limited_workflows": scored_length, "length_limit_outcome": config.get("length_limit_outcome", "stop"),
        "numerical_health": {"snapshots": len(health), "all_finite_spd": all(h["numerically_healthy"] for h in health) if health else None,
                             "maximum_condition_number": max(condition_values, default=None), "convergence_established": False},
        "coverage": {"candidate_generation": bool(generated), "combinatorial_selection": bool(kinds["search"]),
                     "new_elite_route_feedback": bool(generated & registered & routed), "router_updates": len(kinds["router"]),
                     "scheduler_decisions": len(kinds["scheduler"]), "curvature_observations": len(kinds["curvature_observation"])},
        "evaluation": {"status": "deferred" if config["evaluation"]["status"] == "deferred" else "incomplete"},
        "limits": frozen["config"]["limits"], "exact_source_reproduction": False,
        "protocol_hash_excluding_rounds": frozen["config"]["provenance"]["protocol_hash_excluding_rounds"],
        "formal_training_automatically_started": False}
    if kinds["evaluation_finished"]:
        finish = kinds["evaluation_finished"][-1]["payload"]
        if finish["state_digest_before"] != finish["state_digest_after"]:
            raise IntegrityError("evaluation state changed")
        executions = {e["id"]: e["payload"] for e in kinds["execution"]}
        contract = kinds["main_contract"][0]["payload"]
        expected_tasks = [t["id"] for t in contract["partitions"]["evaluation"]]
        source_indices = contract["provenance"].get("evaluation_source_indices", list(range(len(expected_tasks))))
        expected = {f"evaluation:weight:{wi}:task:{ti}:sample:{s}"
                    for wi in range(len(config["policy"]["weight_grid"]))
                    for ti in source_indices
                    for s in range(config["evaluation"]["samples_per_task"])}
        routes = {e["id"] for e in kinds["evaluation_route"]}
        if routes != expected or set(finish["execution_ids"]) != expected or len(finish["execution_ids"]) != len(expected):
            raise IntegrityError("completed evaluation does not cover the full frozen task/weight/sample grid")
        boundary = kinds["evaluation_started"][0]["seq"]
        if any(e["kind"] in {"feedback", "router", "archive", "scheduler", "proxy_train", "state_snapshot"} for e in events[boundary:]):
            raise IntegrityError("learning event found after terminal evaluation started")
        grouped = defaultdict(lambda: defaultdict(list))
        for event in kinds["evaluation_route"]:
            ex = executions.get(event["id"])
            route = event["payload"]
            if ex is None or ex["task_id"] != route["task_id"] or ex["role"] != "evaluation":
                raise IntegrityError("completed evaluation has an unexecuted route")
            grouped[route["weight_index"]][ex["task_id"]].append(ex)
        weights = []
        for wi, tasks in sorted(grouped.items()):
            if set(tasks) != set(expected_tasks):
                raise IntegrityError("evaluation task membership changed")
            qualities, costs = [], []
            api_costs, local_costs, failures, parse_failures = [], [], 0, 0
            pk = {str(k): [] for k in config["evaluation"]["pass_k"]}
            for task, rows in tasks.items():
                if len(rows) != config["evaluation"]["samples_per_task"]:
                    raise IntegrityError("incomplete evaluation sampling")
                quality = [r["evaluation"]["quality"] if r["status"] == "ok" else 0. for r in rows]
                qualities.extend(quality)
                costs.extend(sum(calls[c]["accounted_usd"] for c in r["calls"]) for r in rows)
                api_costs.extend(sum((calls[c].get("api_charge") or {}).get("usd", 0.) for c in r["calls"]) for r in rows)
                local_costs.extend(sum(calls[c].get("local_accounting_usd") or 0. for c in r["calls"]) for r in rows)
                failures += sum(r["status"] != "ok" for r in rows)
                parse_failures += sum(r["status"] == "ok" and r["evaluation"]["metrics"].get("parsed") is False for r in rows)
                for k in config["evaluation"]["pass_k"]:
                    pk[str(k)].append(pass_at_k(len(rows), sum(q == 1. for q in quality), k))
            mean_quality, mean_cost = sum(qualities) / len(qualities), sum(costs) / len(costs)
            weights.append({"weight_index": wi, "weights": config["policy"]["weight_grid"][wi], "tasks": len(tasks),
                "mean_quality": mean_quality, "mean_accounting_usd_per_query": mean_cost,
                "mean_api_tariff_usd_per_query": sum(api_costs) / len(costs),
                "mean_local_accounting_usd_per_query": sum(local_costs) / len(costs),
                "workflow_executions": len(costs), "execution_failures": failures,
                "execution_failure_rate": failures / len(costs), "parse_failures": parse_failures,
                "cost_reward": 1 / (1 + mean_cost / config["policy"]["cost_scale_usd"]),
                "pass_at_k": {k: sum(values) / len(values) for k, values in pk.items()}})
        summary["evaluation"] = {"status": "complete", "learning_state_unchanged": True,
            "expected_workflows": len(expected), "completed_workflows": len(routes), "task_count": len(expected_tasks),
            "weight_results": weights, "pass_definition": "quality == 1; execution failures count as unsuccessful and are separately reported",
            "hv_quality_inverse_cost": hypervolume([(w["mean_quality"], w["cost_reward"]) for w in weights], (0, 0)),
            "hv_reference": [0, 0], "cost_transform": "1/(1+mean_query_accounting_usd/cost_scale_usd)"}
    benchmark = config["evaluation"].get("benchmark")
    if config.get('research_revision'):
        summary['research_revision'] = config['research_revision']
        summary['stage'] = frozen['config'].get('stage')
        summary['stage_rounds_completed'] = len(kinds['cycle_finished'])
        summary['cost_scope'] = 'this_stage_only; ancestral costs remain in linked source run'
        summary['source_theoretical_guarantees_verified'] = False
        summary['parent_credit_updates'] = len(kinds['parent_credit'])
    if benchmark in {"math500_full", "math500_quick35"}:
        quick = benchmark == "math500_quick35"
        summary["evaluation"].update(benchmark=benchmark, diagnostic_subset=quick,
            full_math500=not quick, configured_task_count=35 if quick else 500,
            interpretation=("Fixed 35-stratum diagnostic subset; not a full MATH-500 score or a population-weighted estimate."
                            if quick else "Complete published MATH-500 under the frozen text protocol."))
    active_writer = False
    with (directory / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_UN)
        except BlockingIOError:
            active_writer = True
    summary["writer_active"] = active_writer
    if summary["unknown_cost_calls"] or (summary["unknown_calls"] and not active_writer):
        summary["status"] = "needs_manual_recovery"
    elif summary["unknown_calls"]:
        summary["status"] = "call_in_progress"
    atomic_json(directory / "summary.json", summary)
    atomic_json(directory / "parameter_stability.json", {"snapshots": health, "statistical_convergence_established": False})
    for name in ("call_finished", "execution", "scheduler", "operation_plan", "curvature_observation", "evaluation_route", "resource_envelope", "operation_finished"):
        (directory / f"{name}.jsonl").write_text("".join(json.dumps({"id": e["id"], **e["payload"]}, ensure_ascii=False) + "\n" for e in kinds[name]))
    (directory / "report.md").write_text("# NicheFlow 主实验运行记录\n\n"
        + f"状态：**{summary['status']}**。轮数：{summary['rounds_completed']}/{config['rounds']}。合成后端：{summary['synthetic_backend']}；合成数据：{summary['synthetic_data']}。\n\n"
        + ("本报告使用合成后端，只验证工程链路，所有分数均不是模型性能证据。\n\n" if summary['synthetic_backend'] else "")
        + ("本次为七科×五难度共35题的固定诊断子集；分数不是完整MATH-500成绩，也不是按全体题目比例加权的估计。\n\n" if benchmark == "math500_quick35" else "")
        + "数值健康不等于统计收敛；评估待配置时不产生正式性能结论。API按返回用量和冻结费率核算，未核对实际账单。\n\n"
        + ("本轮优化外部API支出：本地算力不计入货币目标，推理耗时单列；这不是零能耗或零总体运营成本的声明。\n\n"
           if config.get('cost_objective') == 'external_api_spend' else "本地费用按推理秒数乘声明费率核算，非实际租卡账单。\n\n")
        + "```json\n" + json.dumps(summary, ensure_ascii=False, indent=2) + "\n```\n")
    return summary
