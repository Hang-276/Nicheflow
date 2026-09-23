#!/usr/bin/env python3
"""Audit frozen smoke evidence on CPU. No model, network, or subprocess fallback.

Replays historical responses in a disposable directory; this is NOT a new run.
Run: .venv/bin/python scripts/audit_smoke_offline.py
"""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow.datasets import load_tasks
from nicheflow.graph import WorkflowGraph
from nicheflow.ledger import Journal
from nicheflow.policy import MinimalPolicy, router_features
from nicheflow.runtime import GraphExecutor
from nicheflow.scoring import evaluate
from nicheflow.spec import digest


def deny_external_work(event, args):
    if event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn", "os.posix_spawn")) or event == "os.system":
        raise RuntimeError(f"Offline audit prohibits {event}")
    if event == "import" and args[0].split(".")[0] in {"torch", "transformers", "vllm", "openai", "anthropic"}:
        raise RuntimeError("Offline audit prohibits model libraries")


sys.addaudithook(deny_external_work)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def same(actual, expected, path="value"):
    """Exact structure/text; tolerate only floating-point arithmetic differences."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        require(actual.keys() == expected.keys(), f"{path}: keys differ")
        for key in actual:
            same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f"{path}: lengths differ")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f"{path}[{i}]")
    elif type(actual) in (float, int) and type(expected) in (float, int):
        require(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10), f"{path}: {actual} != {expected}")
    else:
        require(actual == expected, f"{path}: {actual!r} != {expected!r}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floating_deltas(a, b):
    if isinstance(a, dict):
        return [d for key in a for d in floating_deltas(a[key], b[key])]
    if isinstance(a, list):
        return [d for x, y in zip(a, b) for d in floating_deltas(x, y)]
    if type(a) in (int, float) and type(b) in (int, float):
        return [abs(a - b)] if a != b else []
    return []


class CachedResponsesOnly:
    def __init__(self, events):
        self.starts = [e for e in events if e["kind"] == "call_started"]
        self.ends = {e["id"]: e["payload"] for e in events if e["kind"] == "call_finished"}
        self.index = 0
        self.model_id = self.starts[0]["payload"]["model"]
        require(all(e["payload"]["model"] == self.model_id for e in self.starts), "mixed model identities")

    def generate(self, messages, **params):
        require(self.index < len(self.starts), "cache exhausted; real inference forbidden")
        expected = self.starts[self.index]
        request = {"messages": messages, "params": params, "model": self.model_id}
        require(digest(request) == digest(expected["payload"]), f"cached request mismatch at {expected['id']}")
        self.index += 1
        return copy.deepcopy(self.ends[expected["id"]])


def hv(points):
    # Independent exact rectangle-union calculation for nonnegative 2D points.
    xs = sorted({0.0, *(p[0] for p in points)})
    return sum((b - a) * max((y for x, y in points if x >= b), default=0.)
               for a, b in zip(xs, xs[1:]))


def new_head(d, p):
    return {"v": np.eye(d) * p["ridge"], "b": np.zeros(d), "count": 0, "regularization": p["ridge"]}


def state(head):
    return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in head.items()}


def update(head, x, reward):
    head["v"] += np.outer(x, x)
    head["b"] += x * reward
    head["count"] += 1


def prediction(head, x, p):
    v, b, reg = head["v"], head["b"], head["regularization"]
    beta = p["noise_bound"] * math.sqrt(np.linalg.slogdet(v)[1] - len(b) * math.log(reg)
                                        + 2 * math.log(1 / p["delta"])) + math.sqrt(reg) * p["parameter_bound"]
    mean = float(x @ np.linalg.solve(v, b))
    width = float(beta * math.sqrt(x @ np.linalg.solve(v, x)))
    return {"mean": mean, "width": width, "ucb": mean + width, "lcb": mean - width}, beta


def main():
    run = ROOT / "runs/smoke_v030"
    frozen = json.loads((run / "config.json").read_text())
    config, code = frozen["config"]["settings"], frozen["config"]["code"]
    protected = [*run.glob("*"), *(ROOT / name for name in code), ROOT / config["tasks"],
                 ROOT / "README.md", ROOT / "configs/release_manifest.json"]
    before = {str(p): sha(p) for p in protected if p.is_file()}
    for name, expected in code.items():
        require(sha(ROOT / name) == expected, f"frozen source changed: {name}")
    require(sha(ROOT / config["tasks"]) == config["tasks_sha256"], "task hash mismatch")
    events = Journal.read(run / "events.jsonl")
    by_kind = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)
    tasks = load_tasks(ROOT / config["tasks"])
    task_by_id = {t.id: t for t in tasks}
    require(all(t.role == "development" and t.split == "train" for t in tasks), "unexpected data role")
    for t in tasks:
        # Changing only private labels must leave the complete public input identical.
        same(t.model_input(), replace(t, gold="AUDIT_PRIVATE_GOLD", private={"secret": "AUDIT_PRIVATE_TEST"}).model_input())

    cache = CachedResponsesOnly(events)
    with tempfile.TemporaryDirectory(prefix="nicheflow-offline-audit-") as temp:
        with Journal(temp, frozen["config"], frozen["max_calls"], frozen["seconds"]) as journal:
            policy = MinimalPolicy(GraphExecutor(journal, {"local": cache}, config["decode"]), config, tasks)
            completion = policy.driver.run()
            require(completion["status"] == "full_smoke_pass", "historical replay did not complete")
            require(cache.index == len(cache.starts), "unused historical responses")
            replay_events = Journal.read(journal.path)
            same(len(replay_events), len(events), "event_count")
            exact_payloads = 0
            deltas = []
            for original, replay in zip(events, replay_events):
                same(replay["kind"], original["kind"], "event_kind")
                same(replay["id"], original["id"], "event_id")
                same(replay["payload"], original["payload"], f"{original['kind']}:{original['id']}")
                exact_payloads += digest(original["payload"]) == digest(replay["payload"])
                deltas.extend(floating_deltas(original["payload"], replay["payload"]))
            final_meta = {name: head.state() for name, head in policy.meta.items()}

    # Fail-closed behavior is checked without invoking any model backend.
    first_request = cache.starts[0]["payload"]
    for candidate, messages in [(cache, first_request["messages"]), (CachedResponsesOnly(events), [])]:
        try:
            candidate.generate(messages, **first_request["params"])
        except AssertionError:
            pass
        else:
            raise AssertionError("missing/mismatched cache did not stop")

    calls = {e["id"]: e["payload"] for e in by_kind["call_finished"]}
    executions = {e["id"]: e["payload"] for e in by_kind["execution"]}
    for id, ex in executions.items():
        same(evaluate(task_by_id[ex["task_id"]], ex["final_output"]), ex["evaluation"], f"rescore:{id}")
    p = config["policy"]
    search_checks = []
    for event in by_kind["search"]:
        s = event["payload"]
        k, features = np.array(s["kernel"]), np.array(s["features"])
        ucbs = np.array([s["trace"][0]["scores"][str(i)]["ucb"] for i in s["ids"]])
        expected = ucbs[:, None] * np.exp(-p["kernel_bandwidth"] * ((features[:, None] - features[None, :]) ** 2).sum(axis=2)) * ucbs[None, :]
        same(k.tolist(), expected.tolist(), "independent_kernel")
        minimum_eigenvalue = float(np.linalg.eigvalsh(k).min())
        require(minimum_eigenvalue >= -1e-10, "kernel is not PSD")
        positions = {n: i for i, n in enumerate(s["ids"])}
        def logdet(ids):
            if not ids:
                return 0.
            sign, value = np.linalg.slogdet(k[np.ix_([positions[i] for i in ids], [positions[i] for i in ids])])
            require(sign > 0, "nonpositive determinant in observed greedy step")
            return float(value)
        for step in s["trace"]:
            for candidate, score in step["scores"].items():
                gain = logdet(step["selected_before"] + [int(candidate)]) - logdet(step["selected_before"])
                same(gain, score["logdet_gain"], "logdet_gain")
                same(score["ucb"] + p["diversity_weight"] * gain, score["score"], "greedy_score")
            winner = int(max(step["scores"], key=lambda c: step["scores"][c]["score"]))
            same(winner, step["winner"], "greedy_winner")
        search_checks.append({"id": event["id"], "selected": s["selected"], "minimum_kernel_eigenvalue": minimum_eigenvalue})

    arms, graphs, route_checks = {}, {}, []
    meta = {name: new_head(4, p) for name in ("search", "deploy")}
    carry, shadow_carry = .5, .5
    meta_x, last_decision = None, None
    scheduler_checks, curvature_points = [], []
    old_vendi, old_hv = 0., 0.
    for e in events:
        kind, value = e["kind"], e["payload"]
        if kind == "workflow":
            graphs[e["id"]] = WorkflowGraph.from_dict(value)
        elif kind == "router_arm":
            arms[e["id"]] = {name: new_head(11, p) for name in ("quality", "cost")}
        elif kind == "route_decision":
            ex = executions[e["id"] + ":execution"]
            public = task_by_id[ex["task_id"]].model_input()
            weights = p["weight_grid"][len(route_checks) % len(p["weight_grid"])]
            same(value["weights"], weights)
            cold_scores = {}
            for arm, heads in arms.items():
                x = np.array(router_features(public, graphs[arm]))
                q, _ = prediction(heads["quality"], x, p)
                c, _ = prediction(heads["cost"], x, p)
                same(q, value["scores"][arm]["quality"], "quality_prediction")
                same(c, value["scores"][arm]["cost_reward"], "cost_prediction")
                same(min(weights[0] * q["ucb"], weights[1] * c["ucb"]), value["scores"][arm]["score"])
                cold, _ = prediction(new_head(11, p), x, p)
                cold_scores[arm] = min(weights) * cold["ucb"]
            winner = max(value["scores"], key=lambda a: value["scores"][a]["score"])
            same(winner, value["arm"], "route_winner")
            cold_winner = max(cold_scores, key=cold_scores.get)
            route_checks.append({"id": e["id"], "chosen": winner, "chosen_previous_updates": arms[winner]["quality"]["count"],
                                 "all_heads_reset_winner": cold_winner, "changes_if_all_heads_reset": winner != cold_winner})
        elif kind == "router":
            ex = executions[value["execution_receipt"]]
            arm = ex["workflow"]
            x = np.array(router_features(task_by_id[ex["task_id"]].model_input(), graphs[arm]))
            cost = sum(calls[c]["elapsed_seconds"] for c in ex["calls"]) * p["reference_usd_per_gpu_hour"] / 3600
            update(arms[arm]["quality"], x, ex["evaluation"]["quality"])
            update(arms[arm]["cost"], x, 1 / (1 + cost / p["cost_scale_usd"]))
            same({a: {name: state(h) for name, h in heads.items()} for a, heads in arms.items()}, value["state"]["arms"], "ridge_updates")
        elif kind == "curvature_observation":
            points = list(value["conditioning_points"].values())
            same(hv(points), value["baseline"], "hv_baseline")
            same(hv(points + [value["point"]]) - hv(points), value["marginal"], "hv_marginal")
            same(hv([value["point"]]), value["empty_marginal"], "hv_singleton")
            if value["status"] == "observed":
                curvature_points.append(e)
        elif kind == "scheduler":
            cycle = int(e["id"].split(":")[1])
            context = value["context"]
            meta_x = np.array([context["remaining_research_budget"] / context["total_blocks"], context["vendi"] / 100,
                               context["hv"], min(context["epoch"] / 100, 1)])
            last_decision = value
            for name, head in meta.items():
                pred, beta = prediction(head, meta_x, p)
                logged = value["meta_linear_ucb"][name]
                same(state(head), logged["head_before_update"], "meta_history")
                same(pred, {k: logged[k] for k in pred}, "meta_prediction")
                same(beta, logged["beta"], "meta_beta")
                same(meta_x.tolist(), logged["features"], "meta_features")
            window = curvature_points[-p["curvature_window"]:]
            empirical = max(1 - o["payload"]["rounded_marginal"] / o["payload"]["empty_marginal"] for o in window)
            epsilon = p["curvature_confidence_constant"] * math.sqrt(math.log(cycle / p["delta"]) / len(window))
            fallback = epsilon > p["epsilon_max"]
            raw_gamma = 1. if fallback else empirical + epsilon
            gamma = min(1., raw_gamma)
            h = 1. if gamma == 0 else -math.expm1(-gamma) / gamma
            same({"empirical": empirical, "epsilon": epsilon, "gamma": gamma, "h": h, "m": len(window),
                  "fallback": fallback, "capped_at_one": raw_gamma > 1},
                 {k: value["estimate"][k] for k in ["empirical", "epsilon", "gamma", "h", "m", "fallback", "capped_at_one"]})
            s, d = [max(0, value["meta_linear_ucb"][name]["ucb"]) for name in ("search", "deploy")]
            beta = h * s / (h * s + d) if h * s + d else .5
            same(beta, value["allocation"]["search"], "allocation")
            shadow_beta = s / (s + d) if s + d else .5
            scheduler_checks.append({"id": e["id"], "gamma": gamma, "epsilon": epsilon, "empirical": empirical,
                                     "actual_search_fraction": beta, "gamma_zero_frozen_history_search_fraction": shadow_beta})
        elif kind == "operation_plan":
            blocks = min(p["blocks_per_cycle"], last_decision["context"]["remaining_research_budget"])
            target = carry + blocks * last_decision["allocation"]["search"]
            count = min(blocks, math.floor(target))
            carry = target - count
            same(["search"] * count + ["deploy"] * (blocks - count), value["operations"], "operation_rounding")
            shadow_target = shadow_carry + blocks * scheduler_checks[-1]["gamma_zero_frozen_history_search_fraction"]
            shadow_count = min(blocks, math.floor(shadow_target))
            shadow_carry = shadow_target - shadow_count
            scheduler_checks[-1].update(actual_operations=value["operations"],
                                       gamma_zero_frozen_history_operations=["search"] * shadow_count + ["deploy"] * (blocks - shadow_count))
        elif kind == "feedback":
            stage = value["stage"]
            if stage in meta:
                reward = (value["vendi"] - old_vendi) / 100 if stage == "search" else value["deployment_hv"] - old_hv
                same(reward, value["stage_reward"], "meta_reward")
                update(meta[stage], meta_x, reward)
            old_vendi, old_hv = value["vendi"], value["deployment_hv"]
    same({name: state(head) for name, head in meta.items()}, final_meta, "final_meta_state")

    repeats = {}
    for id, ex in executions.items():
        repeats.setdefault((ex["workflow"], ex["task_id"]), []).append({"execution_id": id, "quality": ex["evaluation"]["quality"]})
    repeated_pairs = [{"workflow": key[0], "task_id": key[1], "observations": vals,
                       "score_changed": len({v["quality"] for v in vals}) > 1}
                      for key, vals in repeats.items() if len(vals) > 1]
    require(before == {name: sha(Path(name)) for name in before}, "original evidence or frozen release modified")
    result = {
        "status": "offline_checks_pass", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "new_model_calls": 0, "new_model_tokens": 0, "network_or_subprocess_allowed": False,
        "source_run": str(run), "source_events_sha256": before[str(run / "events.jsonl")],
        "frozen_source_files_verified": len(code), "original_evidence_and_frozen_release_unchanged": True,
        "cached_replay": {"responses_reused": cache.index, "exact_request_matches": cache.index,
                          "events_checked": len(events), "bit_exact_payloads": exact_payloads,
                          "maximum_absolute_numeric_difference": max(deltas, default=0.),
                          "mismatched_and_exhausted_cache_refused": True,
                          "numeric_relative_tolerance": 1e-9, "numeric_absolute_tolerance": 1e-10,
                          "temporary_journal_deleted": True, "is_new_real_or_performance_run": False},
        "rescored_executions": len(executions), "private_label_public_input_invariance_tasks": len(tasks),
        "search_checks": search_checks, "router_checks": route_checks, "scheduler_checks": scheduler_checks,
        "curvature_marginals_recomputed": len(curvature_points), "repeated_workflow_task_pairs": repeated_pairs,
        "limitations": ["Cache replay validates deterministic processing, not new model behavior or recovery after a new failure.",
                        "Independent numerical checks share logged inputs and public feature construction; not an independent algorithm implementation.",
                        "Gamma-zero and reset-head comparisons freeze actual history; they are arithmetic sensitivity checks, not counterfactual performance results.",
                        "Original smoke only uses MATH development tasks, one model, and engineering completions; no theorem or generalization claim."]}
    output = ROOT / "reports/smoke_v030_offline_checks.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
