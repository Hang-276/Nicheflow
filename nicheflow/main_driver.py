"""One main-method driver for one-round validation and full-length runs."""
from pathlib import Path
import numpy as np
from .graph import WorkflowGraph
from .ledger import atomic_json
from .main_policy import stability
from .policy import router_features
from .router import BiGLinUCB
from .spec import BudgetStop, EnvironmentBlocked, ExecutionStop, IntegrityError, SpecGap, digest


def frozen_router(state):
    first = next(iter(state["router"]["arms"].values()))["quality"]
    router = BiGLinUCB(len(first["b"]), first["regularization"])
    for version, pair in state["router"]["arms"].items():
        router.add_arm(version)
        for name, head in zip(("quality", "cost"), router.arms[version]):
            record = pair[name]
            head.v, head.b = np.asarray(record["v"], float), np.asarray(record["b"], float)
            head.count = record["count"]
    router.epoch = state["router"]["epoch"]
    return router


class MainDriver:
    def __init__(self, policy, partitions, provenance):
        self.policy, self.tasks, self.provenance = policy, partitions, provenance
        self.executor, self.journal, self.config = policy.executor, policy.journal, policy.config
        self.limits = self.journal.limits
        self.executor.failure_limit = 3
        self.executor.generation_model = self.config["generation_model"]

    def feedback(self, id, stage, results):
        value = self.policy.observe(stage, results)
        paid = {e["id"] for e in self.journal.events if e["kind"] == "call_finished"}
        if not set(value["receipt_ids"]) <= paid:
            raise IntegrityError("feedback references missing receipts")
        self.journal.append("feedback", id, {"stage": stage, "profile": self.config["profile"], **value})

    def snapshot(self, round, previous):
        value = self.policy.export_state()
        health = stability(value, previous)
        self.journal.append("parameter_health", f"round:{round}", health)
        if not health["numerically_healthy"]:
            raise ExecutionStop("non-finite or invalid parameter matrices")
        identity = digest(value)
        self.journal.append("state_snapshot", f"round:{round}", {"digest": identity, "state": value})
        folder = self.journal.directory / "checkpoints"
        folder.mkdir(exist_ok=True)
        atomic_json(folder / f"round_{round:04d}.json", {"digest": identity, "state": value})
        return value

    def scope(self, name, calls):
        return self.journal.scope(name, calls, calls * self.limits["max_call_accounting_usd"])

    def evaluate_frozen(self, frozen):
        before = digest(self.policy.export_state())
        router = frozen_router(frozen)
        router_before = digest(router.state())
        graphs = {v: WorkflowGraph.from_dict(g) for v, g in frozen["graphs"].items()}
        p = self.config["policy"]
        evidence = []
        source_indices = self.provenance.get("evaluation_source_indices", list(range(len(self.tasks["evaluation"]))))
        with self.scope("evaluation", self.limits["evaluation_calls"]):
            self.journal.append("evaluation_started", "main:evaluation", {
                "state_digest": before, "task_ids": [t.id for t in self.tasks["evaluation"]],
                "samples_per_task": self.config["evaluation"]["samples_per_task"],
                "selection": self.config["evaluation"]["selection"], "learning_updates_allowed": False})
            for wi, weights in enumerate(p["weight_grid"]):
                for ti, task in zip(source_indices, self.tasks["evaluation"]):
                    if task.role != "evaluation":
                        raise IntegrityError("terminal evaluation requires held-out tasks")
                    features = {a: self.policy.features(task.model_input(), graphs[a]) for a in router.arms}
                    confidence = self.policy.router_confidence(router, ti + 1)
                    choice = router.choose(features, weights=weights, beta_quality=confidence[0], beta_cost=confidence[1],
                        cost_reward_definition="frozen main quality/cost reward", fallback_threshold=0., safe_arm=frozen["seed_versions"][0])
                    for sample in range(self.config["evaluation"]["samples_per_task"]):
                        id = f"evaluation:weight:{wi}:task:{ti}:sample:{sample}"
                        self.journal.append("evaluation_route", id, {"task_id": task.id, "weight_index": wi,
                            "weights": weights, "sample": sample, "decision": choice, "state_digest": before})
                        result = self.executor.execute(graphs[choice["arm"]], task, id)
                        self.executor.check_failures()
                        self.journal.reserve(0)
                        evidence.append(id)
                        if len(evidence) % 25 == 0:
                            progress = {"completed": len(evidence), "total": self.limits["evaluation_workflows"],
                                        "learning_state_digest": before}
                            self.journal.append("evaluation_progress", str(len(evidence)), progress)
                            print(f"Evaluation workflows: {progress['completed']}/{progress['total']}", flush=True)
            if digest(self.policy.export_state()) != before or digest(router.state()) != router_before:
                raise IntegrityError("evaluation changed learned state")
            self.journal.append("evaluation_finished", "main:evaluation", {
                "state_digest_before": before, "state_digest_after": digest(self.policy.export_state()),
                "execution_ids": evidence, "learning_state_unchanged": True})

    def run(self):
        if self.journal.lookup("run_finished", "main:done") is not None:
            return self.journal.lookup("run_finished", "main:done")
        contract = {"profile": self.config["profile"], "rounds": self.config["rounds"],
                    "partitions": {k: [t.record() for t in ts] for k, ts in self.tasks.items()},
                    "seeds": [g.version for g in self.policy.seeds], "provenance": self.provenance,
                    "limits": self.limits, "synthetic": self.policy.synthetic}
        self.journal.append("main_contract", "main", {"digest": digest(contract), **contract})
        try:
            initial = self.policy.export_state()
            with self.scope("bootstrap", self.limits["bootstrap_calls"]):
                for i, graph in enumerate(self.policy.seeds):
                    id = f"bootstrap:{i}"
                    result = self.policy.search.evaluate(graph, self.tasks["development"], id)
                    if result["status"] != "evaluated":
                        raise ExecutionStop("bootstrap evaluation incomplete; no hidden replacement/retry")
                    self.feedback(id, "bootstrap", [result])
            previous = self.snapshot(0, initial)
            query_index = 0
            for round in range(1, self.config["rounds"] + 1):
                id = f"round:{round}"
                context = self.policy.context()
                self.journal.append("cycle_context", id, context)
                decision = self.policy.scheduler.decide(round, context, context["remaining_research_budget"])
                self.journal.append("scheduler", id, decision)
                operations = self.policy.operations(decision, context)
                self.journal.append("operation_plan", id, {"operations": operations,
                    "accounting_usd_per_block": self.limits["block_accounting_usd"],
                    "allocation_semantics": "reserved upper-bound resource envelopes, not a claim of equal realized spend"})
                for step, kind in enumerate(operations):
                    op = f"{id}:{step}:{kind}"
                    with self.scope(op, self.limits["block_calls"]):
                        if kind == "search":
                            selection, parents = self.policy.selection()
                            results = self.policy.search.run(op, selection, parents, self.tasks["development"],
                                max_nodes=self.config["generation_max_nodes"])
                        else:
                            results = []
                            for q in range(self.config["queries_per_deploy_block"]):
                                task = self.tasks["calibration"][query_index % len(self.tasks["calibration"])]
                                query_index += 1
                                results.append(self.policy.deployment.run(f"{op}:query:{q}", task, self.policy.graphs, query_index))
                                self.executor.check_failures()
                        self.feedback(op, kind, results)
                        self.executor.check_failures()
                        self.journal.append("operation_finished", op, {"kind": kind, "resource_usage": self.journal.spending(op)})
                previous = self.snapshot(round, previous)
                self.journal.append("cycle_finished", id, {"operations": operations, "state_digest": digest(previous),
                                                          "calibration_queries_seen": query_index})
            frozen_digest = digest(previous)
            self.journal.append("learning_finished", "main:learning", {"rounds": self.config["rounds"], "state_digest": frozen_digest})
            atomic_json(self.journal.directory / "trained_state.json", {"digest": frozen_digest, "state": previous})
            failed = [e["id"] for e in self.journal.events if e["kind"] == "execution" and not self.executor.assessment_complete(e["payload"])]
            if self.config["evaluation"]["status"] == "deferred":
                return self.journal.append("run_finished", "main:done", {
                    "status": "main_completed_with_execution_errors" if failed else "learning_complete_evaluation_pending", "synthetic": self.policy.synthetic,
                    "rounds_completed": self.config["rounds"], "state_digest": frozen_digest,
                    "failed_executions": failed,
                    "evaluation_complete": False, "reason": "user deferred evaluation dataset scope; no evaluation score claimed",
                    "numerical_health_only_not_convergence": True, "formal_training_automatically_started": False})
            self.evaluate_frozen(previous)
            failed = [e["id"] for e in self.journal.events if e["kind"] == "execution" and not self.executor.assessment_complete(e["payload"])]
            return self.journal.append("run_finished", "main:done", {
                "status": "main_completed_with_execution_errors" if failed else "main_run_complete",
                "synthetic": self.policy.synthetic, "rounds_completed": self.config["rounds"],
                "state_digest": frozen_digest, "failed_executions": failed, "evaluation_complete": True,
                "formal_training_automatically_started": False, "numerical_health_only_not_convergence": True})
        except (BudgetStop, EnvironmentBlocked, ExecutionStop, SpecGap, ValueError) as exc:
            # Integrity failures are deliberately not caught/relabelled as normal stops.
            return self.journal.append("run_finished", "main:done", {
                "status": {BudgetStop: "budget_stopped", EnvironmentBlocked: "environment_blocked", ExecutionStop: "execution_stopped",
                           SpecGap: "spec_blocked", ValueError: "implementation_failure"}[type(exc)],
                "synthetic": self.policy.synthetic, "evaluation_complete": False,
                "rounds_completed": sum(e["kind"] == "cycle_finished" for e in self.journal.events),
                "error": f"{type(exc).__name__}: {exc}"})
