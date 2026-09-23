"""Bounded adaptive driver; research policies are explicit, separately sourced inputs.

The driver can be tested with synthetic policies without declaring those policies
to be the advisor's algorithm. It never supplies an allocation formula itself.
Fresh service instances plus journal replay rebuild learning state after a crash.
"""
from dataclasses import dataclass
from .graph import WorkflowGraph
from .spec import BudgetStop, EnvironmentBlocked, ExecutionStop, IntegrityError, SpecGap, digest


@dataclass(frozen=True)
class SmokeBindings:
    # Each callback is instantiated with fresh state for run/resume. It must be
    # deterministic from frozen configuration and the feedback already supplied.
    selection: object       # () -> (CS-DPP-NI trace, niche -> parent graph)
    observe: object         # (stage, outcomes) -> auditable feedback; updates policy state
    context: object         # () -> public meta-context, never held-out labels
    operations: object      # (scheduler decision, context) -> ordered Search/Deploy blocks
    source: str
    synthetic: bool

    def validate(self):
        if not all(callable(x) for x in (self.selection, self.observe, self.context, self.operations)) or not self.source:
            raise SpecGap("selection, feedback, public context and budget-block mapping must be bound")


class SmokeDriver:
    def __init__(self, executor, search, deployment, scheduler, bindings, *, seeds,
                 search_tasks, deployment_tasks, cycles, generation_max_nodes):
        bindings.validate()
        if not seeds or not search_tasks or not deployment_tasks or not 1 <= cycles <= 6:
            raise ValueError("nonempty seeds/tasks and 1..6 bounded cycles required")
        if not bindings.synthetic and cycles < 4:
            raise ValueError("real smoke requires at least four scheduled cycles")
        if not 1 <= generation_max_nodes <= 12:
            raise ValueError("generation node bound must be 1..12")
        if search.executor is not executor or deployment.executor is not executor or search.router is not deployment.router:
            raise IntegrityError("search and deployment must share the executor, budget and router")
        for task in [*search_tasks, *deployment_tasks]:
            task.require_learning()
        for graph in seeds:
            graph.validate(executor.backends)
        self.executor, self.journal = executor, executor.journal
        self.search, self.deployment, self.scheduler, self.bindings = search, deployment, scheduler, bindings
        self.seeds, self.search_tasks, self.deployment_tasks = list(seeds), list(search_tasks), list(deployment_tasks)
        self.cycles, self.max_nodes = cycles, generation_max_nodes
        self.workflows = {g.version: g for g in seeds}
        self.executor.failure_limit = 3

    def _feedback(self, id, stage, results):
        feedback = self.bindings.observe(stage, results)
        if not isinstance(feedback, dict):
            raise ValueError("feedback callback must return an audit record")
        # Call IDs are evidence references, not additional charges. Reject invented
        # observations even in fixtures (which have recorded synthetic receipts).
        evidence = set(feedback.get("receipt_ids", []))
        paid = {e["id"] for e in self.journal.events if e["kind"] == "call_finished"}
        if not evidence <= paid:
            raise IntegrityError("scheduler/search feedback references an unrecorded call")
        self.journal.append("feedback", id, {"stage": stage, "source": self.bindings.source, **feedback})

    def _coverage(self):
        events = self.journal.events
        generated = {e["payload"]["version"] for e in events
                     if e["kind"] == "generation" and e["payload"]["status"] == "valid"}
        admitted = {e["payload"]["candidate"] for e in events if e["kind"] == "archive"
                    and e["payload"]["accepted"] and e["payload"]["candidate"] in generated}
        admitted |= {e["id"] for e in events if e["kind"] == "router_arm" and e["id"] in generated}
        routed = {e["payload"]["decision"]["arm"] for e in events if e["kind"] == "router"}
        return {
            "candidate_generation": bool(generated),
            "combinatorial_selection": any(e["kind"] == "search" and len(e["payload"]["selected"]) > 1 for e in events),
            "candidate_evaluation": any(e["kind"] == "execution" and e["payload"]["status"] == "ok"
                                        and e["payload"]["workflow"] in generated for e in events),
            "candidate_archive_decision": any(e["kind"] == "archive" and e["payload"]["candidate"] in generated for e in events),
            "new_elite_route_feedback": bool(admitted & routed),
            "budget_actions_executed": any(e["kind"] == "cycle_finished" and e["payload"]["operations"] for e in events),
            "cycles_completed": sum(e["kind"] == "cycle_finished" for e in events),
            "curvature_observations": sum(e["kind"] == "curvature_observation" and e["payload"]["status"] == "observed" for e in events),
            "curvature_nonfallback_decisions": sum(e["kind"] == "scheduler" and not e["payload"]["estimate"]["fallback"]
                                                  and not e["payload"]["estimate"].get("capped_at_one", False) for e in events),
        }

    def run(self):
        contract = {"seeds": [g.version for g in self.seeds], "search_tasks": [t.record() for t in self.search_tasks],
                    "deployment_tasks": [t.record() for t in self.deployment_tasks], "cycles": self.cycles,
                    "generation_max_nodes": self.max_nodes, "policy_source": self.bindings.source,
                    "synthetic": self.bindings.synthetic}
        self.journal.append("driver_contract", "smoke", {"digest": digest(contract), **contract})
        if self.journal.lookup("run_finished", "smoke:done") is not None:
            return self.journal.lookup("run_finished", "smoke:done")
        if self.journal.audit()["unknown_calls"]:
            raise IntegrityError("unknown call outcome; no automated replay")
        try:
            if self.journal.lookup("bootstrap_started", "smoke") is None:
                bound = sum(g.model_calls for g in self.seeds) * len(self.search_tasks)
                self.journal.reserve(bound)
                self.journal.append("bootstrap_started", "smoke", {"max_calls": bound})
            for i, graph in enumerate(self.seeds):
                id = f"bootstrap:{i}"
                result = self.search.evaluate(graph, self.search_tasks, id)
                self._feedback(id, "bootstrap", [result])
            if not self.deployment.router.arms:
                raise ExecutionStop("bootstrap produced no admissible evaluated workflow")
            deployment_index = 0
            for cycle in range(1, self.cycles + 1):
                id = f"cycle:{cycle}"
                historical = self.journal.lookup("cycle_context", id)
                context = self.bindings.context()
                # Preserve the original decision-time engineering cap on replay.
                # It is separate from the policy's research-budget unit.
                if historical is None:
                    self.journal.reserve(0)
                    context = {**context, "remaining_call_cap": self.journal.max_calls - self.journal.attempts}
                    self.journal.append("cycle_context", id, context)
                else:
                    expected = {**context, "remaining_call_cap": historical["remaining_call_cap"]}
                    if digest(expected) != digest(historical):
                        raise IntegrityError("replayed meta-context differs from the original decision")
                    context = historical
                if "remaining_research_budget" not in context:
                    raise SpecGap("policy must identify its own research budget, independently of call cap")
                decision = self.scheduler.decide(cycle, context, context["remaining_research_budget"])
                self.journal.append("scheduler", id, decision)
                operations = self.bindings.operations(decision, context)
                if not isinstance(operations, list) or not 1 <= len(operations) <= 32 or any(x not in {"search", "deploy"} for x in operations):
                    raise ValueError("budget mapping must give 1..32 explicit search/deploy blocks")
                self.journal.append("operation_plan", id, {"operations": operations, "source": self.bindings.source})
                for step, kind in enumerate(operations):
                    op = f"{id}:{step}:{kind}"
                    if kind == "search":
                        selection, parents = self.bindings.selection()
                        results = self.search.run(op, selection, parents, self.search_tasks, max_nodes=self.max_nodes)
                        for result in results:
                            if "graph" in result and result["graph"] is not None:
                                graph = WorkflowGraph.from_dict(result["graph"])
                                self.workflows[graph.version] = graph
                    else:
                        task = self.deployment_tasks[deployment_index % len(self.deployment_tasks)]
                        deployment_index += 1
                        results = [self.deployment.run(op, task, self.workflows, deployment_index)]
                    self._feedback(op, kind, results)
                    self.executor.check_failures()
                self.journal.append("cycle_finished", id, {"operations": operations})
            coverage = self._coverage()
            complete = all(coverage[key] for key in ("candidate_generation", "combinatorial_selection", "candidate_evaluation",
                "candidate_archive_decision", "new_elite_route_feedback", "budget_actions_executed"))
            status = ("adaptive_fixture_pass" if self.bindings.synthetic else "full_smoke_pass") if complete else "partial_smoke_pass"
            return self.journal.append("run_finished", "smoke:done", {"status": status,
                "synthetic": self.bindings.synthetic, "coverage": coverage, "policy_source": self.bindings.source})
        except (BudgetStop, EnvironmentBlocked, ExecutionStop, SpecGap, ValueError) as exc:
            status = {BudgetStop: "budget_stopped", EnvironmentBlocked: "environment_blocked",
                      ExecutionStop: "execution_stopped", SpecGap: "spec_blocked", ValueError: "implementation_failure"}[type(exc)]
            return self.journal.append("run_finished", "smoke:done", {"status": status,
                "synthetic": self.bindings.synthetic, "coverage": self._coverage(),
                "error": f"{type(exc).__name__}: {exc}"})
