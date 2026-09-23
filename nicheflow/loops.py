"""Search/deployment orchestration with required, explicit research-policy dependencies.

The explicit minimal completion lives in policy.py; these services do not hide
research choices inside execution or accounting code.
"""
from .archive import Evaluation
from .graph import WorkflowGraph
from .spec import SpecGap, IntegrityError


class SearchService:
    def __init__(self, executor, archive, router, *, describe, accept_elite, proxy_screen,
                 measure_usd, policy_source, batch_screen=None):
        if not all([describe, accept_elite, proxy_screen, measure_usd, policy_source]):
            raise SpecGap("search requires descriptor, elite, proxy and cost policies")
        self.executor, self.archive, self.router = executor, archive, router
        self.describe, self.accept, self.screen, self.cost = describe, accept_elite, proxy_screen, measure_usd
        self.source = policy_source
        self.batch_screen = batch_screen
        self.max_evaluations = None
        self.parent_feedback = None

    def evaluate(self, graph, tasks, operation_id):
        """Shared bootstrap/candidate evaluation, with per-query measured cost."""
        if not tasks:
            raise ValueError("evaluation tasks cannot be empty")
        for task in tasks:
            task.require_learning()
        journal = self.executor.journal
        ids = [f"{operation_id}:task:{j}" for j in range(len(tasks))]
        remaining = sum(self.executor.remaining_calls(graph, id) for id in ids)
        if remaining:
            journal.reserve(remaining)
        results = []
        for task, id in zip(tasks, ids):
            result = self.executor.execute(graph, task, id)
            results.append(result)
            if not self.executor.assessment_complete(result):
                return {"status": "evaluation_incomplete", "workflow": graph.version,
                        "execution_ids": ids[:len(results)], "receipts": [c for r in results for c in r["calls"]]}
        receipts = tuple(c for r in results for c in r["calls"])
        quality = sum(self.executor.assessment_quality(r) for r in results) / len(results)
        # measure_usd totals the actual calls; the descriptor is USD per query.
        total_usd = self.cost(receipts)
        per_query_usd = total_usd / len(results)
        cell, descriptor_version = self.describe(graph, per_query_usd)
        evaluated = Evaluation(graph.version, cell, quality, per_query_usd, receipts,
                               "development", descriptor_version)
        accepted = self.archive.evaluate(evaluated, paid_receipts={e["id"] for e in journal.events
                                          if e["kind"] == "call_finished"},
                                         acceptance=self.accept, policy_source=self.source)
        if accepted is not None:
            journal.append("archive", operation_id, accepted)
            if accepted["accepted"] and self.router.add_arm(graph.version):
                journal.append("router_arm", graph.version, {"epoch": self.router.epoch,
                                                              "source": operation_id, "receipts": receipts})
        return {"status": "evaluated", "workflow": graph.version, "quality": quality,
                "cell": list(cell), "total_usd": total_usd, "usd_per_query": per_query_usd,
                "execution_ids": ids, "accepted": accepted, "receipts": list(receipts)}

    def run(self, operation_id, selection, parents_by_niche, tasks, *, max_nodes=12):
        for task in tasks:
            task.require_learning()
        if not tasks:
            raise ValueError("search evaluation tasks cannot be empty")
        journal = self.executor.journal
        # Selection is a computed CS-DPP-NI result, not an arbitrary candidate list.
        if len(selection["selected"]) < 2 or len(selection["trace"]) != len(selection["selected"]):
            raise IntegrityError("combinatorial search requires its full decision trace")
        if journal.lookup("search", operation_id) is None:
            k = len(selection["selected"])
            evaluations = k if self.max_evaluations is None else self.max_evaluations
            journal.reserve(k + evaluations * max_nodes * len(tasks))
        journal.append("search", operation_id, selection)
        events, candidates = [], []
        for index, niche in enumerate(selection["selected"]):
            extra = {"feedback": self.parent_feedback(parents_by_niche[niche])} if self.parent_feedback else {}
            generation = self.executor.generate_candidate([parents_by_niche[niche]], f"{operation_id}:{index}",
                model=getattr(self.executor, "generation_model", "local"), max_nodes=max_nodes, **extra)
            if generation["status"] != "valid":
                events.append({**generation, "parent_niche": niche})
                continue
            graph = WorkflowGraph.from_dict(generation["graph"])
            candidates.append((index, niche, graph))
        if self.batch_screen is not None:
            screens = self.batch_screen([graph for _, _, graph in candidates]) if candidates else []
            if len(screens) != len(candidates):
                raise IntegrityError("batch proxy must return one ordered decision per candidate")
        else:
            screens = [self.screen(graph) for _, _, graph in candidates]
        for (index, niche, graph), screen in zip(candidates, screens):
            if type(screen.get("selected")) is not bool:
                raise IntegrityError("proxy decision must be a boolean")
            if "candidate" in screen and screen["candidate"] != graph.version:
                raise IntegrityError("proxy decision refers to another candidate")
            journal.append("proxy", f"{operation_id}:{index}", screen)
            if not screen["selected"]:
                events.append({"status": "proxy_rejected", "workflow": graph.version, "parent_niche": niche})
                continue
            previous = next((population[graph.version] for population in self.archive.population.values()
                             if graph.version in population), None)
            if previous is not None:
                # Generating the identical immutable candidate is paid, but it is
                # not a new independent evaluation or an additional population member.
                events.append({"status": "duplicate_candidate", "workflow": graph.version,
                               "previous_receipts": list(previous.receipt_ids), "parent_niche": niche})
                continue
            result = self.evaluate(graph, tasks, f"{operation_id}:{index}")
            events.append({**result, "parent_niche": niche, "graph": graph.definition()})
        return events


class DeploymentService:
    def __init__(self, executor, router, *, features, weights, confidence, cost_reward, cost_definition):
        if not all([features, weights, confidence, cost_reward, cost_definition]):
            raise SpecGap("deployment requires explicit feature/weight/confidence/cost policies")
        self.executor, self.router = executor, router
        self.features, self.weights, self.confidence = features, weights, confidence
        self.cost_reward, self.cost_definition = cost_reward, cost_definition
        self.fallback_threshold, self.safe_arm = None, None

    def run(self, operation_id, task, workflows, t):
        task.require_learning()
        # Only the public task object is available to the feature function.
        features = {a: self.features(task.model_input(), workflows[a]) for a in self.router.arms}
        bq, bc = self.confidence(self.router, t)
        decision = self.router.choose(features, weights=self.weights(t), beta_quality=bq, beta_cost=bc,
                                      cost_reward_definition=self.cost_definition,
                                      fallback_threshold=self.fallback_threshold, safe_arm=self.safe_arm)
        self.executor.journal.append("route_decision", operation_id, decision)
        arm = decision["arm"]
        result = self.executor.execute(workflows[arm], task, operation_id + ":execution")
        if self.executor.assessment_complete(result):
            self.router.update(arm, features[arm], self.executor.assessment_quality(result),
                               self.cost_reward(result["calls"]), operation_id + ":execution")
            self.executor.journal.append("router", operation_id, {"decision": decision, "state": self.router.state(),
                                                                   "execution_receipt": operation_id + ":execution"})
        return result
