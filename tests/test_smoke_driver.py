"""Synthetic integration tests, including interruption at durable event boundaries.

The small policy below deliberately uses fixture descriptors and allocations.
It is NOT a source-backed NicheFlow policy and cannot unlock full-smoke.
"""
import json
import math
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
from dataclasses import replace

from nicheflow.archive import Archive, Evaluation, cell_id, pareto_admission
from nicheflow.cli import main
from nicheflow.datasets import Task
from nicheflow.graph import Node, WorkflowGraph
from nicheflow.ledger import Journal
from nicheflow.loops import SearchService, DeploymentService
from nicheflow.reporting import report
from nicheflow.router import BiGLinUCB
from nicheflow.runtime import GraphExecutor
from nicheflow.scheduler import CASBMS, CurvatureWindow
from nicheflow.search import QualityEstimate, rbf_ucb_kernel, select_niches
from nicheflow.smoke import SmokeBindings, SmokeDriver
from nicheflow.spec import IntegrityError


class SyntheticBackend:
    model_id = "synthetic-driver-test"

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def generate(self, messages, **params):
        self.calls.append(messages)
        if self.fail:
            return {"status": "error", "error": "synthetic failure", "finish_reason": "error", "text": "",
                    "input_tokens": 0, "output_tokens": 0}
        if "Propose one executable workflow DAG" in messages[0]["content"]:
            parent = json.loads(messages[1]["content"])["parents"][0]
            node = parent["nodes"][0]
            candidate = {"nodes": [{"id": "a", "operator": "Generate", "model": "local",
                                    "prompt": node["prompt"] + f" mutated {params['seed']}"}], "output": "a"}
            text = json.dumps(candidate)
        else:
            text = r"\boxed{1}"
        return {"status": "ok", "finish_reason": "stop", "text": text, "input_tokens": 2, "output_tokens": 3}


def fixture_driver(journal, backend, cycles=4, reject_new=False):
    executor = GraphExecutor(journal, {"local": backend}, {"seed": 7, "max_new_tokens": 64})
    archive, router = Archive(), BiGLinUCB(1, 1)
    seeds = [WorkflowGraph((Node("a", "Generate", f"branch{i}"),), "a") for i in range(2)]
    tasks = [Task(f"synthetic/{i}", "math", "train", "development", "Compute 2-1.", "1", "rational") for i in range(2)]
    graphs = {g.version: g for g in seeds}
    estimates, seen_feedback, feedback_count = {}, set(), [0]

    def describe(graph, usd):
        # Synthetic cells exercise plumbing, not production descriptor semantics.
        return (int(graph.nodes[0].prompt.startswith("branch1")), 0, 0), "synthetic-cell-policy"

    def total_usd(receipts):
        return sum(.02 if id.startswith("bootstrap:") else .01 for id in receipts)

    search = SearchService(executor, archive, router, describe=describe,
        accept_elite=(lambda new, old: old is None) if reject_new else pareto_admission,
        proxy_screen=lambda graph: {"selected": True}, measure_usd=total_usd,
        policy_source="synthetic-fixture", batch_screen=lambda gs: [
            {"candidate": g.version, "selected": True} for g in gs])
    deployment = DeploymentService(executor, router,
        features=lambda task, graph: [1 + len(graph.parents) + graph.nodes[0].prompt.count("mutated")],
        weights=lambda t: [1, 1], confidence=lambda r, t: (1, 1),
        cost_reward=lambda receipts: .5, cost_definition="synthetic utility; not dollar cost")

    def observe(stage, results):
        receipts = []
        for result in results:
            receipts.extend(result.get("receipts", result.get("calls", [])))
            if result["status"] != "evaluated":
                continue
            workflow = result["workflow"]
            if "graph" in result:
                graphs[workflow] = WorkflowGraph.from_dict(result["graph"])
            key = tuple(result["receipts"])
            if key not in seen_feedback:
                seen_feedback.add(key)
                # This explicit fixture attributes feedback to the child's niche.
                niche = cell_id(tuple(result["cell"]))
                estimates.setdefault(niche, QualityEstimate()).observe(result["quality"])
        feedback_count[0] += 1
        return {"receipt_ids": receipts, "feedback_count": feedback_count[0],
                "search_counts": {str(i): e.count for i, e in estimates.items()}}

    def selection():
        ids = sorted(archive.elites)
        kernel = rbf_ucb_kernel([[i] for i in ids], [estimates[i].ucb(2, 1) for i in ids], 1)
        selected = select_niches(ids, {i: len(archive.population[i]) for i in ids}, dict.fromkeys(ids, 0),
            estimates, kernel, t=2, k=2, population_threshold=1, rho0=1, alpha=.1,
            diversity_weight=.1, exploration=1)
        return selected, {i: graphs[archive.elites[i].workflow] for i in ids}

    estimator = CurvatureWindow(4, .05, .1, .8, "synthetic fixture utility")
    scheduler = CASBMS(estimator, allocation_policy=lambda c, e, b: {"search": .5, "deploy": .5},
                      policy_source="fixed synthetic fixture; not CA-SBMS")
    bindings = SmokeBindings(selection, observe,
        lambda: {"epoch": router.epoch, "remaining_research_budget": 100 - feedback_count[0]},
        lambda decision, context: ["search", "deploy"], "synthetic-fixture", True)
    driver = SmokeDriver(executor, search, deployment, scheduler, bindings, seeds=seeds,
                         search_tasks=tasks, deployment_tasks=tasks, cycles=cycles, generation_max_nodes=1)
    return driver, estimates


class SmokeDriverTests(unittest.TestCase):
    def test_four_cycles_update_search_and_route_real_candidate_receipts(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {"mode": "adaptive-fixture"}, 60, 60) as j:
            backend = SyntheticBackend()
            driver, estimates = fixture_driver(j, backend)
            result = driver.run()
            self.assertEqual(result["status"], "adaptive_fixture_pass")
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["coverage"]["cycles_completed"], 4)
            self.assertTrue(result["coverage"]["new_elite_route_feedback"])
            self.assertEqual([e.count for e in estimates.values()], [5, 5])
            self.assertEqual(j.attempts, 32)  # 4 bootstrap + 4*(2 generation + 4 evaluation + 1 route)
            self.assertEqual(j.attempts, len(backend.calls))
            initial = [e["payload"] for e in j.events if e["kind"] == "archive" and e["id"].startswith("bootstrap")]
            self.assertEqual(len(initial), 2)
            self.assertAlmostEqual(driver.search.archive.population[0][driver.seeds[0].version].usd_cost, .02)

    def test_finished_resume_has_no_new_calls_or_events_even_after_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            with Journal(d, {}, 60, 10, clock=lambda: 100) as j:
                fixture_driver(j, SyntheticBackend())[0].run()
                before = Path(d, "events.jsonl").read_bytes()
            with Journal(d, {}, 60, 10, clock=lambda: 200) as j:
                backend = SyntheticBackend()
                result = fixture_driver(j, backend)[0].run()
                self.assertEqual(result["status"], "adaptive_fixture_pass")
                self.assertEqual(backend.calls, [])
                self.assertEqual(before, Path(d, "events.jsonl").read_bytes())

    def test_interruptions_replay_same_decisions_without_duplicate_calls(self):
        # A crash can occur after durable execution but before an in-memory update,
        # or between feedback, decision and cycle completion. Cover each boundary.
        for kind in ["call_finished", "execution", "archive", "feedback", "scheduler", "generation", "router", "cycle_finished"]:
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                with Journal(d, {}, 60, 60) as j:
                    original_append = j.append
                    interrupted = [False]
                    def crash_once(event_kind, id, payload):
                        result = original_append(event_kind, id, payload)
                        if event_kind == kind and not interrupted[0]:
                            interrupted[0] = True
                            raise KeyboardInterrupt("synthetic durable-boundary crash")
                        return result
                    j.append = crash_once
                    with self.assertRaises(KeyboardInterrupt):
                        fixture_driver(j, SyntheticBackend())[0].run()
                    first_calls = j.attempts
                with Journal(d, {}, 60, 60) as j:
                    backend = SyntheticBackend()
                    result = fixture_driver(j, backend)[0].run()
                    self.assertEqual(result["status"], "adaptive_fixture_pass")
                    self.assertEqual(first_calls + len(backend.calls), 32)
                    self.assertEqual(j.attempts, 32)

    def test_budget_stop_does_not_start_unaffordable_evaluation_batch(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 7, 60) as j:
            result = fixture_driver(j, SyntheticBackend())[0].run()
            self.assertEqual(result["status"], "budget_stopped")
            self.assertEqual(j.attempts, 4)
            self.assertEqual(result["coverage"]["cycles_completed"], 0)

    def test_rejected_new_elites_remain_uncovered_not_forced(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 60, 60) as j:
            result = fixture_driver(j, SyntheticBackend(), reject_new=True)[0].run()
            self.assertEqual(result["status"], "partial_smoke_pass")
            self.assertFalse(result["coverage"]["new_elite_route_feedback"])

    def test_unknown_call_does_not_retry(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 60, 60) as j:
            j.append("call_started", "unknown", {"messages": [], "params": {}, "model": "synthetic-driver-test"})
            backend = SyntheticBackend()
            with self.assertRaises(IntegrityError): fixture_driver(j, backend)[0].run()
            self.assertEqual(backend.calls, [])

    def test_three_execution_failures_stop_before_fourth(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 60, 60) as j:
            backend = SyntheticBackend(fail=True)
            driver, _ = fixture_driver(j, backend)
            driver.seeds += [WorkflowGraph((Node("a", "Generate", f"extra{i}"),), "a") for i in range(2)]
            result = driver.run()
            self.assertEqual(result["status"], "execution_stopped")
            self.assertEqual(j.attempts, 3)

    def test_batch_screen_sees_all_candidates_before_any_evaluation(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 60, 60) as j:
            driver, _ = fixture_driver(j, SyntheticBackend(), cycles=1)
            seen = []
            def screen(graphs):
                seen.append((len(graphs), len([e for e in j.events if e["kind"] == "generation"])))
                return [{"candidate": g.version, "selected": i == 0} for i, g in enumerate(graphs)]
            driver.search.batch_screen = screen
            driver.run()
            self.assertEqual(seen, [(2, 2)])
            self.assertEqual(len([e for e in j.events if e["kind"] == "proxy" and e["payload"]["selected"]]), 1)

    def test_different_driver_contract_refuses_resume(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 60, 60) as j:
            fixture_driver(j, SyntheticBackend())[0].run()
            with self.assertRaises(IntegrityError): fixture_driver(j, SyntheticBackend(), cycles=2)[0].run()

    def test_identical_candidate_is_not_paid_or_observed_twice(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 60, 60) as j:
            backend = SyntheticBackend()
            original = backend.generate
            def fixed_generation(messages, **params):
                return original(messages, **{**params, "seed": 1})
            backend.generate = fixed_generation
            driver, estimates = fixture_driver(j, backend, reject_new=True)
            result = driver.run()
            self.assertEqual(result["status"], "partial_smoke_pass")
            self.assertEqual(j.attempts, 20)  # repeated proposals paid; identical evaluations skipped
            self.assertEqual([e.count for e in estimates.values()], [2, 2])

    def test_synthetic_completion_never_reports_real_smoke_pass(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {"mode": "adaptive-fixture"}, 60, 60) as j:
            fixture_driver(j, SyntheticBackend())[0].run()
            summary = report(d)
            self.assertFalse(summary["real_adaptive_smoke_pass"])
            self.assertEqual(summary["completion"]["coverage"]["cycles_completed"], 4)

    def test_missing_run_status_does_not_create_evidence_or_infer_zero_calls(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d, "missing")
            result = report(missing)
            self.assertEqual(result["status"], "run_not_created")
            self.assertIsNone(result["calls_attempted"])
            self.assertFalse(missing.exists())

    def test_unreleased_full_smoke_keeps_preflight_evidence_without_backend_load(self):
        with tempfile.TemporaryDirectory() as d, patch("nicheflow.backend_process.LocalWorker") as backend:
            config = Path(d, "smoke.json")
            config.write_text('{"profile":"unreleased"}')
            directory = Path(d, "run")
            with redirect_stdout(StringIO()):
                code = main(["run", "--mode", "full-smoke", "--config", str(config), "--run-dir", str(directory)])
            self.assertEqual(code, 2)
            backend.assert_not_called()
            result = report(directory)
            self.assertEqual(result["status"], "configuration_blocked")
            self.assertFalse(result["run_started"])
            self.assertEqual(result["calls_attempted"], 0)
            self.assertFalse((directory / "events.jsonl").exists())


class SourcePrimitiveTests(unittest.TestCase):
    def test_pareto_dominance_retains_ties_and_incomparables(self):
        old = Evaluation("old", (0, 0, 0), .6, .2, ("paid",), "development", "v")
        self.assertTrue(pareto_admission(old, None))
        self.assertFalse(pareto_admission(replace(old, workflow="tie"), old))
        self.assertFalse(pareto_admission(replace(old, quality=.8, usd_cost=.3), old))
        self.assertTrue(pareto_admission(replace(old, quality=.8), old))
        self.assertTrue(pareto_admission(replace(old, usd_cost=.1), old))

    def test_rbf_quality_kernel_matches_two_point_formula(self):
        kernel = rbf_ucb_kernel([[0], [2]], [2, 3], .5)
        self.assertEqual(kernel[0, 0], 4)
        self.assertEqual(kernel[1, 1], 9)
        self.assertAlmostEqual(kernel[0, 1], 6 * math.exp(-2))


if __name__ == "__main__":
    unittest.main()
