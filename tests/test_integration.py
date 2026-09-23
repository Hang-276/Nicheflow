import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from nicheflow.archive import Archive, Evaluation, cell_id, cost_bin, cost_boundaries, heterogeneity_bin, occupancy_density
from nicheflow.datasets import REGISTRY, Task, normalize
from nicheflow.fixtures import run_fixture
from nicheflow.graph import Node, WorkflowGraph, graph_messages, parse_proposal, probe_graphs
from nicheflow.ledger import Journal
from nicheflow.metrics import logdet_subset, hypervolume, vendi_score, pass_at_k
from nicheflow.proxy import MLPProxy
from nicheflow.router import RidgeHead, BiGLinUCB
from nicheflow.runtime import GraphExecutor
from nicheflow.scheduler import CurvatureWindow, MarginalObservation, CASBMS, curvature_ratio
from nicheflow.search import QualityEstimate, select_niches
from nicheflow.scoring import evaluate
from nicheflow.spec import BudgetStop, EnvironmentBlocked, IntegrityError, SpecGap, full_smoke_readiness
from nicheflow_probe.backends import ReplayBackend, decoding_kwargs


class NumericalTests(unittest.TestCase):
    def test_logdet_is_not_identity_plus_kernel(self):
        self.assertAlmostEqual(logdet_subset([[2, .5], [.5, 3]], [0, 1]), math.log(5.75))
        self.assertEqual(logdet_subset([[1]], []), 0)

    def test_singular_kernel_is_not_repaired(self):
        with self.assertRaises(ValueError):
            logdet_subset([[1, 1], [1, 1]], [0, 1])

    def test_vendi_and_hv_analytic_examples(self):
        self.assertAlmostEqual(vendi_score(np.eye(4)), 4)
        self.assertAlmostEqual(vendi_score(np.ones((4, 4))), 1)
        self.assertEqual(hypervolume([(1, 2), (2, 1), (.5, .5)], (0, 0)), 3)
        self.assertEqual(hypervolume([(2, 3)], (1, 1)), 2)

    def test_pass_k_is_sample_estimator(self):
        self.assertAlmostEqual(pass_at_k(4, 1, 2), .5)
        with self.assertRaises(ValueError): pass_at_k(2, 1, 3)

    def test_dual_ridge_matches_independent_batch_solution(self):
        x = np.array([[1, 2], [2, -1], [.3, .7]])
        y = np.array([.2, .8, .4])
        h = RidgeHead(2, .5)
        for a, b in zip(x, y): h.update(a, b)
        expected = np.linalg.solve(.5 * np.eye(2) + x.T @ x, x.T @ y)
        self.assertAlmostEqual(h.predict([1, 1], 0)["mean"], sum(expected))
        self.assertGreater(h.oful_beta(noise_bound=1, parameter_bound=1, delta=.05), 0)

    def test_router_growth_leaves_old_state_and_no_duplicate_feedback(self):
        r = BiGLinUCB(2, 1)
        r.add_arm("old")
        r.update("old", [1., 0.], .8, .2, "receipt1")
        before = r.state()["arms"]["old"]
        r.add_arm("new")
        self.assertEqual(before, r.state()["arms"]["old"])
        self.assertEqual(r.arms["new"][0].count, 0)
        self.assertFalse(r.update("old", [1., 0.], .8, .2, "receipt1"))
        with self.assertRaises(IntegrityError): r.update("old", [1., 0.], .7, .2, "receipt1")

    def test_chebyshev_not_weighted_sum_and_explicit_cost(self):
        r = BiGLinUCB(1, 1)
        for a, q, c in [("unbalanced", .99, .01), ("balanced", .4, .4)]:
            r.add_arm(a); r.update(a, [1], q, c, a)
        decision = r.choose({a: [1] for a in r.arms}, weights=[1, 1], beta_quality=0, beta_cost=0,
                            cost_reward_definition="test utility, both maximize")
        self.assertEqual(decision["arm"], "balanced")
        with self.assertRaises(SpecGap):
            r.choose({a: [1] for a in r.arms}, weights=[1, 1], beta_quality=0, beta_cost=0, cost_reward_definition=None)

    def test_occupancy_is_not_communication_density(self):
        neighborhood = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
        self.assertAlmostEqual(occupancy_density((0, 0, 0), {(0, 1, 0)}, lambda _: neighborhood), 1/3)

    def test_100_cells_cost_ties_and_heterogeneity(self):
        self.assertEqual(len({cell_id((a, b, c)) for a in range(5) for b in range(5) for c in range(4)}), 100)
        self.assertEqual(cost_bin(0, cost_boundaries([0, 0, 0])), 0)
        self.assertEqual([heterogeneity_bin(n, 10) for n in [0, 3, 7, 8]], [0, 1, 2, 3])

    def test_search_manual_greedy_and_filter(self):
        args = dict(ids=["a", "b", "c"], populations={"a": 1, "b": 1, "c": 1},
                    densities={"a": 0, "b": 0, "c": .9}, estimates={a: QualityEstimate(1, q) for a, q in [("a", .8), ("b", .7), ("c", 1)]},
                    kernel=np.eye(3), t=2, k=2, population_threshold=1, rho0=1, alpha=1, diversity_weight=1, exploration=0)
        result = select_niches(**args)
        self.assertEqual(result["selected"], ["a", "b"])
        self.assertEqual(result["pool"], ["a", "b"])
        args["densities"]["b"] = 1
        with self.assertRaises(SpecGap): select_niches(**args)

    def test_no_hidden_zero_visit_ucb_default(self):
        with self.assertRaises(SpecGap): QualityEstimate().ucb(1, 1)

    def test_curvature_confidence_fallback_and_no_clipping(self):
        w = CurvatureWindow(3, .05, .1, .5, "f")
        self.assertTrue(w.estimate(1)["fallback"])
        obs = MarginalObservation("m", "i", ("j",), .8, 1, "f", ("r",))
        w.observe(obs, {"r"})
        e = w.estimate(1)
        self.assertAlmostEqual(e["gamma"], .2 + .1 * math.sqrt(math.log(20)))
        self.assertFalse(e["fallback"])
        self.assertFalse(w.observe(obs, {"r"}))
        with self.assertRaises(IntegrityError): CurvatureWindow(3, .05, .1, .5, "f").observe(obs, set())
        w.observe(MarginalObservation("m2", "k", (), 0, 1, "f", ("r2",)), {"r2"})
        with self.assertRaises(SpecGap): w.estimate(2)

    def test_curvature_zero_denominator_and_undefined_allocation(self):
        w = CurvatureWindow(3, .05, .1, .5, "f")
        with self.assertRaises(ValueError):
            w.observe(MarginalObservation("m", "i", (), .1, 0, "f", ("r",)), {"r"})
        with self.assertRaises(SpecGap): CASBMS(w).decide(1, {}, 10)
        self.assertEqual(curvature_ratio(0), 1)
        self.assertAlmostEqual(curvature_ratio(1), 1-math.exp(-1))

    def test_proxy_learns_signal_from_paid_development_only(self):
        p = MLPProxy(1, 4, 1)
        x, y = [[-1], [-.8], [.8], [1]], [0, 0, 1, 1]
        with self.assertRaises(IntegrityError):
            p.fit(x, y, receipt_ids=list("abcd"), roles=["evaluation"]*4, paid_receipts=list("abcd"), steps=1, learning_rate=.1)
        p.fit(x, y, receipt_ids=list("abcd"), roles=["development"]*4, paid_receipts=list("abcd"), steps=200, learning_rate=.1)
        self.assertLess(p.predict([[-1]])[0], .2)
        self.assertGreater(p.predict([[1]])[0], .8)


class GraphAndDataTests(unittest.TestCase):
    def test_cycle_unknown_operator_dead_node_rejected(self):
        for graph in [WorkflowGraph((Node("a", "Generate", inputs=("b",)), Node("b", "Generate", inputs=("a",))), "a"),
                      WorkflowGraph((Node("a", "Shell"),), "a"),
                      WorkflowGraph((Node("a", "Generate"), Node("b", "Generate")), "a")]:
            with self.assertRaises(ValueError): graph.validate()

    def test_independent_branches_only_merge_after_execution(self):
        graph = WorkflowGraph((Node("a", "Generate"), Node("b", "Generate"), Node("c", "Ensemble", inputs=("a", "b"))), "c", topology_primitive="NGT-Independent")
        graph.validate()
        payload = graph_messages(graph.nodes[1], {"question": "Q"}, {"a": "SECRET_BRANCH"})
        self.assertNotIn("SECRET_BRANCH", json.dumps(payload))

    def test_task_gold_private_fields_never_reach_prompt(self):
        task = Task("id", "math", "train", "development", "Q", "SECRET_GOLD", "rational", private={"solution": "SECRET_SOLUTION", "level": "SECRET_LEVEL"})
        content = json.dumps(graph_messages(probe_graphs()[0].nodes[0], task.model_input(), {}))
        self.assertNotIn("SECRET", content)

    def test_original_probe_prompts_are_preserved(self):
        self.assertEqual([g.model_calls for g in probe_graphs()], [1, 1, 2])

    def test_rejected_model_graph_not_silently_unwrapped_or_pruned(self):
        bad = {"dag": {"nodes": [{"id": "a", "operator": "Generate"}], "output": "a"}}
        with self.assertRaises(ValueError): parse_proposal(json.dumps(bad), probe_graphs()[:1], ["local"])
        dead = {"nodes": [{"id": "a", "operator": "Generate"}, {"id": "b", "operator": "Generate"}], "output": "b"}
        with self.assertRaises(ValueError): parse_proposal(json.dumps(dead), probe_graphs()[:1], ["local"])

    def test_ten_registrations_and_eval_only_benchmarks(self):
        self.assertEqual(len(REGISTRY), 10)
        with self.assertRaises(IntegrityError):
            normalize("gpqa", {}, 1, role="development")
        with self.assertRaises(IntegrityError):
            normalize("math", {}, 1, split="test", role="development")

    def test_mbpp_tests_separated(self):
        task = normalize("mbpp", {"task_id": 601, "text": "Q", "code": "PRIVATE_CODE", "test_list": ["public", "HIDDEN_TEST"]}, 601)
        self.assertEqual(task.public_tests, ("public",))
        self.assertNotIn("HIDDEN_TEST", json.dumps(task.model_input()))

    def test_gpqa_choice_shuffle_preserves_label(self):
        row = {"Question": "q", "Correct Answer": "yes", "Incorrect Answer 1": "a", "Incorrect Answer 2": "b", "Incorrect Answer 3": "c"}
        a, b = normalize("gpqa", row, 7), normalize("gpqa", row, 7)
        self.assertEqual(a, b)
        self.assertEqual(a.options[ord(a.gold)-65], "yes")

    def test_math_full_reference_normalization(self):
        task = normalize("math", {"problem": "q", "solution": r"\boxed{\sqrt{2}}"}, 1)
        self.assertEqual(evaluate(task, r"\boxed{\sqrt2}")["quality"], 1)
        self.assertEqual(evaluate(task, r"\boxed{\sqrt3}")["quality"], 0)

    def test_choice_parser_wont_guess_letter_in_reasoning(self):
        task = normalize("mmlu_pro", {"question": "q", "options": ["a", "b"], "answer": "A"}, 1)
        self.assertEqual(evaluate(task, "A is mentioned but no final answer")["quality"], 0)
        self.assertEqual(evaluate(task, "ANSWER: A")["quality"], 1)

    def test_hotpot_joint_and_answer_metrics(self):
        task = normalize("hotpotqa", {"question": "Q", "answer": "yes", "context": [], "supporting_facts": [["Page", 0]]}, 1)
        result = evaluate(task, '{"answer":"yes","sp":[["Page",0]]}')
        self.assertEqual(result["metrics"]["joint_em"], 1)
        self.assertEqual(evaluate(task, '{"answer":"no","sp":[]}')["quality"], 0)

    def test_drop_numbers_and_multiple_spans(self):
        task = normalize("drop", {"question": "Q", "passage": "P", "answers_spans": {"spans": ["10", "blue"]}}, 1)
        self.assertEqual(evaluate(task, '{"answers":["blue","10.0"]}')["quality"], 1)
        self.assertEqual(evaluate(task, '{"answers":["blue","20"]}')["quality"], 0)

    def test_missing_code_and_planning_environment_are_not_wrong_answers(self):
        code = normalize("mbpp", {"text": "Q", "code": "x=1", "test_list": ["assert x==1"]}, 601)
        with self.assertRaises(EnvironmentBlocked): evaluate(code, "x=1")
        travel = normalize("travelplanner", {"query": "Q"}, 1)
        with self.assertRaises(EnvironmentBlocked): evaluate(travel, "plan")

    def test_zero_temperature_selects_greedy(self):
        self.assertEqual(decoding_kwargs(0, .8, 10), {"do_sample": False, "max_new_tokens": 10})
        self.assertTrue(decoding_kwargs(.7, .8, 10)["do_sample"])

    def test_gaia_numeric_list_and_missing_test_label(self):
        task = normalize("gaia", {"Question": "Q", "Final answer": "10", "file_name": ""}, 1)
        self.assertEqual(evaluate(task, "$10")["quality"], 1)
        task = normalize("gaia", {"Question": "Q", "Final answer": "blue, 10", "file_name": ""}, 2)
        self.assertEqual(evaluate(task, "blue; 10")["quality"], 1)
        task = normalize("gaia", {"Question": "Q", "file_name": ""}, 3, split="test", role="evaluation")
        with self.assertRaises(EnvironmentBlocked): evaluate(task, "answer")

    def test_subgroup_indirect_information_leak_rejected(self):
        graph = WorkflowGraph((Node("a", "Generate", subgroup="one"),
            Node("bridge", "Identity", inputs=("a",), model=None),
            Node("b", "Generate", inputs=("bridge",), subgroup="two"),
            Node("final", "Ensemble", inputs=("a", "b"))), "final", topology_primitive="Subgroup")
        with self.assertRaises(ValueError): graph.validate()

    def test_code_node_roundtrip_keeps_source_and_temperature(self):
        graph = WorkflowGraph((Node("a", "Generate", temperature=0),
            Node("b", "Code", inputs=("a",), model=None, source="print('{}')", signature=("json", "json"))), "b")
        graph.validate()
        self.assertEqual(graph.version, WorkflowGraph.from_dict(json.loads(json.dumps(graph.definition()))).version)


class DurabilityTests(unittest.TestCase):
    def test_duplicate_call_reuses_result_and_counts_once(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 2, 60) as j:
            b = ReplayBackend()
            a = j.call("c", b, [], {"seed": 1})
            self.assertEqual(a, j.call("c", b, [], {"seed": 1}))
            self.assertEqual(len(b.calls), 1)
            self.assertEqual(j.audit()["attempts"], 1)
            with self.assertRaises(IntegrityError): j.call("c", b, [], {"seed": 2})

    def test_inflight_call_blocks_resume_without_reissue(self):
        with tempfile.TemporaryDirectory() as d:
            with Journal(d, {}, 2, 60) as j:
                j.append("call_started", "c", {"messages": [], "params": {"seed": 1}, "model": "replay"})
            with Journal(d, {}, 2, 60) as j:
                b = ReplayBackend()
                with self.assertRaises(IntegrityError): j.call("c", b, [], {"seed": 1})
                self.assertEqual(len(b.calls), 0)
                self.assertEqual(j.audit()["unknown_calls"], ["c"])

    def test_configuration_change_and_second_writer_blocked(self):
        with tempfile.TemporaryDirectory() as d:
            with Journal(d, {"mode": "fixture"}, 2, 60) as j:
                with self.assertRaises(IntegrityError): Journal(d, {"mode": "fixture"}, 2, 60)
            with self.assertRaises(IntegrityError): Journal(d, {"mode": "changed"}, 2, 60)

    def test_operation_budget_and_wall_clock_survive_resume(self):
        with tempfile.TemporaryDirectory() as d:
            with Journal(d, {}, 2, 10, clock=lambda: 100) as j:
                with self.assertRaises(BudgetStop): j.reserve(3)
            with Journal(d, {}, 2, 10, clock=lambda: 111) as j:
                with self.assertRaises(BudgetStop): j.reserve(1)

    def test_failed_calls_consume_attempt_and_unknown_tokens_remain_unknown(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 1, 60) as j:
            b = ReplayBackend({1: {"status": "error", "output_tokens": None}})
            j.call("c", b, [], {"seed": 1})
            self.assertEqual(j.audit()["unknown_token_calls"], 1)
            with self.assertRaises(BudgetStop): j.call("d", b, [], {"seed": 1})

    def test_tampered_journal_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with Journal(d, {}, 1, 60) as j: j.append("x", "1", {"value": 1})
            path = Path(d) / "events.jsonl"
            path.write_text(path.read_text().replace('"value": 1', '"value": 2'))
            with self.assertRaises(IntegrityError): Journal.read(path)

    def test_fixture_resume_adds_no_calls_or_events(self):
        with tempfile.TemporaryDirectory() as d:
            with Journal(d, {}, 4, 60) as j:
                run_fixture(j)
                count = len(j.events)
            with Journal(d, {}, 4, 60) as j:
                run_fixture(j)
                self.assertEqual(count, len(j.events))
                self.assertEqual(j.attempts, 2)

    def test_new_archive_entry_is_not_forced_elite(self):
        a = Archive()
        e = Evaluation("w", (0,0,0), .5, .2, ("r",), "development", "v1")
        result = a.evaluate(e, paid_receipts={"r"}, acceptance=lambda *_: False, policy_source="test")
        self.assertFalse(result["accepted"])
        self.assertEqual(a.elites, {})

    def test_code_task_fails_preflight_without_gpu_spend(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 4, 60) as j:
            task = normalize("mbpp", {"text": "Q", "code": "x=1", "test_list": ["assert x==1"]}, 601)
            e = GraphExecutor(j, {"local": ReplayBackend()}, {"seed": 1})
            with self.assertRaises(EnvironmentBlocked): e.execute(probe_graphs()[0], task, "x")
            self.assertEqual(j.attempts, 0)

    def test_full_smoke_gate_cannot_be_enabled_by_config(self):
        self.assertFalse(full_smoke_readiness()["ready"])
        self.assertFalse(full_smoke_readiness({"resolved": True})["ready"])
        self.assertIn("scheduler_allocation", full_smoke_readiness()["engineering_completions"])

    def test_generation_resume_rejects_different_parent(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 4, 60) as j:
            e = GraphExecutor(j, {"local": ReplayBackend()}, {"seed": 1})
            e.generate_candidate(probe_graphs()[:1], "g")
            with self.assertRaises(IntegrityError): e.generate_candidate(probe_graphs()[1:2], "g")
            self.assertEqual(j.attempts, 1)

    def test_renaming_curvature_evidence_does_not_add_samples(self):
        w = CurvatureWindow(5, .05, .1, .5, "f")
        w.observe(MarginalObservation("a", "i", (), .8, 1, "f", ("r",)), {"r"})
        with self.assertRaises(IntegrityError):
            w.observe(MarginalObservation("b", "i", (), .8, 1, "f", ("r",)), {"r"})
        self.assertEqual(len(w.window), 1)


if __name__ == "__main__": unittest.main()
