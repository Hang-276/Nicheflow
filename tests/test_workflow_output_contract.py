import json
from pathlib import Path
import tempfile
import unittest
import io
from contextlib import redirect_stdout

from nicheflow.datasets import Task
from nicheflow.graph import Node, WorkflowGraph, graph_messages, parse_proposal, proposal_messages
from nicheflow.ledger import Journal
from nicheflow.runtime import GraphExecutor
from nicheflow.spec import IntegrityError
from nicheflow.workflow_contracts import PRESERVE_POLICY, format_output


class ScriptedBackend:
    model_id = "offline-workflow-contract-fixture"

    def __init__(self, texts, finish_reason="stop"):
        self.texts = iter(texts)
        self.calls = []
        self.finish_reason = finish_reason

    def generate(self, messages, **params):
        self.calls.append(messages)
        return {"text": next(self.texts), "status": "ok", "finish_reason": self.finish_reason,
                "input_tokens": 1, "output_tokens": 1, "elapsed_seconds": .01,
                "accounted_usd": .001}


class WorkflowOutputContractTests(unittest.TestCase):
    def run_graph(self, texts, graph, gold, policy=PRESERVE_POLICY, finish_reason="stop"):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        journal = Journal(temp.name, {"workflow_output_policy": policy}, 20, 100)
        self.addCleanup(journal.close)
        backend = ScriptedBackend(texts, finish_reason)
        executor = GraphExecutor(journal, {"local": backend}, {"seed": 1}, workflow_output_policy=policy)
        task = Task("fixture", "math", "train", "development", "A generic fixture question.", gold,
                    "math_official", private={"solution": "PRIVATE_REFERENCE_MUST_NOT_LEAK"})
        result = executor.execute(graph, task, "case")
        return executor, journal, backend, task, result

    @staticmethod
    def format_graph():
        return WorkflowGraph((Node("a", "Generate"), Node("f", "Format", inputs=("a",))), "f")

    def test_preserves_latex_with_raw_paid_receipt_and_no_retry(self):
        executor, journal, backend, task, result = self.run_graph(
            [r"Answer: \boxed{\text{hyperbola}}", r"\boxed{hyperbola}"], self.format_graph(), r"\text{hyperbola}")
        self.assertEqual(result["evaluation"]["quality"], 1)
        self.assertEqual(result["final_output"], r"\boxed{\text{hyperbola}}")
        self.assertEqual(journal.lookup("call_finished", "case:f")["text"], r"\boxed{hyperbola}")
        self.assertTrue(journal.lookup("format_output_decision", "case:f")["applied"])
        self.assertEqual(len(backend.calls), 2)
        self.assertNotIn("PRIVATE_REFERENCE", json.dumps(backend.calls))
        self.assertEqual(executor.execute(self.format_graph(), task, "case"), result)
        self.assertEqual(len(backend.calls), 2)
        other = GraphExecutor(journal, {"local": backend}, {"seed": 1})
        with self.assertRaises(IntegrityError):
            other.execute(self.format_graph(), task, "case")

    def test_preservation_does_not_use_gold_to_fix_wrong_upstream(self):
        _, journal, backend, _, result = self.run_graph([r"\boxed{17}", r"\boxed{19}"], self.format_graph(), "19")
        self.assertEqual(result["final_output"], r"\boxed{17}")
        self.assertEqual(result["evaluation"]["quality"], 0)
        self.assertEqual(journal.audit()["attempts"], 2)
        self.assertNotIn('"gold"', json.dumps(backend.calls))

    def test_legacy_keeps_original_format_behavior(self):
        _, journal, _, _, result = self.run_graph([r"\boxed{17}", r"\boxed{19}"], self.format_graph(), "19", policy="legacy")
        self.assertEqual(result["final_output"], r"\boxed{19}")
        self.assertFalse(any(e["kind"] == "format_output_decision" for e in journal.events))

    def test_missing_conflicting_and_non_boxed_inputs_are_not_guessed(self):
        node = Node("f", "Format", inputs=("a", "b"))
        raw = r"\boxed{23}"
        for inputs, contract, reason in [
            ({"a": r"\boxed{17}", "b": r"\boxed{19}"}, r"Final in \boxed{...}", "upstream_answers_disagree"),
            ({"a": r"\boxed{17}", "b": "unfinished"}, r"Final in \boxed{...}", "upstream_answer_missing"),
            ({"a": r"\boxed{17}", "b": r"\boxed{17}"}, "Return JSON", "non_boxed_output_contract")]:
            effective, audit = format_output(node, {"output_contract": contract}, inputs, raw, PRESERVE_POLICY)
            self.assertEqual(effective, raw)
            self.assertFalse(audit["applied"])
            self.assertEqual(audit["reason"], reason)

    def test_agreed_inputs_and_nested_boxed_answer_preserved(self):
        node = Node("f", "Format", inputs=("a", "b"))
        answer = r"\frac{\sqrt{5}}{3}"
        effective, audit = format_output(node, {"output_contract": r"\boxed{...}"},
            {"a": r"\boxed{0} then \boxed{" + answer + "}", "b": r"\boxed{" + answer + "}"}, "plain result", PRESERVE_POLICY)
        self.assertEqual(effective, r"\boxed{" + answer + "}")
        self.assertTrue(audit["applied"])

    def test_review_may_correct_or_regress_without_reference_oracle(self):
        graph = WorkflowGraph((Node("a", "Generate"), Node("r", "Review&Revise", inputs=("a",)),
                               Node("f", "Format", inputs=("r",))), "f")
        for upstream, revised, expected in [("17", "19", 1), ("19", "17", 0)]:
            _, journal, backend, _, result = self.run_graph(
                [r"\boxed{" + x + "}" for x in [upstream, revised, revised]], graph, "19")
            observation = journal.lookup("review_answer_observation", "case:r")
            self.assertTrue(observation["answer_changed"])
            self.assertFalse(observation["correctness_assessed"])
            self.assertEqual(result["evaluation"]["quality"], expected)
            self.assertIn("specific error", backend.calls[1][0]["content"])
            self.assertEqual(len(backend.calls), 3)

    def test_truncation_never_gets_restored_into_success(self):
        _, journal, backend, _, result = self.run_graph([r"\boxed{19}"], self.format_graph(), "19", finish_reason="length")
        self.assertEqual(result["status"], "truncated")
        self.assertIsNone(result["final_output"])
        self.assertFalse(any(e["kind"] == "format_output_decision" for e in journal.events))
        self.assertEqual(len(backend.calls), 1)

    def test_deterministic_prompt_contract_and_actionable_validation(self):
        parent = WorkflowGraph((Node("a", "Generate"),), "a")
        payload = json.loads(proposal_messages([parent], ["local"], allowed_operators=["Generate", "Identity"])[1]["content"])
        self.assertEqual(payload["deterministic_nodes"]["prompt"], "")
        self.assertIn("never null", payload["operator_contracts"]["Identity"]["prompt"])
        candidate = {"nodes": [{"id": "a", "operator": "Generate", "model": "local"},
                               {"id": "copy", "operator": "Identity", "model": None, "inputs": ["a"], "prompt": None}], "output": "copy"}
        with self.assertRaisesRegex(ValueError, "node copy.*Identity.*string"):
            parse_proposal(json.dumps(candidate), [parent], ["local"])
        candidate["nodes"][1]["prompt"] = ""
        self.assertEqual(parse_proposal(json.dumps(candidate), [parent], ["local"]).model_calls, 1)

    def test_invalid_policy_rejected_before_calls(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 2, 10) as journal:
            with self.assertRaisesRegex(ValueError, "unknown workflow"):
                GraphExecutor(journal, {}, {"seed": 1}, workflow_output_policy="typo")
            self.assertEqual(journal.attempts, 0)

    def test_main_entry_records_opt_in_and_preserves_frozen_evaluation(self):
        from nicheflow.main_entry import run_main
        from test_main import fixture, ROOT, SyntheticPool
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()):
            args, _ = fixture(d)
            config_path = Path(args.config)
            config = json.loads(config_path.read_text())
            config["workflow_output_policy"] = PRESERVE_POLICY
            config_path.write_text(json.dumps(config))
            summary, code = run_main(args, ROOT, SyntheticPool)
            self.assertEqual(code, 0)
            self.assertTrue(summary["evaluation"]["learning_state_unchanged"])
            frozen = json.loads((Path(args.run_dir)/"config.json").read_text())
            self.assertEqual(frozen["config"]["settings"]["workflow_output_policy"], PRESERVE_POLICY)
            events = Journal.read(Path(args.run_dir)/"events.jsonl")
            self.assertTrue(any(e["kind"] == "review_answer_observation" for e in events))
