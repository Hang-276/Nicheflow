import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nicheflow_probe import report  # noqa: E402
from nicheflow_probe.backends import ReplayBackend, TransformersChatBackend  # noqa: E402
from nicheflow_probe.runner import Runner, build_plan, load_tasks, node_seed  # noqa: E402

SECRET_SOLUTION = "REFERENCE-SOLUTION-MUST-NOT-LEAK"
SECRET_GOLD = "987654"


def fake_task(index: int) -> dict:
    return {
        "id": f"algebra/train/{index}",
        "subject": "algebra",
        "level": "Level 3" if index % 2 else "Level 4",
        "question": f"Compute {index} + 1.",
        "gold": SECRET_GOLD if index == 0 else str(index + 1),
        "reference_solution": SECRET_SOLUTION,
    }


class SilentBackend(ReplayBackend):
    """Any call is a test failure: used to prove a resume does not re-execute."""

    def generate(self, *args, **kwargs):
        raise AssertionError("a completed execution was recomputed")


class FailingBackend(ReplayBackend):
    def generate(self, messages, seed, max_new_tokens=1536, temperature=0.7, top_p=0.8):
        self.calls.append({"messages": messages, "seed": seed})
        return {"text": "", "input_tokens": 12, "output_tokens": 0, "elapsed_seconds": 0.5,
                "finish_reason": "error", "status": "error", "error": "CUDA out of memory",
                "transient": False, "context_limit": False, "seed": seed}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "run"
        self.tasks = [fake_task(i) for i in range(12)]
        self.repeat_ids = [self.tasks[3]["id"], self.tasks[0]["id"], self.tasks[10]["id"],
                           self.tasks[1]["id"]]
        self.plan = build_plan(self.tasks, self.repeat_ids)
        self.config = {"model_id": "replay", "max_new_tokens": 1536, "temperature": 0.7,
                       "top_p": 0.8, "generation_seed": 73129}

    def tearDown(self):
        self.tmp.cleanup()

    def runner(self, backend):
        return Runner(self.run_dir, backend, self.tasks, self.config, self.plan, "sha", "unit")

    def test_plan_is_frozen_at_48_slots_and_64_calls(self):
        self.assertEqual(len(self.plan), 48)
        self.assertEqual(sum(len(entry["node_ids"]) for entry in self.plan), 64)
        self.assertEqual(sum(1 for e in self.plan if e["stage"] == "A"), 18)
        self.assertEqual(sum(1 for e in self.plan if e["stage"] == "B"), 18)
        self.assertEqual(sum(1 for e in self.plan if e["stage"] == "C"), 12)
        self.assertEqual(len({e["execution_id"] for e in self.plan}), 48)
        self.assertEqual(sorted({e["task_id"] for e in self.plan if e["repeat_index"] == 1}),
                         sorted(self.repeat_ids))

    def test_seed_rule_matches_the_task_document(self):
        import hashlib
        key = "73129|algebra/train/26|revise|0|draft"
        expected = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31)
        self.assertEqual(node_seed("algebra/train/26", "revise", 0, "draft"), expected)

    def test_revise_runs_two_independent_calls_and_pays_for_both(self):
        backend = ReplayBackend()
        runner = self.runner(backend)
        entry = next(e for e in self.plan if e["workflow_id"] == "revise" and e["repeat_index"] == 0)
        record = runner._run_execution(entry)
        calls = [json.loads(line) for line in (self.run_dir / "calls.jsonl").read_text().splitlines()]
        self.assertEqual(record["model_calls"], 2)
        self.assertEqual([call["node_id"] for call in calls], ["draft", "answer"])
        self.assertNotEqual(calls[0]["seed"], calls[1]["seed"])
        draft_prompt = calls[1]["messages"][1]["content"]
        self.assertIn("Draft solution:", draft_prompt)
        self.assertIn(calls[0]["raw_output"], draft_prompt)
        self.assertNotIn("Draft solution:", calls[0]["messages"][1]["content"])
        self.assertEqual({call["execution_id"] for call in calls}, {record["execution_id"]})
        self.assertEqual(record["total_output_tokens"], sum(call["output_tokens"] for call in calls))
        self.assertEqual(record["total_node_seconds"],
                         sum(call["elapsed_seconds"] for call in calls))

    def test_reference_solution_and_gold_never_reach_the_model(self):
        backend = ReplayBackend()
        runner = self.runner(backend)
        runner.run(stages=("A",))
        calls = [json.loads(line) for line in (self.run_dir / "calls.jsonl").read_text().splitlines()]
        self.assertTrue(calls)
        for call in calls:
            blob = json.dumps(call["messages"], ensure_ascii=False)
            self.assertNotIn(SECRET_SOLUTION, blob)
            self.assertNotIn(SECRET_GOLD, blob)
            self.assertNotIn("reference_solution", blob)
            self.assertNotIn("gold", blob)
            self.assertIn("Problem:", blob)

    def test_resume_does_not_recount_completed_executions(self):
        runner = self.runner(ReplayBackend())
        first = runner.run(stages=("A",))
        self.assertEqual(first["completed_slots"], 18)
        again = self.runner(SilentBackend()).run(stages=("A",))
        self.assertEqual(again["completed_slots"], 18)
        records = (self.run_dir / "executions.jsonl").read_text().splitlines()
        self.assertEqual(len(records), 18)

    def test_backend_failure_is_logged_and_is_not_a_wrong_answer(self):
        runner = self.runner(FailingBackend())
        result = runner.run(stages=("A",))
        # Three consecutive execution failures stop the batch instead of burning the budget.
        self.assertEqual(result["stopped_for"], "three_consecutive_execution_failures")
        executions = [json.loads(line) for line in
                      (self.run_dir / "executions.jsonl").read_text().splitlines()]
        self.assertEqual(len(executions), 3)
        self.assertTrue(all(e["status"] == "failed" for e in executions))
        self.assertTrue(all(e["final_output"] is None for e in executions))
        self.assertTrue(all(e["error"] for e in executions))
        self.assertTrue(all(not e["parsed"] for e in executions))
        calls = [json.loads(line) for line in (self.run_dir / "calls.jsonl").read_text().splitlines()]
        self.assertTrue(all(call["status"] == "error" and call["error"] for call in calls))
        summary = summarise(self.run_dir, self.tasks, self.plan)
        self.assertEqual(sum(s["successful_executions"] for s in summary["per_workflow"].values()), 0)
        self.assertEqual(sum(s["correct_over_planned_slots"]
                             for s in summary["per_workflow"].values()), 0)
        self.assertEqual(summary["per_workflow"]["direct"]["planned_slots_main"], 12)
        self.assertEqual(summary["per_workflow"]["direct"]["planned_slots_total"], 16)
        self.assertEqual(sum(s["planned_slots_total"]
                             for s in summary["per_workflow"].values()), 48)

    def test_resume_refuses_a_different_plan(self):
        self.runner(ReplayBackend()).initialise()
        other = build_plan(self.tasks, self.repeat_ids)[:-1]
        broken = Runner(self.run_dir, ReplayBackend(), self.tasks, self.config, other, "sha", "unit")
        with self.assertRaises(RuntimeError):
            broken.initialise()


class DataChecks(unittest.TestCase):
    def test_shipped_data_is_intact_and_scoreable(self):
        tasks = load_tasks(ROOT / "data" / "tasks.jsonl")
        self.assertEqual(len(tasks), 12)
        self.assertTrue(all(task["gold"] for task in tasks))

    def test_transient_detection(self):
        self.assertTrue(TransformersChatBackend.is_transient("Connection reset by peer"))
        self.assertFalse(TransformersChatBackend.is_transient("CUDA out of memory"))


def summarise(run_dir, tasks, plan):
    return report.build_summary(run_dir, tasks, plan)


if __name__ == "__main__":
    unittest.main()
