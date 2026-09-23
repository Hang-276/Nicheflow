import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from io import StringIO
from contextlib import redirect_stdout

from nicheflow.cli import main
from nicheflow.datasets import Task
from nicheflow.graph import WorkflowGraph, Node
from nicheflow.ledger import Journal
from nicheflow.policy import MinimalPolicy, bootstrap_cfg, topology_bin, neighbors
from nicheflow.reporting import report
from nicheflow.runtime import GraphExecutor
from nicheflow.spec import full_smoke_readiness, digest
from test_smoke_driver import SyntheticBackend

ROOT = Path(__file__).resolve().parents[1]


class TimedSyntheticBackend(SyntheticBackend):
    is_synthetic = True
    environment = {"model": {"id": "synthetic-policy-test"}, "synthetic": True}

    def __init__(self, *args):
        super().__init__()
        self.closed = False

    def generate(self, messages, **params):
        result = super().generate(messages, **params)
        return {**result, "elapsed_seconds": 10.}

    def close(self):
        self.closed = True


def make_policy(j, backend):
    config = json.loads((ROOT / "configs/smoke.json").read_text())
    tasks = [Task(f"synthetic/{i}", "math", "train", "development", "Compute 2-1.", "1", "rational") for i in range(3)]
    return MinimalPolicy(GraphExecutor(j, {"local": backend}, config["decode"]), config, tasks, synthetic=True)


class PolicyTests(unittest.TestCase):
    def test_actual_production_policies_complete_four_cycles_on_synthetic_backend(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {"mode": "adaptive-fixture"}, 146, 3600) as j:
            p = make_policy(j, TimedSyntheticBackend())
            result = p.driver.run()
            self.assertIn(result["status"], {"adaptive_fixture_pass", "partial_smoke_pass"})
            self.assertEqual(result["coverage"]["cycles_completed"], 4)
            self.assertEqual(p.completed_blocks, 8)
            self.assertGreater(result["coverage"]["curvature_observations"], 0)
            self.assertLessEqual(j.attempts, 146)
            self.assertEqual(sum(e.count for e in p.estimates.values()), len(p.records))
            self.assertEqual(len(p.archive.population), 100)
            self.assertTrue(all(e.cell[2] == 0 for e in p.archive.evaluations.values()))
            decisions = [e["payload"] for e in j.events if e["kind"] == "scheduler"]
            self.assertEqual(len(decisions), 4)
            for decision in decisions:
                self.assertAlmostEqual(sum(decision["allocation"].values()), 1)
                self.assertIn("head_before_update", decision["meta_linear_ucb"]["search"])
            self.assertFalse(report(d)["real_adaptive_smoke_pass"])

    def test_policy_replay_after_rebin_and_meta_updates_matches_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as base:
            with Journal(Path(base, "complete"), {}, 146, 3600) as j:
                make_policy(j, TimedSyntheticBackend()).driver.run()
                expected = [(e["kind"], e["id"], digest(e["payload"])) for e in j.events]
            for kind in ["archive_rebin", "curvature_observation", "feedback", "scheduler", "cycle_finished"]:
                with self.subTest(kind=kind):
                    directory = Path(base, kind)
                    with Journal(directory, {}, 146, 3600) as j:
                        append, counts = j.append, [0]
                        def crash(k, id, payload):
                            result = append(k, id, payload)
                            if k == kind:
                                counts[0] += 1
                                if counts[0] == (2 if kind in {"archive_rebin", "curvature_observation", "feedback"} else 1):
                                    raise KeyboardInterrupt("synthetic crash")
                            return result
                        j.append = crash
                        with self.assertRaises(KeyboardInterrupt): make_policy(j, TimedSyntheticBackend()).driver.run()
                        initial_calls = j.attempts
                    with Journal(directory, {}, 146, 3600) as j:
                        backend = TimedSyntheticBackend()
                        make_policy(j, backend).driver.run()
                        self.assertEqual(initial_calls + len(backend.calls), j.attempts)
                        self.assertEqual(expected, [(e["kind"], e["id"], digest(e["payload"])) for e in j.events])

    def test_curvature_changes_allocation_and_signed_meta_feedback_is_not_clipped(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 146, 3600) as j:
            p = make_policy(j, TimedSyntheticBackend())
            context = {"total_blocks": 8, "vendi": 1, "hv": 0, "epoch": 2}
            low_h = p.allocate(context, {"h": .632}, 8)["search"]
            high_h = p.allocate(context, {"h": 1.}, 8)["search"]
            self.assertGreater(high_h, low_h)
            self.assertAlmostEqual(high_h, .5)
            p.meta["search"].update(p.meta_x, -.2)
            self.assertLess(p.meta["search"].predict(p.meta_x, 0)["mean"], 0)

    def test_cost_transform_rewards_lower_cost_and_keeps_units_separate(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 146, 3600) as j:
            p = make_policy(j, TimedSyntheticBackend())
            self.assertGreater(p.cost_reward(.001), p.cost_reward(.01))
            self.assertEqual(p.cost_reward(0), 1)

    def test_cfg_and_descriptor_order_preserve_independent_primitive(self):
        a, b = bootstrap_cfg(7), bootstrap_cfg(7)
        self.assertEqual([g.version for g in a], [g.version for g in b])
        independent = next(g for g in a if g.topology_primitive == "NGT-Independent")
        self.assertEqual(topology_bin(independent, .67), 3)
        self.assertEqual(len(list(neighbors((0, 0, 0)))), 3)
        self.assertEqual(len(list(neighbors((2, 2, 2)))), 6)

    def test_smoke_config_is_ready_without_claiming_exact_reproduction_or_real_pass(self):
        config = json.loads((ROOT / "configs/smoke.json").read_text())
        ready = full_smoke_readiness(config)
        self.assertTrue(ready["ready"])
        self.assertFalse(ready["exact_source_reproduction"])
        self.assertFalse(ready["real_smoke_passed"])
        config["max_calls"] = 147
        self.assertFalse(full_smoke_readiness(config)["ready"])

    def test_real_cli_path_reaches_driver_and_completed_resume_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as d:
            config = json.loads((ROOT / "configs/smoke.json").read_text())
            config["model_path"] = d
            path = Path(d, "config.json")
            path.write_text(json.dumps(config))
            run = Path(d, "run")
            args = ["run", "--mode", "full-smoke", "--config", str(path), "--run-dir", str(run)]
            backend = TimedSyntheticBackend()
            with patch("nicheflow.backend_process.LocalWorker", return_value=backend) as create, redirect_stdout(StringIO()):
                self.assertEqual(main(args), 0)
                create.assert_called_once()
                self.assertGreater(len(backend.calls), 0)
                self.assertTrue(backend.closed)
            summary = report(run)
            self.assertFalse(summary["real_adaptive_smoke_pass"])
            self.assertEqual(summary["completion"]["coverage"]["cycles_completed"], 4)
            original = Path(run, "events.jsonl").read_bytes()
            with patch("nicheflow.backend_process.LocalWorker") as create, redirect_stdout(StringIO()):
                self.assertEqual(main(args), 0)
                create.assert_not_called()
            self.assertEqual(original, Path(run, "events.jsonl").read_bytes())

    def test_operator_and_node_limits_reject_graph_instead_of_accepting_error(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 10, 60) as j:
            class InvalidBackend(TimedSyntheticBackend):
                def generate(self, messages, **params):
                    return {"status": "ok", "finish_reason": "stop", "text": json.dumps({
                        "nodes": [{"id": "a", "operator": "Generate"},
                                  {"id": "b", "operator": "Format", "inputs": ["a"]}], "output": "b"}),
                        "input_tokens": 1, "output_tokens": 1, "elapsed_seconds": 1}
            executor = GraphExecutor(j, {"local": InvalidBackend()}, {"seed": 1})
            result = executor.generate_candidate(bootstrap_cfg(7)[:1], "too_many", max_nodes=1)
            self.assertEqual(result["status"], "invalid")
            self.assertIsNone(result["graph"])
            executor.allowed_operators = ["Generate"]
            result = executor.generate_candidate(bootstrap_cfg(7)[:1], "wrong_operator", max_nodes=3)
            self.assertEqual(result["status"], "invalid")
            self.assertIsNone(result["graph"])


if __name__ == "__main__":
    unittest.main()
