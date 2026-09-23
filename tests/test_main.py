import copy
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from nicheflow.cli import main
from nicheflow.datasets import Task
from nicheflow.ledger import Journal
from nicheflow.main_config import load_protocol, derive_limits, validate_main
from nicheflow.main_entry import run_main, plan
from nicheflow.main_models import MainAPI, api_charge
from nicheflow.main_policy import main_seeds, stability
from nicheflow.spec import IntegrityError, digest, file_hash

ROOT = Path(__file__).resolve().parents[1]


class SyntheticMainBackend:
    is_synthetic = True

    def __init__(self, name, profile):
        self.model_id, self.profile = name, profile
        self.environment = {"model": {"id": name}, "synthetic": True}
        self.calls = 0

    def generate(self, messages, **params):
        self.calls += 1
        if "Propose one executable" in messages[0]["content"]:
            text = json.dumps({"nodes": [{"id": "a", "operator": "Generate", "model": "strong",
                                         "prompt": f"Solve carefully. Variant {params['seed']}."}], "output": "a"})
        else:
            text = "\\boxed{1}"
        cost = .0001 + (params["seed"] % 13) * .000001
        return {"status": "ok", "finish_reason": "stop", "text": text, "input_tokens": 20,
                "output_tokens": 10, "elapsed_seconds": cost * 3600, "accounted_usd": cost,
                "local_accounting_usd": cost if self.profile["kind"] == "local" else 0.,
                "api_charge": {"usd": cost} if self.profile["kind"] == "api" else None}


class SyntheticPool:
    is_synthetic = True
    instances = []

    def __init__(self, config):
        self.backends = {k: SyntheticMainBackend(k, v) for k, v in config["models"].items()}
        self.closed = False
        self.instances.append(self)

    def close(self):
        self.closed = True


def fixture(directory, rounds=1, deferred=False):
    directory = Path(directory)
    config = json.loads((ROOT / "configs/main_math.json").read_text())
    config.update(rounds=rounds, queries_per_deploy_block=3)
    config["evaluation"].update(status="deferred" if deferred else "configured", samples_per_task=2, pass_k=[1, 2])
    config["evaluation"].pop("benchmark", None)
    config["evaluation"].pop("subset_manifest", None)
    config["evaluation"].pop("subset_manifest_sha256", None)
    parts = {}
    for role in ("development", "calibration", "evaluation"):
        path = directory / f"{role}.jsonl"
        tasks = [Task(f"fixture/{role}/{i}", "math", "test" if role == "evaluation" else "train", role,
                      f"Compute 2 minus 1. Exercise {role} {i}.", "1", "rational") for i in range(3)]
        path.write_text("".join(json.dumps(t.record()) + "\n" for t in tasks))
        parts[role] = {"path": path.name, "count": len(tasks), "sha256": file_hash(path)}
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"partitions": parts, "synthetic": True}))
    config["data_manifest"], config["data_manifest_sha256"] = str(manifest), file_hash(manifest)
    path = directory / "main.json"
    path.write_text(json.dumps(config))
    args = SimpleNamespace(config=str(path), rounds=None, run_dir=str(directory / "run"))
    return args, config


class MainTests(unittest.TestCase):
    def test_integer_niche_keys_and_mutable_snapshots_remain_hash_valid(self):
        with tempfile.TemporaryDirectory() as d, Journal(d, {}, 1, 60) as journal:
            value = {"scores": {3: .3, 20: .2}, "history": [1]}
            journal.append("snapshot", "a", value)
            value["history"].append(2)
            read = Journal.read(Path(d) / "events.jsonl")
            self.assertEqual(read[0]["payload"]["scores"], {"3": .3, "20": .2})
            self.assertEqual(journal.lookup("snapshot", "a")["history"], [1])

    def test_api_cap_stops_before_external_call_and_keeps_local_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            args, config = fixture(d, deferred=True)
            config["limits"].update(api_usd_fixed=1e-8, api_usd_per_round=1e-8)
            Path(args.config).write_text(json.dumps(config))
            summary, code = run_main(args, ROOT, SyntheticPool)
            self.assertEqual(code, 5)
            self.assertEqual(summary["status"], "budget_stopped")
            self.assertEqual(SyntheticPool.instances[-1].backends["strong"].calls, 0)
            self.assertEqual(summary["api_tariff_usd"], 0.)
            instances = len(SyntheticPool.instances)
            repeated, repeated_code = run_main(args, ROOT, SyntheticPool)
            self.assertEqual(repeated_code, 5)
            self.assertEqual(repeated["status"], "budget_stopped")
            self.assertEqual(len(SyntheticPool.instances), instances)

    def test_deferred_evaluation_does_not_hide_learning_execution_error(self):
        class OneFailurePool(SyntheticPool):
            def __init__(self, config):
                super().__init__(config)
                self.failed = False
                for backend in self.backends.values():
                    generate = backend.generate
                    def once(messages, _generate=generate, **params):
                        result = _generate(messages, **params)
                        if not self.failed and "Exercise calibration" in json.dumps(messages):
                            self.failed = True
                            result.update(status="error", finish_reason="error", error="synthetic known failure")
                        return result
                    backend.generate = once
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d, deferred=True)
            result, code = run_main(args, ROOT, OneFailurePool)
            self.assertEqual(code, 6)
            self.assertEqual(result["rounds_completed"], 1)
            self.assertEqual(result["status"], "main_completed_with_execution_errors")
            self.assertEqual(result["evaluation"]["status"], "deferred")
            self.assertEqual(len(result["execution_errors"]), 1)

    def test_unknown_finished_api_charge_cannot_be_replayed_as_free(self):
        class UnknownCostPool(SyntheticPool):
            def __init__(self, config):
                super().__init__(config)
                api = self.backends["strong"]
                generate = api.generate
                def unknown(*args, **kwargs):
                    result = generate(*args, **kwargs)
                    result["accounted_usd"] = None
                    return result
                api.generate = unknown
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d)
            with self.assertRaises(IntegrityError):
                run_main(args, ROOT, UnknownCostPool)
            count = len(SyntheticPool.instances)
            with self.assertRaises(IntegrityError):
                run_main(args, ROOT, SyntheticPool)
            self.assertEqual(len(SyntheticPool.instances), count)

    def test_readonly_plan_does_not_construct_or_call_backends(self):
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d, deferred=True)
            with patch("nicheflow.main_entry.ModelPool", side_effect=AssertionError("must not load")):
                result = plan(args.config, ROOT, 1)
            self.assertEqual(result["model_calls"], 0)
            self.assertFalse(result["model_loaded"])
            self.assertEqual(result["evaluation_status"], "deferred")

    def test_one_round_runs_both_layers_freezes_and_evaluates_without_learning_leak(self):
        with tempfile.TemporaryDirectory() as d:
            args, config = fixture(d)
            summary, code = run_main(args, ROOT, SyntheticPool)
            self.assertEqual(code, 0)
            self.assertEqual(summary["status"], "main_run_complete")
            self.assertEqual(summary["rounds_completed"], 1)
            self.assertTrue(summary["synthetic"])
            self.assertTrue(summary["numerical_health"]["all_finite_spd"])
            self.assertEqual(summary["evaluation"]["status"], "complete")
            self.assertTrue(summary["evaluation"]["learning_state_unchanged"])
            events = Journal.read(Path(args.run_dir) / "events.jsonl")
            boundary = next(e["seq"] for e in events if e["kind"] == "evaluation_started")
            self.assertFalse(any(e["kind"] in {"feedback", "router", "archive", "scheduler"} for e in events[boundary:]))
            self.assertEqual(sum(e["kind"] == "evaluation_route" for e in events), 3 * 3 * 2)
            self.assertTrue(any(e["kind"] == "search" for e in events))
            self.assertEqual(sum(e["kind"] == "router" for e in events), 3)
            self.assertTrue(SyntheticPool.instances[-1].closed)
            self.assertTrue(all(w["pass_at_k"]["2"] == 1 for w in summary["evaluation"]["weight_results"]))

    def test_round_override_changes_only_horizon_and_derived_limits_and_exceeds_smoke_cap(self):
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d, deferred=True)
            a, tasks, pa = load_protocol(args.config, ROOT, 1)
            b, _, pb = load_protocol(args.config, ROOT, 7)
            self.assertEqual(pa, pb)
            self.assertEqual({k:v for k,v in a.items() if k != "rounds"}, {k:v for k,v in b.items() if k != "rounds"})
            la, lb = [derive_limits(c, tasks, main_seeds(c)) for c in (a,b)]
            self.assertEqual(la["bootstrap_calls"], lb["bootstrap_calls"])
            self.assertEqual(la["evaluation_calls"], lb["evaluation_calls"])
            self.assertEqual(lb["max_calls"] - la["max_calls"], 6 * a["policy"]["blocks_per_cycle"] * la["block_calls"])
            args.rounds = 7
            summary, code = run_main(args, ROOT, SyntheticPool)
            self.assertEqual((code, summary["rounds_completed"]), (0, 7))
            self.assertEqual(summary["status"], "learning_complete_evaluation_pending")
            self.assertEqual(summary["evaluation"], {"status": "deferred"})

    def test_completed_resume_uses_zero_backends_and_preserves_journal(self):
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d)
            run_main(args, ROOT, SyntheticPool)
            path = Path(args.run_dir) / "events.jsonl"
            original = path.read_bytes()
            class NeverLoad(SyntheticPool):
                def __init__(self, _):
                    raise AssertionError("completed resume loaded a backend")
            summary, code = run_main(args, ROOT, NeverLoad)
            self.assertEqual(code, 0)
            self.assertEqual(path.read_bytes(), original)

    def test_resume_at_learning_and_evaluation_boundaries_is_identical_and_does_not_repeat_calls(self):
        for kind in ("resource_reservation", "call_finished", "feedback", "state_snapshot", "learning_finished", "evaluation_route", "evaluation_finished"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                args, _ = fixture(d)
                append = Journal.append
                crashed = [False]
                def interrupt(j, k, id, payload):
                    result = append(j, k, id, payload)
                    if k == kind and not crashed[0]:
                        crashed[0] = True
                        raise KeyboardInterrupt("simulated process crash")
                    return result
                with patch.object(Journal, "append", interrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        run_main(args, ROOT, SyntheticPool)
                old = Journal.read(Path(args.run_dir) / "events.jsonl")
                existing = sum(e["kind"] == "call_started" for e in old)
                summary, code = run_main(args, ROOT, SyntheticPool)
                self.assertEqual(code, 0)
                new = sum(b.calls for b in SyntheticPool.instances[-1].backends.values())
                self.assertEqual(summary["calls_attempted"], existing + new)
                ids = [(e["kind"], e["id"]) for e in Journal.read(Path(args.run_dir) / "events.jsonl")]
                self.assertEqual(len(ids), len(set(ids)))

    def test_unknown_started_call_stops_before_backend_and_never_retries(self):
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d)
            append = Journal.append
            def crash(j, kind, id, payload):
                result = append(j, kind, id, payload)
                if kind == "call_started":
                    raise KeyboardInterrupt()
                return result
            with patch.object(Journal, "append", crash):
                with self.assertRaises(KeyboardInterrupt):
                    run_main(args, ROOT, SyntheticPool)
            count = len(SyntheticPool.instances)
            with self.assertRaises(IntegrityError):
                run_main(args, ROOT, SyntheticPool)
            self.assertEqual(len(SyntheticPool.instances), count)

    def test_changed_data_and_changed_rounds_refuse_old_run_resume(self):
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d)
            run_main(args, ROOT, SyntheticPool)
            args.rounds = 2
            with self.assertRaises(IntegrityError):
                run_main(args, ROOT, SyntheticPool)
            Path(d, "development.jsonl").write_text("{}\n")
            with self.assertRaises(IntegrityError):
                load_protocol(args.config, ROOT)

    def test_duplicate_public_query_cannot_cross_splits_with_a_renamed_id(self):
        with tempfile.TemporaryDirectory() as d:
            args, config = fixture(d)
            dev = json.loads(Path(d, "development.jsonl").read_text().splitlines()[0])
            cal = Path(d, "calibration.jsonl")
            rows = [json.loads(l) for l in cal.read_text().splitlines()]
            rows[0]["question"] = dev["question"]
            cal.write_text("".join(json.dumps(r) + "\n" for r in rows))
            manifest = Path(config["data_manifest"])
            m = json.loads(manifest.read_text()); m["partitions"]["calibration"]["sha256"] = file_hash(cal)
            manifest.write_text(json.dumps(m)); config["data_manifest_sha256"] = file_hash(manifest)
            Path(args.config).write_text(json.dumps(config))
            with self.assertRaises(IntegrityError):
                load_protocol(args.config, ROOT)

    def test_parameter_health_detects_non_spd_and_reports_delta_without_claiming_convergence(self):
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d)
            run_main(args, ROOT, SyntheticPool)
            state = json.loads(Path(args.run_dir, "trained_state.json").read_text())["state"]
            self.assertTrue(stability(state)["numerically_healthy"])
            bad = copy.deepcopy(state); bad["meta"]["search"]["v"][0][0] = -100
            result = stability(bad, state)
            self.assertFalse(result["numerically_healthy"])
            self.assertFalse(result["statistical_convergence_established"])

    def test_main_cli_uses_actual_main_driver(self):
        with tempfile.TemporaryDirectory() as d:
            args, _ = fixture(d, deferred=True)
            with patch("nicheflow.main_entry.ModelPool", SyntheticPool), patch("sys.stdout", io.StringIO()):
                code = main(["run", "--mode", "main", "--config", args.config, "--rounds", "1", "--run-dir", args.run_dir])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(Path(args.run_dir, "summary.json").read_text())["mode"], "main")


class ProviderTests(unittest.TestCase):
    def test_deepseek_peak_and_cache_accounting(self):
        p = json.loads((ROOT / "configs/main_math.json").read_text())["models"]["strong"]
        # Monday UTC 02:00 is peak; Sunday is off-peak.
        peak = datetime(2026, 9, 21, 2, tzinfo=timezone.utc).timestamp()
        off = datetime(2026, 9, 20, 2, tzinfo=timezone.utc).timestamp()
        usage = {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 900, "completion_tokens": 50}
        a, b = [api_charge(usage, p["pricing"], t) for t in (peak, off)]
        self.assertAlmostEqual(a["usd"], (100*.3 + 900*.006 + 50*1.2)/1e6)
        self.assertAlmostEqual(a["usd"], b["usd"] * 2)
        self.assertIsNone(api_charge({}, p["pricing"], peak))

    def test_api_payload_omits_unsupported_seed_and_top_p_and_records_actual_usage(self):
        p = json.loads((ROOT / "configs/main_math.json").read_text())["models"]["strong"]
        body = {"model": "deepseek-flash", "id": "fixture", "created": 1789900000,
                "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 50, "prompt_cache_hit_tokens": 10, "completion_tokens": 7}}
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "not-a-real-key"}), patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(body).encode())) as http:
            result = MainAPI(p).generate([{"role": "user", "content": "hello"}], seed=9, max_new_tokens=20, temperature=.7, top_p=.8)
        sent = json.loads(http.call_args.args[0].data)
        self.assertNotIn("seed", sent); self.assertNotIn("top_p", sent)
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertFalse(result["provider_seed_supported"])
        self.assertIsNotNone(result["accounted_usd"])
        self.assertEqual(result["usage"], body["usage"])

    def test_api_timeout_has_unknown_charge_and_no_retry(self):
        p = json.loads((ROOT / "configs/main_math.json").read_text())["models"]["strong"]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "not-a-real-key"}), patch("urllib.request.urlopen", side_effect=TimeoutError()) as http:
            result = MainAPI(p).generate([{"role": "user", "content": "hello"}], seed=9, max_new_tokens=20, temperature=.7, top_p=.8)
        self.assertEqual(http.call_count, 1)
        self.assertIsNone(result["accounted_usd"])
        self.assertTrue(result["billing_outcome_unknown"])


if __name__ == "__main__":
    unittest.main()
