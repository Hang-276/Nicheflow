"""Explicit synthetic integration fixture: never a real smoke result."""
from .archive import Archive, Evaluation
from .graph import WorkflowGraph, Node, probe_graphs
from .metrics import hypervolume, vendi_score
from .router import BiGLinUCB
from .runtime import GraphExecutor
from .scheduler import CurvatureWindow, MarginalObservation, CASBMS
from .search import QualityEstimate, select_niches
from .datasets import Task
from nicheflow_probe.backends import ReplayBackend


def run_fixture(journal):
    candidate = WorkflowGraph((Node("answer", "Generate", "Solve the problem. Give a boxed answer."),), "answer")
    import json
    replay = ReplayBackend({1: {"text": json.dumps(candidate.definition()), "status": "ok"}})
    executor = GraphExecutor(journal, {"local": replay}, {"seed": 71, "temperature": .7, "top_p": .8, "max_new_tokens": 64})
    generation = executor.generate_candidate(probe_graphs()[:1], "fixture:proposal")
    graph = WorkflowGraph.from_dict(generation["graph"])
    task = Task("fixture/1", "math", "train", "development", "Compute 2-1.", "1", "rational")
    execution = executor.execute(graph, task, "fixture:execution")
    receipt_ids = tuple(execution["calls"])
    archive = Archive()
    ev = Evaluation(graph.version, (0, 0, 0), execution["evaluation"]["quality"], 0.01,
                    receipt_ids, "development", "synthetic_fixture_v1")
    archive_event = archive.evaluate(ev, paid_receipts=receipt_ids, acceptance=lambda new, old: old is None or new.quality > old.quality,
                                     policy_source="fixture_only_not_source_defined")
    journal.append("archive", "fixture:archive", archive_event)
    router = BiGLinUCB(2, 1)
    router.add_arm(graph.version)
    choice = router.choose({graph.version: [1., .2]}, weights=[1., 1.], beta_quality=1, beta_cost=1,
                           cost_reward_definition="synthetic fixture utility, not raw dollar cost")
    router.update(graph.version, [1., .2], ev.quality, .5, receipt_ids[0])
    journal.append("router", "fixture:router", {"decision": choice, "state": router.state()})
    estimates = {str(i): QualityEstimate(2, float(i) / 2) for i in range(3)}
    search = select_niches(list(estimates), dict.fromkeys(estimates, 1), dict.fromkeys(estimates, .1), estimates,
                           [[1, .1, .2], [.1, 1, .3], [.2, .3, 1]], t=2, k=2,
                           population_threshold=1, rho0=1, alpha=.1, diversity_weight=.5, exploration=1)
    journal.append("search", "fixture:search", search)
    estimator = CurvatureWindow(5, .05, .1, .8, "synthetic fixture utility")
    estimator.observe(MarginalObservation("m1", "i", ("j",), .8, 1., "synthetic fixture utility", receipt_ids), receipt_ids)
    scheduler = CASBMS(estimator, allocation_policy=lambda context, estimate, budget: {"search": .5, "deploy": .5},
                       policy_source="fixed fixture policy: NOT CA-SBMS allocation")
    journal.append("scheduler", "fixture:scheduler", scheduler.decide(2, {"epoch": 1}, 10))
    journal.append("metrics", "fixture:metrics", {"vendi": vendi_score([[1, 0], [0, 1]]),
                                                  "hv": hypervolume([(1, 2), (2, 1)], (0, 0))})
    journal.append("run_finished", "fixture:done", {"status": "integration_fixture_pass", "synthetic": True,
                   "not_covered": ["real candidate generation", "source-defined archive policy", "source-defined adaptive scheduler"]})
    return journal.audit()
