"""Minimal engineering completions authorized 2026-09-20; NOT advisor-specified formulas.

README documents every completion. Existing 3.4 selection, dual-head routing and
curvature equations are retained. No theorem is claimed for these bindings.
"""
from __future__ import annotations
import math
import random
import numpy as np
from .archive import Archive, cell_id, cost_bin, cost_boundaries, occupancy_density, pareto_admission
from .graph import Node, WorkflowGraph
from .loops import SearchService, DeploymentService
from .metrics import hypervolume, vendi_score
from .proxy import MLPProxy
from .router import BiGLinUCB, RidgeHead
from .scheduler import CASBMS, CurvatureWindow, MarginalObservation
from .search import QualityEstimate, rbf_ucb_kernel, select_niches
from .smoke import SmokeDriver, SmokeBindings
from .spec import IntegrityError, digest

PROFILE = "minimal_engineering_v1"
SOURCE = "NicheFlow 3.4 technical plan + user-authorized minimal engineering completions 2026-09-20; see README §7"
UTILITY = "snapshot_hv_quality_vs_inverse_cost_per_niche_v1"
ALLOWED = ["Generate", "Format", "Review&Revise", "Ensemble", "Identity", "Join"]


def graph_stats(graph):
    depth = {}
    for n in graph.order():
        depth[n.id] = 1 + max((depth[p] for p in n.inputs), default=0)
    v = len(graph.nodes)
    return {"vertices": v, "edges": sum(len(n.inputs) for n in graph.nodes), "depth": depth[graph.output],
            "roots": sum(not n.inputs for n in graph.nodes), "calls": graph.model_calls}


def topology_bin(graph, broad_depth_ratio):
    s = graph_stats(graph)
    if graph.topology_primitive == "NGT-Independent":
        return 3
    if s["depth"] == s["vertices"] and s["edges"] == s["vertices"] - 1:
        return 0
    if s["edges"] / s["vertices"] > 2.5:
        return 2
    if s["edges"] / s["vertices"] < 1.5 and s["depth"] / s["vertices"] <= broad_depth_ratio:
        return 1
    return 4


def neighbors(cell):
    for axis, size in enumerate((5, 5, 4)):
        for shift in (-1, 1):
            n = list(cell)
            n[axis] += shift
            if 0 <= n[axis] < size:
                yield tuple(n)


def bootstrap_cfg(seed):
    """Finite typed grammar, one seed per production in a seeded sampled order.

    Prompts are task independent. All six productions are validated DAGs; the
    sampled order and grammar are deterministic, frozen engineering parameters.
    """
    solve = "Solve the supplied problem. Finish with a single exact answer in \\boxed{}; no decimal approximation."
    reason = "Work through the supplied problem carefully, checking the arithmetic. Finish with a single exact answer in \\boxed{}."
    review = "Check the upstream reasoning and correct any mistakes. Solve independently if needed. Finish with one exact answer in \\boxed{}."
    merge = "Compare the independent upstream solutions, verify the result, and finish with one exact answer in \\boxed{}."
    graphs = [
        WorkflowGraph((Node("a", "Generate", solve),), "a"),
        WorkflowGraph((Node("a", "Generate", reason),), "a"),
        WorkflowGraph((Node("a", "Generate", solve), Node("b", "Review&Revise", review, ("a",))), "b"),
        WorkflowGraph((Node("a", "Generate", solve), Node("b", "Generate", reason),
                       Node("c", "Ensemble", merge, ("a", "b"))), "c", topology_primitive="NGT-Independent"),
        WorkflowGraph((Node("a", "Generate", reason), Node("b", "Generate", solve),
                       Node("c", "Ensemble", merge, ("a", "b"))), "c"),
        WorkflowGraph((Node("a", "Generate", solve, subgroup="one"), Node("b", "Generate", reason, subgroup="two"),
                       Node("c", "Review&Revise", review, ("a",), subgroup="one"),
                       Node("d", "Review&Revise", review, ("b",), subgroup="two"),
                       Node("e", "Ensemble", merge, ("c", "d"))), "e", topology_primitive="Subgroup"),
    ]
    random.Random(seed).shuffle(graphs)
    return [g.validate({"local"}) for g in graphs]


def structural_features(graph):
    s = graph_stats(graph)
    return [min(s["vertices"] / 12, 1), min(s["edges"] / 66, 1), min(s["depth"] / 12, 1),
            min(s["roots"] / 12, 1), min(s["calls"] / 12, 1),
            float(graph.topology_primitive == "NGT-Independent"), float(graph.topology_primitive == "Subgroup"),
            min(sum(len(n.prompt) for n in graph.nodes) / 12000, 1)]


def router_features(public_task, graph):
    question = public_task["question"]
    return [1., min(len(question) / 2048, 1), min(sum(c.isdigit() for c in question) / 128, 1),
            *structural_features(graph)]


class PolicyScheduler(CASBMS):
    def __init__(self, owner):
        super().__init__(owner.curvature, owner.allocate, SOURCE)
        self.owner = owner

    def decide(self, t, public_context, remaining_budget):
        result = super().decide(t, public_context, remaining_budget)
        return {**result, "meta_linear_ucb": self.owner.last_meta,
                "engineering_formula": "beta_s=h*max(UCB_s,0)/(h*max(UCB_s,0)+max(UCB_d,0)); zero sum -> 1/2"}


class MinimalPolicy:
    def __init__(self, executor, config, tasks, *, synthetic=False):
        self.validate_settings(config, len(tasks))
        self.executor, self.journal, self.config = executor, executor.journal, config
        self.p = config["policy"]
        executor.allowed_operators = self.operators(config)
        self.synthetic = synthetic
        self.archive, self.router = Archive(), BiGLinUCB(self.router_dimension(), self.p["ridge"])
        self.seeds = self.bootstrap(config)
        self.graphs = {g.version: g for g in self.seeds}
        self.boundaries, self.records, self.estimates = [], {}, {}
        self.proxy = MLPProxy(self.workflow_dimension(), self.p["proxy_hidden"], self.p["seed"])
        self.meta = {a: RidgeHead(4, self.p["ridge"]) for a in ("search", "deploy")}
        self.curvature = CurvatureWindow(self.p["curvature_window"], self.p["delta"],
            self.p["curvature_confidence_constant"], self.p["epsilon_max"], UTILITY, boundary_policy="cap_at_one")
        self.meta_x, self.last_meta, self.completed_blocks, self.feedback_index = None, {}, 0, 0
        self.rounding_carry = .5
        self.deploy_observations = {}
        self.search_vendi, self.deploy_hv = 0., 0.
        self.search = SearchService(executor, self.archive, self.router, describe=self.describe,
            accept_elite=pareto_admission, proxy_screen=lambda _: None, measure_usd=self.total_reference_usd,
            policy_source=SOURCE, batch_screen=self.screen)
        self.search.max_evaluations = math.ceil(self.p["k"] * self.p["proxy_fraction"])
        self.deployment = DeploymentService(executor, self.router, features=self.features,
            weights=lambda t: self.p["weight_grid"][(t - 1) % len(self.p["weight_grid"])],
            confidence=self.router_confidence, cost_reward=lambda receipts: self.cost_reward(self.total_reference_usd(receipts)),
            cost_definition="r_c=1/(1+reference_USD_per_query/c0); reference cost is measured call seconds * assumed USD/GPU-hour/3600")
        self.deployment.fallback_threshold, self.deployment.safe_arm = 0., self.seeds[0].version
        self.scheduler = PolicyScheduler(self)
        self.driver = self.make_driver(tasks)

    def validate_settings(self, config, task_count):
        validate_config(config, task_count)

    def router_dimension(self):
        return 11

    def workflow_dimension(self):
        return 8

    def workflow_features(self, graph):
        return structural_features(graph)

    def features(self, public_task, graph):
        return router_features(public_task, graph)

    def operators(self, config):
        return ALLOWED

    def bootstrap(self, config):
        return bootstrap_cfg(config["policy"]["seed"])

    def make_driver(self, tasks):
        config = self.config
        return SmokeDriver(self.executor, self.search, self.deployment, self.scheduler,
            SmokeBindings(self.selection, self.observe, self.context, self.operations, SOURCE, self.synthetic),
            seeds=self.seeds, search_tasks=[tasks[i] for i in config["search_task_indices"]],
            deployment_tasks=tasks, cycles=config["cycles"], generation_max_nodes=config["generation_max_nodes"])

    def total_reference_usd(self, receipts):
        total = 0.
        for receipt in receipts:
            call = self.journal.lookup("call_finished", receipt)
            if call is None or call.get("elapsed_seconds") is None:
                raise IntegrityError("measured call duration is required for reference GPU cost")
            duration = call["elapsed_seconds"]
            if not math.isfinite(duration) or duration < 0:
                raise IntegrityError("invalid measured call duration")
            total += duration * self.p["reference_usd_per_gpu_hour"] / 3600
        return total

    def cost_reward(self, cost):
        return 1 / (1 + cost / self.p["cost_scale_usd"])

    def describe(self, graph, usd):
        boundaries = self.boundaries or cost_boundaries([usd])
        return (cost_bin(usd, boundaries), topology_bin(graph, self.p["broad_depth_ratio"]), 0), digest(boundaries)

    def _rebin(self):
        self.boundaries = cost_boundaries([r["usd_per_query"] for r in self.records.values()])
        changes = self.archive.rebin(lambda e: self.describe(self.graphs[e.workflow], e.usd_cost), pareto_admission)
        self.estimates = {}
        for e in self.archive.evaluations.values():
            self.estimates.setdefault(cell_id(e.cell), QualityEstimate()).observe(e.quality)
        for e in self.archive.elites.values():
            if self.router.add_arm(e.workflow):
                self.journal.append("router_arm", e.workflow, {"epoch": self.router.epoch,
                    "source": "quantile_rebin", "receipts": list(e.receipt_ids)})
        self.journal.append("archive_rebin", f"feedback:{self.feedback_index}", {
            "boundaries": self.boundaries, "moves": changes,
            "elites": {str(i): e.workflow for i, e in self.archive.elites.items()}, "source": SOURCE})

    def _proxy_fit(self):
        records = list(self.records.values())
        self.proxy.fit([self.workflow_features(self.graphs[r["workflow"]]) for r in records],
            [r["quality"] for r in records], receipt_ids=[r["receipts"][0] for r in records],
            roles=["development"] * len(records), paid_receipts=[c for r in records for c in r["receipts"]],
            steps=self.p["proxy_steps"], learning_rate=self.p["proxy_learning_rate"])

    def screen(self, graphs):
        return self.proxy.screen_fraction([g.version for g in graphs],
            [self.workflow_features(g) for g in graphs], self.p["proxy_fraction"])

    def selection(self):
        ids = sorted(self.archive.elites)
        occupied = {e.cell for e in self.archive.elites.values()}
        features = [[c / n for c, n in zip(self.archive.elites[i].cell, (4, 4, 3))]
                    + self.workflow_features(self.graphs[self.archive.elites[i].workflow]) for i in ids]
        t = 1 + sum(e.count for e in self.estimates.values())
        kernel = rbf_ucb_kernel(features, [self.estimates[i].ucb(t, self.p["search_exploration"]) for i in ids], self.p["kernel_bandwidth"])
        result = select_niches(ids, {i: len(self.archive.population[i]) for i in ids},
            {i: occupancy_density(self.archive.elites[i].cell, occupied, neighbors) for i in ids}, self.estimates,
            kernel, t=t, k=self.p["k"], population_threshold=self.p["population_threshold"],
            rho0=self.p["rho0"], alpha=self.p["alpha"], diversity_weight=self.p["diversity_weight"],
            exploration=self.p["search_exploration"])
        result.update({"clock": "1 + distinct evaluated workflows, including bootstrap", "kernel": kernel.tolist(),
                       "features": features, "ids": ids})
        return result, {i: self.graphs[self.archive.elites[i].workflow] for i in ids}

    def router_confidence(self, router, t):
        kwargs = {"noise_bound": self.p["noise_bound"], "parameter_bound": self.p["parameter_bound"], "delta": self.p["delta"]}
        return ({a: q.oful_beta(**kwargs) for a, (q, c) in router.arms.items()},
                {a: c.oful_beta(**kwargs) for a, (q, c) in router.arms.items()})

    def _vendi(self):
        x = np.array([self.workflow_features(self.graphs[e.workflow]) for e in self.archive.elites.values()])
        if not len(x):
            return 0.
        similarity = np.exp(-self.p["kernel_bandwidth"] * ((x[:, None] - x[None, :]) ** 2).sum(axis=2))
        return vendi_score(similarity)

    def _curvature_observation(self, record):
        entry = self.archive.evaluations[record["workflow"]]
        i = cell_id(entry.cell)
        snapshot = {str(j): [e.quality, self.cost_reward(e.usd_cost)] for j, e in self.archive.elites.items() if j != i}
        point = [entry.quality, self.cost_reward(entry.usd_cost)]
        empty = hypervolume([point], (0, 0))
        baseline = hypervolume(snapshot.values(), (0, 0))
        marginal = hypervolume([*snapshot.values(), point], (0, 0)) - baseline
        id = record["workflow"]
        evidence = {"element": str(i), "conditioning_points": snapshot, "point": point,
                    "conditioning_workflows": {str(j): e.workflow for j, e in self.archive.elites.items() if j != i},
                    "baseline": baseline, "marginal": marginal, "empty_marginal": empty,
                    "receipt_ids": record["receipts"], "utility_definition": UTILITY}
        if empty <= 0:
            self.journal.append("curvature_observation", id, {**evidence, "status": "skipped_zero_denominator"})
            return
        # Floating point subtraction may produce only a tiny negative roundoff.
        if -1e-12 < marginal < 0:
            marginal = 0.
        if empty < marginal < empty + 1e-12:
            marginal = empty
        observation = MarginalObservation(id, str(i), tuple(snapshot), marginal, empty, UTILITY, tuple(record["receipts"]))
        self.curvature.observe(observation, {e["id"] for e in self.journal.events if e["kind"] == "call_finished"})
        self.journal.append("curvature_observation", id, {**evidence, "status": "observed", "rounded_marginal": marginal})

    def observe(self, stage, results):
        self.feedback_index += 1
        receipts, new = [], []
        old_vendi, old_hv = self.search_vendi, self.deploy_hv
        for r in results:
            receipts.extend(r.get("receipts", r.get("calls", [])))
            if r["status"] == "evaluated" and r["workflow"] not in self.records:
                if "graph" in r:
                    self.graphs[r["workflow"]] = WorkflowGraph.from_dict(r["graph"])
                self.records[r["workflow"]] = r
                new.append(r)
            elif stage == "deploy" and self.executor.assessment_complete(r):
                history = self.deploy_observations.setdefault(r["workflow"], [])
                history.append((self.executor.assessment_quality(r), self.cost_reward(self.total_reference_usd(r["calls"]))))
        if new:
            self._rebin()
            self._proxy_fit()
            for r in new:
                self._curvature_observation(r)
        self.search_vendi = self._vendi()
        points = [np.mean(values, axis=0).tolist() for values in self.deploy_observations.values()]
        self.deploy_hv = hypervolume(points, (0, 0))
        reward = None
        if stage in self.meta:
            reward = (self.search_vendi - old_vendi) / 100 if stage == "search" else self.deploy_hv - old_hv
            self.meta[stage].update(self.meta_x, reward)
            self.completed_blocks += 1
        return {"receipt_ids": receipts, "stage_reward": reward,
                "vendi": self.search_vendi, "deployment_hv": self.deploy_hv,
                "search_visit_counts": {str(i): e.count for i, e in self.estimates.items()},
                "proxy_label_groups": {r["workflow"]: r["receipts"] for r in new},
                "completed_blocks": self.completed_blocks, "new_observation_count": len(new),
                "synthetic": self.synthetic}

    def context(self):
        total = self.config["cycles"] * self.p["blocks_per_cycle"]
        return {"remaining_research_budget": total - self.completed_blocks, "research_budget_unit": "operation_block",
                "total_blocks": total, "vendi": self.search_vendi, "hv": self.deploy_hv, "epoch": self.router.epoch}

    def allocate(self, context, estimate, remaining_budget):
        self.meta_x = [remaining_budget / context["total_blocks"], context["vendi"] / 100,
                       context["hv"], min(context["epoch"] / 100, 1)]
        self.last_meta = {}
        for name, head in self.meta.items():
            beta = head.oful_beta(noise_bound=self.p["noise_bound"], parameter_bound=self.p["parameter_bound"], delta=self.p["delta"])
            self.last_meta[name] = {**head.predict(self.meta_x, beta), "features": self.meta_x,
                                    "beta": beta, "head_before_update": head.state()}
        s = estimate["h"] * max(0., self.last_meta["search"]["ucb"])
        d = max(0., self.last_meta["deploy"]["ucb"])
        beta_s = s / (s + d) if s + d > 0 else .5
        return {"search": beta_s, "deploy": 1 - beta_s}

    def operations(self, decision, context):
        blocks = min(self.p["blocks_per_cycle"], context["remaining_research_budget"])
        target = self.rounding_carry + blocks * decision["allocation"]["search"]
        searches = min(blocks, math.floor(target))
        self.rounding_carry = target - searches
        return ["search"] * searches + ["deploy"] * (blocks - searches)


def validate_config(config, task_count=None):
    if config.get("profile") != PROFILE:
        raise ValueError(f"smoke profile must be {PROFILE}")
    if not 4 <= config["cycles"] <= 6 or not 1 <= config["max_calls"] <= 146 or not 0 < config["max_seconds"] <= 3600:
        raise ValueError("smoke caps: 4..6 cycles, <=146 new attempts, <=3600 seconds")
    if not 1 <= config["generation_max_nodes"] <= 4:
        raise ValueError("this smoke profile limits generated DAGs to four nodes")
    indices = config["search_task_indices"]
    if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 or (task_count is not None and i >= task_count) for i in indices):
        raise ValueError("invalid frozen development task indices")
    p = config["policy"]
    for key in ["ridge", "proxy_learning_rate", "kernel_bandwidth", "search_exploration", "rho0", "alpha",
                "cost_scale_usd", "reference_usd_per_gpu_hour", "curvature_confidence_constant", "noise_bound", "parameter_bound"]:
        if not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"positive finite {key} required")
    for key in ["k", "population_threshold", "proxy_hidden", "proxy_steps", "curvature_window", "blocks_per_cycle"]:
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"positive integer {key} required")
    if p["k"] != 2 or p["blocks_per_cycle"] != 2:
        raise ValueError("frozen minimal profile uses k=2 and two blocks per cycle")
    if not 0 < p["proxy_fraction"] <= 1 or not 0 < p["delta"] < 1 or not 0 < p["broad_depth_ratio"] < 1:
        raise ValueError("invalid fraction/confidence/descriptor parameter")
    if not math.isfinite(p["diversity_weight"]) or p["diversity_weight"] < 0 or not math.isfinite(p["epsilon_max"]) or p["epsilon_max"] < 0:
        raise ValueError("invalid diversity/confidence parameter")
    if not p["weight_grid"] or any(len(w) != 2 or any(not math.isfinite(v) or v <= 0 for v in w) for w in p["weight_grid"]):
        raise ValueError("nonempty positive quality/cost weight grid required")
    for field in ["model_path", "tasks", "tasks_sha256", "decode", "max_context_tokens"]:
        if field not in config:
            raise ValueError(f"missing runtime configuration {field}")
