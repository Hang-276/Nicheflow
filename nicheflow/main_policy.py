"""Main-method bindings reuse the verified three layers, without smoke-only caps."""
from dataclasses import asdict, replace
import math
import numpy as np
from .archive import cost_bin, cost_boundaries, heterogeneity_bin
from .graph import WorkflowGraph, LLM_OPERATORS
from .main_config import validate_main
from .policy import MinimalPolicy, bootstrap_cfg, topology_bin
from .spec import IntegrityError, digest


def main_seeds(config):
    prompts = config["seed_prompts"]
    profiles = list(config["models"])
    base = config["seed_model"]
    seeds = []
    for index, graph in enumerate(bootstrap_cfg(config["policy"]["seed"])):
        nodes = []
        for j, node in enumerate(graph.nodes):
            prompt = prompts["reason"] if "carefully" in node.prompt else prompts["solve"]
            if node.operator == "Review&Revise":
                prompt = prompts["review"]
            elif node.operator == "Ensemble":
                prompt = prompts["merge"]
            model = base if index == 0 else profiles[(index + j) % len(profiles)]
            nodes.append(replace(node, prompt=prompt, model=model))
        seeds.append(replace(graph, nodes=tuple(nodes)).validate(config["models"]))
    return seeds


class MainPolicy(MinimalPolicy):
    def __init__(self, executor, config, tasks, *, synthetic=False):
        super().__init__(executor, config, tasks, synthetic=synthetic)
        self.deployment.cost_definition = "r_c=1/(1+C/C0); C=API usage*tariff + local measured inference time*declared GPU rate; not verified invoice"
        if config.get('cost_objective') == 'external_api_spend':
            self.deployment.cost_definition = "r_c=1/(1+C/C0); C=external API usage*tariff; existing local capacity excluded from monetary objective; local inference seconds reported separately; not total operating cost or verified invoice"

    def validate_settings(self, config, task_count):
        validate_main(config)

    def operators(self, config):
        return config["allowed_operators"]

    def bootstrap(self, config):
        return main_seeds(config)

    def make_driver(self, tasks):
        # MainDriver owns stage boundaries and evaluation; no shadow SmokeDriver.
        return None

    def total_reference_usd(self, receipts):
        total = 0.
        for id in receipts:
            call = self.journal.lookup("call_finished", id)
            amount = None if call is None else call.get("accounted_usd")
            if amount is None or not math.isfinite(amount) or amount < 0:
                raise IntegrityError("main policy requires known measured/accounted call cost")
            total += amount
        return total

    def describe(self, graph, usd):
        boundaries = self.boundaries or cost_boundaries([usd])
        # Source denominator is all workflow nodes, including deterministic nodes.
        noncheap = sum(n.operator in LLM_OPERATORS and self.config["models"][n.model]["noncheap"] for n in graph.nodes)
        return (cost_bin(usd, boundaries), topology_bin(graph, self.p["broad_depth_ratio"]),
                heterogeneity_bin(noncheap, len(graph.nodes))), digest(boundaries)

    def context(self):
        blocks = self.config["rounds"] * self.p["blocks_per_cycle"]
        cap = self.journal.limits["block_accounting_usd"]
        return {"remaining_research_budget": (blocks - self.completed_blocks) * cap,
                "total_research_budget": blocks * cap, "research_budget_unit": "reserved_accounting_usd",
                "total_blocks": blocks, "remaining_blocks": blocks - self.completed_blocks,
                "block_accounting_usd": cap, "vendi": self.search_vendi, "hv": self.deploy_hv, "epoch": self.router.epoch}

    def allocate(self, context, estimate, remaining_budget):
        self.meta_x = [remaining_budget / context["total_research_budget"], context["vendi"] / 100,
                       context["hv"], min(context["epoch"] / 100, 1)]
        self.last_meta = {}
        for name, head in self.meta.items():
            beta = head.oful_beta(noise_bound=self.p["noise_bound"], parameter_bound=self.p["parameter_bound"], delta=self.p["delta"])
            self.last_meta[name] = {**head.predict(self.meta_x, beta), "features": self.meta_x,
                                    "beta": beta, "head_before_update": head.state()}
        s = estimate["h"] * max(0., self.last_meta["search"]["ucb"])
        d = max(0., self.last_meta["deploy"]["ucb"])
        beta = s / (s + d) if s + d else .5
        return {"search": beta, "deploy": 1 - beta}

    def operations(self, decision, context):
        blocks = min(self.p["blocks_per_cycle"], context["remaining_blocks"])
        target = self.rounding_carry + blocks * decision["allocation"]["search"]
        searches = min(blocks, math.floor(target))
        self.rounding_carry = target - searches
        return ["search"] * searches + ["deploy"] * (blocks - searches)

    def export_state(self):
        """No task text or evaluation labels. Sufficient for independent frozen routing."""
        return {"schema": "nicheflow_state_v1", "synthetic": self.synthetic,
                "graphs": {v: g.definition() for v, g in self.graphs.items()},
                "seed_versions": [g.version for g in self.seeds], "router": self.router.state(),
                "archive": {"evaluations": {v: asdict(e) for v, e in self.archive.evaluations.items()},
                            "elites": {str(i): e.workflow for i, e in self.archive.elites.items()}, "boundaries": self.boundaries},
                "proxy": {"w1": self.proxy.w1.tolist(), "b1": self.proxy.b1.tolist(), "w2": self.proxy.w2.tolist(),
                          "b2": float(self.proxy.b2), "training_receipts": list(self.proxy.training_receipts)},
                "search_estimates": {str(i): asdict(e) for i, e in self.estimates.items()},
                "meta": {name: head.state() for name, head in self.meta.items()},
                "curvature": {"window": [asdict(o) for o in self.curvature.window],
                              "seen": {id: asdict(o) for id, o in self.curvature.seen.items()}},
                "completed_blocks": self.completed_blocks, "feedback_index": self.feedback_index,
                "rounding_carry": self.rounding_carry, "search_vendi": self.search_vendi, "deploy_hv": self.deploy_hv,
                "deploy_observations": self.deploy_observations,
                "records": {v: {k: val for k, val in r.items() if k != "graph"} for v, r in self.records.items()},
                "policy_config_digest": digest({k: v for k, v in self.config.items() if k != "rounds"})}


def stability(state, previous=None):
    """Numerical health + observed drift, never a claim of statistical convergence."""
    heads = {f"meta/{name}": h for name, h in state["meta"].items()}
    heads.update({f"router/{arm}/{name}": h for arm, pair in state["router"]["arms"].items() for name, h in pair.items()})
    old = {} if previous is None else {f"meta/{name}": h for name, h in previous["meta"].items()}
    if previous:
        old.update({f"router/{arm}/{name}": h for arm, pair in previous["router"]["arms"].items() for name, h in pair.items()})
    checks, healthy = {}, True
    for name, head in heads.items():
        v, b = np.asarray(head["v"]), np.asarray(head["b"])
        finite = bool(np.isfinite(v).all() and np.isfinite(b).all())
        eig = np.linalg.eigvalsh(v) if finite else np.array([0.])
        symmetric = bool(np.allclose(v, v.T, atol=1e-10, rtol=0)) if finite else False
        positive = bool(eig.min() > 0)
        condition = float(np.linalg.cond(v)) if finite and positive else None
        theta = np.linalg.solve(v, b) if finite and positive else None
        delta = None
        if name in old and theta is not None:
            previous_theta = np.linalg.solve(np.asarray(old[name]["v"]), np.asarray(old[name]["b"]))
            delta = float(np.linalg.norm(theta - previous_theta))
        ok = finite and symmetric and positive and theta is not None and bool(np.isfinite(theta).all())
        healthy &= ok
        checks[name] = {"finite": finite, "symmetric": symmetric, "positive_definite": positive,
                        "min_eigenvalue": float(eig.min()), "condition_number": condition,
                        "theta_norm": float(np.linalg.norm(theta)) if theta is not None else None,
                        "theta_delta_l2": delta, "observations": head["count"],
                        "condition_warning": condition is not None and condition > 1e12}
    def proxy_vector(s):
        return np.concatenate([np.asarray(s["proxy"][k]).reshape(-1) for k in ("w1", "b1", "w2", "b2")])
    proxy = proxy_vector(state)
    proxy_finite = bool(np.isfinite(proxy).all())
    return {"numerically_healthy": bool(healthy and proxy_finite), "heads": checks,
            "proxy_finite": proxy_finite, "proxy_norm": float(np.linalg.norm(proxy)) if proxy_finite else None,
            "proxy_delta_l2": float(np.linalg.norm(proxy - proxy_vector(previous))) if previous is not None and proxy_finite else None,
            "arms": len(state["router"]["arms"]), "elites": len(state["archive"]["elites"]),
            "new_arms": len(set(state["router"]["arms"]) - set(previous["router"]["arms"])) if previous else None,
            "statistical_convergence_established": False,
            "interpretation": "finite/SPD checks detect numerical failure; drift is descriptive, not a convergence test"}
