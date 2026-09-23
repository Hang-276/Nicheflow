"""Per-stage resource envelopes over the shared immutable call journal."""
from contextlib import contextmanager
import math
from .ledger import Journal
from .spec import BudgetStop, IntegrityError
from .main_config import max_call_usd


class ResourceJournal(Journal):
    def __init__(self, *args, limits, **kwargs):
        super().__init__(*args, **kwargs)
        self.limits, self.active_scope = limits, None
        self._resource_cursor, self._reservations, self._resource_totals = 0, {}, {}

    def spending(self, scope=None):
        # Incrementally rebuildable indexes keep per-call checks independent of run length.
        def total(name):
            return self._resource_totals.setdefault(name, {"calls": 0, "accounted_usd": 0., "api_usd": 0., "unknown": set()})
        for event in self.events[self._resource_cursor:]:
            id, kind, value = event["id"], event["kind"], event["payload"]
            if kind == "resource_reservation":
                self._reservations[id] = value
            elif kind in {"call_started", "call_finished"}:
                if id not in self._reservations:
                    raise IntegrityError("main call without a recorded resource reservation")
                reservation = self._reservations[id]
                group = "__evaluation__" if reservation["scope"] == "evaluation" else "__learning__"
                for name in (None, reservation["scope"], group):
                    item = total(name)
                    if kind == "call_started":
                        item["calls"] += 1
                        item["unknown"].add(id)
                    elif value.get("accounted_usd") is not None:
                        amount = value["accounted_usd"]
                        if not math.isfinite(amount) or amount < 0:
                            raise IntegrityError("non-finite/negative recorded cost")
                        if amount > reservation["maximum_accounting_usd"] + 1e-9:
                            raise IntegrityError("call exceeded its reserved cost bound")
                        item["unknown"].discard(id)
                        item["accounted_usd"] += amount
                        item["api_usd"] += (value.get("api_charge") or {}).get("usd", 0.)
            self._resource_cursor += 1
        item = total(scope)
        return {"calls": item["calls"], "accounted_usd": item["accounted_usd"], "api_usd": item["api_usd"],
                "unknown_cost_calls": sorted(item["unknown"])}

    @contextmanager
    def scope(self, name, call_cap, usd_cap):
        if self.active_scope is not None:
            raise IntegrityError("nested resource envelopes are not allowed")
        self.append("resource_envelope", name, {"call_cap": call_cap, "accounting_usd_cap": usd_cap})
        self.active_scope = name
        try:
            if name == "evaluation" and self.lookup("resource_stage_started", "evaluation") is None:
                self.append("resource_stage_started", "evaluation", {"started_at": self.clock()})
            yield
        finally:
            self.active_scope = None

    def remaining_seconds(self):
        overall = super().remaining_seconds()
        start = self.lookup("resource_stage_started", "evaluation")
        if start:
            stage = self.limits["evaluation_seconds"] - (self.clock() - start["started_at"])
        else:
            stage = self.limits["learning_seconds"] - (self.clock() - self.started)
        return min(overall, stage)

    def reserve(self, count):
        super().reserve(count)
        overall = self.spending()
        if overall["unknown_cost_calls"]:
            raise IntegrityError("unknown billed outcome; no automatic continuation")
        if overall["accounted_usd"] > self.limits["max_accounting_usd"] + 1e-9:
            raise BudgetStop("total accounting USD cap")
        if self.active_scope:
            spec = self.lookup("resource_envelope", self.active_scope)
            spent = self.spending(self.active_scope)
            if spent["calls"] + count > spec["call_cap"]:
                raise BudgetStop("stage call cap: " + self.active_scope)
            if spent["accounted_usd"] + count * self.limits["max_call_accounting_usd"] > spec["accounting_usd_cap"] + 1e-9:
                raise BudgetStop("stage accounting USD reservation: " + self.active_scope)

    def call(self, id, backend, messages, params):
        if self.lookup("call_started", id) is None:
            if self.active_scope is None:
                raise IntegrityError("main experiment calls require an explicit resource envelope")
            self.reserve(1)
            if getattr(backend, "profile", {}).get("kind") == "api":
                bound = max_call_usd(backend.profile, params["max_new_tokens"])
                if self.spending()["api_usd"] + bound > self.limits["max_api_usd"] + 1e-9:
                    raise BudgetStop("external API dollar cap; no automatic increase")
                phase = "evaluation" if self.active_scope == "evaluation" else "learning"
                if self.spending(f"__{phase}__")["api_usd"] + bound > self.limits[f"{phase}_api_usd"] + 1e-9:
                    raise BudgetStop(f"{phase} API cap; other stage's allowance cannot be borrowed")
            self.append("resource_reservation", id, {"scope": self.active_scope,
                        "maximum_accounting_usd": self.limits["max_call_accounting_usd"]})
            if getattr(backend, "profile", {}).get("call_timeout_seconds"):
                # A near-expired learning stage must not permanently shorten evaluation calls.
                backend.call_timeout = backend.profile["call_timeout_seconds"]
        return super().call(id, backend, messages, params)
