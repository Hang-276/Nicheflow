"""Explicit local/API profiles with frozen decoding and auditable cost components."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import urllib.request
import urllib.error
from .spec import EnvironmentBlocked, IntegrityError
from .main_config import model_selection_problems


def load_credentials(path, models):
    """Parse only the requested variable names, never execute a shell/env file."""
    names = {m["key_environment"] for m in models.values() if m["kind"] == "api"}
    if not names or not path or not Path(path).exists():
        return
    p = Path(path)
    if p.stat().st_mode & 0o077:
        raise EnvironmentBlocked("credentials file must not be readable by group/others")
    for line in p.read_text().splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if separator and name in names and not os.environ.get(name):
            os.environ[name] = value.strip()


def model_readiness(config):
    load_credentials(config.get("credentials_file"), config["models"])
    problems = model_selection_problems(config)
    local_count = 0
    for name, p in config["models"].items():
        if p.get('configuration_status') == 'pending_selection':
            continue
        if p["kind"] == "local":
            local_count += 1
            if not Path(p["path"]).is_dir():
                problems.append(f"{name}: local model snapshot missing")
        elif not os.environ.get(p["key_environment"]):
            problems.append(f"{name}: credential variable {p['key_environment']} is absent")
    if local_count > 1:
        problems.append("single-GPU profile supports one resident local model; use APIs for other tiers")
    return problems


def peak_period(timestamp):
    t = datetime.fromtimestamp(timestamp, timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


def api_charge(usage, pricing, timestamp):
    inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if type(inputs) is not int or type(outputs) is not int or min(inputs, outputs) < 0:
        return None
    cached = usage.get("prompt_cache_hit_tokens", usage.get("prompt_tokens_details", {}).get("cached_tokens", 0))
    if type(cached) is not int or not 0 <= cached <= inputs:
        raise IntegrityError("invalid API cache token accounting")
    period = "peak" if pricing.get("utc_peak_offpeak") and peak_period(timestamp) else "off_peak"
    rates = pricing.get(period, pricing["peak"])
    cost = ((inputs - cached) * rates["input_per_million_usd"] + cached * rates.get("cached_input_per_million_usd", rates["input_per_million_usd"])
            + outputs * rates["output_per_million_usd"]) / 1e6
    return {"usd": cost, "period": period, "rates": rates, "cached_input_tokens": cached,
            "uncached_input_tokens": inputs - cached, "source": pricing["source"],
            "basis": "provider_usage_times_frozen_tariff", "actual_invoice_verified": False}


class MainAPI:
    def __init__(self, profile):
        if profile.get('configuration_status') == 'pending_selection' or not profile.get('model'):
            raise EnvironmentBlocked('API model selection is pending')
        self.profile, self.model_id = profile, profile["model"]
        self.call_timeout = profile["call_timeout_seconds"]
        self.environment = {"model": {"id": self.model_id, "profile": profile}, "synthetic": False}

    def generate(self, messages, seed, max_new_tokens, temperature, top_p):
        p = self.profile
        # A conservative text-byte bound prevents context overflow without a new tokenizer dependency.
        bound = sum(len(str(m["content"]).encode()) + 64 for m in messages) + 1024 + max_new_tokens
        if bound > p["max_context_tokens"]:
            return {"status": "error", "text": "", "finish_reason": "context_limit", "error": "conservative input bound exceeded",
                    "input_tokens": 0, "output_tokens": 0, "elapsed_seconds": 0., "accounted_usd": 0., "provider_request_sent": False}
        payload = {"model": self.model_id, "messages": messages, "max_tokens": max_new_tokens}
        if p.get("provider") == "deepseek":
            payload["thinking"] = {"type": p["thinking"]}
            if p["thinking"] == "enabled":
                payload["reasoning_effort"] = p["reasoning_effort"]
                payload["top_p"] = max(.95, top_p)
            else:
                payload["temperature"] = temperature
            # No undocumented seed field; DeepSeek does not promise seeded generation.
        else:
            payload.update(temperature=temperature, top_p=top_p)
            if p.get("supports_seed", False):
                payload["seed"] = seed
        key = os.environ.get(p["key_environment"])
        if not key:
            raise EnvironmentBlocked("configured API credential disappeared")
        request = urllib.request.Request(p["endpoint"], data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.call_timeout) as response:
                data = json.load(response)
            choice = data["choices"][0]
            message, usage = choice["message"], data.get("usage", {})
            charge = api_charge(usage, p["pricing"], data.get("created", start))
            returned = data.get("model")
            status = "ok" if isinstance(message.get("content"), str) and returned in p["accepted_response_models"] else "error"
            return {"text": message.get("content") or "", "status": status, "finish_reason": choice["finish_reason"],
                    "error": None if status == "ok" else "missing final content or unapproved returned model identity",
                    "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                    "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                    "usage": usage, "elapsed_seconds": time.time() - start, "accounted_usd": charge["usd"] if charge else None,
                    "api_charge": charge, "local_accounting_usd": 0., "response_id": data.get("id"), "returned_model": returned,
                    "system_fingerprint": data.get("system_fingerprint"), "provider_created": data.get("created"),
                    "provider_request_sent": True, "requested_seed": seed, "provider_seed_supported": bool(p.get("supports_seed", False)),
                    "effective_decode": {k: v for k, v in payload.items() if k not in {"messages", "model"}}}
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, IndexError, TypeError) as exc:
            # Never print headers, credential values or provider-controlled error bodies.
            return {"text": "", "status": "error", "finish_reason": "error", "error": type(exc).__name__,
                    "input_tokens": None, "output_tokens": None, "accounted_usd": None,
                    "elapsed_seconds": time.time() - start, "provider_request_sent": True,
                    "billing_outcome_unknown": True}

    def close(self):
        pass


class AccountedLocal:
    def __init__(self, profile):
        from .backend_process import LocalWorker
        self.profile = profile
        self.worker = LocalWorker(profile["path"], profile["max_context_tokens"], call_timeout=profile["call_timeout_seconds"])
        self.model_id, self.environment = self.worker.model_id, self.worker.environment
        self.call_timeout = profile["call_timeout_seconds"]

    def generate(self, messages, **params):
        self.worker.call_timeout = self.call_timeout
        result = self.worker.generate(messages, **params)
        seconds = result.get("elapsed_seconds")
        cost = seconds * self.profile["accounting_usd_per_gpu_hour"] / 3600 if seconds is not None and math.isfinite(seconds) and seconds >= 0 else None
        return {**result, "accounted_usd": cost, "local_accounting_usd": cost,
                "local_cost_basis": ("external_api_spend_only_local_capacity_excluded" if self.profile['accounting_usd_per_gpu_hour'] == 0
                                     else "measured_inference_seconds_times_declared_rate"), "actual_invoice_verified": False,
                "api_charge": None}

    def close(self):
        self.worker.close()


class ModelPool:
    def __init__(self, config):
        problems = model_readiness(config)
        if problems:
            raise EnvironmentBlocked("; ".join(problems))
        self.backends = {}
        try:
            for name, profile in config["models"].items():
                self.backends[name] = AccountedLocal(profile) if profile["kind"] == "local" else MainAPI(profile)
        except BaseException:
            self.close()
            raise

    def close(self):
        for backend in self.backends.values():
            backend.close()
