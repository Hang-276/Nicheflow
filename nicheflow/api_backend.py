"""Optional cheap/strong model profiles; no provider is selected or called by default."""
import json
import os
import time
import urllib.error
import urllib.request
from .spec import EnvironmentBlocked


class ChatCompletionBackend:
    def __init__(self, *, model, endpoint, key_environment, timeout, pricing=None):
        if not endpoint.startswith("https://"):
            raise ValueError("API endpoint must use HTTPS")
        self.model_id, self.endpoint = model, endpoint
        self.key_environment, self.call_timeout = key_environment, timeout
        self.pricing = pricing

    def generate(self, messages, seed, max_new_tokens=1536, temperature=.7, top_p=.8):
        key = os.environ.get(self.key_environment)
        if not key:
            raise EnvironmentBlocked(f"API credential environment variable {self.key_environment} is absent")
        payload = {"model": self.model_id, "messages": messages, "max_tokens": max_new_tokens,
                   "temperature": temperature, "top_p": top_p, "seed": seed}
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=self.call_timeout) as response:
                data = json.load(response)
            choice = data["choices"][0]
            usage = data.get("usage", {})
            inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
            cost = None
            if self.pricing and inputs is not None and outputs is not None:
                cost = (inputs * self.pricing["input_per_million_usd"] + outputs * self.pricing["output_per_million_usd"]) / 1e6
            return {"text": choice["message"]["content"], "status": "ok", "finish_reason": choice["finish_reason"],
                    "input_tokens": inputs, "output_tokens": outputs, "usd_cost": cost,
                    "pricing": self.pricing, "elapsed_seconds": time.time() - started,
                    "response_id": data.get("id"), "returned_model": data.get("model")}
        except urllib.error.HTTPError as exc:
            # Do not echo request headers, credentials, or an arbitrary provider error body.
            return {"text": "", "status": "error", "finish_reason": "error", "error": f"provider HTTP {exc.code}",
                    "input_tokens": None, "output_tokens": None, "usd_cost": None,
                    "elapsed_seconds": time.time() - started}
