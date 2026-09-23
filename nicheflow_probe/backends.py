"""Single local Qwen2.5-7B-Instruct 4-bit backend.

Only the local model branch from the proposal is exercised.  The backend turns
chat messages into text and returns termination/resource metadata; it never
receives reference answers, quality labels or dataset levels.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

# A call that fails for a transient reason may be retried once by the runner.
TRANSIENT_MARKERS = (
    "connection", "timeout", "timed out", "temporarily", "server error",
    "cuda error", "device-side", "unavailable", "busy", "try again",
)


class BackendError(RuntimeError):
    """The backend could not produce a result for this call."""

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


def decoding_kwargs(temperature, top_p, max_new_tokens):
    """Temperature zero is greedy; never pass it into sampled generation."""
    import math
    if not math.isfinite(temperature) or not 0 <= temperature <= 2 or not 0 < top_p <= 1 or max_new_tokens < 1:
        raise ValueError("invalid decoding parameters")
    kwargs = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens}
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p)
    return kwargs


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_inventory(model_dir: Path) -> list[dict]:
    """Model file names, sizes and hashes, as load-time identity evidence."""
    inventory = []
    for path in sorted(model_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".incomplete"):
            continue
        entry = {"file": str(path.relative_to(model_dir)), "bytes": path.stat().st_size}
        entry["sha256"] = _sha256(path)
        inventory.append(entry)
    return inventory


class TransformersChatBackend:
    """Greedy-free sampled decoding with the model's own chat template."""

    def __init__(self, model_id: str, max_context_tokens: int = 8192, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.load_seconds = None
        self.model_dir = None
        self.identity = {}
        self._torch = torch

        started = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map={"": device}, torch_dtype="auto", low_cpu_mem_usage=True
        )
        self.model.eval()
        self.load_seconds = time.time() - started

        self.model_dir = self._resolve_model_dir(model_id)
        self.identity = self._collect_identity()

    @staticmethod
    def _resolve_model_dir(model_id: str) -> Path:
        """Local directory actually loaded, resolved through the HF cache."""
        if Path(model_id).exists():
            return Path(model_id)
        try:
            from huggingface_hub import snapshot_download

            return Path(snapshot_download(model_id, local_files_only=True,
                                          allow_patterns=["*.json", "*.safetensors",
                                                          "*.txt", "*.model"]))
        except Exception:  # pragma: no cover - only if the cache lookup fails
            return Path(model_id)

    # ---------------------------------------------------------------- identity
    def _collect_identity(self) -> dict:
        quantization = None
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            quantization = config.get("quantization_config")
        index_path = self.model_dir / "model.safetensors.index.json"
        on_disk_bytes = None
        if index_path.exists():
            on_disk_bytes = json.loads(index_path.read_text()).get("metadata", {}).get("total_size")
        packed_parameters = int(sum(p.numel() for p in self.model.parameters()))
        return {
            "requested_model_id": self.model_id,
            "resolved_path": str(self.model_dir),
            "architecture": self.model.config.architectures,
            "model_type": self.model.config.model_type,
            "packed_parameter_tensors": packed_parameters,
            "packed_parameter_note": (
                "module-tree parameter count of the 4-bit checkpoint; int4 weights are packed, "
                "so this is smaller than the 7.6B parameters of the unquantized model"
            ),
            "original_weight_bytes_from_index": on_disk_bytes,
            "quantization_config": quantization,
            "weight_dtype_of_embeddings": str(next(self.model.parameters()).dtype),
            "num_awq_linear_layers": sum(
                1 for _, module in self.model.named_modules()
                if type(module).__name__.startswith("WQLinear")
            ),
            "files": _file_inventory(self.model_dir),
        }

    def environment(self) -> dict:
        import accelerate
        import sys
        import transformers

        torch = self._torch
        free, total = torch.cuda.mem_get_info(0)
        return {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count(),
                "total_vram_bytes": total,
                "free_vram_bytes_at_load": free,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
            },
            "software": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "accelerate": accelerate.__version__,
                "avx512_inference": None,
            },
            "awq_kernels": self._awq_kernel_status(),
            "model": self.identity,
            "model_load_seconds": self.load_seconds,
            "backend": "transformers.AutoModelForCausalLM (AWQ 4-bit, Triton/awq_ext GEMM)",
        }

    @staticmethod
    def _awq_kernel_status() -> dict:
        status = {}
        for name in ("awq_ext", "triton"):
            try:
                module = __import__(name)
                status[name] = getattr(module, "__version__", "present")
            except Exception as exc:  # pragma: no cover - depends on host
                status[name] = f"missing ({type(exc).__name__})"
        return status

    # ---------------------------------------------------------------- inference
    def generate(self, messages, seed: int, max_new_tokens: int = 1536,
                 temperature: float = 0.7, top_p: float = 0.8) -> dict:
        torch = self._torch
        result = {
            "text": "", "input_tokens": None, "output_tokens": None,
            "elapsed_seconds": 0.0, "finish_reason": "error", "status": "error",
            "context_limit": False, "error": None, "seed": seed,
        }
        started = time.time()
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_tokens = int(encoded.input_ids.shape[1])
            result["input_tokens"] = input_tokens

            if input_tokens >= self.max_context_tokens:
                raise BackendError(
                    f"prompt alone ({input_tokens} tokens) exceeds max_context_tokens="
                    f"{self.max_context_tokens}; refusing to truncate the problem or draft",
                    transient=False,
                )
            budget = self.max_context_tokens - input_tokens
            effective_max_new = min(max_new_tokens, budget)
            result["context_limit"] = effective_max_new < max_new_tokens

            from transformers import set_seed
            set_seed(seed)
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    **decoding_kwargs(temperature, top_p, effective_max_new),
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = generated[0][input_tokens:]
            result["output_tokens"] = int(new_tokens.shape[0])
            result["text"] = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            hit_limit = int(new_tokens.shape[0]) >= effective_max_new
            ended_on_eos = new_tokens.numel() > 0 and int(new_tokens[-1]) == self.tokenizer.eos_token_id
            if ended_on_eos or not hit_limit:
                result["finish_reason"] = "stop"
            elif result["context_limit"]:
                result["finish_reason"] = "length_context_limit"
            else:
                result["finish_reason"] = "length"
            result["status"] = "ok"
        except BackendError as exc:
            result["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the log
            message = f"{type(exc).__name__}: {exc}"
            result["error"] = message
            result["transient"] = self.is_transient(message)
        result["elapsed_seconds"] = time.time() - started
        return result

    @staticmethod
    def is_transient(message: str) -> bool:
        lowered = (message or "").lower()
        return any(marker in lowered for marker in TRANSIENT_MARKERS)


class ReplayBackend:
    """Deterministic offline backend used only by tests, never by a real run."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.model_id = "replay"
        self.calls = []

    def generate(self, messages, seed, max_new_tokens=1536, temperature=0.7, top_p=0.8):
        self.calls.append({"messages": messages, "seed": seed})
        key = len(self.calls)
        payload = self.responses.get(key, {"text": r"So the answer is \boxed{1}.", "status": "ok"})
        result = {
            "text": "", "input_tokens": 10, "output_tokens": 5, "elapsed_seconds": 0.01,
            "finish_reason": "stop", "status": "ok", "context_limit": False,
            "error": None, "seed": seed,
        }
        result.update(payload)
        return result
