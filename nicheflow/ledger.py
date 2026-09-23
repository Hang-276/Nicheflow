"""Durable single-writer journal. A started, uncompleted call is never reissued."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import time
from .spec import BudgetStop, IntegrityError, digest


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Journal:
    def __init__(self, directory, frozen_config, max_calls, seconds, clock=time.time):
        self.directory, self.clock = Path(directory), clock
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise IntegrityError("another writer is running") from None
        try:
            if max_calls < 1 or seconds <= 0:
                raise ValueError("positive budget required")
            self.path = self.directory / "events.jsonl"
            self.events = self.read(self.path)
            self._index = {(e["kind"], e["id"]): e["payload"] for e in self.events}
            if len(self._index) != len(self.events):
                raise IntegrityError("duplicate event identity in journal")
            self._attempts = sum(e["kind"] == "call_started" for e in self.events)
            contract = {"config": frozen_config, "max_calls": max_calls, "seconds": seconds}
            path = self.directory / "config.json"
            if path.exists():
                old = json.loads(path.read_text())
                if old["fingerprint"] != digest(contract):
                    raise IntegrityError("run configuration/code/data changed; resume refused")
                self.started = old["started"]
            else:
                if self.events:
                    raise IntegrityError("journal exists without frozen config")
                self.started = clock()
                atomic_json(path, {**contract, "fingerprint": digest(contract), "started": self.started})
            self.max_calls, self.seconds = max_calls, seconds
        except BaseException:
            self.close()
            raise

    @staticmethod
    def read(path):
        path = Path(path)
        rows, previous = [], "0" * 64
        if not path.exists():
            return rows
        for line in path.read_text().splitlines():
            if not line.strip():
                raise IntegrityError("blank/partial journal entry; manual recovery required")
            try:
                r = json.loads(line)
                content = {k: v for k, v in r.items() if k != "hash"}
                if r["previous"] != previous or r["seq"] != len(rows) or digest(content) != r["hash"]:
                    raise IntegrityError("journal chain mismatch")
            except (ValueError, KeyError) as exc:
                raise IntegrityError("corrupt/partial journal; refusing automatic replay") from exc
            previous = r["hash"]
            rows.append(r)
        return rows

    def close(self):
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def append(self, kind, id, payload):
        # Hash exactly the JSON value that will be read back. Integer mapping
        # keys sort differently before/after JSON conversion (e.g. 3 vs 20).
        # This also detaches historical snapshots from mutable learning state.
        payload = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        existing = self.lookup(kind, id)
        if existing is not None:
            if digest(existing) != digest(payload):
                raise IntegrityError(f"conflicting {kind} event {id}")
            return existing
        event = {"seq": len(self.events), "previous": self.events[-1]["hash"] if self.events else "0" * 64,
                 "kind": kind, "id": id, "payload": payload, "time": self.clock()}
        event["hash"] = digest(event)
        with self.path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.events.append(event)
        self._index[(kind, id)] = payload
        self._attempts += kind == "call_started"
        return payload

    def lookup(self, kind, id):
        return self._index.get((kind, id))

    @property
    def attempts(self):
        return self._attempts

    def remaining_seconds(self):
        return self.seconds - (self.clock() - self.started)

    def reserve(self, count):
        if self.attempts + count > self.max_calls:
            raise BudgetStop("insufficient call budget for entire operation")
        if self.remaining_seconds() <= 0:
            raise BudgetStop("wall-clock limit")

    def call(self, id, backend, messages, params):
        request = {"messages": messages, "params": params, "model": backend.model_id}
        result = self.lookup("call_finished", id)
        start = self.lookup("call_started", id)
        if start is not None:
            if digest(start) != digest(request):
                raise IntegrityError("call id reused with changed input")
            if result is None:
                raise IntegrityError("previous call outcome is unknown; refusing a duplicate paid call")
            return result
        self.reserve(1)
        self.append("call_started", id, request)
        started = self.clock()
        try:
            if hasattr(backend, "call_timeout"):
                backend.call_timeout = min(backend.call_timeout, max(.01, self.remaining_seconds()))
            result = backend.generate(messages, **params)
        except Exception as exc:
            result = {"status": "error", "finish_reason": "error", "text": "",
                      "error": f"{type(exc).__name__}: {exc}", "input_tokens": None,
                      "output_tokens": None, "elapsed_seconds": self.clock() - started}
        result = {**result, "call_id": id}
        return self.append("call_finished", id, result)

    def audit(self):
        starts = {e["id"] for e in self.events if e["kind"] == "call_started"}
        ends = {e["id"] for e in self.events if e["kind"] == "call_finished"}
        if not ends <= starts or self.attempts > self.max_calls:
            raise IntegrityError("budget journal invariant failed")
        results = [e["payload"] for e in self.events if e["kind"] == "call_finished"]
        return {"attempts": len(starts), "completed_calls": len(ends), "unknown_calls": sorted(starts - ends),
                "input_tokens": sum(r.get("input_tokens") or 0 for r in results),
                "output_tokens": sum(r.get("output_tokens") or 0 for r in results),
                "unknown_token_calls": sum(r.get("output_tokens") is None for r in results),
                "remaining_calls": self.max_calls - len(starts), "hash_chain_valid": True}
