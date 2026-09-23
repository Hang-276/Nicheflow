"""Local GPU worker with a parent-enforced time limit; no dependency changes."""
import multiprocessing as mp
import time
from .spec import EnvironmentBlocked


def _worker(connection, model_path, context_tokens):
    try:
        from nicheflow_probe.backends import TransformersChatBackend
        backend = TransformersChatBackend(model_path, max_context_tokens=context_tokens)
        connection.send({"loaded": True, "environment": backend.environment()})
        while True:
            message = connection.recv()
            if message is None:
                break
            messages, params = message
            connection.send(backend.generate(messages, **params))
    except BaseException as exc:
        try:
            connection.send({"loaded": False, "error": f"{type(exc).__name__}: {exc}"})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        connection.close()


class LocalWorker:
    def __init__(self, model_path, context_tokens=8192, load_timeout=180, call_timeout=180):
        self.model_id, self.call_timeout = model_path, call_timeout
        ctx = mp.get_context("spawn")
        self.connection, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, model_path, context_tokens))
        self.process.start()
        child.close()
        if not self.connection.poll(load_timeout):
            self.close(graceful=False)
            raise EnvironmentBlocked("model loading timed out")
        try:
            response = self.connection.recv()
        except EOFError:
            self.close(graceful=False)
            raise EnvironmentBlocked("model worker exited while loading") from None
        if not response.get("loaded"):
            self.close(graceful=False)
            raise EnvironmentBlocked(response.get("error"))
        self.environment = response["environment"]

    def generate(self, messages, **params):
        started = time.time()
        self.connection.send((messages, params))
        if not self.connection.poll(self.call_timeout):
            self.close(graceful=False)
            return {"status": "error", "finish_reason": "timeout", "text": "", "error": "worker terminated at time budget",
                    "input_tokens": None, "output_tokens": None, "elapsed_seconds": time.time() - started}
        return self.connection.recv()

    def close(self, graceful=True):
        if graceful and self.process.is_alive():
            try:
                self.connection.send(None)
                self.process.join(5)
            except (EOFError, BrokenPipeError, OSError):
                pass
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(5)
        self.connection.close()
