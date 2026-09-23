"""Code execution is only available inside an explicitly provisioned container."""
import json
import re
import shutil
import subprocess
import tempfile
import uuid
import selectors
import os
import time
from .spec import EnvironmentBlocked


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else text.strip()


class PythonSandbox:
    def __init__(self, image=None, timeout=10):
        self.image, self.timeout = image, timeout

    def readiness(self):
        if not self.image or not re.fullmatch(r"[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}", self.image):
            return {"ready": False, "reason": "pinned Docker image digest required"}
        if not shutil.which("docker"):
            return {"ready": False, "reason": "Docker executable unavailable"}
        result = subprocess.run(["docker", "image", "inspect", self.image], capture_output=True, timeout=10)
        return {"ready": result.returncode == 0, "reason": "local image inspection; no automatic pull"}

    def run(self, code, tests, setup=""):
        status = self.readiness()
        if not status["ready"]:
            raise EnvironmentBlocked(status["reason"])
        script = setup + "\n" + code + "\n" + "\n".join(tests)
        name = "nicheflow-" + uuid.uuid4().hex
        command = ["docker", "run", "--name", name, "--rm", "--pull=never", "--network=none", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=32",
                   "--memory=256m", "--cpus=1", "--user=65534:65534", "--tmpfs=/tmp:rw,noexec,nosuid,size=16m",
                   "-i", self.image, "python", "-I", "-c", script]
        # No host mounts, environment secrets or eval() in the parent process.
        try:
            p = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=self.timeout)
            if p.returncode in {125, 126, 127}:
                raise EnvironmentBlocked("container runtime/image could not execute Python")
            return {"passed": p.returncode == 0, "status": "ok", "exit_code": p.returncode}
        except subprocess.TimeoutExpired:
            return {"passed": False, "status": "timeout"}
        finally:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)

    def transform(self, source, payload):
        status = self.readiness()
        if not status["ready"]:
            raise EnvironmentBlocked(status["reason"])
        name = "nicheflow-" + uuid.uuid4().hex
        command = ["docker", "run", "--name", name, "--rm", "--pull=never", "--network=none", "--read-only",
                   "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=32", "--memory=256m",
                   "--cpus=1", "--user=65534:65534", "--ulimit=fsize=1048576:1048576", "-i", self.image,
                   "python", "-I", "-c", source]
        try:
            with tempfile.TemporaryFile() as stdin:
                stdin.write(json.dumps(payload).encode()); stdin.seek(0)
                p = subprocess.Popen(command, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                raw = bytearray()
                deadline = time.monotonic() + self.timeout
                try:
                    with selectors.DefaultSelector() as selector:
                        selector.register(p.stdout, selectors.EVENT_READ)
                        while True:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0:
                                raise TimeoutError("code node timeout")
                            if not selector.select(min(remaining, .1)):
                                continue
                            chunk = os.read(p.stdout.fileno(), 65536)
                            if not chunk:
                                break
                            raw.extend(chunk)
                            if len(raw) > 1048576:
                                raise ValueError("code output exceeded limit")
                    p.wait(timeout=max(.01, deadline-time.monotonic()))
                    if p.returncode in {125, 126, 127}:
                        raise EnvironmentBlocked("container runtime/image could not execute Python")
                    if p.returncode:
                        raise ValueError("deterministic code node failed")
                    return json.loads(raw)
                finally:
                    if p.poll() is None:
                        p.kill(); p.wait(timeout=5)
                    p.stdout.close()
        finally:
            subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)
