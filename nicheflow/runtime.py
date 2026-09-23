"""The same graph executor serves fixtures, component probes and adaptive runs."""
from __future__ import annotations
import json
from .graph import LLM_OPERATORS, graph_messages, proposal_messages, parse_proposal
from .scoring import evaluate
from .spec import BudgetStop, EnvironmentBlocked, ExecutionStop, digest
from .sandbox import extract_code
from .workflow_contracts import LEGACY_POLICY, PRESERVE_POLICY, validate_output_policy, format_output


class GraphExecutor:
    def __init__(self, journal, backends, decode, sandbox=None, official_adapters=None, failure_limit=None, allowed_operators=None, workflow_output_policy=LEGACY_POLICY):
        self.journal, self.backends, self.decode, self.sandbox = journal, backends, decode, sandbox
        self.official_adapters = official_adapters or {}
        self.failure_limit = failure_limit
        self.allowed_operators = allowed_operators
        self.length_limit_as_zero = False
        self.workflow_output_policy = validate_output_policy(workflow_output_policy)

    def scored_length_failure(self, result):
        return self.length_limit_as_zero and result["status"] == "truncated" and result.get("error") == "length"

    def assessment_complete(self, result):
        return result["status"] == "ok" or self.scored_length_failure(result)

    def assessment_quality(self, result):
        return 0. if self.scored_length_failure(result) else result["evaluation"]["quality"]

    def check_failures(self):
        if self.failure_limit is None:
            return
        streak = 0
        for event in reversed(self.journal.events):
            if event["kind"] == "execution":
                if self.assessment_complete(event["payload"]):
                    break
                streak += 1
        if streak >= self.failure_limit:
            raise ExecutionStop(f"{streak} consecutive execution failures; no automatic retry")

    def remaining_calls(self, graph, execution_id):
        if self.journal.lookup("execution", execution_id) is not None:
            return 0
        return sum(self.journal.lookup("call_finished", execution_id + ":" + n.id) is None
                   for n in graph.nodes if n.operator in LLM_OPERATORS)

    def execute(self, graph, task, execution_id):
        graph.validate(self.backends)
        existing = self.journal.lookup("execution", execution_id)
        contract = digest([graph.version, task.record()])
        if self.workflow_output_policy != LEGACY_POLICY:
            contract = digest([graph.version, task.record(), self.workflow_output_policy])
        if existing is not None:
            if existing["contract"] != contract:
                from .spec import IntegrityError
                raise IntegrityError("execution id reused for a different graph/task")
            return existing
        self.check_failures()
        if task.attachments:
            raise EnvironmentBlocked("attachment handling is not provisioned for this task")
        if task.evaluator == "travelplanner_official":
            adapter = self.official_adapters.get(task.evaluator)
            if adapter is None or not adapter.readiness()["ready"]:
                raise EnvironmentBlocked("official TravelPlanner environment required before model calls")
        needs_sandbox = task.evaluator == "python_tests" or any(n.operator in {"Test", "Code"} for n in graph.nodes)
        if needs_sandbox and (self.sandbox is None or not self.sandbox.readiness()["ready"]):
            raise EnvironmentBlocked("isolated code execution required before spending model calls")
        completed = sum(self.journal.lookup("call_finished", execution_id + ":" + n.id) is not None
                        for n in graph.nodes if n.operator in LLM_OPERATORS)
        self.journal.reserve(graph.model_calls - completed)
        self.journal.append("workflow", graph.version, graph.definition())
        outputs, calls, status, error = {}, [], "ok", None
        for n in graph.order():
            if self.journal.remaining_seconds() <= 0:
                raise BudgetStop("wall-clock limit before graph node")
            if n.operator in LLM_OPERATORS:
                params = dict(self.decode)
                params["seed"] = int(digest([execution_id, n.id, params.pop("seed")])[:8], 16) % (2 ** 31)
                if n.temperature is not None:
                    params["temperature"] = n.temperature
                call_id = execution_id + ":" + n.id
                result = self.journal.call(call_id, self.backends[n.model], graph_messages(n, task.model_input(), outputs, self.workflow_output_policy), params)
                calls.append(call_id)
                if result["status"] != "ok":
                    status, error = "execution_failed", result.get("error")
                    break
                outputs[n.id] = result["text"]
                if result.get("finish_reason") != "stop":
                    status, error = "truncated", result.get("finish_reason")
                    break
                effective, decision = format_output(n, task.model_input(), outputs, result["text"], self.workflow_output_policy)
                if decision is not None:
                    self.journal.append("format_output_decision", call_id, decision)
                outputs[n.id] = effective
                if n.operator == "Review&Revise" and self.workflow_output_policy == PRESERVE_POLICY:
                    from nicheflow_probe.evaluation import extract_boxed
                    upstream_answer = extract_boxed(outputs[n.inputs[0]])
                    reviewed_answer = extract_boxed(effective)
                    self.journal.append("review_answer_observation", call_id, {
                        "policy": self.workflow_output_policy, "upstream_answer": upstream_answer,
                        "reviewed_answer": reviewed_answer, "answer_changed": upstream_answer != reviewed_answer,
                        "correctness_assessed": False})
            else:
                node_id = execution_id + ":" + n.id
                deterministic = self.journal.lookup("code_node", node_id)
                if deterministic is None:
                    try:
                        if n.operator == "Identity":
                            value = outputs[n.inputs[0]]
                        elif n.operator == "Join":
                            value = json.dumps({i: outputs[i] for i in n.inputs})
                        elif n.operator == "Code":
                            value = json.dumps(self.sandbox.transform(n.source, {
                                "task": task.model_input(), "inputs": {i: outputs[i] for i in n.inputs}}))
                        else:
                            if not task.public_tests:
                                raise ValueError("Test operator needs permitted public tests")
                            value = json.dumps(self.sandbox.run(extract_code(outputs[n.inputs[0]]), task.public_tests))
                        deterministic = {"status": "ok", "output": value}
                    except Exception as exc:
                        deterministic = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
                    self.journal.append("code_node", node_id, deterministic)
                if deterministic["status"] != "ok":
                    status, error = "execution_failed", deterministic["error"]
                    break
                outputs[n.id] = deterministic["output"]
        final = outputs.get(graph.output) if status == "ok" else None
        score = None
        if final is not None:
            try:
                score = evaluate(task, final, self.sandbox, self.official_adapters)
            except EnvironmentBlocked as exc:
                status, error = "evaluation_blocked", str(exc)
            except Exception as exc:
                status, error = "evaluation_error", f"{type(exc).__name__}: {exc}"
        record = {"contract": contract, "task_id": task.id, "workflow": graph.version,
                  "role": task.role, "status": status, "error": error, "calls": calls,
                  "final_output": final, "evaluation": score}
        return self.journal.append("execution", execution_id, record)

    def generate_candidate(self, parents, attempt_id, model="local", max_nodes=12, feedback=None):
        contract = digest({"parents": [p.version for p in parents], "model": model,
                           "backend": self.backends[model].model_id, "decode": self.decode,
                           "max_nodes": max_nodes, "allowed_models": list(self.backends),
                           "allowed_operators": self.allowed_operators, "feedback": feedback})
        existing = self.journal.lookup("generation", attempt_id)
        if existing is not None:
            if existing.get("contract") != contract:
                from .spec import IntegrityError
                raise IntegrityError("generation id reused with a changed parent/model/protocol")
            return existing
        self.check_failures()
        if self.journal.lookup("call_started", attempt_id + ":generate") is None:
            self.journal.reserve(1)
        payload = proposal_messages(parents, self.backends, max_nodes, self.allowed_operators, feedback=feedback)
        params = dict(self.decode)
        params["seed"] = int(digest([attempt_id, params.pop("seed")])[:8], 16) % (2 ** 31)
        result = self.journal.call(attempt_id + ":generate", self.backends[model], payload, params)
        graph, error = None, None
        if result["status"] == "ok" and result.get("finish_reason") == "stop":
            try:
                graph = parse_proposal(result["text"], parents, self.backends)
                graph.validate(self.backends, max_nodes=max_nodes)
                if self.allowed_operators is not None and any(n.operator not in self.allowed_operators for n in graph.nodes):
                    raise ValueError("operator outside the frozen smoke grammar")
                if feedback is not None:
                    from .mutations import validate_mutation
                    mutation = validate_mutation(parents, graph)
            except (ValueError, KeyError, TypeError) as exc:
                graph = None
                error = str(exc)
        else:
            error = result.get("error") or result.get("finish_reason")
        record = {"contract": contract, "parents": [p.version for p in parents], "call_id": attempt_id + ":generate",
                  "status": "valid" if graph else "invalid", "error": error,
                  "graph": graph.definition() if graph else None,
                  "version": graph.version if graph else None}
        if feedback is not None:
            record.update(feedback=feedback, mutation=mutation if graph else None)
        return self.journal.append("generation", attempt_id, record)
