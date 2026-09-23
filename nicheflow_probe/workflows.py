"""Explicit node graphs using Generate and Review & Revise from the proposal."""
from dataclasses import dataclass, asdict

SYSTEM = "You solve mathematics problems. Follow the requested answer format."
FORMAT = r"Put your final answer inside \boxed{...}."


@dataclass(frozen=True)
class Node:
    id: str
    operator: str
    prompt: str
    inputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Workflow:
    id: str
    description: str
    nodes: tuple[Node, ...]

    def definition(self):
        return asdict(self)


WORKFLOWS = (
    Workflow("direct", "Generate: concise direct solution", (
        Node("answer", "Generate", "Solve the following problem. Be concise.\n" + FORMAT),)),
    Workflow("reason", "Generate: explicit step-by-step solution", (
        Node("answer", "Generate", "Solve the following problem step by step. Check your arithmetic.\n" + FORMAT),)),
    Workflow("revise", "Generate followed by Review & Revise", (
        Node("draft", "Generate", "Solve the following problem step by step.\n" + FORMAT),
        Node("answer", "Review & Revise", "Independently check the draft solution for mathematical mistakes and missing cases. Correct it if necessary and give the final solution.\n" + FORMAT, ("draft",)))),
)


def messages(node: Node, question: str, outputs: dict[str, str]) -> list[dict]:
    # No reference solution, quality label, or dataset level is passed to the model.
    content = node.prompt + "\n\nProblem:\n" + question
    for parent in node.inputs:
        content += "\n\nDraft solution:\n" + outputs[parent]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]

