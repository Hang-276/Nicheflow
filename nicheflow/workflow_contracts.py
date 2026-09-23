"""Explicit, opt-in workflow output semantics; never consult reference answers."""
from nicheflow_probe.evaluation import extract_boxed

LEGACY_POLICY = "legacy"
PRESERVE_POLICY = "preserve_boxed_v1"


def validate_output_policy(policy):
    if policy not in {LEGACY_POLICY, PRESERVE_POLICY}:
        raise ValueError("unknown workflow output policy")
    return policy


def operator_instruction(operator, policy):
    validate_output_policy(policy)
    if policy == LEGACY_POLICY:
        return ""
    if operator == "Format":
        return ("You only format the supplied answer; do not solve the task again or revise its value. "
                "If the upstream inputs have the same final boxed answer, copy its contents exactly, "
                "including all LaTeX commands, braces, signs and units. Do not simplify or strip text commands. "
                "If inputs disagree, do not claim they agree.")
    if operator == "Review&Revise":
        return ("Verify the upstream reasoning independently. A revision is not required. "
                "Before changing its final answer, identify a specific error and verify the replacement "
                "against the original problem, for example by substitution or a separate calculation. "
                "If you cannot establish an error, preserve the existing final boxed answer exactly. "
                "Do not rewrite a correct solution merely to make a revision.")
    return ""


def format_output(node, public_input, outputs, raw_text, policy):
    """Preserve an unambiguous upstream boxed value, including an incorrect one.

    Format remains a paid LLM call. The caller records both its original receipt
    and this explicit output decision. No correction uses gold or task metadata.
    """
    validate_output_policy(policy)
    if policy == LEGACY_POLICY or node.operator != "Format":
        return raw_text, None
    audit = {"policy": policy, "inputs": list(node.inputs), "applied": False}
    if "\\boxed" not in public_input.get("output_contract", ""):
        return raw_text, {**audit, "reason": "non_boxed_output_contract"}
    answers = [extract_boxed(outputs[i]) for i in node.inputs]
    if not answers or any(a is None or not a.strip() for a in answers):
        return raw_text, {**audit, "reason": "upstream_answer_missing"}
    if len(set(answers)) != 1:
        return raw_text, {**audit, "reason": "upstream_answers_disagree"}
    original = answers[0]
    formatted = extract_boxed(raw_text)
    audit.update(upstream_answer=original, model_answer=formatted)
    if formatted == original:
        return raw_text, {**audit, "reason": "answer_preserved"}
    effective = "\\boxed{" + original + "}"
    return effective, {**audit, "applied": True, "reason": "restored_upstream_answer",
                       "effective_output": effective}
