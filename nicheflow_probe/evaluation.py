"""Exact rational scoring without evaluating model-generated code.

The first probe deliberately includes only MATH questions with scalar rational
answers. Unsupported or missing predictions are separately marked unparseable.
"""
import ast
import re
from fractions import Fraction


def extract_boxed(text: str) -> str | None:
    starts = list(re.finditer(r"\\(?:boxed|fbox)\s*\{", text))
    for match in reversed(starts):
        depth = 1
        for pos in range(match.end(), len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[match.end():pos]
    return None


def rational(value: str) -> Fraction | None:
    value = value.strip().replace("$", "").replace("\\left", "").replace("\\right", "")
    value = value.replace("\\!", "").replace("\\,", "").replace(" ", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    for _ in range(6):
        new = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", value)
        if new == value:
            break
        value = new
    if len(value) > 160 or not re.fullmatch(r"[\d.()+*/\-]+", value):
        return None
    try:
        tree = ast.parse(value, mode="eval")

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                # Source text preserves exact decimal notation instead of binary rounding.
                return Fraction(ast.get_source_segment(value, node))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                a, b = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add): return a + b
                if isinstance(node.op, ast.Sub): return a - b
                if isinstance(node.op, ast.Mult): return a * b
                return a / b
            raise ValueError("unsupported expression")

        return visit(tree.body)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError, OverflowError):
        return None


def score(output: str, gold: str) -> dict:
    target = rational(gold)
    if target is None:
        raise ValueError("Dataset contains an unsupported reference answer")
    answer = extract_boxed(output)
    predicted = rational(answer) if answer is not None else None
    return {"answer": answer, "parsed": predicted is not None,
            "correct": predicted == target if predicted is not None else False}

