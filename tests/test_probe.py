import unittest
from fractions import Fraction
from nicheflow_probe.evaluation import extract_boxed, rational, score
from nicheflow_probe.workflows import WORKFLOWS, messages


class EvaluationTests(unittest.TestCase):
    def test_nested_and_last_box(self):
        text = r"Old: \boxed{9}; corrected: \boxed{\frac{1}{2}}"
        self.assertEqual(extract_boxed(text), r"\frac{1}{2}")
        self.assertTrue(score(text, "0.5")["correct"])

    def test_safe_parser(self):
        for expr in ["__import__('os').system('id')", "2**999999", "1/0", "True", "sqrt(4)"]:
            self.assertIsNone(rational(expr))
        self.assertEqual(rational(r"-\frac{3}{4}"), Fraction(-3, 4))

    def test_no_answer_guess_from_reasoning(self):
        self.assertFalse(score("We obtained 42 but could not finish.", "42")["parsed"])
        self.assertFalse(score(r"\boxed{\sqrt{2}}", "2")["correct"])

    def test_prompt_boundary_and_dependencies(self):
        for workflow in WORKFLOWS:
            outputs = {}
            for node in workflow.nodes:
                payload = messages(node, "Public problem text", outputs)
                self.assertIn("Public problem text", payload[1]["content"])
                self.assertNotIn("reference_solution", str(payload))
                outputs[node.id] = "A draft"
        revise = WORKFLOWS[-1].nodes[-1]
        self.assertIn("A draft", messages(revise, "Problem", {"draft": "A draft"})[1]["content"])


if __name__ == "__main__":
    unittest.main()
