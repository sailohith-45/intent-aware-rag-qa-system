"""
Unit tests for Faithfulness Checker component.
"""

import unittest
from unittest.mock import MagicMock
import torch

from src.faithfulness import FaithfulnessChecker


class TestFaithfulnessChecker(unittest.TestCase):
    def test_formulate_hypothesis_for_short_answers(self):
        checker = FaithfulnessChecker.__new__(FaithfulnessChecker)

        # Short answers / fragments must be expanded to complete propositions
        q1 = "What is the complexity?"
        a1 = "O(log n)"
        h1 = checker.formulate_hypothesis(q1, a1)
        self.assertEqual(h1, "The answer to the question 'What is the complexity?' is O(log n).")

        # Long answers are retained directly
        q2 = "What is a process?"
        a2 = "A process is a program in execution containing code and memory state."
        h2 = checker.formulate_hypothesis(q2, a2)
        self.assertEqual(h2, a2)

    def test_faithfulness_calculation_with_mock(self):
        checker = FaithfulnessChecker.__new__(FaithfulnessChecker)
        checker.threshold = 0.30
        checker.entailment_idx = 2
        checker.neutral_idx = 1
        checker.contradiction_idx = 0

        mock_model = MagicMock()
        mock_model.device = "cpu"
        # Entailment logits: [contradiction, neutral, entailment]
        mock_model.return_value.logits = torch.tensor([[0.1, 0.2, 0.7]])
        checker.model = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        checker.tokenizer = mock_tokenizer

        res = checker.calculate_support("Context text", "Supported answer", "Question?")
        self.assertTrue(res["is_faithful"])
        self.assertGreater(res["entailment_prob"], 0.30)


if __name__ == "__main__":
    unittest.main()
