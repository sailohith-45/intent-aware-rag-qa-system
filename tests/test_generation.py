"""
Unit tests for Generator prompt and context management.
"""

import unittest
from unittest.mock import MagicMock
from src.generator import Generator


class MockTokenizer:
    """Mock tokenizer simulating truncation."""
    def __call__(self, text, truncation=True, max_length=350, **kwargs):
        words = text.split()
        truncated_words = words[:max_length]
        return {"input_ids": [101] * len(truncated_words)}

    def decode(self, token_ids, **kwargs):
        return f"decoded_{len(token_ids)}_tokens"


class TestGenerator(unittest.TestCase):
    def test_generator_prompt_and_intent_styling(self):
        gen = Generator.__new__(Generator)
        gen.tokenizer = MockTokenizer()
        gen.max_context_tokens = 350
        gen.max_new_tokens = 120

        question = "What is a process?"
        intent = "Definition"
        context = "A process is a program in execution."

        prompt = gen.construct_prompt(question, intent, context)
        self.assertIn("Give a clear, precise, and direct definition", prompt)
        self.assertIn(f"Question: {question}", prompt)
        self.assertIn(f"Context:\n{context}", prompt)

    def test_generator_effective_context_truncation(self):
        gen = Generator.__new__(Generator)
        gen.tokenizer = MockTokenizer()
        gen.max_context_tokens = 50

        passages = [
            {"text": "Passage 1 content about operating systems."},
            {"text": "Passage 2 content about thread scheduling."},
        ]
        eff_ctx = gen.get_effective_context(passages, num_passages=2)
        self.assertIn("decoded_", eff_ctx)


if __name__ == "__main__":
    unittest.main()
