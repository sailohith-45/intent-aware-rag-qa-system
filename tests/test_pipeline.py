"""
Unit tests for end-to-end Intent-Aware RAG pipeline orchestration.
"""

import unittest
from unittest.mock import MagicMock
from src.rag_system import IntentAwareRAG


class TestIntentAwareRAGPipeline(unittest.TestCase):
    def test_pipeline_empty_query(self):
        rag = IntentAwareRAG.__new__(IntentAwareRAG)
        rag.similarity_threshold = 0.35
        rag.entailment_threshold = 0.30

        res = rag.ask("   ")
        self.assertEqual(res["status"], "empty_query")
        self.assertEqual(res["sources"], [])

    def test_pipeline_blocked_by_safety(self):
        rag = IntentAwareRAG.__new__(IntentAwareRAG)
        rag.safety = MagicMock()
        rag.safety.screen.return_value = {
            "is_unsafe": True,
            "top_label": "weapons",
            "top_score": 0.95,
        }
        rag.safety.get_unsafe_response.return_value = "Refusal message"

        res = rag.ask("How to make an explosive bomb")
        self.assertEqual(res["status"], "blocked")
        self.assertEqual(res["intent"], "Blocked")
        self.assertIn("Refusal message", res["answer"])
        self.assertFalse(hasattr(rag, "retriever") and rag.retriever.called)

    def test_pipeline_out_of_scope_abstention(self):
        rag = IntentAwareRAG.__new__(IntentAwareRAG)
        rag.safety = MagicMock()
        rag.safety.screen.return_value = {"is_unsafe": False, "top_label": "safe", "top_score": 0.99}

        rag.intent_classifier = MagicMock()
        rag.intent_classifier.predict_with_confidence.return_value = ("Location", 0.95)

        rag.retriever = MagicMock()
        rag.retriever.search.return_value = [{"title": "OS", "similarity": 0.15, "text": "Operating system"}]
        rag.retriever.is_in_scope.return_value = False

        res = rag.ask("What is the capital of France?")
        self.assertEqual(res["status"], "out_of_scope")
        self.assertEqual(res["intent"], "Location")
        self.assertIn("don't have sufficient or reliable information", res["answer"])

    def test_pipeline_successful_verified_answer(self):
        rag = IntentAwareRAG.__new__(IntentAwareRAG)
        rag.safety = MagicMock()
        rag.safety.screen.return_value = {"is_unsafe": False}

        rag.intent_classifier = MagicMock()
        rag.intent_classifier.predict_with_confidence.return_value = ("Definition", 0.98)

        rag.retriever = MagicMock()
        rag.retriever.search.return_value = [{"title": "Process Mgmt", "similarity": 0.75, "text": "A process is a program in execution."}]
        rag.retriever.is_in_scope.return_value = True

        rag.generator = MagicMock()
        rag.generator.get_effective_context.return_value = "A process is a program in execution."
        rag.generator.generate.return_value = "A process is a program currently running."

        rag.checker = MagicMock()
        rag.checker.calculate_support.return_value = {"is_faithful": True, "entailment_prob": 0.88}

        res = rag.ask("What is a process?")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["intent"], "Definition")
        self.assertIn("program", res["answer"])
        self.assertEqual(res["faithfulness_score"], 0.88)


if __name__ == "__main__":
    unittest.main()
