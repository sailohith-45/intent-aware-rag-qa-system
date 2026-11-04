"""
Unit tests for Intent Classifier component.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.config import INTENT_MAP
from src.intent_classifier import IntentClassifier, clean_text


class DummyEmbedder:
    """Fast deterministic embedder for unit tests."""
    def encode(self, texts, **kwargs):
        vectors = []
        for t in texts:
            l = len(t)
            v = np.array([l % 5, (l * 2) % 7, (l * 3) % 11, 1.0], dtype="float32")
            norm = np.linalg.norm(v)
            vectors.append(v / (norm + 1e-8))
        return np.array(vectors)


class TestIntentClassifier(unittest.TestCase):
    def test_clean_text(self):
        raw = "   What   is   VIRTUAL    Memory?? \n\t "
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "what is virtual memory??")

    def test_intent_map_completeness(self):
        self.assertEqual(len(INTENT_MAP), 10)
        self.assertIn("DESC:definition", INTENT_MAP)
        self.assertIn("DESC:comparison", INTENT_MAP)
        self.assertEqual(INTENT_MAP["ABBR"], "Abbreviation")

    def test_classifier_fit_predict_save_load(self):
        embedder = DummyEmbedder()
        clf = IntentClassifier(embedder=embedder)

        train_questions = [
            "What is an operating system?",
            "How does binary search work?",
            "What is the difference between TCP and UDP?",
            "What does CPU stand for?",
        ]
        train_labels = ["DESC:definition", "DESC:procedure", "DESC:comparison", "ABBR"]

        clf.fit(train_questions, train_labels, show_progress=False)
        self.assertTrue(clf.is_fitted)

        pred = clf.predict("What is an operating system?")
        self.assertIn(pred, INTENT_MAP.values())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir) / "test_clf.joblib"
            saved = clf.save(tmp_path)
            self.assertTrue(saved.exists())

            new_clf = IntentClassifier(embedder=embedder)
            new_clf.load(tmp_path)
            self.assertTrue(new_clf.is_fitted)
            self.assertEqual(new_clf.predict("What is an operating system?"), pred)


if __name__ == "__main__":
    unittest.main()
