"""
Unit tests for FAISS Retriever component.
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.retriever import Retriever


class DummyEmbedder:
    """Mock embedder providing deterministic normalized 4-dim embeddings."""
    def encode(self, texts, **kwargs):
        vectors = []
        for t in texts:
            words = t.lower().split()
            v = np.array([len(words), "cpu" in t.lower(), "memory" in t.lower(), 1.0], dtype="float32")
            norm = np.linalg.norm(v)
            vectors.append(v / (norm + 1e-8))
        return np.array(vectors, dtype="float32")


class TestRetriever(unittest.TestCase):
    def test_retriever_build_search_save_load(self):
        embedder = DummyEmbedder()
        retriever = Retriever(embedder=embedder, similarity_threshold=0.30)

        data = pd.DataFrame([
            {"id": "doc1", "title": "Process Mgmt", "topic": "OS", "text": "A process executes on the CPU.", "chunk_id": 1},
            {"id": "doc2", "title": "Virtual Memory", "topic": "OS", "text": "Virtual memory abstracts physical RAM.", "chunk_id": 2},
        ])

        retriever.build_index_from_dataframe(data, text_column="text", show_progress=False)
        self.assertIsNotNone(retriever.index)
        self.assertEqual(retriever.index.ntotal, 2)

        # Query search
        res = retriever.search("What runs on CPU?", k=2)
        self.assertEqual(len(res), 2)
        self.assertIn("similarity", res[0])
        self.assertIn(res[0]["title"], ("Process Mgmt", "Virtual Memory"))

        # In-scope check
        self.assertTrue(retriever.is_in_scope(res))

        # Test persistence
        with tempfile.TemporaryDirectory() as tmpdir:
            retriever.save(tmpdir)
            loaded = Retriever(embedder=embedder)
            loaded.load(tmpdir)
            self.assertEqual(loaded.index.ntotal, 2)
            loaded_res = loaded.search("What runs on CPU?", k=1)
            self.assertEqual(len(loaded_res), 1)
            self.assertEqual(loaded_res[0]["id"], res[0]["id"])

    def test_retriever_empty_input_handling(self):
        embedder = DummyEmbedder()
        retriever = Retriever(embedder=embedder)
        empty_df = pd.DataFrame({"text": ["", "   "]})
        with self.assertRaises(ValueError):
            retriever.build_index_from_dataframe(empty_df, text_column="text")


if __name__ == "__main__":
    unittest.main()
