"""
Semantic Retrieval Component: FAISS Vector Index over Normalized Dense Embeddings.
Manages passage indexing, cosine similarity ranking, metadata preservation, and index persistence.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.config import ARTIFACTS_DIR, DEFAULT_SIMILARITY_THRESHOLD, EMBED_MODEL, KB_CSV


class Retriever:
    """Dense vector retriever using FAISS IndexFlatIP over unit-normalized embeddings."""

    def __init__(
        self,
        embedder: Optional[SentenceTransformer] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ):
        self.embedder = embedder or SentenceTransformer(EMBED_MODEL)
        self.similarity_threshold = similarity_threshold
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: List[Dict[str, Any]] = []
        self.embedding_dimension: int = 384

    def build_index_from_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        show_progress: bool = True,
    ) -> "Retriever":
        """Build FAISS index and preserve document metadata from a DataFrame."""
        if text_column not in df.columns:
            raise ValueError(f"DataFrame must contain '{text_column}' column.")

        # Filter out empty or null texts
        valid_df = df.dropna(subset=[text_column]).copy()
        valid_df[text_column] = valid_df[text_column].astype(str).str.strip()
        valid_df = valid_df[valid_df[text_column] != ""]

        # Drop exact duplicate texts to prevent redundant retrieval
        valid_df = valid_df.drop_duplicates(subset=[text_column]).reset_index(drop=True)

        if valid_df.empty:
            raise ValueError("Knowledge base DataFrame contains no valid text documents.")

        texts = valid_df[text_column].tolist()
        self.documents = valid_df.to_dict(orient="records")

        # Encode passages with L2 unit normalization
        embeddings = self.embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        ).astype("float32")

        self.embedding_dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.embedding_dimension)
        self.index.add(embeddings)
        return self

    def build_index_from_csv(
        self,
        csv_path: Union[str, Path] = KB_CSV,
        text_column: str = "text",
        show_progress: bool = True,
    ) -> "Retriever":
        """Load knowledge base from CSV and construct the vector index."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base CSV not found at {path}")
        df = pd.read_csv(path)
        return self.build_index_from_dataframe(df, text_column=text_column, show_progress=show_progress)

    def search(
        self,
        question: str,
        k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Retrieve the top-k most semantically relevant passages for a query."""
        if self.index is None or not self.documents:
            raise RuntimeError("Retriever index is not initialized or contains no documents.")

        query_text = str(question).strip()
        if not query_text:
            return []

        # Encode query with unit normalization for exact cosine similarity
        query_vector = self.embedder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc_record = self.documents[idx].copy()
                doc_record["similarity"] = float(score)
                results.append(doc_record)
        return results

    def is_in_scope(self, passages: List[Dict[str, Any]]) -> bool:
        """Verify whether top retrieved passage meets the minimum relevance threshold."""
        if not passages:
            return False
        return passages[0]["similarity"] >= self.similarity_threshold

    def save(self, directory: Optional[Union[str, Path]] = None) -> Path:
        """Persist FAISS vector index and document metadata to disk."""
        if self.index is None or not self.documents:
            raise RuntimeError("Cannot save an unbuilt index.")

        target_dir = Path(directory) if directory else ARTIFACTS_DIR
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "faiss_index.bin"
        metadata_file = target_dir / "kb_metadata.json"

        faiss.write_index(self.index, str(index_file))
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": self.documents,
                    "embedding_dimension": self.embedding_dimension,
                    "threshold": self.similarity_threshold,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return target_dir

    def load(self, directory: Optional[Union[str, Path]] = None) -> "Retriever":
        """Load persisted FAISS vector index and metadata into memory."""
        target_dir = Path(directory) if directory else ARTIFACTS_DIR
        index_file = target_dir / "faiss_index.bin"
        metadata_file = target_dir / "kb_metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"FAISS index or metadata missing in {target_dir}")

        self.index = faiss.read_index(str(index_file))
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.documents = meta["documents"]
            self.embedding_dimension = meta.get("embedding_dimension", self.index.d)
            self.similarity_threshold = meta.get("threshold", self.similarity_threshold)

        return self
