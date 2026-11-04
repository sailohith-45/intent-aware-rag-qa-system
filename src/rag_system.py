"""
Main RAG Pipeline Orchestrator: IntentAwareRAG.
Coordinates safety screening, intent detection, intent-conditioned FAISS retrieval,
grounded FLAN-T5 answer generation, and NLI faithfulness verification.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_ENTAILMENT_THRESHOLD,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEVICE,
    EMBED_MODEL,
    INTENT_MAP,
    INTENT_TOP_K,
    KB_CSV,
    OUT_OF_SCOPE_MESSAGE,
    TEST_CSV,
    TRAIN_CSV,
    UNSAFE_MESSAGE,
    UNVERIFIED_MESSAGE,
)
from src.faithfulness import FaithfulnessChecker
from src.generator import Generator
from src.intent_classifier import IntentClassifier
from src.retriever import Retriever
from src.safety import SafetyGuard


class IntentAwareRAG:
    """End-to-end Intent-Aware Retrieval-Augmented Generation Question Answering System."""

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        entailment_threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
        safety_threshold: float = DEFAULT_SAFETY_THRESHOLD,
        rebuild_artifacts: bool = False,
    ):
        self.similarity_threshold = similarity_threshold
        self.entailment_threshold = entailment_threshold
        self.safety_threshold = safety_threshold

        print(f"[Init] Initializing Intent-Aware RAG System on device: {DEVICE}")

        # Shared embedding model for both intent classifier and retriever
        self.embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

        # 1. Safety Guard
        print("[Init] Loading SafetyGuard (BART-large-MNLI)...")
        self.safety = SafetyGuard(threshold=self.safety_threshold)

        # 2. Faithfulness Checker (shares the BART-MNLI model from safety to conserve VRAM/RAM)
        print("[Init] Initializing FaithfulnessChecker...")
        self.checker = FaithfulnessChecker(
            threshold=self.entailment_threshold,
            tokenizer=self.safety.classifier.tokenizer,
            model=self.safety.classifier.model,
        )

        # 3. Intent Classifier (with persistence support)
        classifier_artifact = ARTIFACTS_DIR / "intent_classifier.joblib"
        self.intent_classifier = IntentClassifier(embedder=self.embedder)

        if not rebuild_artifacts and classifier_artifact.exists():
            print(f"[Init] Loading existing intent classifier from {classifier_artifact}...")
            self.intent_classifier.load(classifier_artifact)
        else:
            print("[Init] Fitting intent classifier on training split...")
            if not TRAIN_CSV.exists() or not TEST_CSV.exists():
                raise FileNotFoundError(
                    f"Datasets {TRAIN_CSV} and {TEST_CSV} are required to train the classifier. "
                    "Run 'python prepare_data.py' first."
                )
            X_train, y_train = IntentClassifier.load_dataset(TRAIN_CSV)
            self.intent_classifier.fit(X_train, y_train, show_progress=False)
            self.intent_classifier.save(classifier_artifact)
            print(f"[Init] Classifier saved to {classifier_artifact}")

        # 4. Retriever (with FAISS persistence support)
        faiss_artifact = ARTIFACTS_DIR / "faiss_index.bin"
        metadata_artifact = ARTIFACTS_DIR / "kb_metadata.json"
        self.retriever = Retriever(
            embedder=self.embedder,
            similarity_threshold=self.similarity_threshold,
        )

        if not rebuild_artifacts and faiss_artifact.exists() and metadata_artifact.exists():
            print(f"[Init] Loading existing FAISS index from {faiss_artifact}...")
            self.retriever.load(ARTIFACTS_DIR)
        else:
            print("[Init] Building FAISS index from knowledge base...")
            if not KB_CSV.exists():
                raise FileNotFoundError(
                    f"Knowledge base {KB_CSV} is required to build the FAISS index. "
                    "Run 'python prepare_data.py' first."
                )
            self.retriever.build_index_from_csv(KB_CSV, show_progress=False)
            self.retriever.save(ARTIFACTS_DIR)
            print(f"[Init] FAISS index saved to {ARTIFACTS_DIR}")

        # 5. Generator (FLAN-T5)
        print("[Init] Loading Generator (google/flan-t5-base)...")
        self.generator = Generator()
        print("[Init] System ready!\n")

    def ask(self, question: str) -> Dict[str, Any]:
        """Process a user question through the full intent-aware RAG pipeline."""
        clean_q = str(question).strip()
        if not clean_q:
            return {
                "question": question,
                "intent": "None",
                "answer": "Please ask a question.",
                "sources": [],
                "status": "empty_query",
                "retrieval_score": 0.0,
                "faithfulness_score": 0.0,
            }

        # STEP 1: Safety Screening (Screen FIRST to avoid wasteful downstream compute)
        safety_res = self.safety.screen(clean_q)
        if safety_res["is_unsafe"]:
            return {
                "question": question,
                "intent": "Blocked",
                "answer": self.safety.get_unsafe_response(),
                "sources": [],
                "status": "blocked",
                "safety_details": safety_res,
                "retrieval_score": 0.0,
                "faithfulness_score": 0.0,
            }

        # STEP 2: Intent Classification
        intent, intent_confidence = self.intent_classifier.predict_with_confidence(clean_q)

        # STEP 3: Intent-Aware Semantic Retrieval
        top_k = INTENT_TOP_K.get(intent, 3)
        passages = self.retriever.search(clean_q, k=top_k)

        # STEP 4: Relevance & In-Scope Validation
        top_similarity = passages[0]["similarity"] if passages else 0.0
        if not self.retriever.is_in_scope(passages):
            return {
                "question": question,
                "intent": intent,
                "intent_confidence": intent_confidence,
                "answer": OUT_OF_SCOPE_MESSAGE,
                "sources": passages,
                "status": "out_of_scope",
                "retrieval_score": top_similarity,
                "faithfulness_score": 0.0,
            }

        # STEP 5: Context Alignment & Grounded Generation
        effective_context = self.generator.get_effective_context(passages)
        answer = self.generator.generate(clean_q, intent, passages)

        # STEP 6: NLI Faithfulness Verification (evaluated over the EXACT effective context)
        faithfulness_res = self.checker.calculate_support(
            context=effective_context,
            answer=answer,
            question=clean_q,
        )

        if not faithfulness_res["is_faithful"]:
            return {
                "question": question,
                "intent": intent,
                "intent_confidence": intent_confidence,
                "answer": UNVERIFIED_MESSAGE,
                "unverified_generated_answer": answer,
                "sources": passages,
                "status": "unverified",
                "retrieval_score": top_similarity,
                "faithfulness_score": faithfulness_res["entailment_prob"],
                "faithfulness_details": faithfulness_res,
            }

        # STEP 7: Verified Response with Attribution
        return {
            "question": question,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "answer": answer,
            "sources": passages,
            "status": "ok",
            "retrieval_score": top_similarity,
            "faithfulness_score": faithfulness_res["entailment_prob"],
            "faithfulness_details": faithfulness_res,
        }


def main():
    parser = argparse.ArgumentParser(description="Query the Intent-Aware RAG QA System.")
    parser.add_argument("query", nargs="?", type=str, help="Question to ask the system.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild model and FAISS artifacts.")
    args = parser.parse_args()

    system = IntentAwareRAG(rebuild_artifacts=args.rebuild)

    if args.query:
        result = system.ask(args.query)
        print(f"\nQuestion: {result['question']}")
        print(f"Status:   {result['status'].upper()}")
        print(f"Intent:   {result['intent']}")
        print(f"Answer:   {result['answer']}")
        if result["sources"]:
            print("\nSources:")
            for idx, src in enumerate(result["sources"][:2], 1):
                title = src.get("title", "Unknown")
                sim = src.get("similarity", 0.0)
                print(f"  [{idx}] {title} (Cosine similarity: {sim:.3f})")
                print(f"      {src['text'][:140]}...")
    else:
        print("\nInteractive Mode (Type 'exit' to quit)\n" + "-" * 40)
        while True:
            try:
                user_input = input("\nEnter Question: ").strip()
                if user_input.lower() in ("exit", "quit", "q"):
                    break
                if not user_input:
                    continue
                res = system.ask(user_input)
                print(f"Status: {res['status'].upper()} | Intent: {res['intent']}")
                print(f"Answer: {res['answer']}\n")
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
