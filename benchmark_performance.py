"""
Empirical Performance Benchmarking Suite for Intent-Aware RAG Question Answering System.
Measures and records genuine runtime metrics across:
1. Cold Startup (from-scratch dataset loading, model fitting, and FAISS indexing)
2. Warm Startup (deserializing pre-built artifacts and instantiating transformer pipelines)
3. Component-Level Loading Times (SentenceTransformer, FAISS, IntentClassifier, BART-MNLI, FLAN-T5)
4. Per-Stage Inference Latency across multiple iterations (Safety, Intent, Retrieval, Generation, Faithfulness)
5. First-Query vs. Subsequent-Query Latency (Median, Mean, Min, Max)
"""

import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from src.config import (
    ARTIFACTS_DIR,
    DEVICE,
    EMBED_MODEL,
    GEN_MODEL,
    KB_CSV,
    SAFETY_MODEL,
    TEST_CSV,
    TRAIN_CSV,
)
from src.faithfulness import FaithfulnessChecker
from src.generator import Generator
from src.intent_classifier import IntentClassifier
from src.rag_system import IntentAwareRAG
from src.retriever import Retriever
from src.safety import SafetyGuard


def log_environment_details():
    """Display and record environment hardware and software specifications."""
    print("=" * 75)
    print("BENCHMARK ENVIRONMENT SPECIFICATIONS")
    print("=" * 75)
    print(f"Platform / OS:        {platform.platform()}")
    print(f"Processor:            {platform.processor() or platform.machine()}")
    print(f"Python Version:       {platform.python_version()}")
    print(f"PyTorch Version:      {torch.__version__}")
    print(f"Execution Device:     {DEVICE}")
    print(f"CUDA Available:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name:     {torch.cuda.get_device_name(0)}")
    print("=" * 75)


def benchmark_component_loading() -> Dict[str, float]:
    """Measure individual component loading and initialization latencies."""
    print("\n--- 1. Component Loading Breakdown ---")
    timings = {}

    # 1. SentenceTransformer embedder
    t0 = time.perf_counter()
    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    timings["SentenceTransformer (all-MiniLM-L6-v2)"] = time.perf_counter() - t0
    print(f"  - SentenceTransformer loading:      {timings['SentenceTransformer (all-MiniLM-L6-v2)']:.2f} s")

    # 2. FAISS Index & Metadata loading
    t0 = time.perf_counter()
    retriever = Retriever(embedder=embedder)
    retriever.load(ARTIFACTS_DIR)
    timings["FAISS Index & Metadata"] = time.perf_counter() - t0
    print(f"  - FAISS index & metadata loading:   {timings['FAISS Index & Metadata']:.3f} s")

    # 3. Intent Classifier joblib loading
    t0 = time.perf_counter()
    clf = IntentClassifier(embedder=embedder)
    clf.load(ARTIFACTS_DIR / "intent_classifier.joblib")
    timings["Intent Classifier (.joblib)"] = time.perf_counter() - t0
    print(f"  - Intent classifier loading:        {timings['Intent Classifier (.joblib)']:.3f} s")

    # 4. SafetyGuard (BART-large-MNLI) loading
    t0 = time.perf_counter()
    safety = SafetyGuard()
    timings["SafetyGuard (BART-large-MNLI)"] = time.perf_counter() - t0
    print(f"  - SafetyGuard (BART-large-MNLI):    {timings['SafetyGuard (BART-large-MNLI)']:.2f} s")

    # 5. Generator (FLAN-T5-base) loading
    t0 = time.perf_counter()
    generator = Generator()
    timings["Generator (FLAN-T5-base)"] = time.perf_counter() - t0
    print(f"  - Generator (google/flan-t5-base):  {timings['Generator (FLAN-T5-base)']:.2f} s")

    return timings


def benchmark_startups() -> Dict[str, float]:
    """Measure and compare cold startup vs warm startup times."""
    print("\n--- 2. End-to-End System Startup Measurements ---")
    timings = {}

    # Warm Startup: Loads all persisted models, indexes, and metadata
    t0 = time.perf_counter()
    warm_sys = IntentAwareRAG(rebuild_artifacts=False)
    t_warm = time.perf_counter() - t0
    timings["Warm Startup"] = t_warm
    print(f"  - Warm Startup (loading pre-built artifacts): {t_warm:.2f} s")

    # Cold Startup: Rebuilds classifier from CSV and rebuilds FAISS index from CSV
    t0 = time.perf_counter()
    cold_sys = IntentAwareRAG(rebuild_artifacts=True)
    t_cold = time.perf_counter() - t0
    timings["Cold Startup"] = t_cold
    print(f"  - Cold Startup (rebuilding artifacts):        {t_cold:.2f} s")

    return timings, warm_sys


def benchmark_inference_stages(system: IntentAwareRAG, num_iterations: int = 5):
    """
    Measure stage-by-stage inference latency over multiple iterations.
    Separates first-query (cold inference) from subsequent queries (warm cache).
    """
    print(f"\n--- 3. Inference Latency Breakdown ({num_iterations} iterations) ---")

    benchmark_queries = [
        "What is virtual memory in an operating system?",
        "How does the TCP three-way handshake establish a connection?",
        "What is the difference between symmetric and asymmetric encryption?",
        "Why do databases use write-ahead logging?",
        "What does ACID stand for in databases?",
    ]

    stage_timings: Dict[str, List[float]] = {
        "Safety Screening": [],
        "Intent Classification": [],
        "FAISS Retrieval": [],
        "FLAN-T5 Generation": [],
        "Faithfulness Verification": [],
        "Total End-to-End": [],
    }

    first_query_latency = None

    for i in range(num_iterations):
        q = benchmark_queries[i % len(benchmark_queries)]

        t_start = time.perf_counter()

        # Stage 1: Safety
        t0 = time.perf_counter()
        safety_res = system.safety.screen(q)
        t_safety = time.perf_counter() - t0

        # Stage 2: Intent
        t0 = time.perf_counter()
        intent = system.intent_classifier.predict(q)
        t_intent = time.perf_counter() - t0

        # Stage 3: Retrieval
        t0 = time.perf_counter()
        passages = system.retriever.search(q, k=3)
        t_retrieval = time.perf_counter() - t0

        # Stage 4: Generation
        t0 = time.perf_counter()
        effective_context = system.generator.get_effective_context(passages)
        answer = system.generator.generate(q, intent, passages)
        t_gen = time.perf_counter() - t0

        # Stage 5: Faithfulness
        t0 = time.perf_counter()
        faith_res = system.checker.calculate_support(effective_context, answer, question=q)
        t_faith = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        if i == 0:
            first_query_latency = t_total

        stage_timings["Safety Screening"].append(t_safety)
        stage_timings["Intent Classification"].append(t_intent)
        stage_timings["FAISS Retrieval"].append(t_retrieval)
        stage_timings["FLAN-T5 Generation"].append(t_gen)
        stage_timings["Faithfulness Verification"].append(t_faith)
        stage_timings["Total End-to-End"].append(t_total)

    print(f"\nFirst-Query Latency (cold weights / JIT compilation): {first_query_latency:.2f} s")

    print(f"\n{'Pipeline Stage':<28}{'Median':<12}{'Mean':<12}{'Min':<10}{'Max':<10}")
    print("-" * 72)

    for stage, times in stage_timings.items():
        med = statistics.median(times)
        mean_val = statistics.mean(times)
        min_val = min(times)
        max_val = max(times)
        print(f"{stage:<28}{med:.3f} s     {mean_val:.3f} s     {min_val:.3f} s    {max_val:.3f} s")


def main():
    log_environment_details()
    benchmark_component_loading()
    startup_timings, warm_system = benchmark_startups()
    benchmark_inference_stages(warm_system, num_iterations=5)
    print("\n" + "=" * 75)
    print("PERFORMANCE BENCHMARKING COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
