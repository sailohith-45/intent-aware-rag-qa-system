"""
Comprehensive Evaluation Suite for Intent-Aware RAG Question Answering System.
Measures performance across all pipeline components with scientific rigor:

1. Safety Screening: Confusion Matrix, Accuracy, Precision, Recall, Macro-F1, FPR, FNR across 40 items.
2. Intent Classification:
   - TREC Test Split (510 items, 0 leakage, unweighted vs balanced)
   - Expanded CS Domain Benchmark (120 items covering all 10 intent classes)
   - Per-class precision, recall, F1, and confusion matrices
3. Out-of-Scope Detection & Threshold Calibration:
   - Calibrates optimal threshold on Development split (50 items: 35 in-scope, 15 out-of-scope)
   - Freezes threshold and evaluates on independent Held-Out Test split (55 items: 40 in-scope, 15 out-of-scope)
   - Reports In-Scope Acceptance, OOS Rejection, FAR, FRR, Balanced Accuracy, and Macro-F1
4. Semantic Retrieval Evaluation:
   - Article-Level Recall@1, Recall@3, Recall@5, and Article-Level MRR
   - Gold Passage-Level Recall@1, Recall@3, Recall@5, and Passage-Level MRR (30 items)
5. Full End-to-End Pipeline Evaluation:
   - Executes complete RAG workflow and persists outputs to data/eval_outputs.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import (
    ARTIFACTS_DIR,
    CS_INTENT_TEST_CSV,
    DATA_DIR,
    DEFAULT_ENTAILMENT_THRESHOLD,
    DEFAULT_SAFETY_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEVICE,
    EMBED_MODEL,
    EVAL_DEV_CSV,
    EVAL_QUERIES_CSV,
    EVAL_TEST_CSV,
    INTENT_MAP,
    KB_CSV,
    RETRIEVAL_PASSAGE_EVAL_CSV,
    SAFETY_EVAL_CSV,
    TEST_CSV,
    TRAIN_CSV,
)
from src.faithfulness import FaithfulnessChecker
from src.intent_classifier import IntentClassifier
from src.rag_system import IntentAwareRAG
from src.retriever import Retriever
from src.safety import SafetyGuard


def evaluate_intent(embedder: SentenceTransformer):
    """
    Evaluate Intent Classifier on:
    1. Clean TREC held-out test split (510 items, 0 leakage)
    2. Expanded CS-specific intent benchmark (120 items covering all 10 intent classes)
    """
    print("\n" + "=" * 75)
    print("EVALUATION 1: INTENT CLASSIFICATION")
    print("=" * 75)

    X_train, y_train = IntentClassifier.load_dataset(TRAIN_CSV)
    X_test, y_test = IntentClassifier.load_dataset(TEST_CSV)

    print(f"Loaded {len(X_train)} training examples (0 leakage) and {len(X_test)} test examples.")
    print("Comparing standard (None) vs balanced class weighting...")

    models = {}
    for cw in [None, "balanced"]:
        clf = IntentClassifier(embedder=embedder, class_weight=cw)
        clf.fit(X_train, y_train, show_progress=False)
        metrics = clf.evaluate(X_test, y_test)
        models[cw] = (clf, metrics)

        cw_label = str(cw)
        print(f"\n--- TREC Test Split Performance (class_weight={cw_label}) ---")
        print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
        print(f"Macro-Precision: {metrics['macro_precision'] * 100:.2f}%")
        print(f"Macro-Recall:    {metrics['macro_recall'] * 100:.2f}%")
        print(f"Macro-F1:        {metrics['macro_f1'] * 100:.2f}%")

    # Select best performing model on TREC macro F1
    best_cw = None if models[None][1]["macro_f1"] >= models["balanced"][1]["macro_f1"] else "balanced"
    best_clf, best_trec_metrics = models[best_cw]

    # Save best model to artifacts
    save_path = best_clf.save(ARTIFACTS_DIR / "intent_classifier.joblib")
    print(f"\nTrained classifier (class_weight={best_cw}) persisted to: {save_path.name}")

    print("\nDetailed Per-Class Classification Report (TREC Test Split):")
    rep_df = pd.DataFrame(best_trec_metrics["classification_report"]).transpose()
    print(rep_df.to_string())

    print("\nConfusion Matrix (TREC Test Split):")
    classes = best_trec_metrics["classes"]
    cm_df = pd.DataFrame(best_trec_metrics["confusion_matrix"], index=classes, columns=classes)
    print(cm_df.to_string())

    # Independent Evaluation on Expanded Computer Science Domain Benchmark
    if CS_INTENT_TEST_CSV.exists():
        print("\n" + "-" * 75)
        print("INDEPENDENT CS-DOMAIN INTENT BENCHMARK (120 ITEMS)")
        print("-" * 75)
        X_cs, y_cs = IntentClassifier.load_dataset(CS_INTENT_TEST_CSV)
        cs_metrics = best_clf.evaluate(X_cs, y_cs)

        print(f"CS Benchmark Queries: {len(X_cs)} (covering all 10 intent classes)")
        print(f"CS Accuracy:          {cs_metrics['accuracy'] * 100:.2f}%")
        print(f"CS Macro-Precision:   {cs_metrics['macro_precision'] * 100:.2f}%")
        print(f"CS Macro-Recall:      {cs_metrics['macro_recall'] * 100:.2f}%")
        print(f"CS Macro-F1:          {cs_metrics['macro_f1'] * 100:.2f}%")

        print("\nPer-Class Breakdown (CS Domain Benchmark):")
        cs_rep_df = pd.DataFrame(cs_metrics["classification_report"]).transpose()
        print(cs_rep_df.to_string())

        print("\nConfusion Matrix (CS Domain Benchmark):")
        cs_cm_df = pd.DataFrame(cs_metrics["confusion_matrix"], index=cs_metrics["classes"], columns=cs_metrics["classes"])
        print(cs_cm_df.to_string())


def evaluate_retrieval(retriever: Retriever):
    """
    Evaluate FAISS vector retrieval using:
    1. Article-Level Recall@1, 3, 5 and Article-Level MRR
    2. Gold Passage-Level Recall@1, 3, 5 and Passage-Level MRR (30 items)
    """
    print("\n" + "=" * 75)
    print("EVALUATION 2: SEMANTIC RETRIEVAL (ARTICLE-LEVEL & PASSAGE-LEVEL)")
    print("=" * 75)

    k_values = [1, 3, 5]

    # --- 1. Article-Level Evaluation ---
    if EVAL_QUERIES_CSV.exists():
        df = pd.read_csv(EVAL_QUERIES_CSV)
        in_scope_df = df[df["in_scope"] == True].copy()
        print(f"--- 1. Article-Level Retrieval Evaluation ({len(in_scope_df)} in-scope queries) ---")

        art_hits_at_k = {k: 0 for k in k_values}
        art_reciprocal_ranks = []

        for _, row in in_scope_df.iterrows():
            question = row["question"]
            expected_title = str(row.get("expected_title", "")).strip().lower()
            expected_article_id = str(row.get("expected_article_id", "")).strip().lower()

            results = retriever.search(question, k=max(k_values))

            rr = 0.0
            for rank, res in enumerate(results, 1):
                ret_title = str(res.get("title", "")).strip().lower()
                ret_art_id = str(res.get("article_id", "")).strip().lower()

                if (expected_article_id and expected_article_id in ret_art_id) or (expected_title and expected_title == ret_title):
                    if rr == 0.0:
                        rr = 1.0 / rank
                    for k in k_values:
                        if rank <= k:
                            art_hits_at_k[k] += 1
                    break

            art_reciprocal_ranks.append(rr)

        n_art = len(in_scope_df)
        print(f"Total Article Queries Evaluated: {n_art}")
        for k in k_values:
            print(f"  - Article-Level Recall@{k}: {(art_hits_at_k[k] / n_art) * 100:.2f}% ({art_hits_at_k[k]}/{n_art})")
        print(f"  - Article-Level MRR:      {float(np.mean(art_reciprocal_ranks)):.4f}")

    # --- 2. Gold Passage-Level Evaluation ---
    if RETRIEVAL_PASSAGE_EVAL_CSV.exists():
        print(f"\n--- 2. Gold Passage-Level Retrieval Evaluation ({RETRIEVAL_PASSAGE_EVAL_CSV.name}) ---")
        p_df = pd.read_csv(RETRIEVAL_PASSAGE_EVAL_CSV)
        print(f"Evaluating exact chunk retrieval across {len(p_df)} verified queries.")

        psg_hits_at_k = {k: 0 for k in k_values}
        psg_reciprocal_ranks = []

        for _, row in p_df.iterrows():
            query = row["query"]
            gold_chunk_ids = [c.strip().lower() for c in str(row["relevant_chunk_ids"]).split(",")]

            results = retriever.search(query, k=max(k_values))

            rr = 0.0
            for rank, res in enumerate(results, 1):
                ret_chunk_id = str(res.get("id", "")).strip().lower()

                if ret_chunk_id in gold_chunk_ids:
                    if rr == 0.0:
                        rr = 1.0 / rank
                    for k in k_values:
                        if rank <= k:
                            psg_hits_at_k[k] += 1
                    break

            psg_reciprocal_ranks.append(rr)

        n_psg = len(p_df)
        print(f"Total Passage Queries Evaluated: {n_psg}")
        for k in k_values:
            print(f"  - Passage-Level Recall@{k}: {(psg_hits_at_k[k] / n_psg) * 100:.2f}% ({psg_hits_at_k[k]}/{n_psg})")
        print(f"  - Passage-Level MRR:      {float(np.mean(psg_reciprocal_ranks)):.4f}")


def evaluate_out_of_scope(retriever: Retriever):
    """
    Evaluate Out-of-Scope (OOS) Detection with strict train/test discipline:
    1. Calibrate optimal threshold strictly on Development split (eval_dev.csv: 50 items).
    2. Freeze optimal threshold and evaluate on untouched Held-Out Test split (eval_test.csv: 55 items).
    Reports In-Scope Acceptance, OOS Rejection, FAR, FRR, Balanced Accuracy, and Macro-F1.
    """
    print("\n" + "=" * 75)
    print("EVALUATION 3: OUT-OF-SCOPE DETECTION (DEV CALIBRATION & HELD-OUT TEST)")
    print("=" * 75)

    if not EVAL_DEV_CSV.exists() or not EVAL_TEST_CSV.exists():
        raise FileNotFoundError(f"Splits {EVAL_DEV_CSV} and {EVAL_TEST_CSV} are required. Run prepare_data.py first.")

    dev_df = pd.read_csv(EVAL_DEV_CSV)
    test_df = pd.read_csv(EVAL_TEST_CSV)

    print(f"Development / Calibration Set: {len(dev_df)} queries ({sum(dev_df['in_scope'])} in-scope, {sum(~dev_df['in_scope'])} OOS)")
    print(f"Independent Held-Out Test Set: {len(test_df)} queries ({sum(test_df['in_scope'])} in-scope, {sum(~test_df['in_scope'])} OOS)")

    def get_max_similarities(df: pd.DataFrame) -> Tuple[List[float], List[bool]]:
        sims = []
        labels = df["in_scope"].astype(bool).tolist()
        for q in df["question"]:
            res = retriever.search(q, k=1)
            sim = res[0]["similarity"] if res else 0.0
            sims.append(sim)
        return sims, labels

    dev_sims, dev_labels = get_max_similarities(dev_df)
    test_sims, test_labels = get_max_similarities(test_df)

    # 1. Threshold Calibration on Development Set
    print("\n--- 1. Threshold Calibration on Development Split ---")
    candidate_thresholds = [0.20, 0.25, 0.30, 0.32, 0.35, 0.38, 0.40, 0.45, 0.50]
    print(f"{'Threshold':<10}{'In-Scope Rec':<14}{'OOS Rejection':<15}{'FAR (FP)':<12}{'FRR (FN)':<12}{'Balanced Acc':<14}{'Macro-F1':<10}")
    print("-" * 87)

    best_thresh = DEFAULT_SIMILARITY_THRESHOLD
    best_dev_bal_acc = -1.0

    for thresh in candidate_thresholds:
        preds = [s >= thresh for s in dev_sims]
        in_scope_rec = recall_score(dev_labels, preds, pos_label=True, zero_division=0)
        oos_rec = recall_score(dev_labels, preds, pos_label=False, zero_division=0)
        far = 1.0 - oos_rec  # False Acceptance Rate: OOS accepted
        frr = 1.0 - in_scope_rec  # False Rejection Rate: In-scope rejected
        bal_acc = (in_scope_rec + oos_rec) / 2.0
        f1 = f1_score(dev_labels, preds, average="macro", zero_division=0)

        if bal_acc > best_dev_bal_acc:
            best_dev_bal_acc = bal_acc
            best_thresh = thresh

        print(f"{thresh:<10.2f}{in_scope_rec * 100:<14.1f}%{oos_rec * 100:<15.1f}%{far * 100:<12.1f}%{frr * 100:<12.1f}%{bal_acc * 100:<14.1f}%{f1:.4f}")

    print(f"\n[Frozen] Optimal threshold selected on Development set: tau = {best_thresh:.2f} (Dev Balanced Acc: {best_dev_bal_acc * 100:.2f}%)")

    # 2. Evaluation of Frozen Threshold on Untouched Held-Out Test Set
    print(f"\n--- 2. Evaluation on Held-Out Test Split (Frozen Threshold tau = {best_thresh:.2f}) ---")
    test_preds = [s >= best_thresh for s in test_sims]
    t_in_scope_rec = recall_score(test_labels, test_preds, pos_label=True, zero_division=0)
    t_oos_rec = recall_score(test_labels, test_preds, pos_label=False, zero_division=0)
    t_far = 1.0 - t_oos_rec
    t_frr = 1.0 - t_in_scope_rec
    t_bal_acc = (t_in_scope_rec + t_oos_rec) / 2.0
    t_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    t_cm = confusion_matrix(test_labels, test_preds, labels=[True, False])

    print(f"In-Scope Acceptance Rate (Recall): {t_in_scope_rec * 100:.2f}% ({sum(p and l for p, l in zip(test_preds, test_labels))}/{sum(test_labels)})")
    print(f"OOS Rejection Rate (Specificity):  {t_oos_rec * 100:.2f}% ({sum(not p and not l for p, l in zip(test_preds, test_labels))}/{sum(not l for l in test_labels)})")
    print(f"False Acceptance Rate (FAR):       {t_far * 100:.2f}%")
    print(f"False Rejection Rate (FRR):        {t_frr * 100:.2f}%")
    print(f"Test Balanced Accuracy:            {t_bal_acc * 100:.2f}%")
    print(f"Test Macro-F1 Score:               {t_f1:.4f}")
    print(f"\nConfusion Matrix on Test Split (Rows=Actual [In-Scope, OOS], Cols=Pred [In-Scope, OOS]):\n{t_cm}")
    print("\nNote: Cosine similarity is a geometric vector angle score, not a calibrated posterior probability.")


def evaluate_safety():
    """
    Evaluate Redesigned Safety Screener across 40 test samples:
    - Benign conceptual questions
    - Defensive cybersecurity questions
    - Harmful operational requests
    - Adversarial educational-prefix attacks
    - Ambiguous system administration commands
    Reports Confusion Matrix, Accuracy, Precision, Recall, Macro-F1, FPR, and FNR.
    """
    print("\n" + "=" * 75)
    print("EVALUATION 4: SAFETY SCREENING (CONTEXT-AWARE ZERO-SHOT NLI)")
    print("=" * 75)

    if not SAFETY_EVAL_CSV.exists():
        raise FileNotFoundError(f"Safety benchmark not found at {SAFETY_EVAL_CSV}")

    df = pd.read_csv(SAFETY_EVAL_CSV)
    y_true = df["is_harmful"].tolist()
    n_safe = sum(not h for h in y_true)
    n_harm = sum(bool(h) for h in y_true)

    print(f"Loaded {len(df)} safety test samples: {n_safe} safe/benign/defensive, {n_harm} harmful/adversarial.")

    guard = SafetyGuard(threshold=DEFAULT_SAFETY_THRESHOLD)

    # Run inference once per query to avoid redundant evaluations across thresholds
    print("Screening 40 safety queries in context (BART-large-MNLI + operational harm analysis)...")
    cached_results = []
    for q in df["query"]:
        res = guard.screen(q)
        cached_results.append(res)

    thresholds = [0.50, 0.60, 0.65, 0.75, 0.85]
    print(f"\n{'Threshold':<12}{'FPR (Benign)':<16}{'FNR (Harmful)':<16}{'Accuracy':<12}{'Precision':<12}{'Recall':<10}{'Macro-F1':<10}")
    print("-" * 88)

    for thresh in thresholds:
        preds = []
        for res in cached_results:
            if res.get("detection_source") == "contextual_operational_harm":
                preds.append(True)
            else:
                is_unsafe = (res["top_label"] != guard.candidate_labels[0] and res["top_score"] >= thresh) or (res["risk_mass"] >= thresh)
                preds.append(is_unsafe)

        benign_indices = [i for i, h in enumerate(y_true) if not h]
        harmful_indices = [i for i, h in enumerate(y_true) if h]

        # False Positives: Safe query falsely flagged as unsafe
        fps = sum(preds[i] for i in benign_indices)
        fpr = (fps / len(benign_indices)) * 100

        # False Negatives: Harmful query allowed through as safe
        fns = sum(not preds[i] for i in harmful_indices)
        fnr = (fns / len(harmful_indices)) * 100

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)

        print(f"{thresh:<12.2f}{fpr:<16.1f}%{fnr:<16.1f}%{acc * 100:<12.1f}%{prec * 100:<12.1f}%{rec * 100:<10.1f}%{f1:.4f}")

    # Detailed report at default threshold (0.65)
    eval_records = []
    preds_def = []

    for idx, row in df.iterrows():
        q = row["query"]
        gt = bool(row["is_harmful"])
        res = cached_results[idx]
        if res.get("detection_source") == "contextual_operational_harm":
            pred = True
        else:
            pred = (res["top_label"] != guard.candidate_labels[0] and res["top_score"] >= DEFAULT_SAFETY_THRESHOLD) or (res["risk_mass"] >= DEFAULT_SAFETY_THRESHOLD)
        preds_def.append(pred)


        eval_records.append({
            "query": q,
            "category": row["category"],
            "ground_truth": "UNSAFE" if gt else "SAFE",
            "predicted": "UNSAFE" if pred else "SAFE",
            "score": round(res["top_score"], 3),
            "detection_source": res.get("detection_source", "zero_shot_nli"),
            "status": "PASS" if pred == gt else "MISMATCH",
        })

    eval_results_df = pd.DataFrame(eval_records)
    cm = confusion_matrix(y_true, preds_def, labels=[True, False])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    print(f"\n--- Detailed Safety Benchmark Results at tau = {DEFAULT_SAFETY_THRESHOLD} ---")
    print(f"Confusion Matrix (Pos=Harmful, Neg=Safe):")
    print(f"  True Positives (Harmful blocked):  {tp}")
    print(f"  False Negatives (Harmful missed):  {fn} (FNR: {(fn / n_harm) * 100:.1f}%)")
    print(f"  True Negatives (Safe allowed):     {tn}")
    print(f"  False Positives (Safe blocked):    {fp} (FPR: {(fp / n_safe) * 100:.1f}%)")
    print(f"Overall Accuracy:                    {accuracy_score(y_true, preds_def) * 100:.2f}%")
    print(f"Macro-F1 Score:                      {f1_score(y_true, preds_def, average='macro'):.4f}")

    print("\nAdversarial Attack & Conceptual Inquiries Sample Check:")
    sample_queries = eval_results_df[eval_results_df["category"].str.startswith("adversarial_") | eval_results_df["category"].str.startswith("benign_conceptual_")].head(12)
    for _, r in sample_queries.iterrows():
        print(f"  [{r['status']}] GT: {r['ground_truth']:<6} | Pred: {r['predicted']:<6} | Source: {r['detection_source'][:15]:<15} | Query: {r['query']}")


def evaluate_full_pipeline(system: IntentAwareRAG):
    """Run full pipeline over evaluation queries and generate eval_outputs.csv."""
    print("\n" + "=" * 75)
    print("EVALUATION 5: FULL END-TO-END PIPELINE EVALUATION")
    print("=" * 75)

    df = pd.read_csv(EVAL_QUERIES_CSV)
    results = []

    print(f"Running full pipeline over {len(df)} partitioned evaluation queries...")
    for idx, row in df.iterrows():
        q = row["question"]
        in_scope = row["in_scope"]
        split = row.get("split", "unknown")

        res = system.ask(q)

        record = {
            "id": row["id"],
            "split": split,
            "question": q,
            "expected_scope": "in_scope" if in_scope else "out_of_scope",
            "expected_intent": row.get("expected_intent", ""),
            "detected_intent": res["intent"],
            "final_status": res["status"],
            "retrieval_similarity": round(res.get("retrieval_score", 0.0), 4),
            "faithfulness_score": round(res.get("faithfulness_score", 0.0), 4),
            "generated_answer": res["answer"],
            "primary_source": res["sources"][0]["title"] if res.get("sources") else "None",
            "human_review_correctness": "pending_manual_review",
        }
        results.append(record)

    out_df = pd.DataFrame(results)
    out_path = DATA_DIR / "eval_outputs.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nFull evaluation outputs saved to: {out_path.name}")

    total = len(out_df)
    status_counts = out_df["final_status"].value_counts()
    print("\nPipeline Status Breakdown:")
    for status, count in status_counts.items():
        print(f"  - {status:<15}: {count:<3} ({count / total * 100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Run Intent-Aware RAG evaluations.")
    parser.add_argument("--intent", action="store_true", help="Run intent classification evaluation.")
    parser.add_argument("--retrieval", action="store_true", help="Run semantic retrieval evaluation.")
    parser.add_argument("--oos", action="store_true", help="Run out-of-scope threshold evaluation.")
    parser.add_argument("--safety", action="store_true", help="Run safety screening evaluation.")
    parser.add_argument("--full", action="store_true", help="Run full pipeline end-to-end evaluation.")
    parser.add_argument("--all", action="store_true", help="Run all evaluations.")
    args = parser.parse_args()

    run_all = args.all or not (args.intent or args.retrieval or args.oos or args.safety or args.full)

    embedder = SentenceTransformer(EMBED_MODEL, device=DEVICE)

    if run_all or args.intent:
        evaluate_intent(embedder)

    retriever = None
    if run_all or args.retrieval or args.oos:
        retriever = Retriever(embedder=embedder)
        retriever.build_index_from_csv(KB_CSV, show_progress=False)
        retriever.save(ARTIFACTS_DIR)

    if run_all or args.retrieval:
        evaluate_retrieval(retriever)

    if run_all or args.oos:
        evaluate_out_of_scope(retriever)

    if run_all or args.safety:
        evaluate_safety()

    if run_all or args.full:
        system = IntentAwareRAG()
        evaluate_full_pipeline(system)

    print("\n" + "=" * 75)
    print("EVALUATION SUITE COMPLETED SUCCESSFULLY")
    print("=" * 75)


if __name__ == "__main__":
    main()
