# Datasets & Knowledge Base Documentation

This directory contains the training data, testing splits, computer-science knowledge base, and evaluation benchmarks used by the **Intent-Aware RAG Question Answering System**.

---

## 1. Intent Classification Datasets

The intent classification component uses question classification data adapted from the **UIUC / TREC Question Classification benchmark** (Li & Roth, 2002), supplemented with verified comparison intent examples.

### `train_data.csv` (5,421 samples)
* **Columns:**
  * `question` (string): The natural language query.
  * `intent` (string): Target category label from our 10-class taxonomy.
* **Class Distribution:**
  * `ENTY` (Entity): 1,245
  * `HUM` (Human): 1,215
  * `NUM` (Number): 858
  * `LOC` (Location): 824
  * `DESC:definition` (Definition): 416
  * `DESC:procedure` (Procedure): 275
  * `DESC:description` (Description): 272
  * `DESC:reason` (Reason): 190
  * `ABBR` (Abbreviation): 86
  * `DESC:comparison` (Comparison): 40

### `test.csv` (510 samples)
* **Columns:** `question`, `intent`
* Independent held-out test split from the TREC benchmark + comparison test samples.

### `cs_intent_test.csv` (32 samples)
* **Columns:** `question`, `intent`
* A curated benchmark of 100% Computer Science questions labelled against our 10 intent categories. Used to evaluate domain generalization beyond standard TREC questions.

---

## 2. Computer Science Knowledge Base

### `cse_topics.csv`
* **Purpose:** The external domain corpus retrieved via FAISS to ground answer generation.
* **Columns:**
  * `id` (string): Unique identifier for the chunk (e.g. `os_process_mgmt_01`).
  * `article_id` (string): Stable identifier for the source article.
  * `topic` (string): Computer science sub-discipline (e.g. `Operating Systems`, `Database Systems`, `Computer Networks`, `Cybersecurity`).
  * `title` (string): Title of the topic / concept.
  * `url` (string): Wikipedia / reference source URL.
  * `chunk_id` (integer): Sequential chunk index.
  * `text` (string): Sliding-window passage chunk (120–180 words, 25-word overlap).

---

## 3. Evaluation Benchmarks

### `eval_queries.csv` (105 questions)
* **Purpose:** End-to-end evaluation benchmark measuring retrieval Recall@K, MRR, and out-of-scope abstention.
* **Columns:**
  * `id`: Unique query identifier (`q01` to `q105`).
  * `question`: Natural language inquiry.
  * `expected_intent`: Target intent label.
  * `domain`: Technical or general knowledge domain.
  * `expected_article_id`: Gold ground-truth article ID (for in-scope queries).
  * `expected_title`: Gold ground-truth article title.
  * `in_scope`: Boolean (`True` if answered by the knowledge base; `False` if out-of-scope).
  * `ground_truth_answer`: Ground truth reference answer.

### `safety_eval.csv` (25 samples)
* **Purpose:** Evaluates safety screening false positives on legitimate CS cybersecurity questions vs. false negatives on dangerous/harmful prompts.
* **Columns:**
  * `query`: Input prompt.
  * `is_harmful`: Boolean (`True` = unsafe; `False` = benign CS educational inquiry).
  * `category`: Detailed sub-type (e.g. `benign_cs_security`, `weapons_explosives`, `self_harm`, `illegal_cyberattack`).

---

## Data Generation & Licensing
* All datasets can be regenerated deterministically by running:
  ```bash
  python prepare_data.py
  ```
* **Citations & Attribution:**
  * UIUC / TREC Question Classification: Xin Li and Dan Roth, *"Learning Question Classifiers"*, COLING 2002.
  * Wikipedia articles: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0).
