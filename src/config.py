"""
Configuration and Constants for Intent-Aware RAG Question Answering System.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
TRAIN_CSV = DATA_DIR / "train_data.csv"
TEST_CSV = DATA_DIR / "test.csv"
KB_CSV = DATA_DIR / "cse_topics.csv"
EVAL_QUERIES_CSV = DATA_DIR / "eval_queries.csv"
EVAL_DEV_CSV = DATA_DIR / "eval_dev.csv"
EVAL_TEST_CSV = DATA_DIR / "eval_test.csv"
SAFETY_EVAL_CSV = DATA_DIR / "safety_eval.csv"
CS_INTENT_TEST_CSV = DATA_DIR / "cs_intent_test.csv"
RETRIEVAL_PASSAGE_EVAL_CSV = DATA_DIR / "retrieval_passage_eval.csv"
AUDIT_REPORT_JSON = DATA_DIR / "dataset_audit_report.json"


# Model identifiers
EMBED_MODEL = "all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-base"
SAFETY_MODEL = "facebook/bart-large-mnli"

# Hardware execution device (Torch on this environment is 2.8.0+cpu)
DEVICE = "cpu"


# Default operational thresholds
DEFAULT_SIMILARITY_THRESHOLD = 0.35  # Minimum FAISS cosine similarity for in-scope acceptance
DEFAULT_ENTAILMENT_THRESHOLD = 0.30   # Minimum NLI entailment probability for faithfulness verification
DEFAULT_SAFETY_THRESHOLD = 0.65       # Minimum posterior probability for flagging harmful content

# Full 10-class intent taxonomy mapping
INTENT_MAP = {
    "ABBR": "Abbreviation",
    "ENTY": "Entity",
    "HUM": "Human",
    "LOC": "Location",
    "NUM": "Number",
    "DESC:definition": "Definition",
    "DESC:procedure": "Procedure",
    "DESC:reason": "Reason",
    "DESC:comparison": "Comparison",
    "DESC:description": "Description",
}

# Coarse 9-class subset mapping directly from TREC QC (TREC does not contain comparison)
TREC_INTENT_MAP = {
    "ABBR": "Abbreviation",
    "ENTY": "Entity",
    "HUM": "Human",
    "LOC": "Location",
    "NUM": "Number",
    "DESC:definition": "Definition",
    "DESC:procedure": "Procedure",
    "DESC:reason": "Reason",
    "DESC:description": "Description",
}

# Intent-driven generation guidelines
INTENT_STYLE = {
    "Definition": "Give a clear, precise, and direct definition of the concept.",
    "Procedure": "Explain the step-by-step procedure or mechanism in chronological order.",
    "Reason": "Explain the underlying reason, rationale, or cause clearly.",
    "Comparison": "Compare the key differences, trade-offs, and distinguishing characteristics.",
    "Description": "Provide a concise technical description of the architecture or mechanism.",
    "Abbreviation": "Provide the full expanded form followed by a one-line explanation.",
    "Entity": "Identify the relevant software, hardware, or theoretical entity and describe its role.",
    "Human": "Identify the person, group, or pioneer and summarize their core contribution.",
    "Location": "State the relevant physical or logical location with technical context.",
    "Number": "State the relevant numeric quantity, complexity, or value with proper units.",
}

# Intent-aware retrieval parameter (number of passages retrieved based on intent complexity)
INTENT_TOP_K = {
    "Comparison": 4,   # Requires multiple passages to cover both contrasting concepts
    "Procedure": 3,
    "Reason": 3,
    "Description": 3,
    "Definition": 2,
    "Abbreviation": 2,
    "Entity": 2,
    "Human": 2,
    "Location": 2,
    "Number": 2,
}

# Safety screening candidate labels for zero-shot NLI classification
# Distinguishes conceptual, educational, and defensive technical inquiries from operational harm
SAFETY_CANDIDATE_LABELS = [
    "safe conceptual, educational, defensive, or informational inquiry about computer science, security concepts, or system administration",
    "actionable instructions, exploit code, malware, or attack execution to compromise systems, steal credentials, or damage infrastructure",
    "instructions for manufacturing dangerous weapons, explosives, or lethal chemical poisons",
    "instructions or encouragement for suicide, self-harm, or physical violence",
    "instructions for theft, extortion, fraud, or real-world criminal acts",
]

# Note: BENIGN_CS_EDUCATIONAL_TERMS is deprecated for allowlisting to eliminate bypass vulnerabilities
# Contextual evaluation of requested action and target is performed instead


# System standard response messages
UNSAFE_MESSAGE = (
    "I cannot fulfill this request as it involves instructions for potentially dangerous, "
    "illegal, or harmful activities. For educational cybersecurity concepts or process control, "
    "please rephrase with a focus on defense, mechanisms, or standard operating system commands."
)

OUT_OF_SCOPE_MESSAGE = (
    "I don't have sufficient or reliable information regarding that topic in my computer science knowledge base."
)

UNVERIFIED_MESSAGE = (
    "I retrieved relevant passages from the knowledge base, but the generated answer could not be verified "
    "with sufficient confidence against the retrieved context. The retrieved source passages are provided below."
)
