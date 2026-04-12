"""Config file to help with modularization."""

from pathlib import Path

# Repo root (parent of src/) — stable regardless of process cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Embedding
# Options: "openai" | "pubmed_bert" | "mini_lm"
EMBEDDING_MODEL = "mini_lm"

# LLM settings
LLM_PROVIDER = "vertex"
LLM_MODEL = "gemini-2.0-flash"

# Vector DB = chromadb (absolute path so uvicorn/cron/index_kb share one store)
CHROMA_PATH = str(_REPO_ROOT / "src" / "retrieval" / "vector_storage" / "chroma_db")
CHROMA_COLLECTION = "bowel_prep_kb"
CHROMA_CINICAL_COLLECTION = "clinical_kb"
CHROMA_QA_COLLECTION = "qa_kb"
CHROMA_CONVO_COLLECTION = "conversation_kb"

# Indexing
BATCH_SIZE = 100

# Retrieval # todo: adjust later
CLINICAL_TOP_K = 5
QA_TOP_K = 2
CONVERSATION_TOP_K = 2

# Judge guardrail
# Confidence thresholds vary by patient risk tier — higher-risk patients require
# more certainty before a response is delivered autonomously.
JUDGE_THRESHOLDS = {
    "high":   0.9,
    "medium": 0.85,
    "low":    0.8,
}
JUDGE_DEFAULT_THRESHOLD = 0.85   # used when risk tier is unknown
JUDGE_MAX_RETRIES = 2             # max regeneration attempts before escalation
