"""Config file to help with modularization."""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Repo root (parent of src/) — stable regardless of process cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Embedding
# Options: "openai" | "pubmed_bert" | "mini_lm"
EMBEDDING_MODEL = "mini_lm"

# LLM settings
LLM_PROVIDER = "vertex"
# Primary model: final patient answer + judge (quality / safety).
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
# Cheaper Vertex models for routing, structured JSON extraction, and source titles.
ROUTING_LLM_MODEL = os.getenv("ROUTING_LLM_MODEL", "gemini-2.5-flash-lite")
STRUCTURED_LLM_MODEL = os.getenv("STRUCTURED_LLM_MODEL", "gemini-2.5-flash-lite")
# Short patient-facing labels for RAG “View clinical sources” (any flash-tier model is fine)
CLINICAL_SOURCE_DISPLAY_MODEL = os.getenv("CLINICAL_SOURCE_DISPLAY_MODEL", "gemini-2.5-flash-lite")

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
# CLINICAL_RELEVANCE_THRESHOLD = 0.5  # unused — all observed hits are well under 0.5; delete if confirmed unnecessary

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

# Email settings
DEFAULT_PATIENT_EMAIL = os.getenv("DEFAULT_PATIENT_EMAIL", "mayochatbot1@gmail.com")
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"

SMTP_EMAIL = os.getenv("SMTP_EMAIL", "mayochatbot1@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587