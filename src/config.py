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

# Retrieval # todo: adjust later
TOP_K = 5
