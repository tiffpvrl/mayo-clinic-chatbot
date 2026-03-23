'''Config file to help with modularization'''

# Embedding
# Options: "openai" | "pubmed_bert" | "mini_llm"
EMBEDDING_MODEL = "mini_lm"

# LLM — model string is passed directly to LiteLLM
# OpenAI:    "gpt-4o", "gpt-4o-mini"
# Anthropic: "claude-sonnet-4-6", "claude-haiku-4-5-20251001"
# Google:    "gemini/gemini-2.0-flash"
# Local:     "ollama/llama3"
LLM_MODEL = "gpt-4o"
LLM_TEMPERATURE = 0.3

# Vector DB = chromadb for simplicity
CHROMA_PATH = "src/retrieval/vector_storage/chroma_db"
CHROMA_CINICAL_COLLECTION = "bowel_prep_kb"
CHROMA_QA_COLLECTION = "qa_kb"
CHROMA_CONVO_COLLECTION = "convo_kb"

# Retrieval # todo: adjust later
TOP_K = 5

# Indexing
BATCH_SIZE = 256  # max chunks per ChromaDB .add() call
