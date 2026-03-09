'''Config file to help with modularization'''

# Embedding
# Options: "openai" | "pubmed_bert" | "mini_llm"
EMBEDDING_MODEL = "mini_lm"

# LLM settings
LLM_PROVIDER = "vertex"
LLM_MODEL = "gemini-2.0-flash"

# Vector DB = chromadb for simplicity
CHROMA_PATH = "src/retrieval/vector_storage/chroma_db"
CHROMA_COLLECTION = "bowel_prep_kb"

# Retrieval # todo: adjust later
TOP_K = 5
