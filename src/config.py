'''Config file to help with modularization'''

# Embedding
# Options: "openai" | "pubmed_bert" | "mini_llm"
EMBEDDING_MODEL = "mini_lm"

# LLM
# Options: # todo
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o"           

# Vector DB = chromadb for simplicity
CHROMA_PATH = "src/retrieval/vector_storage/chroma_db"
CHROMA_CINICAL_COLLECTION = "bowel_prep_kb"
CHROMA_QA_COLLECTION = "qa_kb"
CHROMA_CONVO_COLLECTION = "convo_kb"

# Retrieval # todo: adjust later
TOP_K = 5

# Indexing
BATCH_SIZE = 256  # max chunks per ChromaDB .add() call
