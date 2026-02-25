import chromadb
from embedder import Embedding
from data_processing.document_processor import ProcessedChunk
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EMBEDDING_MODEL, CHROMA_PATH, CHROMA_COLLECTION

embedder = Embedding(model_type=EMBEDDING_MODEL)

client = chromadb.PersistentClient(CHROMA_PATH)
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

def index_chunks(chunks: list[ProcessedChunk]) -> None:
    embeddings = np.array(embedder.encode([c.content for c in chunks]))
    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.content for c in chunks],
        metadatas=[
            {
                "document_type": c.metadata.document_type.value,
                "drug_name": c.metadata.drug_name or "",
                "section_title": c.metadata.section_title,
                "tags": ",".join(sorted(c.metadata.tags)),
                "organization": c.metadata.organization or "",
                "publication_year": c.metadata.publication_year or "",
            }
            for c in chunks
        ],
    )

