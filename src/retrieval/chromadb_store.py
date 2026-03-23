from pathlib import Path

import chromadb
import numpy as np

from src.data_processing.document_processor import ProcessedChunk
from src.retrieval.embedder import Embedding
from src.config import CHROMA_PATH, CHROMA_COLLECTION, EMBEDDING_MODEL

embedder = Embedding(model_type=EMBEDDING_MODEL)

client = chromadb.PersistentClient(CHROMA_PATH)
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

def _chunk_metadata_for_chroma(c: ProcessedChunk) -> dict:
    """Flatten ChunkMetadata to Chroma-safe string values (no None)."""
    meta = c.metadata
    src = meta.source_file or ""
    source_basename = Path(src).name if src else ""
    dn = (meta.drug_name or "").strip()
    drug_name_indexed = dn.upper() if dn else ""
    return {
        "document_type": meta.document_type.value,
        "drug_name": drug_name_indexed,
        "section_title": meta.section_title or "",
        "tags": ",".join(sorted(meta.tags)),
        "organization": meta.organization or "",
        "publication_year": meta.publication_year or "",
        "audience_tier": meta.audience_tier or "",
        "content_use_policy": meta.content_use_policy or "",
        "source_category": meta.source_category or "",
        "source_file": source_basename,
    }


def index_chunks(chunks: list[ProcessedChunk]) -> None:
    embeddings = np.array(embedder.encode([c.content for c in chunks]))
    collection.add(
        ids=[c.id for c in chunks],
        embeddings=embeddings,
        documents=[c.content for c in chunks],
        metadatas=[_chunk_metadata_for_chroma(c) for c in chunks],
    )

