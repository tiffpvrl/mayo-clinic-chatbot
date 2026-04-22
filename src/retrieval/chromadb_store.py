from pathlib import Path

import chromadb
import numpy as np

from src.data_processing.document_processor import ProcessedChunk
from src.retrieval.embedder import Embedding
from src.config import (
    CHROMA_PATH,
    CHROMA_CINICAL_COLLECTION,
    CHROMA_QA_COLLECTION,
    CHROMA_CONVO_COLLECTION,
    EMBEDDING_MODEL,
    BATCH_SIZE,
)

embedder = Embedding(model_type=EMBEDDING_MODEL)

client = chromadb.PersistentClient(CHROMA_PATH)

clinical_collection = client.get_or_create_collection(
    name=CHROMA_CINICAL_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

qa_collection = client.get_or_create_collection(
    name=CHROMA_QA_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

conversation_collection = client.get_or_create_collection(
    name=CHROMA_CONVO_COLLECTION,
    metadata={"hnsw:space": "cosine"},
)


def _clinical_metadata_for_chroma(c: ProcessedChunk) -> dict:
    """Flatten ChunkMetadata to Chroma-safe string values (no None)."""
    meta = c.metadata
    src = meta.source_file or ""
    source_basename = Path(src).name if src else ""
    dn = (meta.drug_name or "").strip()
    return {
        "document_type": meta.document_type.value,
        "drug_name": dn.upper() if dn else "",
        "section_title": meta.section_title or "",
        "tags": ",".join(sorted(meta.tags)),
        "organization": meta.organization or "",
        "publication_year": meta.publication_year or "",
        "audience_tier": meta.audience_tier or "",
        "content_use_policy": meta.content_use_policy or "",
        "source_category": meta.source_category or "",
        "source_file": source_basename,
        "patient_source_label": meta.patient_source_label or "",
    }


def index_clinical_chunks(chunks: list[ProcessedChunk]) -> None:
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        embed_inputs = [
            f"{c.metadata.section_title}\n\n{c.content}" if c.metadata.section_title else c.content
            for c in batch
        ]
        embeddings = np.array(embedder.encode(embed_inputs))
        clinical_collection.upsert(
            ids=[c.id for c in batch],
            embeddings=embeddings,
            documents=[c.content for c in batch],
            metadatas=[_clinical_metadata_for_chroma(c) for c in batch],
        )
        print(f"  clinical_collection: indexed {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")


def index_qa_chunks(chunks: list[dict]) -> None:
    """
    Index turn-level Q&A chunks into qa_collection in batches.
    Expects chunks.turn_level from process_conversational_dialogues() — no filtering needed.
    Embeds only patient_message for semantic search so query-to-question matching
    is not distorted by the response text. clinician_response is stored in metadata
    for retrieval-time tone reference.
    """
    if not chunks:
        return

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        embeddings = np.array(embedder.encode([c["metadata"]["patient_message"] for c in batch]))
        qa_collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["metadata"]["patient_message"] for c in batch],
            metadatas=[
                {
                    "document_type": c["metadata"]["document_type"],
                    "chunk_type": c["metadata"]["chunk_type"],
                    "conversation_id": c["metadata"]["conversation_id"],
                    "turn_number": c["metadata"]["turn_number"],
                    "risk_tier": c["metadata"]["risk_tier"],
                    "prep_type": c["metadata"]["prep_type"],
                    "appointment_time": c["metadata"]["appointment_time"],
                    "is_follow_up": c["metadata"]["is_follow_up"],
                    "patient_message": c["metadata"]["patient_message"],
                    "clinician_response": c["metadata"]["clinician_response"],
                    "escalated_to_clinician": c["metadata"]["escalated_to_clinician"],
                    "escalation_reason": c["metadata"]["escalation_reason"],
                    "tags": ",".join(c["metadata"]["tags"]),
                }
                for c in batch
            ],
        )
        print(f"  qa_collection: indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")


def index_conversation_chunks(chunks: list[dict]) -> None:
    """
    Index conversation-level chunks into conversation_collection in batches.
    Expects chunks.conversation_level from process_conversational_dialogues() — no filtering needed.
    Embeds the full multi-turn thread so semantically similar conversations are retrieved together.
    """
    if not chunks:
        return

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        embeddings = np.array(embedder.encode([c["content"] for c in batch]))
        conversation_collection.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings,
            documents=[c["content"] for c in batch],
            metadatas=[
                {
                    "document_type": c["metadata"]["document_type"],
                    "chunk_type": c["metadata"]["chunk_type"],
                    "conversation_id": c["metadata"]["conversation_id"],
                    "num_turns": c["metadata"]["num_turns"],
                    "risk_tier": c["metadata"]["risk_tier"],
                    "prep_type": c["metadata"]["prep_type"],
                    "appointment_time": c["metadata"]["appointment_time"],
                    "demonstrates_multi_turn": c["metadata"]["demonstrates_multi_turn"],
                    "any_escalated": c["metadata"]["any_escalated"],
                    "tags": ",".join(c["metadata"]["tags"]),
                }
                for c in batch
            ],
        )
        print(f"  conversation_collection: indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
