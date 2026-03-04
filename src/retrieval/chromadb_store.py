import chromadb
from embedder import Embedding
from data_processing.document_processor import ProcessedChunk
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import EMBEDDING_MODEL, CHROMA_PATH, CHROMA_CINICAL_COLLECTION, CHROMA_QA_COLLECTION, CHROMA_CONVO_COLLECTION, BATCH_SIZE

embedder = Embedding(model_type=EMBEDDING_MODEL)

client = chromadb.PersistentClient(CHROMA_PATH)

# 3 seprate ChromaDB collections
clinical_collection = client.get_or_create_collection(
    name=CHROMA_CINICAL_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

qa_collection = client.get_or_create_collection(
    name=CHROMA_QA_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

conversation_collection = client.get_or_create_collection(
    name=CHROMA_CONVO_COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

def index_clinical_chunks(chunks: list[ProcessedChunk]) -> None:
    embeddings = np.array(embedder.encode([c.content for c in chunks]))
    clinical_collection.add(
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


def index_qa_chunks(chunks: list[dict]) -> None:
    """
    Index turn-level Q&A chunks into qa_collection in batches.
    Expects chunks.turn_level from process_conversational_dialogues() — no filtering needed.
    Embeds only patient_message for semantic search so query-to-question matching
    is not distorted by the response text. chatbot_response is stored in metadata
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
                    "query_category": c["metadata"]["query_category"],
                    "prep_type": c["metadata"]["prep_type"],
                    "appointment_time": c["metadata"]["appointment_time"],
                    "days_relative_to_procedure": c["metadata"]["days_relative_to_procedure"],
                    "is_follow_up": c["metadata"]["is_follow_up"],
                    "patient_message": c["metadata"]["patient_message"],
                    "chatbot_response": c["metadata"]["chatbot_response"],
                    "tags": ",".join(c["metadata"]["tags"]),
                    "timestamp": c["metadata"].get("timestamp") or "",
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
                    "prep_type": c["metadata"]["prep_type"],
                    "appointment_time": c["metadata"]["appointment_time"],
                    "demonstrates_multi_turn": c["metadata"]["demonstrates_multi_turn"],
                    "conversation_flow": c["metadata"]["conversation_flow"],
                    "query_categories": ",".join(c["metadata"]["query_categories"]),
                    "tags": ",".join(c["metadata"]["tags"]),
                }
                for c in batch
            ],
        )
        print(f"  conversation_collection: indexed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")

