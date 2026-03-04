"""
One-time script to process the knowledge base and populate ChromaDB
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from data_processing.document_processor import process_patient_kb
from data_processing.conversation_processor import process_conversational_dialogues
from chromadb_store import (
    client,
    index_clinical_chunks, index_qa_chunks, index_conversation_chunks,
    clinical_collection, qa_collection, conversation_collection,
)
from config import CHROMA_CINICAL_COLLECTION, CHROMA_QA_COLLECTION, CHROMA_CONVO_COLLECTION

CONV_INPUT = Path("src/data_processing/patient_kb/conversations/mayo_clinic_chatbot_dialogues.xlsx")
CONV_OUTPUT = Path("src/data_processing/patient_kb/processed_chunks/conversational_chunks.json")


def _wipe(collection, collection_name: str) -> None:
    """
    Delete and recreate a collection to guarantee a clean state before re-indexing.
    Using collection.delete(where=...) on large collections leaves ChromaDB's HNSW
    index and metadata segment out of sync, causing compaction errors on the next add().
    """
    count = collection.count()
    if count > 0:
        print(f"  '{collection_name}' has {count} existing chunks — dropping for re-index...")
        client.delete_collection(collection_name)
        client.get_or_create_collection(collection_name, metadata={"hnsw:space": "cosine"})
        print(f"  '{collection_name}' recreated.")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Clinical chunks → clinical_collection
    # ------------------------------------------------------------------
    print("--- Clinical indexing ---")
    _wipe(clinical_collection, CHROMA_CINICAL_COLLECTION)
    clinical_chunks = process_patient_kb(kb_dir="src/data_processing/patient_kb")
    index_clinical_chunks(clinical_chunks)
    print(f"Indexed {len(clinical_chunks)} clinical chunks\n")

    # ------------------------------------------------------------------
    # 2. Conversational chunks → qa_collection + conversation_collection
    # ------------------------------------------------------------------
    print("--- Conversational indexing ---")
    _wipe(qa_collection, CHROMA_QA_COLLECTION)
    _wipe(conversation_collection, CHROMA_CONVO_COLLECTION)

    conv = process_conversational_dialogues(input_file=CONV_INPUT, output_path=CONV_OUTPUT)

    index_qa_chunks(conv.turn_level)
    print(f"Indexed {len(conv.turn_level)} turn-level chunks into qa_collection")

    index_conversation_chunks(conv.conversation_level)
    print(f"Indexed {len(conv.conversation_level)} conversation-level chunks into conversation_collection")

    print("\nFinished indexing all collections into ChromaDB")