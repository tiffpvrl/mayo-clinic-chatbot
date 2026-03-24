"""
One-time script to process all knowledge bases and populate ChromaDB.

Run from the repository root:
  python -m src.retrieval.index_kb
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing.document_processor import process_patient_kb
from src.data_processing.conversation_processor import process_conversational_dialogues
from src.retrieval.chromadb_store import (
    clinical_collection,
    qa_collection,
    conversation_collection,
    index_clinical_chunks,
    index_qa_chunks,
    index_conversation_chunks,
)

KB_DIR = ROOT / "src/data_processing/patient_kb"
CLINICAL_OUT_JSON = ROOT / "src/data_processing/patient_kb/processed_chunks/clinical_processed_chunks.json"
CONV_INPUT = ROOT / "src/data_processing/patient_kb/conversations/mayo_clinic_patient_clinician_dialogues.xlsx"
CONV_OUT_JSON = ROOT / "src/data_processing/patient_kb/processed_chunks/conversational_chunks.json"


def _clear_collection(collection, name: str) -> None:
    existing = collection.count()
    if existing > 0:
        print(f"{name} already has {existing} chunks. Re-indexing...")
        collection.delete(where={"document_type": {"$ne": ""}})


if __name__ == "__main__":
    use_cached = "--reprocess" not in sys.argv #! YOU NEED TO INCLUDED THIS ARGUMENT IF YOU WANT TO RERUN THE DOCUMENT PROCESSING FOR CLINICAL

    # ── Clinical KB ────────────────────────────────────────────────────────────
    _clear_collection(clinical_collection, "clinical_collection")
    clinical_chunks = process_patient_kb(kb_dir=KB_DIR, output_path=CLINICAL_OUT_JSON, use_cached=use_cached)
    index_clinical_chunks(clinical_chunks)
    print(f"Finished indexing {len(clinical_chunks)} chunks into clinical_collection\n")

    # ── Conversational KB ──────────────────────────────────────────────────────
    _clear_collection(qa_collection, "qa_collection")
    _clear_collection(conversation_collection, "conversation_collection")
    conv_chunks = process_conversational_dialogues(input_file=CONV_INPUT, output_path=CONV_OUT_JSON)
    index_qa_chunks(conv_chunks.turn_level)
    print(f"Finished indexing {len(conv_chunks.turn_level)} chunks into qa_collection\n")
    index_conversation_chunks(conv_chunks.conversation_level)
    print(f"Finished indexing {len(conv_chunks.conversation_level)} chunks into conversation_collection\n")

    print("All collections indexed successfully.")
