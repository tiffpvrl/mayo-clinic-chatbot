"""
One-time script to process the knowledge base and populate ChromaDB.

Run from the repository root, e.g.:
  python -m src.retrieval.index_kb
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root (parent of src/)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing.document_processor import process_patient_kb
from src.retrieval.chromadb_store import collection, index_chunks

KB_DIR = ROOT / "src/data_processing/patient_kb"
OUT_JSON = ROOT / "src/data_processing/patient_kb/processed_chunks/processed_chunks.json"


if __name__ == "__main__":
    existing = collection.count()
    if existing > 0:
        print(f"Collection already has {existing} chunks. Re-indexing...")
        collection.delete(where={"document_type": {"$ne": ""}})

    chunks = process_patient_kb(kb_dir=KB_DIR, output_path=OUT_JSON)
    index_chunks(chunks)
    print(f"Finished indexing {len(chunks)} chunks into ChromaDB")
