"""
One-time script to process the knowledge base and populate ChromaDB
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_processing.document_processor import process_patient_kb
from chromadb_store import index_chunks, collection

if __name__ == "__main__":
    # Wipe existing collection so re-runs don't hit duplicate ID errors
    existing = collection.count()
    if existing > 0:
        print(f"Collection already has {existing} chunks. Re-indexing...")
        collection.delete(where={"document_type": {"$ne": ""}})  # delete all

    chunks = process_patient_kb(kb_dir="src/data_processing/patient_kb")
    index_chunks(chunks)
    print("Finished indexing chunks into ChromaDB")