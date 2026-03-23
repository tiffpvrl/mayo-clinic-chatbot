"""
Manual smoke test for RAG (run from repo root).

  PYTHONPATH=. python src/retrieval/unit_test.py [patient_id]

Requires a valid patient_id in BigQuery for full context; otherwise patient_record may be None.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.rag import retrieve_for_query


def main() -> None:
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "REPLACE_WITH_VALID_PATIENT_ID"
    query = "can I drive after my colonoscopy?"
    patient_record, hits, context = retrieve_for_query(query, patient_id)
    print("patient_record:", "found" if patient_record else None)
    for h in hits:
        print(h.get("distance"), h.get("id"))
    print("--- context ---")
    print(context)


if __name__ == "__main__":
    main()
