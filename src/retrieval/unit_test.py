import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retrieval.rag import retrieve_for_query


def main() -> None:
    patient_id = sys.argv[1] if len(sys.argv) > 1 else "REPLACE_WITH_VALID_PATIENT_ID"
    query = "can I drive after my colonoscopy?"

    result = retrieve_for_query(query, patient_id)

    print("patient_record:", "found" if result.patient_record else None)

    print("\n--- clinical hits ---")
    for h in result.clinical_hits:
        print(h.get("distance"), h.get("id"))

    print("\n--- Q&A hits ---")
    for h in result.qa_hits:
        print(h.get("distance"), h.get("id"))

    print("\n--- conversation hits ---")
    for h in result.conversation_hits:
        print(h.get("distance"), h.get("id"))

    print("\n--- combined context ---")
    print(result.combined_context)


if __name__ == "__main__":
    main()
