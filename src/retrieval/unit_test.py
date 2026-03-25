
""" Local unit test to check RAG results """
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.retrieval.rag import (
    retrieve_clinical,
    retrieve_qa,
    retrieve_conversations,
    format_clinical_context,
    format_qa_context,
    format_conversation_context,
)
from src.patient_data.patient_context import build_patient_context

# Mock patient record — no BigQuery needed
MOCK_PATIENT_RECORD = {
    "patient_id": "test-patient",
    "diabetes": True,
    "heart_failure": False,
    "cirrhosis": False,
    "ibd_diagnosis": False,
    "chronic_constipation": False,
    "ckd_stage": None,
    "colonoscopy_datetime": "2026-03-24T10:00:00",
    "prep_agent": "MiraLAX",
    "risk_tier": None,
}


def main() -> None:
    query = "I have diabetes, is there anything I should do regarding prep?"

    patient_record = MOCK_PATIENT_RECORD
    patient_context = build_patient_context(patient_record)

    print("patient_record:", "found (mock)" if patient_record else None)

    clinical_hits = retrieve_clinical(query, patient_record=patient_record)
    qa_hits = retrieve_qa(query)
    conversation_hits = retrieve_conversations(query)

    print("\n--- clinical hits ---")
    for h in clinical_hits:
        print(h.get("distance"), h.get("id"))

    print("\n--- Q&A hits ---")
    for h in qa_hits:
        print(h.get("distance"), h.get("id"))

    print("\n--- conversation hits ---")
    for h in conversation_hits:
        print(h.get("distance"), h.get("id"))

    combined_context = f"""
PATIENT-SPECIFIC CONTEXT
{patient_context}

CLINICAL KNOWLEDGE BASE
{format_clinical_context(clinical_hits)}

SIMILAR Q&A EXAMPLES
{format_qa_context(qa_hits)}

SIMILAR CONVERSATION FLOWS
{format_conversation_context(conversation_hits)}
""".strip()

    print("\n--- combined context ---")
    print(combined_context)


if __name__ == "__main__":
    main()
