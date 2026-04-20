"""
GCP smoke test for the full RAG pipeline.

Requires:
  - GCP credentials (Application Default Credentials or service account)
  - ChromaDB populated at CHROMA_PATH
  - BigQuery patient table accessible

Run:
  python tests/test_rag_gcp.py <patient_id>
  python -m pytest tests/test_rag_gcp.py --patient-id <patient_id> -s
"""
from __future__ import annotations

import sys
import os
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
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
from src.patient_data.bigquery_client import get_patient_record
from src.patient_data.patient_context import build_patient_context


def _get_patient_id() -> str:
    # pytest: pass --patient-id <id> on the command line
    for i, arg in enumerate(sys.argv):
        if arg == "--patient-id" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    # direct: python tests/test_rag_gcp.py <patient_id>
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1]
    return "REPLACE_WITH_VALID_PATIENT_ID"


PATIENT_ID = _get_patient_id()
QUERY = "can I drive after my colonoscopy?"


class TestRAGPipelineGCP(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """Fetch patient record and run all three retrievers once; tests share the results."""
        cls.patient_record = get_patient_record(PATIENT_ID)
        cls.patient_context = (
            build_patient_context(cls.patient_record)
            if cls.patient_record
            else "No patient-specific data found."
        )
        cls.clinical_hits = retrieve_clinical(QUERY, patient_record=cls.patient_record)
        cls.qa_hits = retrieve_qa(QUERY)
        cls.conversation_hits = retrieve_conversations(QUERY)
        cls.combined_context = f"""PATIENT-SPECIFIC CONTEXT
{cls.patient_context}

CLINICAL KNOWLEDGE BASE
{format_clinical_context(cls.clinical_hits)}

SIMILAR Q&A EXAMPLES
{format_qa_context(cls.qa_hits)}

SIMILAR CONVERSATION FLOWS
{format_conversation_context(cls.conversation_hits)}""".strip()

    def test_patient_record_found(self) -> None:
        self.assertIsNotNone(self.patient_record, "No patient record returned from BigQuery")

    def test_clinical_hits_returned(self) -> None:
        self.assertGreater(len(self.clinical_hits), 0, "No clinical hits returned")

    def test_qa_hits_returned(self) -> None:
        self.assertGreater(len(self.qa_hits), 0, "No Q&A hits returned")

    def test_conversation_hits_returned(self) -> None:
        self.assertGreater(len(self.conversation_hits), 0, "No conversation hits returned")

    def test_hit_fields_present(self) -> None:
        for hit in self.clinical_hits:
            self.assertIn("id", hit)
            self.assertIn("distance", hit)
            self.assertIn("document", hit)
            self.assertIn("metadata", hit)

    def test_combined_context_non_empty(self) -> None:
        self.assertTrue(self.combined_context.strip())

    def test_no_sentinel_only_context(self) -> None:
        """Fail if every section returned a 'no results' sentinel — indicates empty ChromaDB."""
        sentinels = {
            "No relevant clinical information found.",
            "No similar Q&A examples found.",
            "No similar conversation flows found.",
        }
        self.assertFalse(
            all(s in self.combined_context for s in sentinels),
            "All three collections returned empty — ChromaDB may not be populated",
        )


def main() -> None:
    """Direct run: mirrors unit_test.py output exactly."""
    patient_record = get_patient_record(PATIENT_ID)
    clinical_hits = retrieve_clinical(QUERY, patient_record=patient_record)
    qa_hits = retrieve_qa(QUERY)
    conversation_hits = retrieve_conversations(QUERY)

    print("patient_record:", "found" if patient_record else None)

    print("\n--- clinical hits ---")
    for h in clinical_hits:
        print(h.get("distance"), h.get("id"))

    print("\n--- Q&A hits ---")
    for h in qa_hits:
        print(h.get("distance"), h.get("id"))

    print("\n--- conversation hits ---")
    for h in conversation_hits:
        print(h.get("distance"), h.get("id"))

    print("\n--- combined context ---")
    print(format_clinical_context(clinical_hits))
    print(format_qa_context(qa_hits))
    print(format_conversation_context(conversation_hits))


if __name__ == "__main__":
    main()
