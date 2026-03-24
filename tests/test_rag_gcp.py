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

from src.retrieval.rag import retrieve_for_query


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
        """Run retrieve_for_query once; all tests share the result."""
        cls.result = retrieve_for_query(QUERY, PATIENT_ID)

    def test_patient_record_found(self) -> None:
        self.assertIsNotNone(self.result.patient_record, "No patient record returned from BigQuery")

    def test_clinical_hits_returned(self) -> None:
        self.assertGreater(len(self.result.clinical_hits), 0, "No clinical hits returned")

    def test_qa_hits_returned(self) -> None:
        self.assertGreater(len(self.result.qa_hits), 0, "No Q&A hits returned")

    def test_conversation_hits_returned(self) -> None:
        self.assertGreater(len(self.result.conversation_hits), 0, "No conversation hits returned")

    def test_hit_fields_present(self) -> None:
        for hit in self.result.clinical_hits:
            self.assertIn("id", hit)
            self.assertIn("distance", hit)
            self.assertIn("document", hit)
            self.assertIn("metadata", hit)

    def test_combined_context_non_empty(self) -> None:
        self.assertTrue(self.result.combined_context.strip())

    def test_no_sentinel_only_context(self) -> None:
        """Fail if every section returned a 'no results' sentinel — indicates empty ChromaDB."""
        sentinels = {
            "No relevant clinical information found.",
            "No similar Q&A examples found.",
            "No similar conversation flows found.",
        }
        self.assertFalse(
            all(s in self.result.combined_context for s in sentinels),
            "All three collections returned empty — ChromaDB may not be populated",
        )


def main() -> None:
    """Direct run: mirrors unit_test.py output exactly."""
    result = retrieve_for_query(QUERY, PATIENT_ID)

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
