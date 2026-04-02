"""Tests for RAG retrieval helpers (research exclusion intent + chunk detection)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.research_filters import is_research_background_metadata  # noqa: E402
from retrieval.filters import extract_query_understanding, build_clinical_where  # noqa: E402


class TestResearchIntent(unittest.TestCase):
    """
    Research intent is now extracted by extract_query_understanding() via a
    Gemini Flash LLM call. These tests mock the LLM response to verify that
    the wants_research flag is correctly read and that build_clinical_where()
    is unaffected by it (research intent is a postprocessing signal, not a filter).
    """

    def _mock_understanding(self, wants_research: bool) -> dict:
        return {
            "medication_class": None,
            "is_diabetes_query": False,
            "drug_name": None,
            "document_type": None,
            "procedure_timing": None,
            "wants_research": wants_research,
        }

    @patch("retrieval.filters.GenerativeModel")
    def test_evidence_queries_flagged(self, mock_model_cls) -> None:
        """LLM returning wants_research=True is surfaced correctly."""
        import json
        mock_model_cls.return_value.generate_content.return_value.text = json.dumps(
            self._mock_understanding(wants_research=True)
        )
        result = extract_query_understanding("Are there RCTs for split dose prep?")
        self.assertTrue(result.get("wants_research"))

    @patch("retrieval.filters.GenerativeModel")
    def test_routine_queries_not_flagged(self, mock_model_cls) -> None:
        """LLM returning wants_research=False is surfaced correctly."""
        import json
        mock_model_cls.return_value.generate_content.return_value.text = json.dumps(
            self._mock_understanding(wants_research=False)
        )
        result = extract_query_understanding("When should I stop eating clear liquids?")
        self.assertFalse(result.get("wants_research"))

    @patch("retrieval.filters.GenerativeModel")
    def test_llm_failure_defaults_to_false(self, mock_model_cls) -> None:
        """On LLM error, wants_research defaults to False (safe fallback)."""
        mock_model_cls.return_value.generate_content.side_effect = Exception("API error")
        result = extract_query_understanding("What does the meta-analysis say about prep?")
        self.assertFalse(result.get("wants_research", False))


class TestBuildClinicalWhere(unittest.TestCase):
    """Verify build_clinical_where() maps extracted fields to correct ChromaDB filters."""

    def test_medication_class(self) -> None:
        where = build_clinical_where({"medication_class": "anticoagulants", "is_diabetes_query": False, "wants_research": False})
        self.assertIsNotNone(where)
        self.assertIn("$contains", str(where))
        self.assertIn("anticoagulants", str(where))

    def test_diabetes_query_expands_to_or(self) -> None:
        where = build_clinical_where({"is_diabetes_query": True, "wants_research": False})
        self.assertIsNotNone(where)
        self.assertIn("metabolic_conditions", str(where))
        self.assertIn("diabetes_meds:oral_agents", str(where))
        self.assertIn("diabetes_meds:insulin", str(where))

    def test_valid_drug_name(self) -> None:
        where = build_clinical_where({"drug_name": "SUPREP", "is_diabetes_query": False, "wants_research": False})
        self.assertEqual(where, {"drug_name": {"$eq": "SUPREP"}})

    def test_invalid_drug_name_ignored(self) -> None:
        where = build_clinical_where({"drug_name": "HALLUCINATED_DRUG", "is_diabetes_query": False, "wants_research": False})
        self.assertIsNone(where)

    def test_no_signals_returns_none(self) -> None:
        where = build_clinical_where({"is_diabetes_query": False, "wants_research": False})
        self.assertIsNone(where)

    def test_wants_research_does_not_affect_where(self) -> None:
        """wants_research is a postprocessing flag, not a ChromaDB filter."""
        where_without = build_clinical_where({"is_diabetes_query": False, "wants_research": False})
        where_with = build_clinical_where({"is_diabetes_query": False, "wants_research": True})
        self.assertEqual(where_without, where_with)


class TestResearchChunkDetection(unittest.TestCase):
    def test_audience_tier(self) -> None:
        self.assertTrue(is_research_background_metadata({"audience_tier": "research_education"}))

    def test_tags_fallback(self) -> None:
        self.assertTrue(
            is_research_background_metadata(
                {"audience_tier": "", "tags": "foo,content_policy:research_background,bar"}
            )
        )

    def test_non_research(self) -> None:
        self.assertFalse(is_research_background_metadata({"audience_tier": "clinician_guideline", "tags": ""}))
        self.assertFalse(is_research_background_metadata({"audience_tier": "", "tags": "diet,split_dose"}))


if __name__ == "__main__":
    unittest.main()
