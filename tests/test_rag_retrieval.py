"""Tests for RAG retrieval helpers (research exclusion intent)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from retrieval.research_filters import (  # noqa: E402
    is_research_background_metadata,
    query_requests_research_evidence,
)


class TestResearchIntent(unittest.TestCase):
    def test_evidence_queries(self) -> None:
        self.assertTrue(query_requests_research_evidence("What does the meta-analysis say about prep?"))
        self.assertTrue(query_requests_research_evidence("clinical trial on bowel prep"))
        self.assertTrue(query_requests_research_evidence("Are there RCTs for split dose?"))

    def test_routine_prep_queries(self) -> None:
        self.assertFalse(query_requests_research_evidence("When should I stop eating clear liquids?"))
        self.assertFalse(query_requests_research_evidence("How do I mix MoviPrep?"))


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
