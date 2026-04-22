"""Unit tests for patient-facing RAG source labels."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from src.retrieval.source_display import _heuristic_title_from_stem, build_patient_facing_clinical_sources


class TestHeuristicTitle(unittest.TestCase):
    def test_clevelandclinic_stem(self) -> None:
        self.assertEqual(_heuristic_title_from_stem("clevelandclinic_003"), "Cleveland Clinic")

    def test_study_number(self) -> None:
        self.assertEqual(_heuristic_title_from_stem("study1"), "Clinical trial")

    def test_generic_underscores(self) -> None:
        self.assertEqual(_heuristic_title_from_stem("foo_bar_baz"), "Foo Bar Baz")


class TestBuildPatientSources(unittest.TestCase):
    def test_drug_label_shows_dailymed_or_openfda_only(self) -> None:
        hit = {
            "id": "d1",
            "document": "Peg-3350...",
            "metadata": {
                "document_type": "drug_label",
                "section_title": "DOSAGE",
                "tags": "dosing,label_source:DailyMed",
            },
        }
        out = build_patient_facing_clinical_sources([hit])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["source_name"], "DailyMed")
        self.assertEqual(out[0]["title"], "DOSAGE")

    def test_drug_label_openfda_from_tags(self) -> None:
        hit = {
            "id": "d2",
            "document": "text",
            "metadata": {
                "document_type": "drug_label",
                "section_title": "WARNINGS",
                "tags": "label_source:OpenFDA",
            },
        }
        out = build_patient_facing_clinical_sources([hit])
        self.assertEqual(out[0]["source_name"], "OpenFDA")

    @patch("vertexai.generative_models.GenerativeModel")
    def test_non_drug_uses_llm_title(self, mock_gm: MagicMock) -> None:
        mock_gm.return_value.generate_content.return_value.text = json.dumps(
            {
                "items": [
                    {"stem": "clevelandclinic_003", "display_title": "Cleveland Clinic"},
                ]
            }
        )
        hit = {
            "id": "p1",
            "document": "Some handout text. " * 20,
            "metadata": {
                "document_type": "patient_instructions",
                "source_file": "clevelandclinic_003.md",
                "section_title": "3 Days Before",
            },
        }
        out = build_patient_facing_clinical_sources([hit])
        self.assertEqual(out[0]["source_name"], "Cleveland Clinic")
        self.assertEqual(out[0]["title"], "3 Days Before")
        mock_gm.assert_called_once()

    @patch("vertexai.generative_models.GenerativeModel", side_effect=Exception("no api"))
    def test_non_drug_falls_back_when_llm_fails(self, _mock_gm: MagicMock) -> None:
        hit = {
            "id": "p1",
            "document": "x" * 50,
            "metadata": {
                "document_type": "patient_instructions",
                "source_file": "clevelandclinic_003.md",
                "section_title": "Intro",
            },
        }
        out = build_patient_facing_clinical_sources([hit])
        self.assertEqual(out[0]["source_name"], "Cleveland Clinic")


if __name__ == "__main__":
    unittest.main()
