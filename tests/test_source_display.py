"""Unit tests for patient-facing RAG source labels."""

from __future__ import annotations

import unittest

from src.retrieval.patient_source_label import patient_source_label_from_stem
from src.retrieval.source_display import build_patient_facing_clinical_sources


class TestHeuristicTitle(unittest.TestCase):
    def test_clevelandclinic_stem(self) -> None:
        self.assertEqual(patient_source_label_from_stem("clevelandclinic_003"), "Cleveland Clinic")

    def test_study_number(self) -> None:
        self.assertEqual(patient_source_label_from_stem("study1"), "Clinical trial")

    def test_generic_underscores(self) -> None:
        self.assertEqual(patient_source_label_from_stem("foo_bar_baz"), "Foo Bar Baz")


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

    def test_drug_uses_precomputed_patient_source_label(self) -> None:
        hit = {
            "id": "d3",
            "document": "x",
            "metadata": {
                "document_type": "drug_label",
                "section_title": "X",
                "tags": "label_source:OpenFDA",
                "patient_source_label": "DailyMed",
            },
        }
        out = build_patient_facing_clinical_sources([hit])
        self.assertEqual(out[0]["source_name"], "DailyMed")

    def test_non_drug_uses_precomputed_patient_source_label(self) -> None:
        hit = {
            "id": "p1",
            "document": "Some handout text. " * 20,
            "metadata": {
                "document_type": "patient_instructions",
                "source_file": "clevelandclinic_003.md",
                "section_title": "3 Days Before",
                "patient_source_label": "Cleveland Clinic (Handout)",
            },
        }
        out = build_patient_facing_clinical_sources([hit])
        self.assertEqual(out[0]["source_name"], "Cleveland Clinic (Handout)")
        self.assertEqual(out[0]["title"], "3 Days Before")

    def test_non_drug_heuristic_when_not_precomputed(self) -> None:
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
