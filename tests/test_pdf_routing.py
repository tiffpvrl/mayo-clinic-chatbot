"""Stem-based tests for PDF routing and manifest defaults."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Repo root: tests/ -> mayo/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_processing.document_processor import (  # noqa: E402
    DocumentType,
    AUDIENCE_TIER_RESEARCH_EDUCATION,
    detect_document_type,
    infer_default_pdf_manifest_entry,
    load_pdf_manifest,
    reset_pdf_manifest_cache,
    resolve_pdf_manifest_entry,
)


class TestDetectDocumentType(unittest.TestCase):
    """Real stems from patient_kb/pdf_assets."""

    def _dtype(self, stem: str) -> DocumentType:
        p = Path(f"/tmp/{stem}.pdf")
        return detect_document_type(p, "")

    def test_research_quality_stems_not_patient_instructions(self) -> None:
        for stem in (
            "colonoscopyquality",
            "colonoscopyquality2",
            "typeofbowelprep",
            "typeofbowelprep2",
            "clinicaltrial1",
            "metaanalysis",
            "metaanalysis0",
            "study1",
            "adenoma",
            "missedwork",
        ):
            with self.subTest(stem=stem):
                self.assertEqual(
                    self._dtype(stem),
                    DocumentType.CLINICAL_GUIDELINE,
                    f"{stem} should be guideline/research routing, not patient handout",
                )

    def test_bowel_prep_stems_still_patient_instructions(self) -> None:
        for stem in ("bowelprep2", "bowelprepguide"):
            with self.subTest(stem=stem):
                self.assertEqual(self._dtype(stem), DocumentType.PATIENT_INSTRUCTIONS)

    def test_boston_bowel_prep_is_guideline_not_generic_prep(self) -> None:
        self.assertEqual(self._dtype("bostonbowelprep"), DocumentType.CLINICAL_GUIDELINE)

    def test_society_hospital_stems_guideline(self) -> None:
        for stem in ("usmstf2025_1", "asge2015", "mayoclinic1", "massgeneral1", "clevelandclinic1", "ucsf_2"):
            with self.subTest(stem=stem):
                self.assertEqual(self._dtype(stem), DocumentType.CLINICAL_GUIDELINE)


class TestPdfManifestDefaults(unittest.TestCase):
    def test_trials_meta_study_research_education(self) -> None:
        for stem in ("clinicaltrial3", "metaanalysis7", "study16"):
            with self.subTest(stem=stem):
                d = infer_default_pdf_manifest_entry(stem)
                self.assertEqual(d.get("audience_tier"), AUDIENCE_TIER_RESEARCH_EDUCATION)

    def test_manifest_override_merges(self) -> None:
        reset_pdf_manifest_cache()
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "pdf_manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "files": {"study1": {"audience_tier": "clinician_guideline", "source_type": "other"}},
                    }
                ),
                encoding="utf-8",
            )
            m = load_pdf_manifest(base)
            r = resolve_pdf_manifest_entry("study1", m)
            self.assertEqual(r["audience_tier"], "clinician_guideline")
            r2 = resolve_pdf_manifest_entry("clinicaltrial1", m)
            self.assertEqual(r2["audience_tier"], AUDIENCE_TIER_RESEARCH_EDUCATION)
        reset_pdf_manifest_cache()


if __name__ == "__main__":
    unittest.main()
