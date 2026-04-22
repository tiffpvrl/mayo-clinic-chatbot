"""
Deterministic, precomputable patient-facing labels for document sources (PDF stems, drug registries).
Used at KB processing time and as a fallback when Chroma metadata omits patient_source_label.
"""

from __future__ import annotations

import re
from typing import Any

# Longer prefixes first so clevelandclinic matches before cleveland
_STEM_PREFIX_TITLES: list[tuple[str, str]] = [
    ("clevelandclinic", "Cleveland Clinic"),
    ("mayoclinichealthsystem", "Mayo Clinic"),
    ("mayoclinic", "Mayo Clinic"),
    ("massgeneral", "Mass General Brigham"),
    ("bostonbowelprep", "Boston Bowel Prep"),
    ("colonoscopyquality", "Colonoscopy quality"),
    ("typeofbowelprep", "Types of bowel preparation"),
    ("ucsf", "UCSF"),
    ("sgna", "SGNA"),
    ("metaanalysis", "Meta-analysis"),
    ("clinicaltrial", "Clinical trial"),
    ("usmstf", "US Multi-Society Task Force"),
    ("asge", "ASGE"),
    ("acg", "ACG"),
    ("aga", "AGA"),
]

_DAILYMED = "DailyMed"
_OPENFDA = "OpenFDA"


def drug_patient_source_label(label_source: str | None) -> str | None:
    """Return DailyMed or OpenFDA when applicable; else None."""
    s = (label_source or "").strip()
    if s in (_DAILYMED, _OPENFDA):
        return s
    return None


def patient_source_label_from_stem(stem: str) -> str:
    """Map a filename stem to a short display name (no LLM)."""
    s = stem.lower().strip()
    if not s or s == "source":
        return "Clinical document"
    for prefix, title in _STEM_PREFIX_TITLES:
        if s == prefix or s.startswith(prefix + "_") or s.startswith(prefix + "-"):
            return title
    if re.match(r"^study_?\d+$", s) or s.startswith("study_"):
        return "Clinical trial"
    words = re.split(r"[_\s]+", stem)
    words = [w for w in words if w]
    if not words:
        return "Clinical document"
    return " ".join(w.capitalize() for w in words)


def drug_patient_label_from_metadata(meta: dict[str, Any]) -> str | None:
    """Derive DailyMed / OpenFDA from tags or organization (for legacy chunks)."""
    tags_raw = (meta.get("tags") or "").strip()
    for part in tags_raw.split(","):
        part = part.strip()
        if part.startswith("label_source:"):
            key = part.split(":", 1)[-1].strip()
            if key in (_DAILYMED, _OPENFDA):
                return key
    org = (meta.get("organization") or "").strip()
    if org in (_DAILYMED, _OPENFDA):
        return org
    return None
