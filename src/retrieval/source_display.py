"""
Patient-facing labels for RAG clinical source citations (Chroma hit shape).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from src.config import CLINICAL_SOURCE_DISPLAY_MODEL

logger = logging.getLogger(__name__)

_CONVERSATIONAL_DOC_TYPE = "conversational_example"
_DRUG_LABEL = "drug_label"
_SNIPPET_MAX = 200

_DAILYMED = "DailyMed"
_OPENFDA = "OpenFDA"

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

_TITLES_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stem": {"type": "string"},
                    "display_title": {"type": "string"},
                },
                "required": ["stem", "display_title"],
            },
        }
    },
    "required": ["items"],
}


def _drug_registry_from_metadata(meta: dict[str, Any]) -> str | None:
    """Return exactly DailyMed or OpenFDA when derivable from tags or organization."""
    tags_raw = (meta.get("tags") or "").strip()
    for part in tags_raw.split(","):
        part = part.strip()
        if part.startswith("label_source:"):
            key = part.split(":", 1)[-1].strip()
            if key == _DAILYMED or key == _OPENFDA:
                return key
    org = (meta.get("organization") or "").strip()
    if org in (_DAILYMED, _OPENFDA):
        return org
    return None


def _stem_from_metadata(meta: dict[str, Any]) -> str:
    base = (meta.get("source_file") or "").strip()
    if not base and meta.get("id"):
        base = str(meta.get("id", ""))
    return Path(base).stem if base else "source"


def _heuristic_title_from_stem(stem: str) -> str:
    s = stem.lower().strip()
    if not s or s == "source":
        return "Clinical document"
    for prefix, title in _STEM_PREFIX_TITLES:
        if s == prefix or s.startswith(prefix + "_") or s.startswith(prefix + "-"):
            return title
    # study3, study12
    if re.match(r"^study_?\d+$", s) or s.startswith("study_"):
        return "Clinical trial"
    # split_stem: foo_bar_baz -> title case words
    words = re.split(r"[_\s]+", stem)
    words = [w for w in words if w]
    if not words:
        return "Clinical document"
    return " ".join(w.capitalize() for w in words)


def _llm_display_titles(stem_doc_pairs: list[tuple[str, str]]) -> dict[str, str]:
    """
    One batched call: map filename stem -> short display title for patients.
    """
    if not stem_doc_pairs:
        return {}
    try:
        from vertexai.generative_models import GenerativeModel, GenerationConfig

        model = GenerativeModel(CLINICAL_SOURCE_DISPLAY_MODEL)
        body = [
            {"stem": stem, "document_type": doc_type or "unknown"}
            for stem, doc_type in stem_doc_pairs
        ]
        prompt = f"""You convert internal PDF file stems into short, patient-friendly document titles (1-5 words, Title Case).

Rules:
- Examples: clevelandclinic_003 -> "Cleveland Clinic", clinicaltrial_1 -> "Clinical Trial", usmstf_2020_bowelprep -> "US Multi-Society Task Force", mayoclinic_prep -> "Mayo Clinic".
- Use the document_type as a hint (clinical_guideline, patient_instructions, etc.) but the title should name the source organization or document, not the section.
- No file extensions, no quotation marks, no source numbers like "_003" in the output.

Stems to convert (JSON):
{json.dumps(body, ensure_ascii=True)}
"""
        response = model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=_TITLES_RESPONSE_SCHEMA,
                temperature=0.1,
                max_output_tokens=512,
            ),
        )
        raw = (response.text or "").strip()
        data = json.loads(raw)
        out: dict[str, str] = {}
        for it in data.get("items", []):
            st = (it.get("stem") or "").strip()
            title = (it.get("display_title") or "").strip()
            if st and title:
                out[st] = title
        return out
    except Exception as e:
        logger.warning("[source_display] LLM display titles failed: %s", e)
        return {}


def build_patient_facing_clinical_sources(clinical_hits: list[Any]) -> list[dict[str, str]]:
    """
    Build rows for the patient UI: source_name (document label or DailyMed/OpenFDA),
    section title, and snippet. Skips conversational_example chunks.
    """
    prepared: list[tuple[dict[str, str], str]] = []
    non_drug_stems: set[str] = set()
    stem_to_doctype: dict[str, str] = {}

    for h in clinical_hits:
        meta = h.get("metadata") or {}
        if (meta.get("document_type") or "").strip() == _CONVERSATIONAL_DOC_TYPE:
            continue
        doc = (h.get("document") or "").strip()
        doc_type = (meta.get("document_type") or "").strip()
        section = (meta.get("section_title") or "").strip()
        if section:
            title = section
        elif doc_type and doc_type not in ("", "unknown"):
            title = doc_type.replace("_", " ").title()
        else:
            title = (doc[:60] + "…") if len(doc) > 60 else doc
        if not title:
            title = "Clinical source"

        snippet = doc
        if len(snippet) > _SNIPPET_MAX:
            snippet = snippet[: _SNIPPET_MAX - 1].rstrip() + "…"

        if doc_type == _DRUG_LABEL:
            reg = _drug_registry_from_metadata(meta)
            source_name = reg if reg in (_DAILYMED, _OPENFDA) else ""
            row = {
                "source_name": source_name,
                "title": title,
                "snippet": snippet,
            }
            prepared.append((row, ""))
        else:
            stem = _stem_from_metadata(meta)
            non_drug_stems.add(stem)
            if stem not in stem_to_doctype and doc_type:
                stem_to_doctype[stem] = doc_type
            row = {
                "source_name": "",
                "title": title,
                "snippet": snippet,
            }
            prepared.append((row, stem))

    pairs: list[tuple[str, str]] = []
    for st in sorted(non_drug_stems):
        pairs.append((st, stem_to_doctype.get(st, "")))

    llm_map = _llm_display_titles(pairs)
    for row, stem in prepared:
        if stem:
            display = (llm_map.get(stem) or "").strip() or _heuristic_title_from_stem(stem)
            row["source_name"] = display

    return [p[0] for p in prepared]
