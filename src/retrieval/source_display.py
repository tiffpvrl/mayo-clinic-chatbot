"""
Patient-facing labels for RAG clinical source citations (Chroma hit shape).
Uses precomputed patient_source_label from chunk metadata when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.retrieval.patient_source_label import (
    drug_patient_label_from_metadata,
    patient_source_label_from_stem,
)

_CONVERSATIONAL_DOC_TYPE = "conversational_example"
_DRUG_LABEL = "drug_label"
_SNIPPET_MAX = 200

_DAILYMED = "DailyMed"
_OPENFDA = "OpenFDA"


def _stem_from_metadata(meta: dict[str, Any]) -> str:
    base = (meta.get("source_file") or "").strip()
    if not base and meta.get("id"):
        base = str(meta.get("id", ""))
    return Path(base).stem if base else "source"


def build_patient_facing_clinical_sources(clinical_hits: list[Any]) -> list[dict[str, str]]:
    """
    Build rows for the patient UI: source_name, section title, and snippet.
    Skips conversational_example chunks. No LLM — uses patient_source_label from
    Chroma (precomputed at index time) with heuristics as fallback.
    """
    rows: list[dict[str, str]] = []
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

        psl = (meta.get("patient_source_label") or "").strip()

        if doc_type == _DRUG_LABEL:
            if psl in (_DAILYMED, _OPENFDA):
                source_name = psl
            else:
                reg = drug_patient_label_from_metadata(meta)
                source_name = reg if reg in (_DAILYMED, _OPENFDA) else ""
        else:
            source_name = psl or patient_source_label_from_stem(_stem_from_metadata(meta))

        rows.append(
            {
                "source_name": source_name,
                "title": title,
                "snippet": snippet,
            }
        )
    return rows
