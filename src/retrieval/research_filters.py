"""
Research intent detection for the RAG pipeline.

is_research_background_metadata() is a lightweight metadata check with no
LLM dependency — safe to import in tests and tooling.

Research intent (wants_research) is now extracted as part of the combined
extract_query_understanding() call in filters.py, eliminating the need for
a separate LLM call here.
"""

from __future__ import annotations

AUDIENCE_RESEARCH_EDUCATION = "research_education"


def is_research_background_metadata(metadata: dict) -> bool:
    """True for trial/meta-analysis chunks (indexed audience_tier or legacy tags substring)."""
    if metadata.get("audience_tier") == AUDIENCE_RESEARCH_EDUCATION:
        return True
    tags = metadata.get("tags") or ""
    return "content_policy:research_background" in tags
