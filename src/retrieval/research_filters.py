"""
Heuristics for excluding trial/meta-analysis chunks from default RAG retrieval.

No heavy dependencies (embedders, Chroma) so tests and tooling can import safely.
"""

from __future__ import annotations

import re

AUDIENCE_RESEARCH_EDUCATION = "research_education"

_RESEARCH_INTENT_KEYWORDS = (
    "clinical trial",
    "clinical trials",
    "randomized",
    "rct",
    "meta-analysis",
    "meta analysis",
    "systematic review",
    "cohort study",
    "study ",
    "studies ",
    "research says",
    "what does the evidence",
    "what does research",
    "published study",
    "journal article",
    "peer-reviewed",
    "peer reviewed",
)


def query_requests_research_evidence(query: str) -> bool:
    """True when the query is likely asking for trial/meta-analysis style information."""
    q = query.lower()
    if any(kw in q for kw in _RESEARCH_INTENT_KEYWORDS):
        return True
    if re.search(r"\b(clinical\s+trial|meta[- ]?analysis|systematic\s+review)\b", q):
        return True
    if re.search(r"\btrials?\b", q):
        return True
    return False


def is_research_background_metadata(metadata: dict) -> bool:
    """True for trial/meta-analysis chunks (indexed audience_tier or legacy tags substring)."""
    if metadata.get("audience_tier") == AUDIENCE_RESEARCH_EDUCATION:
        return True
    tags = metadata.get("tags") or ""
    return "content_policy:research_background" in tags
