"""
RAG pipeline: ties together embedder.py (query encoding) and
chromadb_store.py (vector search) to retrieve relevant chunks
for a given user query.
"""

from __future__ import annotations

import logging
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any
from src.retrieval.embedder import Embedding
from src.retrieval.chromadb_store import clinical_collection, qa_collection, conversation_collection
from src.retrieval.filters import (
    _build_where,
    extract_filters,
    extract_qa_filters,
    extract_conversation_filters,
    extract_patient_filters,
)
from src.config import EMBEDDING_MODEL, CLINICAL_TOP_K, QA_TOP_K, CONVERSATION_TOP_K
from src.retrieval.research_filters import is_research_background_metadata

logger = logging.getLogger(__name__)

embedder = Embedding(model_type=EMBEDDING_MODEL)

# Cosine similarity threshold above which two retrieved hits are considered near-duplicates.
# 1.0 = identical vectors, 0.0 = orthogonal. 0.95 catches paraphrased duplicates
# (e.g. same question with minor wording change) while keeping genuinely different examples.
DIVERSITY_SIMILARITY_THRESHOLD = 0.95
# How many extra candidates to fetch so we have fallbacks after diversity filtering.
DIVERSITY_FETCH_MULTIPLIER = 3


CONTACT_PATTERNS = [
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\bphone\b",
    r"\bcall us\b",
    r"\bcontact\b",
    r"\bappointment\b",
    r"\bscheduling\b",
    r"\bportal\b",
    r"\bmychart\b",
]

ADMIN_PATTERNS = [
    r"\binsurance\b",
    r"\bschedule\b",
    r"\bappointment\b",
    r"\bbook\b",
    r"\bportal\b",
    r"\bmychart\b",
    r"\bcheck in\b",
    r"\bregistration\b",
]

OUTSIDE_HOSPITAL_PATTERNS = [
    r"\bcleveland clinic\b",
    r"\bmgh\b",
    r"\bmassachusetts general\b",
    r"\bmount sinai\b",
    r"\bnyu langone\b",
    r"\bexternal hospital\b",
]

PREFERRED_ORGANIZATIONS = {
    "mayo clinic",
    "mayo",
    "fda",
    "dailymed",
}


def _text_for_checks(hit: dict) -> str:
    metadata = hit.get("metadata", {})
    parts = [
        hit.get("document", ""),
        metadata.get("organization", "") or "",
        metadata.get("section_title", "") or "",
        metadata.get("source_file", "") or "",
    ]
    return " ".join(parts).lower()


def has_contact_info(hit: dict) -> bool:
    text = _text_for_checks(hit)
    return any(re.search(pattern, text) for pattern in CONTACT_PATTERNS)


def is_admin_chunk(hit: dict) -> bool:
    text = _text_for_checks(hit)
    return any(re.search(pattern, text) for pattern in ADMIN_PATTERNS)


def is_outside_hospital_chunk(hit: dict) -> bool:
    text = _text_for_checks(hit)
    org = (hit.get("metadata", {}).get("organization", "") or "").lower()

    if org and org not in PREFERRED_ORGANIZATIONS:
        if "mayo" not in org and "fda" not in org and "dailymed" not in org:
            if "clinic" in org or "hospital" in org or "medical center" in org:
                return True

    return any(re.search(pattern, text) for pattern in OUTSIDE_HOSPITAL_PATTERNS)


def source_priority(hit: dict) -> tuple[int, float]:
    metadata = hit.get("metadata", {})
    org = (metadata.get("organization", "") or "").lower()
    distance = hit.get("distance", 999)

    if "mayo" in org:
        priority = 0
    elif "fda" in org or "dailymed" in org:
        priority = 1
    elif org:
        priority = 2
    else:
        priority = 3

    return (priority, distance)


def postprocess_hits(hits: list[dict], wants_research: bool = False) -> list[dict]:
    """
    Add lightweight safety flags and reorder hits so preferred sources come first.
    Suppress chunks that are mostly contact/admin content, or research-background
    chunks when the query is not explicitly asking for clinical evidence/trials.

    wants_research: pre-computed by extract_query_understanding() in retrieve_rag_node.
    """
    cleaned = []

    for hit in hits:
        metadata = dict(hit.get("metadata", {}) or {})

        contact_flag = has_contact_info(hit)
        admin_flag = is_admin_chunk(hit)
        outside_flag = is_outside_hospital_chunk(hit)

        metadata["has_contact_info"] = contact_flag
        metadata["is_admin_chunk"] = admin_flag
        metadata["needs_source_caution"] = outside_flag

        # Drop chunks that look primarily administrative/contact-oriented
        if contact_flag and admin_flag:
            continue

        # Drop research-background chunks unless the patient is explicitly asking
        # for trial/study evidence — these are not appropriate for routine patient queries
        if not wants_research and is_research_background_metadata(metadata):
            continue

        hit["metadata"] = metadata
        cleaned.append(hit)

    cleaned.sort(key=source_priority)
    return cleaned


# ── 1. Retrieval ───────────────────────────────────────────────────────────────

def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two embedding vectors."""
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _diversify_hits(hits_with_embeddings: list[dict], top_k: int, threshold: float) -> list[dict]:
    """
    Greedy diversity filter: iterate candidates in distance order and add each
    only if its cosine similarity to every already-selected hit is below threshold.
    Falls back gracefully — if fewer than top_k diverse hits exist, returns all found.
    Strips the embedding vector from returned dicts (not needed downstream).
    """
    selected: list[dict] = []
    selected_embeddings: list[list] = []

    for hit in hits_with_embeddings:
        emb = hit.get("embedding")
        if emb is None:
            selected.append(hit)
            continue
        if all(_cosine_similarity(emb, s) < threshold for s in selected_embeddings):
            selected.append(hit)
            selected_embeddings.append(emb)
        if len(selected) >= top_k:
            break

    for h in selected:
        h.pop("embedding", None)
    return selected


def _query_collection(collection: Any, query_embedding: list, top_k: int, where: Any) -> list[dict]:
    """Run a single ChromaDB query and flatten the nested result into a list of dicts."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    docs = results["documents"] or [[]]
    metas = results["metadatas"] or [[]]
    dists = results["distances"] or [[]]
    ids = results["ids"] or [[]]
    for doc, meta, dist, id_ in zip(docs[0], metas[0], dists[0], ids[0]):
        hits.append({"id": id_, "document": doc, "metadata": meta, "distance": dist})
    return hits


def _query_collection_with_embeddings(collection: Any, query_embedding: list, top_k: int, where: Any) -> list[dict]:
    """Like _query_collection but also returns each hit's stored embedding for similarity comparison."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    hits = []
    docs = results["documents"] or [[]]
    metas = results["metadatas"] or [[]]
    dists = results["distances"] or [[]]
    ids = results["ids"] or [[]]
    embs = results["embeddings"] or [[]]
    for doc, meta, dist, id_, emb in zip(docs[0], metas[0], dists[0], ids[0], embs[0]):
        hits.append({"id": id_, "document": doc, "metadata": meta, "distance": dist, "embedding": emb})
    return hits


def _union_query_with_embeddings(collection: Any, query_embedding: list, fetch_k: int, where: Any) -> list[dict]:
    """
    Like _union_query but fetches embeddings for diversity filtering.
    Merges filtered and unfiltered results by best distance per unique chunk ID.
    If no filter is provided, falls through to a single unfiltered query.
    """
    unfiltered = _query_collection_with_embeddings(collection, query_embedding, fetch_k, None)
    if where is None:
        return unfiltered

    filtered = _query_collection_with_embeddings(collection, query_embedding, fetch_k, where)

    best: dict[str, dict] = {}
    for hit in filtered + unfiltered:
        id_ = hit["id"]
        if id_ not in best or hit["distance"] < best[id_]["distance"]:
            best[id_] = hit

    return sorted(best.values(), key=lambda h: h["distance"])[:fetch_k]


def _build_augmented_query(query: str, patient_record: dict) -> str:
    """
    Prepend a short patient summary to the query before embedding so the
    query vector sits closer to chunks relevant to this patient's specific
    conditions, not just the question in the abstract.

    Only the most clinically salient fields are included to keep the
    augmentation concise — a long prefix dilutes the query signal.
    """
    from src.retrieval.filters import _pos
    from datetime import datetime

    parts = []

    # Conditions
    if _pos(patient_record.get("diabetes")) or _pos(patient_record.get("on_diabetes_medication")):
        parts.append("diabetes")
    if _pos(patient_record.get("heart_failure")):
        parts.append("heart failure")
    if _pos(patient_record.get("cirrhosis")):
        parts.append("cirrhosis")
    if _pos(patient_record.get("ibd_diagnosis")):
        parts.append("inflammatory bowel disease")
    if _pos(patient_record.get("chronic_constipation")):
        parts.append("chronic constipation")

    ckd_stage = patient_record.get("ckd_stage")
    if ckd_stage not in (None, "", 0, "0"):
        parts.append(f"chronic kidney disease stage {ckd_stage}")

    # Procedure timing
    colonoscopy_dt = patient_record.get("colonoscopy_datetime")
    if colonoscopy_dt:
        try:
            if isinstance(colonoscopy_dt, str):
                colonoscopy_dt = datetime.fromisoformat(colonoscopy_dt)
            parts.append("morning procedure" if colonoscopy_dt.hour < 12 else "afternoon procedure")
        except Exception:
            pass

    # Prep agent
    prep_agent = patient_record.get("prep_agent")
    if prep_agent:
        parts.append(f"{prep_agent} bowel prep")

    if not parts:
        return query

    patient_summary = ", ".join(parts)
    return f"Patient context: {patient_summary}. Query: {query}"


def retrieve_clinical(
    query: str,
    top_k: int = CLINICAL_TOP_K,
    patient_record: dict | None = None,
    query_where: dict | None = None,
    wants_research: bool = False,
) -> list[dict]:
    """
    Embed the query, run cosine search in clinical_collection, return top_k results.

    patient_record: when provided, patient EHR filters are merged with
        query-based filters so results are scoped to chunks relevant to
        both what the patient asked AND who the patient is.
    query_where: pre-built ChromaDB where clause from build_clinical_where().
        When provided, skips the extract_filters() LLM call — caller is
        responsible for running extract_query_understanding() once upstream.
    wants_research: pre-computed research intent flag, passed through to
        postprocess_hits() to suppress research chunks when not needed.

    Falls back to unfiltered top-k if the filtered query returns no results.
    Each result dict has keys: id, document, metadata, distance.
    """
    augmented_query = _build_augmented_query(query, patient_record) if patient_record else query
    query_embedding = embedder.encode([augmented_query])[0]

    patient_where = None
    where: Any = None
    try:
        if query_where is None:
            logger.warning("[clinical] query_where not provided — falling back to extract_filters (extra LLM call)")
            query_where = extract_filters(query)
        patient_where = extract_patient_filters(patient_record) if patient_record else None
        if query_where and patient_where:
            where = {"$and": [query_where, patient_where]}
        else:
            where = query_where or patient_where
    except Exception as e:
        print(f"[clinical] Filter extraction error: {e}")

    print(f"[clinical] augmented_query={augmented_query!r}")
    print(f"[clinical] patient_filter={patient_where}  combined_filter={where}")
    fetch_k = top_k * DIVERSITY_FETCH_MULTIPLIER
    candidates = _union_query_with_embeddings(clinical_collection, query_embedding, fetch_k, where)
    hits = _diversify_hits(candidates, top_k, DIVERSITY_SIMILARITY_THRESHOLD)
    print(f"[clinical] fetched={len(candidates)}  after_diversity={len(hits)}  threshold={DIVERSITY_SIMILARITY_THRESHOLD}")
    return postprocess_hits(hits, wants_research=wants_research)


def retrieve_qa(query: str, top_k: int = QA_TOP_K, is_follow_up: bool | None = None, risk_tier: str | None = None, augmented_query: str | None = None) -> list[dict]:
    """
    Retrieve turn-level Q&A examples from qa_collection.
    Used for tone/phrasing reference — caller surfaces clinician_response from metadata.

    Over-fetches by DIVERSITY_FETCH_MULTIPLIER then applies cosine-similarity diversity
    filtering so the returned top_k examples are semantically distinct from each other.

    augmented_query: when provided, used for embedding instead of raw query so the
        vector reflects the patient's specific context (conditions, prep agent, timing).
    is_follow_up: when provided, filters examples to the same turn state —
        True  → retrieve follow-up turn examples (turn_number > 1)
        False → retrieve first-turn examples
        None  → no filter (default)
    """
    query_embedding = embedder.encode([augmented_query or query])[0]
    fetch_k = top_k * DIVERSITY_FETCH_MULTIPLIER
    where: Any = None
    try:
        conditions = []
        keyword_where = extract_qa_filters(query, risk_tier=risk_tier)
        if keyword_where:
            conditions.append(keyword_where)
        if is_follow_up is not None:
            conditions.append({"is_follow_up": {"$eq": is_follow_up}})
        where = _build_where(conditions)
    except Exception as e:
        print(f"[qa] Filter extraction error: {e}")

    candidates = _union_query_with_embeddings(qa_collection, query_embedding, fetch_k, where)
    diverse = _diversify_hits(candidates, top_k, DIVERSITY_SIMILARITY_THRESHOLD)
    print(f"[qa] fetched={len(candidates)}  after_diversity={len(diverse)}  threshold={DIVERSITY_SIMILARITY_THRESHOLD}")
    return diverse


def retrieve_conversations(query: str, top_k: int = CONVERSATION_TOP_K, is_follow_up: bool | None = None, risk_tier: str | None = None, augmented_query: str | None = None) -> list[dict]:
    """
    Retrieve full conversation threads from conversation_collection.
    Used for multi-turn flow reference — shows how similar questions were handled end-to-end.

    Over-fetches by DIVERSITY_FETCH_MULTIPLIER then applies cosine-similarity diversity
    filtering so returned threads are semantically distinct from each other.

    augmented_query: when provided, used for embedding instead of raw query so the
        vector reflects the patient's specific context (conditions, prep agent, timing).
    is_follow_up: when True, restricts results to multi-turn conversations
        (demonstrates_multi_turn=True) so the LLM sees flow examples that
        actually demonstrate follow-up handling.
        False/None → no filter on demonstrates_multi_turn.
    """
    query_embedding = embedder.encode([augmented_query or query])[0]
    fetch_k = top_k * DIVERSITY_FETCH_MULTIPLIER
    where: Any = None
    try:
        conditions = []
        keyword_where = extract_conversation_filters(query, risk_tier=risk_tier)
        if keyword_where:
            conditions.append(keyword_where)
        if is_follow_up:
            conditions.append({"demonstrates_multi_turn": {"$eq": True}})
        where = _build_where(conditions)
    except Exception as e:
        print(f"[conversation] Filter extraction error: {e}")

    candidates = _union_query_with_embeddings(conversation_collection, query_embedding, fetch_k, where)
    diverse = _diversify_hits(candidates, top_k, DIVERSITY_SIMILARITY_THRESHOLD)
    print(f"[conversation] fetched={len(candidates)}  after_diversity={len(diverse)}  threshold={DIVERSITY_SIMILARITY_THRESHOLD}")
    return diverse


# ── 2. Context formatting ──────────────────────────────────────────────────────

def format_clinical_context(hits: list[dict]) -> str:
    """
    Format clinical_collection hits into a prompt-ready string.
    Returns a sentinel string when no hits are available.
    """
    if not hits:
        return "No relevant clinical information found."

    blocks = []
    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        org = meta.get("organization") or "Unknown source"
        year = meta.get("publication_year") or ""
        section = meta.get("section_title") or ""
        doc_type = meta.get("document_type") or ""
        drug = meta.get("drug_name") or ""

        parts = [f"source: {org}"]
        if year:
            parts.append(f"year: {year}")
        if doc_type:
            parts.append(f"type: {doc_type}")
        if drug:
            parts.append(f"drug: {drug}")
        if section:
            parts.append(f"section: {section}")

        label = f"[Clinical Source {i+1} | {' | '.join(parts)}]"
        blocks.append(f"{label}\n{hit['document']}")

    return "\n\n---\n\n".join(blocks)


def format_qa_context(hits: list[dict]) -> str:
    """
    Format qa_collection hits (turn-level Q&A pairs) into a prompt-ready string.
    Surfaces clinician_response from metadata for tone/phrasing reference.
    Returns a sentinel string when no hits are available.
    """
    if not hits:
        return "No similar Q&A examples found."

    blocks = []
    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        risk_tier = meta.get("risk_tier") or ""
        turn = meta.get("turn_number") or ""
        patient_msg = meta.get("patient_message") or ""
        clinician_resp = meta.get("clinician_response") or hit["document"]

        label = f"[Q&A Example {i+1} | risk: {risk_tier} | turn: {turn}]"
        blocks.append(
            f"{label}\n"
            f"Similar patient question: {patient_msg}\n"
            f"Example clinician response: {clinician_resp}"
        )

    return "\n\n---\n\n".join(blocks)


def format_qa_context_as_evidence(hits: list[dict]) -> str:
    """
    Format Q&A hits as factual evidence when clinical retrieval has failed.
    Promotes clinician_response to the primary content and labels each block
    as a Clinician Dialogue Example so the generator and judge know the evidence tier.
    """
    if not hits:
        return ""

    blocks = []
    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        risk_tier = meta.get("risk_tier") or ""
        patient_msg = meta.get("patient_message") or ""
        clinician_resp = meta.get("clinician_response") or hit["document"]

        label = f"[Clinician Dialogue Example {i+1} | risk: {risk_tier}]"
        blocks.append(
            f"{label}\n"
            f"Patient asked: {patient_msg}\n"
            f"Clinician responded: {clinician_resp}"
        )

    return "\n\n---\n\n".join(blocks)


def format_conversation_context_as_evidence(hits: list[dict]) -> str:
    """
    Format conversation hits as factual evidence when clinical retrieval has failed.
    Labels each block as a Clinician Dialogue Example so the generator and judge
    know the evidence tier.
    """
    if not hits:
        return ""

    blocks = []
    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        risk_tier = meta.get("risk_tier") or ""
        num_turns = meta.get("num_turns") or ""

        parts = []
        if risk_tier:
            parts.append(f"risk: {risk_tier}")
        if num_turns:
            parts.append(f"turns: {num_turns}")

        label = f"[Clinician Dialogue Example {i+1} | {' | '.join(parts)}]"
        blocks.append(f"{label}\n{hit['document']}")

    return "\n\n---\n\n".join(blocks)


def format_conversation_context(hits: list[dict]) -> str:
    """
    Format conversation_collection hits (full multi-turn threads) into a
    prompt-ready string.
    Returns a sentinel string when no hits are available.
    """
    if not hits:
        return "No similar conversation flows found."

    blocks = []
    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        risk_tier = meta.get("risk_tier") or ""
        num_turns = meta.get("num_turns") or ""
        appt_time = meta.get("appointment_time") or ""

        parts = []
        if risk_tier:
            parts.append(f"risk: {risk_tier}")
        if num_turns:
            parts.append(f"turns: {num_turns}")
        if appt_time:
            parts.append(f"appointment: {appt_time}")

        label = f"[Conversation Example {i+1} | {' | '.join(parts)}]"
        blocks.append(f"{label}\n{hit['document']}")

    return "\n\n---\n\n".join(blocks)
