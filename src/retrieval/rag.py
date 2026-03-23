"""
RAG pipeline: ties together embedder.py (query encoding) and
chromadb_store.py (vector search) to retrieve relevant chunks
for a given user query.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.embedder import Embedding
from src.retrieval.chromadb_store import collection
from src.config import EMBEDDING_MODEL, TOP_K
from src.patient_data.bigquery_client import get_patient_record
from src.patient_data.patient_context import build_patient_context

embedder = Embedding(model_type=EMBEDDING_MODEL)

import re


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


def postprocess_hits(hits: list[dict]) -> list[dict]:
    """
    Add lightweight safety flags and reorder hits so preferred sources come first.
    Suppress chunks that are mostly contact/admin content.
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

        hit["metadata"] = metadata
        cleaned.append(hit)

    cleaned.sort(key=source_priority)
    return cleaned

# ── 1. Query understanding ─────────────────────────────────────────────────────

def extract_filters(query: str) -> dict | None:
    """
    Translate natural-language cues in the query into ChromaDB `where` filters
    so the vector search is scoped before cosine distance is computed.

    Returns a Chroma `where` dict, or None for unfiltered search.
    Multiple conditions are combined with $and.

    Note: `tags` is stored as a comma-joined string in Chroma, so
    {"tags": {"$contains": "med_class:anticoagulants"}} does a substring match.
    """
    q = query.lower()
    conditions = []

    # Medication class keywords (mirrors MEDICATION_CLASSES in document_processor.py)
    MEDICATION_KEYWORDS = {
        "med_class:anticoagulants": ["warfarin", "coumadin", "xarelto", "rivaroxaban",
                                     "eliquis", "apixaban", "pradaxa", "dabigatran",
                                     "lovenox", "enoxaparin", "blood thinner"],
        "med_class:antiplatelet":   ["clopidogrel", "plavix", "ticagrelor", "brilinta",
                                     "prasugrel", "effient", "aspirin"],
        "med_class:diuretics":      ["furosemide", "lasix", "hydrochlorothiazide", "hctz",
                                     "spironolactone"],
        "med_class:ace_inhibitors": ["lisinopril", "enalapril", "ramipril",
                                     "ace inhibitor", "ace-inhibitor"],
        "med_class:sglt2_inhibitors": ["invokana", "canagliflozin", "farxiga",
                                       "dapagliflozin", "jardiance", "empagliflozin", "sglt2"],
        "med_class:nsaids":         ["ibuprofen", "advil", "naproxen", "aleve", "nsaid"],
    }

    for tag, keywords in MEDICATION_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            conditions.append({"tags": {"$contains": tag}})
            break  # one medication class filter at a time is enough

    # ── Specific bowel prep drug names → filter by drug_name field
    BOWEL_PREP_DRUGS = ["suprep", "golytely", "miralax", "moviprep",
                        "prepopik", "clenpiq", "plenvu", "suflave"]
    for drug in BOWEL_PREP_DRUGS:
        if drug in q:
            conditions.append({"drug_name": {"$eq": drug.upper()}})
            break

    # ── Document type keywords
    if any(kw in q for kw in ["drug label", "fda", "prescribing information", "package insert"]):
        conditions.append({"document_type": {"$eq": "drug_label"}})
    elif any(kw in q for kw in ["guideline", "recommend", "consensus", "taskforce", "society"]):
        conditions.append({"document_type": {"$eq": "clinical_guideline"}})
    elif any(kw in q for kw in ["patient instruction", "how to prepare", "preparation steps"]):
        conditions.append({"document_type": {"$eq": "patient_instructions"}})

    # ── Procedure timing
    if "morning" in q:
        conditions.append({"tags": {"$contains": "procedure_time:morning"}})
    elif "afternoon" in q:
        conditions.append({"tags": {"$contains": "procedure_time:afternoon"}})

    # ── Return
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── 2. Retrieval ───────────────────────────────────────────────────────────────


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query, run cosine search in Chroma, return top_k results.

    Each result dict has keys: id, document, metadata, distance.
    """
    query_embedding = embedder.encode([query])[0]
    try:
        where = extract_filters(query)
    except:
        where = None # if there are no filters matched

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten Chroma's nested list response into a list of dicts
    hits = []
    for doc, meta, dist, id_ in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        hits.append({"id": id_, "document": doc, "metadata": meta, "distance": dist})

    return hits


# ── 3. Context formatting ──────────────────────────────────────────────────────

def format_context(hits: list[dict]) -> str:
    """
    Format retrieved chunks into a prompt-ready context string.
    Each block is labelled with source metadata and caution flags.
    """

    blocks = []

    for i, hit in enumerate(hits):
        metadata = hit["metadata"]
        org = metadata.get("organization") or hit["id"]
        doc_type = metadata.get("document_type") or "unknown"
        section = metadata.get("section_title") or "unknown"

        caution_notes = []
        if metadata.get("needs_source_caution"):
            caution_notes.append("external_source")
        if metadata.get("has_contact_info"):
            caution_notes.append("contains_contact_info")
        if metadata.get("is_admin_chunk"):
            caution_notes.append("administrative_content")

        caution_text = (
            " | caution: " + ", ".join(caution_notes)
            if caution_notes
            else ""
        )

        label = (
            f"Source {i+1} | source: {org} | document_type: {doc_type} "
            f"| section: {section}{caution_text}"
        )

        blocks.append(f"[{label}]\n{hit['document']}")

    return "\n\n---\n\n".join(blocks)


# ── 4. Full RAG call (retrieval only — generation wired in orchestration) ──────

def retrieve_for_query(query: str, patient_id: str) -> tuple[dict | None, list[dict], str]:
    """
    Public entry point.  Returns (patient_record, hits, combined_context).

    - patient_record: structured patient data from BigQuery
    - hits: retrieved KB chunks from Chroma
    - combined_context: patient context + KB context for the LLM

    TODO: add re-ranking step here once you have more chunks —
          e.g. cross-encoder on (query, document) pairs to reorder hits
          before trimming to top_k.
    TODO: add query rewriting — expand abbreviations like "UC" → "ulcerative
          colitis" before embedding to improve recall.
    """

    patient_record = get_patient_record(patient_id)
    patient_context = (
        build_patient_context(patient_record)
        if patient_record
        else "No patient-specific data found."
    )

    hits = retrieve(query)
    hits = postprocess_hits(hits)
    kb_context = format_context(hits)

    combined_context = f"""
PATIENT-SPECIFIC CONTEXT
{patient_context}

KNOWLEDGE-BASE CONTEXT
{kb_context}
""".strip()

    return patient_record, hits, combined_context