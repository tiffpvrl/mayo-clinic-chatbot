"""
RAG pipeline: ties together embedder.py (query encoding) and
chromadb_store.py (vector search) to retrieve relevant chunks
for a given user query.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import NamedTuple
from embedder import Embedding
from chromadb_store import clinical_collection, qa_collection, conversation_collection
from config import EMBEDDING_MODEL, TOP_K

embedder = Embedding(model_type=EMBEDDING_MODEL)


# ── 1. Query understanding ─────────────────────────────────────────────────────
# TODO: improve this by using llm-based extraction:

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


def _build_where(conditions: list[dict]) -> dict | None:
    """Collapse a list of Chroma where-clauses into a valid where dict."""
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# Query categories stored in qa_collection / conversation_collection.
# Maps each category value to natural-language trigger keywords.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "timing":     ["when", "how long", "hours before", "what time", "schedule",
                   "timing", "time to take", "how early", "how soon"],
    "dietary":    ["eat", "food", "drink", "diet", "liquid", "clear liquid",
                   "fasting", "avoid eating", "meal", "breakfast", "lunch", "dinner",
                   "snack", "chew", "swallow", "red dye"],
    "medication": ["medicine", "medication", "drug", "pill", "tablet", "dose",
                   "insulin", "aspirin", "warfarin", "metformin", "blood pressure",
                   "hold my", "stop taking", "continue taking"],
    "logistics":  ["drive", "transportation", "ride", "parking", "arrive",
                   "check in", "how long does", "bring", "accompany", "escort",
                   "who should", "what to bring"],
    # "general" is the catch-all — no keyword match needed; just omit the filter
}


def extract_qa_filters(query: str) -> dict | None:
    """
    Translate natural-language cues into ChromaDB `where` filters for qa_collection
    (turn-level Q&A chunks).

    Filterable fields in qa_collection:
      - query_category   : str  exact match ("timing","dietary","medication","logistics","general")
      - appointment_time : str  exact match ("morning" | "afternoon")
      - days_relative_to_procedure : int  (0 = day-of, -1 = day before, -2 = two days before)
      - is_follow_up     : bool (True if turn_number > 1)
    """
    q = query.lower()
    conditions = []

    # ── query_category
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            conditions.append({"query_category": {"$eq": category}})
            break  # one category at a time

    # ── appointment_time
    if "morning" in q:
        conditions.append({"appointment_time": {"$eq": "morning"}})
    elif "afternoon" in q:
        conditions.append({"appointment_time": {"$eq": "afternoon"}})

    # ── days_relative_to_procedure
    if any(kw in q for kw in ["day of", "morning of", "day of procedure"]):
        conditions.append({"days_relative_to_procedure": {"$eq": 0}})
    elif any(kw in q for kw in ["day before", "night before", "eve of"]):
        conditions.append({"days_relative_to_procedure": {"$eq": -1}})
    elif any(kw in q for kw in ["two days before", "2 days before"]):
        conditions.append({"days_relative_to_procedure": {"$eq": -2}})

    return _build_where(conditions)


def extract_conversation_filters(query: str) -> dict | None:
    """
    Translate natural-language cues into ChromaDB `where` filters for
    conversation_collection (full multi-turn thread chunks).

    Filterable fields in conversation_collection:
      - query_categories    : str  comma-joined, use $contains
                              e.g. "timing,dietary" → filter on one category
      - appointment_time    : str  exact match ("morning" | "afternoon")
      - demonstrates_multi_turn : bool
      - tags                : str  comma-joined, use $contains
    """
    q = query.lower()
    conditions = []

    # ── query_categories (comma-joined string — use $contains)
    for category, keywords in _CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            conditions.append({"query_categories": {"$contains": category}})
            break  # one category at a time; conversation chunks often span multiple

    # ── appointment_time
    if "morning" in q:
        conditions.append({"appointment_time": {"$eq": "morning"}})
    elif "afternoon" in q:
        conditions.append({"appointment_time": {"$eq": "afternoon"}})

    # ── demonstrates_multi_turn — prefer threads with back-and-forth when
    #    the query itself signals a multi-step or follow-up scenario
    if any(kw in q for kw in ["and also", "follow up", "follow-up", "and then",
                               "what about", "another question", "additionally"]):
        conditions.append({"demonstrates_multi_turn": {"$eq": True}})

    return _build_where(conditions)


# ── 2. Retrieval ───────────────────────────────────────────────────────────────
# todo: we want to integrate EHR stuff
# todo: format EHR stuff 

def retrieve_clinical(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query, run cosine search in Chroma, return top_k results.

    Each result dict has keys: id, document, metadata, distance.
    """
    query_embedding = embedder.encode([query])[0]
    try:
        where = extract_filters(query)
    except:
        where = None # if there are no filters matched

    results = clinical_collection.query(
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


def retrieve_qa(query: str, top_k: int = TOP_K, is_follow_up: bool | None = None) -> list[dict]:
    """
    Retrieve turn-level Q&A examples from qa_collection.
    Used for tone/phrasing reference — caller surfaces chatbot_response from metadata.

    is_follow_up: when provided, filters examples to the same turn state —
        True  → retrieve follow-up turn examples (turn_number > 1)
        False → retrieve first-turn examples
        None  → no filter (default)

    TODO [EHR integration]: Add `prep_type: str | None = None` parameter and inject
          {"prep_type": {"$eq": prep_type}} as a condition when prep_type is provided.
          prep_type is stored as an uppercase exact string in qa_collection metadata
          (e.g. "SUPREP", "GOLYTELY", "MIRALAX"). This ensures tone/phrasing examples
          are always drawn from the same prep protocol as the patient, preventing
          cross-prep response anchoring. Passed down from retrieve_for_query().
    """
    query_embedding = embedder.encode([query])[0]
    try:
        conditions = []
        keyword_where = extract_qa_filters(query)
        if keyword_where:
            conditions.append(keyword_where)
        if is_follow_up is not None:
            conditions.append({"is_follow_up": {"$eq": is_follow_up}})
        where = _build_where(conditions)
    except Exception:
        where = None

    results = qa_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist, id_ in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        hits.append({"id": id_, "document": doc, "metadata": meta, "distance": dist})
    return hits


def retrieve_conversations(query: str, top_k: int = TOP_K, is_follow_up: bool | None = None) -> list[dict]:
    """
    Retrieve full conversation threads from conversation_collection.
    Used for multi-turn flow reference — shows how similar questions were handled end-to-end.

    is_follow_up: when True, restricts results to multi-turn conversations
        (demonstrates_multi_turn=True) so the LLM sees flow examples that
        actually demonstrate follow-up handling.
        False/None → no filter on demonstrates_multi_turn.

    TODO [EHR integration]: Add `prep_type: str | None = None` parameter and inject
          {"prep_type": {"$eq": prep_type}} as a condition when prep_type is provided.
          prep_type is stored as an uppercase exact string in conversation_collection
          metadata (e.g. "SUPREP", "GOLYTELY", "MIRALAX"), taken from the first turn
          of each conversation. This scopes multi-turn flow examples to the patient's
          actual prep protocol. Passed down from retrieve_for_query().
    """
    query_embedding = embedder.encode([query])[0]
    try:
        conditions = []
        keyword_where = extract_conversation_filters(query)
        if keyword_where:
            conditions.append(keyword_where)
        if is_follow_up:
            conditions.append({"demonstrates_multi_turn": {"$eq": True}})
        where = _build_where(conditions)
    except Exception:
        where = None

    results = conversation_collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

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


def format_clinical_context(hits: list[dict]) -> str:
    """
    Format clinical_collection hits into a prompt-ready string.

    Surfaces the full document text with source provenance so the LLM can
    cite the organisation and section when giving factual answers.
    Returns a sentinel string when no hits are available so the prompt
    section is never silently empty.
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

    Surfaces only the chatbot_response from metadata — not the full embedded
    content — because the LLM needs the response style, not a re-statement
    of the patient question. The patient_message is included as context so
    the LLM understands what situation the response tone was written for.
    Returns a sentinel string when no hits are available.
    """
    if not hits:
        return "No similar Q&A examples found."

    blocks = []
    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        category = meta.get("query_category") or ""
        turn = meta.get("turn_number") or ""
        patient_msg = meta.get("patient_message") or ""
        chatbot_resp = meta.get("chatbot_response") or hit["document"]

        label = f"[Q&A Example {i+1} | category: {category} | turn: {turn}]"
        blocks.append(
            f"{label}\n"
            f"Similar patient question: {patient_msg}\n"
            f"Example response: {chatbot_resp}"
        )

    return "\n\n---\n\n".join(blocks)


def format_conversation_context(hits: list[dict]) -> str:
    """
    Format conversation_collection hits (full multi-turn threads) into a
    prompt-ready string.

    Surfaces the full conversation thread so the LLM can observe multi-turn
    flow, follow-up question patterns, and consistent tone across an exchange.
    Returns a sentinel string when no hits are available.
    """
    if not hits:
        return "No similar conversation flows found."

    blocks = []
    for i, hit in enumerate(hits):
        meta = hit["metadata"]
        flow = meta.get("conversation_flow") or ""
        num_turns = meta.get("num_turns") or ""
        appt_time = meta.get("appointment_time") or ""

        parts = []
        if flow:
            parts.append(f"flow: {flow}")
        if num_turns:
            parts.append(f"turns: {num_turns}")
        if appt_time:
            parts.append(f"appointment: {appt_time}")

        label = f"[Conversation Example {i+1} | {' | '.join(parts)}]"
        blocks.append(f"{label}\n{hit['document']}")

    return "\n\n---\n\n".join(blocks)


# ── 4. Full RAG call (retrieval only — generation wired in orchestration) ──────

class RAGResult(NamedTuple):
    """Structured return type for retrieve_for_query."""
    clinical_hits: list[dict]
    qa_hits: list[dict]
    conversation_hits: list[dict]
    clinical_context: str
    qa_context: str
    conversation_context: str


def retrieve_for_query(query: str, is_follow_up: bool | None = None) -> RAGResult:
    """
    Public entry point.  Returns a RAGResult with hits and formatted context
    strings for all three collections.

    is_follow_up: pass True for any turn after the first in the conversation,
        False for the opening message, or None to skip turn-state filtering.
        The orchestration layer should derive this from conversation history length.

    The caller (orchestration layer) passes the formatted context strings to
    the LLM prompt.

    TODO [EHR integration]: Add `prep_type: str | None = None` parameter once the EHR
          system is available. prep_type should be resolved from the patient's EHR record
          at session start (e.g. "SUPREP", "GOLYTELY", "MIRALAX") and forwarded to
          retrieve_qa() and retrieve_conversations(). Without this filter, a SUPREP
          patient may receive tone/phrasing examples written for a GoLYTELY protocol,
          causing the chatbot to anchor on the wrong dietary or timing framing even when
          the clinical context is factually correct.
          See retrieve_qa() and retrieve_conversations() for where the filter is applied.

    TODO: add re-ranking step here once you have more chunks —
          e.g. cross-encoder on (query, document) pairs to reorder hits
          before trimming to top_k.
    TODO: add query rewriting — expand abbreviations like "UC" → "ulcerative
          colitis" before embedding to improve recall.
    """
    clinical_hits = retrieve_clinical(query)
    qa_hits = retrieve_qa(query, is_follow_up=is_follow_up)
    conversation_hits = retrieve_conversations(query, is_follow_up=is_follow_up)

    return RAGResult(
        clinical_hits=clinical_hits,
        qa_hits=qa_hits,
        conversation_hits=conversation_hits,
        clinical_context=format_clinical_context(clinical_hits),
        qa_context=format_qa_context(qa_hits),
        conversation_context=format_conversation_context(conversation_hits),
    )
