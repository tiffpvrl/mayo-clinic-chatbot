"""
RAG pipeline: ties together embedder.py (query encoding) and
chromadb_store.py (vector search) to retrieve relevant chunks
for a given user query.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from embedder import Embedding
from chromadb_store import collection
from config import EMBEDDING_MODEL, TOP_K

embedder = Embedding(model_type=EMBEDDING_MODEL)


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
# todo: we want to integrate EHR stuff
# todo: format EHR stuff 


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
    Each block is labelled with its source so the LLM can cite it.
    """

    blocks = []
    for i, hit in enumerate(hits):
        metadata = hit["metadata"]
        org = metadata.get('organization') or hit['id']
        label = f"Source {i+1} | source: {org} | document_type: {metadata.get('document_type')} | section: {metadata.get('section_title')}"
        blocks.append(f"[{label}]\n{hit['document']}")

    return "\n\n---\n\n".join(blocks)


# ── 4. Full RAG call (retrieval only — generation wired in orchestration) ──────

def retrieve_for_query(query: str) -> tuple[list[dict], str]:
    """
    Public entry point.  Returns (hits, formatted_context).

    The caller (orchestration layer) passes `formatted_context` to the LLM.

    TODO: add re-ranking step here once you have more chunks —
          e.g. cross-encoder on (query, document) pairs to reorder hits
          before trimming to top_k.
    TODO: add query rewriting — expand abbreviations like "UC" → "ulcerative
          colitis" before embedding to improve recall.
    """
    hits = retrieve(query)
    context = format_context(hits)
    return hits, context
