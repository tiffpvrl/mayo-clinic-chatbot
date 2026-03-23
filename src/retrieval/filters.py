"""
ChromaDB filter builders for the RAG pipeline.

Three filter functions for query-based narrowing (one per collection),
plus one for EHR-based narrowing of the clinical collection.
All share the _build_where / _build_or_where helpers.

TODO: replace keyword matching with LLM-based extraction for better recall.
"""

from datetime import datetime


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_where(conditions: list[dict]) -> dict | None:
    """Collapse a list of Chroma where-clauses into a valid $and where dict."""
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _build_or_where(conditions: list[dict]) -> dict | None:
    """Collapse a list of Chroma where-clauses into a valid $or where dict."""
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$or": conditions}


# ── Query-based filters ────────────────────────────────────────────────────────

def extract_filters(query: str) -> dict | None:
    """
    Translate natural-language cues in the query into ChromaDB `where` filters
    for clinical_collection, scoping the vector search before cosine distance
    is computed.

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

    return _build_where(conditions)


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
            break

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


# ── EHR-based filters ──────────────────────────────────────────────────────────

def _pos(value) -> bool:
    """Positive-value check — mirrors the same logic in patient_context.py."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def extract_patient_filters(patient_record: dict) -> dict | None:
    """
    Translate patient EHR data from BigQuery into ChromaDB `where` filters
    for clinical_collection, surfacing chunks relevant to the patient's
    specific conditions, medications, and procedure context.

    Uses $or across conditions so a chunk matching ANY of the patient's
    risk factors is eligible — avoids over-filtering when a patient has
    multiple comorbidities.

    Filterable tags in clinical_collection (from document_processor.py):
      - metabolic_conditions   : diabetes / G6PD / PKU mentions in chunk
      - heart_failure          : CHF mentions
      - cirrhosis              : liver disease mentions
      - constipation_history   : constipation / prior poor prep mentions
      - renal_disease          : CKD / ESRD / kidney mentions
      - prior_poor_prep        : prior inadequate prep mentions
      - gi_conditions          : IBD / ulcerative colitis / obstruction
      - mobility_frailty       : frailty / wheelchair / limited mobility
      - dehydration_risk       : dehydration mentions
      - diabetes_meds:insulin  : insulin mentions
      - diabetes_meds:oral_agents : metformin / sulfonylurea / SGLT2 mentions
      - procedure_time:morning / procedure_time:afternoon
      - indication:screening / indication:surveillance / indication:diagnostic
      - drug_name              : exact prep agent match (e.g. "SUPREP")
    """
    if not patient_record:
        return None

    conditions = []

    # ── Medical conditions
    if _pos(patient_record.get("diabetes")) or _pos(patient_record.get("on_diabetes_medication")):
        conditions.append({"tags": {"$contains": "metabolic_conditions"}})

    if _pos(patient_record.get("heart_failure")):
        conditions.append({"tags": {"$contains": "heart_failure"}})

    if _pos(patient_record.get("cirrhosis")):
        conditions.append({"tags": {"$contains": "cirrhosis"}})

    if (
        _pos(patient_record.get("chronic_constipation"))
        or _pos(patient_record.get("medication_class_laxative"))
        or _pos(patient_record.get("medication_class_opioid"))
    ):
        conditions.append({"tags": {"$contains": "constipation_history"}})

    ckd_stage = patient_record.get("ckd_stage")
    if ckd_stage not in (None, "", 0, "0"):
        conditions.append({"tags": {"$contains": "renal_disease"}})

    prior_adequate = patient_record.get("prior_prep_adequate_flag")
    if prior_adequate is not None and not _pos(prior_adequate):
        conditions.append({"tags": {"$contains": "prior_poor_prep"}})

    if _pos(patient_record.get("ibd_diagnosis")):
        conditions.append({"tags": {"$contains": "gi_conditions"}})

    mobility = str(patient_record.get("mobility_status") or "").lower()
    if any(kw in mobility for kw in ["limited", "wheelchair", "walker", "frail"]):
        conditions.append({"tags": {"$contains": "mobility_frailty"}})

    if _pos(patient_record.get("high_risk_flag")):
        conditions.append({"tags": {"$contains": "dehydration_risk"}})

    # ── Diabetes medications
    if _pos(patient_record.get("medication_class_glp1_agonist")) or _pos(patient_record.get("on_diabetes_medication")):
        conditions.append({"tags": {"$contains": "diabetes_meds:oral_agents"}})

    # ── Procedure timing derived from colonoscopy datetime
    colonoscopy_dt = patient_record.get("colonoscopy_datetime")
    if colonoscopy_dt:
        try:
            if isinstance(colonoscopy_dt, str):
                colonoscopy_dt = datetime.fromisoformat(colonoscopy_dt)
            if colonoscopy_dt.hour < 12:
                conditions.append({"tags": {"$contains": "procedure_time:morning"}})
            else:
                conditions.append({"tags": {"$contains": "procedure_time:afternoon"}})
        except Exception:
            pass

    # ── Colonoscopy indication
    indication = str(patient_record.get("colonoscopy_indication") or "").lower()
    if "screen" in indication:
        conditions.append({"tags": {"$contains": "indication:screening"}})
    elif "surveillance" in indication or "follow" in indication:
        conditions.append({"tags": {"$contains": "indication:surveillance"}})
    elif "diagnostic" in indication:
        conditions.append({"tags": {"$contains": "indication:diagnostic"}})

    # ── Prep agent → exact drug_name match
    prep_agent = patient_record.get("prep_agent")
    if prep_agent:
        conditions.append({"drug_name": {"$eq": str(prep_agent).upper()}})

    return _build_or_where(conditions)
