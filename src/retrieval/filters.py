"""
ChromaDB filter builders for the RAG pipeline.

Three filter functions for query-based narrowing (one per collection),
plus one for EHR-based narrowing of the clinical collection.
All share the _build_where / _build_or_where helpers.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime

from vertexai.generative_models import GenerativeModel, GenerationConfig
from src.config import STRUCTURED_LLM_MODEL

logger = logging.getLogger(__name__)


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

_FILTER_EXTRACTION_PROMPT = """
You are a filter extractor for a colonoscopy prep chatbot's retrieval system.
Given a patient's question, extract structured search filters and intent signals.

Patient question: {query}

Rules:
- medication_class: identify if the question mentions a medication class by trade name, generic name, or patient vernacular.
  Map to exactly one of: anticoagulants, antiplatelet, diuretics, ace_inhibitors, sglt2_inhibitors, nsaids, or null.
  Examples: "blood thinner" → anticoagulants, "water pill" → diuretics, "Jardiance" → sglt2_inhibitors, "heart medicine" → null (too vague).
- is_diabetes_query: true if the question mentions diabetes, blood sugar, glucose, insulin, metformin, GLP-1, SGLT2, A1C, or hypoglycemia.
- drug_name: if the question names a specific bowel prep agent, return exactly one of:
  SUPREP, GOLYTELY, MIRALAX, MOVIPREP, PREPOPIK, CLENPIQ, PLENVU, SUFLAVE, or null.
- document_type: if the question is clearly asking for FDA/prescribing/drug label info return "drug_label",
  for guidelines/recommendations return "clinical_guideline",
  for step-by-step prep instructions return "patient_instructions", else null.
- procedure_timing: if the question mentions morning or afternoon procedure, return "morning" or "afternoon", else null.
- wants_research: true if the question is asking for clinical study, trial, or research evidence.
  Examples: "Is there any proof that MiraLax works better than GoLytely?" → true,
            "Are there studies showing split-dose prep is more effective?" → true,
            "What does the science say about bowel prep and kidney disease?" → true,
            "What time should I stop eating?" → false,
            "Can I take my blood pressure pill this morning?" → false.
"""

_FILTER_SCHEMA = {
    "type": "object",
    "properties": {
        "medication_class":  {"type": "string"},
        "is_diabetes_query": {"type": "boolean"},
        "drug_name":         {"type": "string"},
        "document_type":     {"type": "string"},
        "procedure_timing":  {"type": "string"},
        "wants_research":    {"type": "boolean"},
    },
    "required": ["is_diabetes_query", "wants_research"],
}

_VALID_MED_CLASSES = {
    "anticoagulants", "antiplatelet", "diuretics",
    "ace_inhibitors", "sglt2_inhibitors", "nsaids",
}
_VALID_DRUG_NAMES = {
    "SUPREP", "GOLYTELY", "MIRALAX", "MOVIPREP",
    "PREPOPIK", "CLENPIQ", "PLENVU", "SUFLAVE",
}
_VALID_DOC_TYPES  = {"drug_label", "clinical_guideline", "patient_instructions"}
_VALID_TIMINGS    = {"morning", "afternoon"}


def extract_query_understanding(query: str) -> dict:
    """
    Single LLM call that returns all query-level signals needed by the pipeline:
    filter fields for ChromaDB (medication_class, drug_name, etc.) and intent
    flags (is_diabetes_query, wants_research).

    Callers should use this once and pass the result to build_clinical_where()
    and retrieve the wants_research flag — avoids a second LLM call.
    Falls back to safe defaults on any error.
    """
    try:
        t0 = time.perf_counter()
        model = GenerativeModel(STRUCTURED_LLM_MODEL)
        response = model.generate_content(
            _FILTER_EXTRACTION_PROMPT.format(query=query),
            generation_config=GenerationConfig(
                response_mime_type="application/json",
                response_schema=_FILTER_SCHEMA,
                temperature=0.0,
                max_output_tokens=128,
            ),
        )
        extracted = json.loads(response.text)
        print(f"[query_understanding] LLM extraction took {(time.perf_counter() - t0) * 1000:.0f}ms | extracted={extracted}")
        return extracted
    except Exception as e:
        logger.warning("[query_understanding] LLM extraction failed, returning defaults: %s", e)
        return {"is_diabetes_query": False, "wants_research": False}


def build_clinical_where(understanding: dict) -> dict | None:
    """
    Convert a pre-extracted query understanding dict (from extract_query_understanding)
    into a ChromaDB `where` clause for clinical_collection.

    Separated from the LLM call so the caller can run extraction once and reuse
    the result without a second round trip.
    """
    conditions = []

    # ── Medication class
    med_class = (understanding.get("medication_class") or "").strip().lower()
    if med_class in _VALID_MED_CLASSES:
        conditions.append({"tags": {"$contains": f"med_class:{med_class}"}})

    # ── Diabetes — $or across condition + medication tags
    if understanding.get("is_diabetes_query"):
        conditions.append(_build_or_where([
            {"tags": {"$contains": "metabolic_conditions"}},
            {"tags": {"$contains": "diabetes_meds:oral_agents"}},
            {"tags": {"$contains": "diabetes_meds:insulin"}},
        ]))

    # ── Bowel prep drug name (enum-constrained, so value is already valid)
    drug_name = (understanding.get("drug_name") or "").strip().upper()
    if drug_name in _VALID_DRUG_NAMES:
        conditions.append({"drug_name": {"$eq": drug_name}})

    # ── Document type
    doc_type = (understanding.get("document_type") or "").strip().lower()
    if doc_type in _VALID_DOC_TYPES:
        conditions.append({"document_type": {"$eq": doc_type}})

    # ── Procedure timing
    timing = (understanding.get("procedure_timing") or "").strip().lower()
    if timing in _VALID_TIMINGS:
        conditions.append({"tags": {"$contains": f"procedure_time:{timing}"}})

    return _build_where(conditions)


def extract_filters(query: str) -> dict | None:
    """
    Convenience wrapper: runs extract_query_understanding + build_clinical_where
    in one call. Use this when you only need the ChromaDB where clause and don't
    need the full understanding dict (e.g. standalone testing).

    For the main pipeline, prefer calling extract_query_understanding() once in
    retrieve_rag_node and passing the result to build_clinical_where() and
    retrieve_clinical(..., filters_from_upstream=True) so a None where clause
    does not trigger a second LLM call.
    """
    understanding = extract_query_understanding(query)
    return build_clinical_where(understanding)


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


def extract_qa_filters(query: str, risk_tier: str | None = None) -> dict | None:
    """
    Translate natural-language cues into ChromaDB `where` filters for qa_collection
    (turn-level Q&A chunks).

    Filterable fields in qa_collection:
      - risk_tier        : str  exact match ("Low" | "Medium" | "High")
      - appointment_time : str  exact match ("morning" | "afternoon")
      - is_follow_up     : bool (True if turn_number > 1)

    risk_tier: when provided (from patient EHR), retrieves examples matching
        the patient's risk level so tone is appropriate for their situation.
    """
    q = query.lower()
    conditions = []

    # ── risk_tier (passed in from patient record — highest-signal filter)
    if risk_tier in ("Low", "Medium", "High"):
        conditions.append({"risk_tier": {"$eq": risk_tier}})

    # ── appointment_time
    if "morning" in q:
        conditions.append({"appointment_time": {"$eq": "morning"}})
    elif "afternoon" in q:
        conditions.append({"appointment_time": {"$eq": "afternoon"}})

    return _build_where(conditions)


def extract_conversation_filters(query: str, risk_tier: str | None = None) -> dict | None:
    """
    Translate natural-language cues into ChromaDB `where` filters for
    conversation_collection (full multi-turn thread chunks).

    Filterable fields in conversation_collection:
      - risk_tier           : str  exact match ("Low" | "Medium" | "High")
      - appointment_time    : str  exact match ("morning" | "afternoon")
      - demonstrates_multi_turn : bool

    risk_tier: when provided (from patient EHR), scopes examples to the
        patient's risk level so tone and clinical urgency are appropriate.
    """
    q = query.lower()
    conditions = []

    # ── risk_tier (passed in from patient record)
    if risk_tier in ("Low", "Medium", "High"):
        conditions.append({"risk_tier": {"$eq": risk_tier}})

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
        conditions.append(_build_or_where([
            {"tags": {"$contains": "metabolic_conditions"}},
            {"tags": {"$contains": "diabetes_meds:oral_agents"}},
            {"tags": {"$contains": "diabetes_meds:insulin"}},
        ]))

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
