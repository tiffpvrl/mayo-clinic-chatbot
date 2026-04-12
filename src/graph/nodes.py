"""
LangGraph node functions for MayoChat.

Each function receives the full ChatState, does its work, and returns a
*partial* dict — LangGraph merges only the returned keys back into state.

Node execution order per intent:

  chitchat  → classify_query → generate_response → finalize
  logistics → classify_query → fetch_patient_data → generate_response → finalize
  medical   → classify_query → fetch_patient_data → score_risk
                             → retrieve_rag → generate_response → judge_response
                               ├── score ≥ threshold      → finalize
                               ├── score < threshold, retries left → generate_response (loop)
                               └── score < threshold, max retries  → escalate → finalize

Key design decisions
--------------------
* generate_response_node does NOT write to chat_history — finalize_node does.
  This means intermediate retry attempts never pollute the conversation log.
* judge_response_node increments retry_count only on failure so the routing
  function can decide whether to loop or escalate without extra bookkeeping.
* classify_query_node resets retry_count to 0 at the start of every new user
  turn so the counter does not bleed across conversation turns.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from vertexai.generative_models import GenerativeModel

from src.config import JUDGE_DEFAULT_THRESHOLD, JUDGE_MAX_RETRIES, JUDGE_THRESHOLDS, LLM_MODEL
from src.graph.state import ChatState
from src.llm.generate_response import generate_response
from src.patient_data.bigquery_client import get_patient_record
from src.patient_data.patient_context import build_patient_context
from src.retrieval.filters import build_clinical_where, extract_query_understanding
from src.retrieval.rag import (
    format_clinical_context,
    format_conversation_context,
    format_qa_context,
    retrieve_clinical,
    retrieve_conversations,
    retrieve_qa,
)
from src.risk.risk_model import get_risk_score

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json_response(text: str, fallback: dict) -> dict:
    """Parse a JSON object from LLM output, stripping markdown fences if present."""
    try:
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)
    except Exception:
        return fallback


def _judge_threshold(risk_tier: str | None) -> float:
    """Return the confidence threshold for the given risk tier."""
    key = (risk_tier or "").lower()
    return JUDGE_THRESHOLDS.get(key, JUDGE_DEFAULT_THRESHOLD)


# ── 1. Query Classifier ────────────────────────────────────────────────────────

_CLASSIFIER_PROMPT = """Classify the patient's message into exactly one of these three categories:

- medical: Questions about medications, symptoms, prep instructions, diet restrictions, side effects, clinical procedures, or anything requiring clinical/medical knowledge
- logistics: Questions about appointment timing, procedure location, what to bring, parking, check-in, how long the procedure takes, or other scheduling and administrative logistics
- chitchat: Greetings, thanks, small talk, acknowledgments, confirmations, or messages with no medical or logistical content

Patient message: {query}

Respond with a JSON object containing a single key "intent" set to one of the three values above.
Example: {{"intent": "medical"}}"""


def classify_query_node(state: ChatState) -> dict:
    """
    Lightweight intent classifier.  Defaults to "medical" on any failure so
    the full pipeline always runs as a safe fallback.

    Also resets retry_count to 0 so the judge retry loop from a previous turn
    does not carry over into the current one.
    """
    query = state["query"]

    prompt = _CLASSIFIER_PROMPT.format(query=query)
    try:
        raw = GenerativeModel(LLM_MODEL).generate_content(prompt)
        result = _parse_json_response(raw.text, {"intent": "medical"})
        intent = result.get("intent", "medical")
        if intent not in ("medical", "logistics", "chitchat"):
            intent = "medical"
    except Exception as exc:
        logger.warning("[classify] Classification failed, defaulting to medical: %s", exc)
        intent = "medical"

    is_follow_up = len(state.get("chat_history") or []) > 0

    print(f"[classify] intent={intent!r}  is_follow_up={is_follow_up}")
    return {
        "query_intent": intent,
        "is_follow_up": is_follow_up,
        "retry_count": 0,       # reset for this turn
        "escalated": False,
        "judge_score": None,
        "judge_reasoning": None,
    }


# ── 2. Patient Data ────────────────────────────────────────────────────────────

def fetch_patient_data_node(state: ChatState) -> dict:
    """Fetch patient record from BigQuery and build patient context string."""
    patient_id = state["patient_id"]
    patient_record = get_patient_record(patient_id)
    patient_context = (
        build_patient_context(patient_record)
        if patient_record
        else "No patient-specific data found."
    )
    print(f"[patient_data] patient_id={patient_id}  found={patient_record is not None}")
    return {"patient_record": patient_record, "patient_context": patient_context}


# ── 3. Risk Scoring ────────────────────────────────────────────────────────────

def score_risk_node(state: ChatState) -> dict:
    """
    Run the joblib risk model to produce a risk tier (Low / Medium / High).
    The tier is stored in state so both the RAG retrieval and judge guardrail
    can read it without re-running the model.
    """
    patient_record = state.get("patient_record")
    if not patient_record:
        return {"risk_tier": None, "risk_probability": None}

    risk_result = get_risk_score(patient_record)
    risk_tier = risk_result["risk_tier"].capitalize()
    risk_probability = risk_result["risk_probability"]

    print(f"[risk] tier={risk_tier}  probability={risk_probability:.3f}  "
          f"judge_threshold={_judge_threshold(risk_tier):.2f}")
    return {"risk_tier": risk_tier, "risk_probability": risk_probability}


# ── 4. RAG Retrieval ───────────────────────────────────────────────────────────

def retrieve_rag_node(state: ChatState) -> dict:
    """
    Run the full three-collection RAG pipeline (clinical, Q&A, conversations).
    Query understanding is extracted once and reused by all three retrievers.
    """
    query = state["query"]
    patient_record: Any = state.get("patient_record")
    risk_tier = state.get("risk_tier")
    is_follow_up = state.get("is_follow_up", False)

    query_understanding = extract_query_understanding(query)
    query_where = build_clinical_where(query_understanding)
    wants_research = bool(query_understanding.get("wants_research", False))

    clinical_hits = retrieve_clinical(
        query,
        patient_record=patient_record,
        query_where=query_where,
        wants_research=wants_research,
    )
    qa_hits = retrieve_qa(query, is_follow_up=is_follow_up, risk_tier=risk_tier)
    conversation_hits = retrieve_conversations(query, is_follow_up=is_follow_up, risk_tier=risk_tier)

    clinical_context = format_clinical_context(clinical_hits)
    qa_context = format_qa_context(qa_hits)
    conversation_context = format_conversation_context(conversation_hits)
    patient_context = state.get("patient_context", "")

    combined_context = f"""PATIENT-SPECIFIC CONTEXT
{patient_context}

CLINICAL KNOWLEDGE BASE
{clinical_context}

SIMILAR Q&A EXAMPLES
{qa_context}

SIMILAR CONVERSATION FLOWS
{conversation_context}""".strip()

    return {
        "query_understanding": query_understanding,
        "query_where": query_where,
        "wants_research": wants_research,
        "clinical_hits": clinical_hits,
        "qa_hits": qa_hits,
        "conversation_hits": conversation_hits,
        "clinical_context": clinical_context,
        "qa_context": qa_context,
        "conversation_context": conversation_context,
        "combined_context": combined_context,
    }


# ── 5. Response Generation ─────────────────────────────────────────────────────

_CHITCHAT_PROMPT = """\
You are MayoChat, a helpful assistant for patients preparing for a colonoscopy at Mayo Clinic.
The patient has sent a casual message — a greeting, thanks, or small talk.
Respond warmly and briefly (1-2 sentences). Remind them you are here to help with questions
about their colonoscopy preparation whenever they are ready.

Patient message: {query}

Response:"""


def generate_response_node(state: ChatState) -> dict:
    """
    Produce the candidate patient-facing response.

    Context selection by intent:
      chitchat  — lightweight direct LLM call, no retrieval context
      logistics — patient context only (structured EHR data, no clinical RAG)
      medical   — full combined context (patient + clinical + Q&A + conversations)

    Retry awareness:
      When retry_count > 0 the judge's reasoning from the previous attempt is
      prepended to the context so the generator knows exactly what to fix.

    NOTE: this node does NOT write to chat_history.  finalize_node handles
    that, ensuring only the final accepted response enters the conversation log.
    """
    query = state["query"]
    intent = state.get("query_intent") or "medical"
    retry_count = state.get("retry_count", 0)
    judge_reasoning = state.get("judge_reasoning")
    judge_score = state.get("judge_score")
    risk_tier = state.get("risk_tier")
    threshold = _judge_threshold(risk_tier)

    def _retry_prefix() -> str:
        if retry_count > 0 and judge_reasoning:
            return (
                f"[REVISION REQUEST — attempt {retry_count + 1}]\n"
                f"Previous response scored {judge_score:.2f} (required ≥ {threshold:.2f}).\n"
                f"Judge feedback: {judge_reasoning}\n"
                f"Please revise the response to address the issues above.\n\n"
            )
        return ""

    if intent == "chitchat":
        prompt = _CHITCHAT_PROMPT.format(query=query)
        try:
            raw = GenerativeModel(LLM_MODEL).generate_content(prompt)
            response_text = raw.text.strip() or "Hello! How can I help you with your colonoscopy preparation?"
        except Exception as exc:
            logger.warning("[generate] Chitchat generation failed: %s", exc)
            response_text = "Hello! How can I help you with your colonoscopy preparation today?"

    elif intent == "logistics":
        patient_context = state.get("patient_context", "No patient data available.")
        context = _retry_prefix() + f"PATIENT-SPECIFIC CONTEXT\n{patient_context}"
        response_text = generate_response(query, context)

    else:
        combined_context = state.get("combined_context", "")
        context = _retry_prefix() + combined_context
        response_text = generate_response(query, context)

    print(f"[generate] intent={intent}  retry={retry_count}  response_len={len(response_text)}")
    return {"response": response_text}


# ── 6. Judge Guardrail ────────────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are a clinical quality reviewer for a patient-facing colonoscopy preparation chatbot.
Score the candidate response on a continuous scale from 0.0 to 1.0.

Scoring dimensions:
1. Factual consistency — Is every claim in the response supported by the retrieved evidence?
   Penalise heavily for hallucinated drug names, dosages, or instructions not in the context.
2. Relevance — Does the response directly answer the patient's question?
3. Absence of unsupported claims — No advice introduced beyond what the context provides.
4. Appropriate tone for patient risk tier ({risk_tier}) — see below.
   Low risk   → concise, reassuring
   Medium risk → explicit, reinforcing
   High risk   → directive, emphasises consequences, suggests care team contact when warranted

Patient question:
{query}

Patient risk tier: {risk_tier} (confidence threshold: {threshold:.2f})

Retrieved evidence (truncated to 2000 chars):
{context}

Candidate response:
{response}

Return a JSON object with exactly two keys:
  "score"     — float 0.0 to 1.0
  "reasoning" — one or two sentences explaining the score and, if below threshold, what must change

Example: {{"score": 0.91, "reasoning": "Response accurately cites SUPREP instructions and uses appropriate high-risk tone."}}"""


def judge_response_node(state: ChatState) -> dict:
    """
    Score the candidate response on a 0.0–1.0 confidence scale.

    The threshold that must be cleared varies by patient risk tier:
        High → 0.90  |  Medium → 0.85  |  Low → 0.80

    On failure (score < threshold) retry_count is incremented so the
    _route_after_judge routing function can decide whether to loop back
    to generate_response or escalate to a clinician.

    Fails open (score=1.0) if the LLM call errors so a judge failure never
    blocks a valid patient response.
    """
    query = state["query"]
    response = state.get("response", "")
    combined_context = state.get("combined_context", "")
    risk_tier = state.get("risk_tier") or "unknown"
    threshold = _judge_threshold(risk_tier)
    current_retry = state.get("retry_count", 0)

    prompt = _JUDGE_PROMPT.format(
        risk_tier=risk_tier,
        threshold=threshold,
        query=query,
        context=combined_context[:2000],
        response=response,
    )

    try:
        raw = GenerativeModel(LLM_MODEL).generate_content(prompt)
        result = _parse_json_response(raw.text, {"score": 1.0, "reasoning": "parse error — failing open"})
        score = float(result.get("score", 1.0))
        score = max(0.0, min(1.0, score))   # clamp to [0, 1]
        reasoning = str(result.get("reasoning", ""))
    except Exception as exc:
        logger.warning("[judge] Evaluation failed, failing open: %s", exc)
        score = 1.0
        reasoning = f"Judge skipped (error: {exc})"

    passed = score >= threshold
    print(f"[judge] score={score:.3f}  threshold={threshold:.2f}  passed={passed}  "
          f"retry={current_retry}  reasoning={reasoning!r}")

    update: dict = {
        "judge_score": score,
        "judge_reasoning": reasoning,
    }

    if not passed:
        # Increment retry_count so routing can decide whether to loop or escalate
        update["retry_count"] = current_retry + 1

    return update


# ── 7. Escalation ─────────────────────────────────────────────────────────────

_ESCALATION_MESSAGE = (
    "I wasn't able to generate a confident enough answer to your question after multiple attempts. "
    "To make sure you receive accurate guidance, your care team has been notified and will follow "
    "up with you directly. Please do not make any changes to your preparation until you hear from them."
)


def escalate_node(state: ChatState) -> dict:
    """
    Triggered when all judge retries are exhausted.  Replaces the candidate
    response with a safe escalation message and flags the turn as escalated
    so downstream systems (e.g. clinician notification) can act on it.
    """
    risk_tier = state.get("risk_tier") or "unknown"
    query = state.get("query", "")
    judge_score = state.get("judge_score") or 0.0   # guard against None
    retry_count = state.get("retry_count", 0)

    print(f"[escalate] Escalating to clinician — risk={risk_tier}  "
          f"score={judge_score:.3f}  retries={retry_count}  query={query!r}")

    return {
        "response": _ESCALATION_MESSAGE,
        "escalated": True,
    }


# ── 8. Finalize ───────────────────────────────────────────────────────────────

def finalize_node(state: ChatState) -> dict:
    """
    Terminal node for every intent path.  Writes the accepted response and
    the user query into the persistent chat_history so multi-turn memory
    reflects only finalized, delivered responses — not retry intermediates.
    """
    query = state["query"]
    response = state.get("response", "")
    escalated = state.get("escalated", False)

    print(f"[finalize] escalated={escalated}  history_len={len(state.get('chat_history') or [])}")

    new_messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ]
    return {"chat_history": new_messages}
