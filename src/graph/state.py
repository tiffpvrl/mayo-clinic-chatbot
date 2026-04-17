"""
ChatState — the single typed dictionary that travels through every node in the
LangGraph pipeline.  Each node reads what it needs and writes its outputs back;
LangGraph merges partial dicts into this shared state object automatically.

chat_history uses operator.add as its reducer so that messages from each turn
are *appended* to the accumulated list across checkpointed conversations rather
than overwriting it.  All other fields are replaced on each update.
"""

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict


class ChatState(TypedDict):
    # ── Input fields (set per turn) ────────────────────────────────────────────
    query: str
    patient_id: str

    # ── Classification ─────────────────────────────────────────────────────────
    # One of "medical" | "logistics" | "chitchat"
    # Determines which pipeline branches are executed.
    query_intent: Optional[Literal["medical", "logistics", "chitchat"]]

    # ── Patient data (fetch_patient_data_node) ─────────────────────────────────
    patient_record: Optional[Dict[str, Any]]
    patient_context: str  # formatted EHR text block for the LLM prompt

    # ── Risk scoring (score_risk_node — medical only) ──────────────────────────
    risk_tier: Optional[str]          # "Low" | "Medium" | "High"
    risk_probability: Optional[float]

    # ── Query understanding (retrieve_rag_node) ────────────────────────────────
    # Computed once per turn and reused by all three retrieval sub-calls so that
    # only a single LLM call is made for filter extraction.
    query_understanding: Optional[Dict[str, Any]]
    query_where: Optional[Dict[str, Any]]  # pre-built ChromaDB where clause
    wants_research: bool

    # ── RAG retrieval (retrieve_rag_node — medical only) ──────────────────────
    clinical_hits: List[Dict[str, Any]]
    qa_hits: List[Dict[str, Any]]
    conversation_hits: List[Dict[str, Any]]
    clinical_context: str
    qa_context: str
    conversation_context: str
    combined_context: str  # all sections merged for the LLM

    # ── Response (generate_response_node) ─────────────────────────────────────
    response: str

    # ── Judge guardrail (judge_response_node — medical only) ──────────────────
    # The judge scores the candidate response on a 0.0–1.0 scale across
    # factual consistency, relevance, and absence of hallucinations.
    # The threshold it must clear varies by patient risk tier:
    #   High → 0.90  |  Medium → 0.85  |  Low → 0.80
    judge_score: Optional[float]       # continuous confidence score 0.0–1.0
    judge_reasoning: Optional[str]     # judge's explanation, re-injected on retry

    # ── Retry loop (judge_response_node / graph routing) ──────────────────────
    # retry_count is incremented each time the judge rejects a response and
    # the pipeline loops back to generate_response.  classify_query_node
    # resets it to 0 at the start of every new user turn so the counter
    # doesn't bleed across conversation turns.
    retry_count: int
    max_retries: int   # configurable ceiling; default set in main.py via config

    # ── Escalation (escalate_node) ────────────────────────────────────────────
    # Set to True when all retries are exhausted and the response is handed off
    # to a human clinician instead of being delivered autonomously.
    escalated: bool

    # ── Multi-turn conversation memory ─────────────────────────────────────────
    # operator.add reducer means each turn's messages are *appended* to the list
    # that LangGraph persists in the checkpoint store.  Reading len > 0 tells any
    # node whether this is a follow-up turn without extra bookkeeping.
    # finalize_node writes to this field — generate_response_node does not —
    # so intermediate retry attempts never pollute the conversation history.
    chat_history: Annotated[List[Dict[str, str]], operator.add]
    is_follow_up: bool

    # ── Timing (classify_query_node → finalize_node) ──────────────────────────
    turn_start_time: Optional[float]  # perf_counter value set at turn start
