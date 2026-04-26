"""
LangGraph stateful graph for MayoChat.

Topology
--------
                      ┌──────────────────────────────────────────────────┐
  START               │                                                  │
    │                 ▼                                                  │
    └──► classify_query                                                  │
              │                                                          │
              ├─ chitchat ──────────────────► generate_response          │
              │                                      │                   │
              └─ logistics / medical ──► fetch_patient_data              │
                                               │                         │
                                     ┌─────────┴──────────┐             │
                                 logistics             medical            │
                                     │                    │              │
                                     │              score_risk            │
                                     │                    │              │
                                     │            retrieve_rag           │
                                     │                    │              │
                                     └────────────────────┘              │
                                               │                         │
                                       generate_response ◄───────────────┤ (retry loop)
                                               │                         │
                                 ┌─────────────┴────────────┐            │
                              medical                 chitchat/logistics  │
                                 │                          │            │
                          judge_response              finalize ──► END   │
                                 │                                       │
                    ┌────────────┼────────────────┐                      │
                score ≥      score <           score <                   │
               threshold   threshold,        threshold,                  │
                            retries left     max retries ─► escalate     │
                    │           │                               │         │
                    └─────────► finalize ◄─────────────────────┘         │
                                    │                                     │
                                   END                                    │
                                                                          │
              (retry loop goes back to generate_response) ────────────────┘

State persistence
-----------------
MemorySaver checkpoints the full ChatState across HTTP requests for the same
thread_id, providing multi-turn memory without an external session store.

    graph.invoke(
        {"query": ..., "patient_id": ..., "chat_history": [],
         "retry_count": 0, "max_retries": 2, "escalated": False},
        config={"configurable": {"thread_id": patient_id}},
    )
"""

from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import JUDGE_DEFAULT_THRESHOLD, JUDGE_MAX_RETRIES, JUDGE_THRESHOLDS
from src.graph.nodes import (
    classify_query_node,
    escalate_node,
    fetch_patient_data_node,
    finalize_node,
    generate_response_node,
    judge_response_node,
    retrieve_rag_node,
)
from src.graph.state import ChatState


# ── Routing functions ──────────────────────────────────────────────────────────

def _route_after_classify(state: ChatState) -> str:
    """Chitchat bypasses patient data and RAG; everything else fetches patient data."""
    intent = state.get("query_intent") or "medical"
    if intent == "chitchat":
        return "generate_response"
    return "fetch_patient_data"


def _route_after_patient_data(state: ChatState) -> str:
    """Logistics stops after patient data; medical continues to RAG retrieval."""
    intent = state.get("query_intent") or "medical"
    if intent == "logistics":
        return "generate_response"
    return "retrieve_rag"


def _route_after_generate(state: ChatState) -> str:
    """Medical and logistics responses go to the judge; chitchat goes straight to finalize."""
    intent = state.get("query_intent") or "medical"
    if intent in ("medical", "logistics"):
        return "judge_response"
    return "finalize"


def _route_after_judge(state: ChatState) -> str:
    """
    Core routing for the judge guardrail loop.

    Thresholds by risk tier (from config):
        High → 0.90  |  Medium → 0.85  |  Low → 0.80

    Decision tree:
        score ≥ threshold           → finalize (deliver to patient)
        score < threshold, retries left → generate_response (retry with feedback)
        score < threshold, exhausted    → escalate (hand off to clinician)
    """
    score = state.get("judge_score") or 0.0
    risk_tier = (state.get("risk_tier") or "").lower()
    threshold = JUDGE_THRESHOLDS.get(risk_tier, JUDGE_DEFAULT_THRESHOLD)

    if score >= threshold:
        return "finalize"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries") or JUDGE_MAX_RETRIES

    if retry_count >= max_retries:
        return "escalate"

    return "generate_response"


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(*, checkpointer: Optional[Any] = None) -> StateGraph:
    builder = StateGraph(ChatState)

    # ── Register nodes ──────────────────────────────────────────────────────────
    builder.add_node("classify_query",    classify_query_node)
    builder.add_node("fetch_patient_data", fetch_patient_data_node)
    builder.add_node("retrieve_rag",      retrieve_rag_node)
    builder.add_node("generate_response", generate_response_node)
    builder.add_node("judge_response",    judge_response_node)
    builder.add_node("escalate",          escalate_node)
    builder.add_node("finalize",          finalize_node)

    # ── Entry point ─────────────────────────────────────────────────────────────
    builder.add_edge(START, "classify_query")

    # ── Post-classification: chitchat skips patient data ────────────────────────
    builder.add_conditional_edges(
        "classify_query",
        _route_after_classify,
        {"generate_response": "generate_response", "fetch_patient_data": "fetch_patient_data"},
    )

    # ── Post-patient-data: logistics skips RAG; medical continues to retrieval ────
    builder.add_conditional_edges(
        "fetch_patient_data",
        _route_after_patient_data,
        {"generate_response": "generate_response", "retrieve_rag": "retrieve_rag"},
    )

    # ── Medical path: fetch_patient_data → retrieve_rag → generate ──────────────
    builder.add_edge("retrieve_rag", "generate_response")

    # ── Post-generation: medical goes to judge; chitchat/logistics to finalize ───
    builder.add_conditional_edges(
        "generate_response",
        _route_after_generate,
        {"judge_response": "judge_response", "finalize": "finalize"},
    )

    # ── Judge routing: pass → finalize | retry → generate | exhausted → escalate ─
    builder.add_conditional_edges(
        "judge_response",
        _route_after_judge,
        {
            "finalize":          "finalize",
            "generate_response": "generate_response",   # retry loop
            "escalate":          "escalate",
        },
    )

    # ── Escalation feeds into finalize so history is always written ─────────────
    builder.add_edge("escalate", "finalize")

    # ── Finalize is the sole exit point ─────────────────────────────────────────
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer or MemorySaver())


def build_graph_with_sqlite(sqlite_path: str | Path) -> StateGraph:
    """
    Build a graph that persists checkpoints to a local SQLite DB.

    This is intended for offline evaluation runs (e.g. RAGAS) so that long jobs
    can resume after rate limiting or process restarts without changing the
    production chatbot's default in-memory behavior.
    """
    sqlite_path = Path(sqlite_path)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as exc:
        raise ImportError(
            "Could not import LangGraph SqliteSaver. Install SQLite checkpoint support with "
            "`pip install langgraph-checkpoint-sqlite`."
        ) from exc

    # Keep the connection open for the lifetime of the compiled graph.
    conn = sqlite3.connect(str(sqlite_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    return build_graph(checkpointer=saver)


# Singleton — compiled once at import time and reused across all requests
graph = build_graph()
