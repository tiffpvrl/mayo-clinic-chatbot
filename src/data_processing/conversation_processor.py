from __future__ import annotations

"""
Conversational Dialogue Processing for Bowel Prep Patient Knowledge Base

Reads Mayo_Clinic_Patient_Clinician_Dialogues.xlsx and produces two chunk types:
1. turn_level        — one Q&A pair per chunk (tone/phrasing examples)
2. conversation_level — full multi-turn thread per chunk (flow examples)

New file schema (9 columns):
  conversation_id, turn_number, risk_tier, prep_type, appointment_time,
  patient_message, clinician_response, escalated_to_clinician, escalation_reason

Output format matches clinical_processed_chunks.json so both can be indexed
by the same ChromaDB pipeline in retrieval/chromadb_store.py.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
import json
from typing import NamedTuple
import pandas as pd

from data_processing.document_processor import DocumentType


VALID_RISK_TIERS = {"Low", "Medium", "High"}


class ConversationalChunks(NamedTuple):
    """Return type for process_conversational_dialogues — pre-split by chunk type."""
    turn_level: list[dict]
    conversation_level: list[dict]


def process_conversational_dialogues(
    input_file: Path | str,
    output_path: Path | str | None = None,
) -> ConversationalChunks:
    """
    Process conversational dialogues from an Excel file into RAG-ready chunks.

    Returns a ConversationalChunks named tuple with .turn_level and .conversation_level
    already separated so callers never need to filter by chunk_type.
    Output format matches existing clinical_processed_chunks.json structure.
    """
    input_file = Path(input_file)
    print("=" * 80)
    print("PROCESSING CONVERSATIONAL DIALOGUES FOR RAG")
    print("=" * 80)

    df = pd.read_excel(input_file)
    print(f"Loaded {len(df):,} dialogue turns across {df['conversation_id'].nunique():,} conversations")

    # Normalise risk_tier: merge any legacy "Very High" into "High"
    df["risk_tier"] = df["risk_tier"].replace("Very High", "High")

    # ------------------------------------------------------------------
    # PART 1: Turn-level chunks (one Q&A pair each)
    # ------------------------------------------------------------------
    print("Creating turn-level chunks...")
    turn_chunks: list[dict] = []

    for _, row in df.iterrows():
        escalated = str(row.get("escalated_to_clinician", "No")).strip()
        escalation_reason = str(row.get("escalation_reason", "")).strip()

        chunk = {
            "id": f"conversation_{row['conversation_id']}_turn_{int(row['turn_number']):02d}",
            "content": (
                f"Patient Question: {row['patient_message']}\n\n"
                f"Clinician Response: {row['clinician_response']}"
            ),
            "metadata": {
                "source_file": str(input_file),
                "document_type": DocumentType.CONVERSATIONAL_EXAMPLE.value,
                "chunk_type": "turn_level",
                "conversation_id": int(row["conversation_id"]),
                "turn_number": int(row["turn_number"]),
                "risk_tier": row["risk_tier"],
                "prep_type": row["prep_type"],
                "appointment_time": row["appointment_time"],
                "patient_message": row["patient_message"],
                "clinician_response": row["clinician_response"],
                "is_follow_up": int(row["turn_number"]) > 1,
                "escalated_to_clinician": escalated,
                "escalation_reason": escalation_reason,
                "tags": [
                    f"risk_{row['risk_tier'].lower()}",
                    "conversational_tone",
                    f"turn_{int(row['turn_number'])}",
                ] + (["escalated"] if escalated == "Yes" else []),
            },
        }
        turn_chunks.append(chunk)

    print(f"  Created {len(turn_chunks):,} turn-level chunks")

    # ------------------------------------------------------------------
    # PART 2: Conversation-level chunks (full multi-turn)
    # ------------------------------------------------------------------
    print("Creating conversation-level chunks...")
    conv_chunks: list[dict] = []

    for conv_id, group in df.groupby("conversation_id"):
        turns_text = []
        for _, row in group.iterrows():
            turns_text.append(
                f"Turn {int(row['turn_number'])}:\n"
                f"Patient: {row['patient_message']}\n"
                f"Clinician: {row['clinician_response']}"
            )

        risk_tier = group.iloc[0]["risk_tier"]
        any_escalated = bool((group["escalated_to_clinician"].str.strip() == "Yes").any())
        escalation_reasons = (
            group.loc[group["escalated_to_clinician"].str.strip() == "Yes", "escalation_reason"]
            .dropna()
            .tolist()
        )

        chunk = {
            "id": f"dialogue_conversation_{conv_id}",
            "content": "\n\n".join(turns_text),
            "metadata": {
                "source_file": str(input_file),
                "document_type": DocumentType.CONVERSATIONAL_EXAMPLE.value,
                "chunk_type": "conversation_level",
                "conversation_id": int(conv_id),
                "num_turns": len(group),
                "risk_tier": risk_tier,
                "prep_type": group.iloc[0]["prep_type"],
                "appointment_time": group.iloc[0]["appointment_time"],
                "demonstrates_multi_turn": len(group) > 1,
                "any_escalated": any_escalated,
                "escalation_reasons": escalation_reasons,
                "tags": (
                    ["multi_turn_example", "conversation_flow", f"{len(group)}_turns",
                     f"risk_{risk_tier.lower()}"]
                    + (["has_escalation"] if any_escalated else [])
                ),
            },
        }
        conv_chunks.append(chunk)

    print(f"  Created {len(conv_chunks):,} conversation-level chunks")

    # ------------------------------------------------------------------
    # PART 3: Save output (combined to single JSON for archival)
    # ------------------------------------------------------------------
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined = turn_chunks + conv_chunks
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(combined):,} conversational chunks to: {output_path}")

        summary_path = output_path.parent / "conversational_processing_summary.json"
        risk_dist = df["risk_tier"].value_counts()
        summary = {
            "total_chunks": len(combined),
            "turn_level_chunks": len(turn_chunks),
            "conversation_level_chunks": len(conv_chunks),
            "total_conversations": int(df["conversation_id"].nunique()),
            "total_turns": len(df),
            "risk_tier_distribution": {
                tier: {"count": int(cnt), "pct": round(cnt / len(df) * 100, 1)}
                for tier, cnt in risk_dist.items()
            },
            "escalation": {
                "total_escalated": int((df["escalated_to_clinician"].str.strip() == "Yes").sum()),
                "escalation_rate_pct": round(
                    (df["escalated_to_clinician"].str.strip() == "Yes").mean() * 100, 1
                ),
            },
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {summary_path}")

    print()
    print("=" * 80)
    print("CONVERSATIONAL CHUNKING SUMMARY")
    print("=" * 80)
    print(f"Turn-level chunks:         {len(turn_chunks):,}")
    print(f"Conversation-level chunks: {len(conv_chunks):,}")
    print(f"Total:                     {len(turn_chunks) + len(conv_chunks):,}")

    return ConversationalChunks(turn_level=turn_chunks, conversation_level=conv_chunks)


# ---------------------------------------------------------------------------
# CLI - for testing only; production indexing runs via retrieval/index_kb.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    conv_input = Path("src/data_processing/patient_kb/conversations/Mayo_Clinic_Patient_Clinician_Dialogues.xlsx")
    conv_out = Path("src/data_processing/patient_kb/processed_chunks/conversational_chunks.json")

    conv = process_conversational_dialogues(input_file=conv_input, output_path=conv_out)
    print(f"\nSample turn-level chunk:")
    if conv.turn_level:
        c = conv.turn_level[0]
        print(f"  Id: {c['id']}")
        print(f"  Risk tier: {c['metadata']['risk_tier']}")
        print(f"  Escalated: {c['metadata']['escalated_to_clinician']}")
        print(f"  Tags: {c['metadata']['tags']}")
        print(f"  Content preview: {c['content'][:200]}...")
    print(f"\nSample conversation-level chunk:")
    if conv.conversation_level:
        c = conv.conversation_level[0]
        print(f"  Id: {c['id']}")
        print(f"  Num turns: {c['metadata']['num_turns']}")
        print(f"  Risk tier: {c['metadata']['risk_tier']}")
        print(f"  Content preview: {c['content'][:200]}...")
