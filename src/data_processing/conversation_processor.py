"""
Conversational Dialogue Processing for Bowel Prep Patient Knowledge Base

Reads mayo_clinic_chatbot_dialogues.xlsx and produces two chunk types:
1. turn_level   — one Q&A pair per chunk (tone/phrasing examples)
2. conversation_level — full multi-turn thread per chunk (flow examples)

Output format matches clinical_processed_chunks.json so both can be indexed
by the same ChromaDB pipeline in retrieval/chromadb_store.py.
"""

from pathlib import Path
import json
from typing import NamedTuple
import pandas as pd

from data_processing.document_processor import DocumentType


class ConversationalChunks(NamedTuple):
    """Return type for process_conversational_dialogues — pre-split by chunk type."""
    turn_level: list[dict]
    conversation_level: list[dict]

# TODO: handle duplicates either through new data
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

    # ------------------------------------------------------------------
    # PART 1: Turn-level chunks (one Q&A pair each)
    # ------------------------------------------------------------------
    print("Creating turn-level chunks...")
    turn_chunks: list[dict] = []

    for _, row in df.iterrows():
        timestamp_val = row["timestamp"]
        timestamp_str = timestamp_val.isoformat() if pd.notna(timestamp_val) else None

        chunk = {
            "id": f"dialogue_turn_{row['conversation_id']}_{int(row['turn_number']):02d}",
            "content": (
                f"Patient Question: {row['patient_message']}\n\n"
                f"Chatbot Response: {row['chatbot_response']}"
            ),
            "metadata": {
                "source_file": str(input_file),
                "document_type": DocumentType.CONVERSATIONAL_EXAMPLE.value,
                "chunk_type": "turn_level",
                "conversation_id": int(row["conversation_id"]),
                "turn_number": int(row["turn_number"]),
                "query_category": row["query_category"],
                "prep_type": row["prep_type"],
                "appointment_time": row["appointment_time"],
                "days_relative_to_procedure": int(row["days_relative_to_procedure"]),
                "timestamp": timestamp_str,
                "patient_message": row["patient_message"],
                "chatbot_response": row["chatbot_response"],
                "is_follow_up": int(row["turn_number"]) > 1,
                "tags": [
                    row["query_category"],
                    "conversational_tone",
                    f"turn_{int(row['turn_number'])}",
                ],
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
                f"Chatbot: {row['chatbot_response']}"
            )

        topics = group["query_category"].tolist()
        unique_topics = list(dict.fromkeys(topics))  # preserve order, deduplicate

        chunk = {
            "id": f"dialogue_conversation_{conv_id}",
            "content": "\n\n".join(turns_text),
            "metadata": {
                "source_file": str(input_file),
                "document_type": DocumentType.CONVERSATIONAL_EXAMPLE.value,
                "chunk_type": "conversation_level",
                "conversation_id": int(conv_id),
                "num_turns": len(group),
                "query_categories": topics,
                "conversation_flow": " -> ".join(topics),
                "prep_type": group.iloc[0]["prep_type"],
                "appointment_time": group.iloc[0]["appointment_time"],
                "demonstrates_multi_turn": len(group) > 1,
                "tags": (
                    ["multi_turn_example", "conversation_flow", f"{len(group)}_turns"]
                    + unique_topics
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
        categories = df["query_category"].value_counts()
        summary = {
            "total_chunks": len(combined),
            "turn_level_chunks": len(turn_chunks),
            "conversation_level_chunks": len(conv_chunks),
            "total_conversations": int(df["conversation_id"].nunique()),
            "total_turns": len(df),
            "query_category_distribution": {
                cat: {"count": int(cnt), "pct": round(cnt / len(df) * 100, 1)}
                for cat, cnt in categories.items()
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
    conv_input = Path("src/data_processing/patient_kb/conversations/mayo_clinic_chatbot_dialogues.xlsx")
    conv_out = Path("src/data_processing/patient_kb/processed_chunks/conversational_chunks.json")

    conv = process_conversational_dialogues(input_file=conv_input, output_path=conv_out)
    print(f"\nSample turn-level chunk:")
    if conv.turn_level:
        c = conv.turn_level[0]
        print(f"  Id: {c['id']}")
        print(f"  Document Type: {c['metadata']['document_type']}")
        print(f"  Chunk Type: {c['metadata']['chunk_type']}")
        print(f"  Tags: {c['metadata']['tags']}")
        print(f"  Content preview: {c['content'][:200]}...")
    print(f"\nSample conversation-level chunk:")
    if conv.conversation_level:
        c = conv.conversation_level[0]
        print(f"  Id: {c['id']}")
        print(f"  Num turns: {c['metadata']['num_turns']}")
        print(f"  Flow: {c['metadata']['conversation_flow']}")
        print(f"  Content preview: {c['content'][:200]}...")
