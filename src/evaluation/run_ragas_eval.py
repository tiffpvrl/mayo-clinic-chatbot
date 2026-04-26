#!/usr/bin/env python3
"""
Run RAGAS evaluation for MayoChat clinician-reviewed dataset.

Input:
  - Excel file with 42 rows: 7 patients x 3 turns x 2 clinicians
  - Required columns (evaluation sheet):
      clinician, patient_id, turn, query, ground_truth
    plus clinician review columns:
      factual, factual_quality, accurate, accuracy_quality, relevant,
      relevance_quality, hallucination, hallucination_quality, harmful,
      harmfulness_quality

What this script does:
  1. Loads the clinician Excel file.
  2. Runs the full MayoChat LangGraph pipeline once per (patient_id, turn),
     reusing the output for both clinicians' ground-truth rows.
  3. Builds RAGAS inputs using:
       - question: Excel query
       - answer: pipeline response
       - contexts: pipeline clinical_hits (retrieved chunks)
       - ground_truth: clinician golden answer
  4. Deduplicates to one unique RAGAS row per question/answer/contexts/ground_truth.
  5. Splits examples into:
       a) rows with retrieved contexts
       b) rows without retrieved contexts
  6. Runs full RAGAS metrics on rows with contexts:
       - faithfulness
       - answer_relevancy
       - context_precision
       - context_recall
  7. Runs answer_relevancy only on rows without contexts.
  8. Merges RAGAS scores back to all clinician rows.
  9. Exports CSV files for analysis.

Example:
python src/evaluation/run_ragas_eval.py \
  --input-xlsx data/ragas_evaluation.xlsx \
  --output-dir outputs/ragas_eval \
  --project YOUR_PROJECT_ID \
  --location us-central1 \
  --judge-model gemini-2.0-flash \
  --embedding-model mini_lm
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd


def clean_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_contexts(x: Any) -> List[str]:
    """
    Convert the contexts cell into a list[str] for RAGAS.

    Handles:
    - NaN / empty cells -> []
    - Python/JSON-looking lists -> list[str]
    - plain strings with multiple source snippets -> list of source-ish snippets

    The splitting logic is intentionally conservative. If we cannot confidently split,
    we keep the full context as one string.
    """
    # Allow callers to pass contexts directly as a list[str]
    if isinstance(x, list):
        return [clean_text(v) for v in x if clean_text(v)]

    if pd.isna(x):
        return []

    s = str(x).strip()
    if not s:
        return []

    # Try literal list first if the cell looks like a list.
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [clean_text(v) for v in parsed if clean_text(v)]
        except Exception:
            pass

    # Normalize whitespace but keep enough text for RAGAS.
    s = re.sub(r"\r\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    # Many of your cells look like concatenated source blocks beginning with source names.
    # If common source names appear repeatedly, split before those names.
    source_pattern = r"(?=\n?(?:Mass General|Cleveland Clinic|UCSF|Mayo Clinic|DailyMed|OpenFDA|USMSTF|AGA|ASGE|ACG)\b)"
    parts = [p.strip() for p in re.split(source_pattern, s) if p.strip()]

    # Use split parts only if it found more than one meaningful block.
    if len(parts) > 1:
        return parts

    return [s]


def build_ragas_input(df: pd.DataFrame) -> pd.DataFrame:
    required = ["query", "answer", "contexts", "ground_truth"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Excel: {missing}")

    work = df.copy()
    work["question"] = work["query"].apply(clean_text)
    work["answer"] = work["answer"].apply(clean_text)
    work["reference"] = work["ground_truth"].apply(clean_text)
    work["contexts_list"] = work["contexts"].apply(parse_contexts)
    work["has_context"] = work["contexts_list"].apply(lambda x: len(x) > 0)
    # Stable, hashable key for dedupe/merge (pandas can't hash list objects).
    work["contexts_key"] = work["contexts_list"].apply(lambda xs: "\n\n---CONTEXT---\n\n".join(xs or []))

    # Deduplicate RAGAS inputs. Clinician labels are merged back later.
    unique_cols = ["question", "answer", "contexts_key", "reference"]
    unique = work.drop_duplicates(subset=unique_cols).copy()
    unique = unique.reset_index(drop=True)
    unique["ragas_case_id"] = [f"RAGAS-{i+1:03d}" for i in range(len(unique))]
    unique["no_context_expected_or_available"] = ~unique["has_context"]

    return unique


def make_hf_dataset(eval_df: pd.DataFrame):
    """
    Convert to Hugging Face Dataset for ragas.evaluate.
    RAGAS commonly expects these column names:
      question, answer, contexts, ground_truth
    """
    from datasets import Dataset

    records = []
    for _, row in eval_df.iterrows():
        records.append(
            {
                "question": row["question"],
                "answer": row["answer"],
                "contexts": row["contexts_list"],
                "ground_truth": row["reference"],
            }
        )
    return Dataset.from_list(records)


def init_vertex_ragas_models(
    project: str,
    location: str,
    judge_model: str,
    embedding_model: str,
    *,
    hf_cache_dir: Path | None = None,
):
    """
    Initialize Gemini/Vertex models for RAGAS.

    This uses LangChain's Vertex wrappers, then wraps them for RAGAS if the installed
    RAGAS version requires wrappers. It supports common RAGAS versions.
    """
    import vertexai
    from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

    vertexai.init(project=project, location=location)

    llm = ChatVertexAI(
        model_name=judge_model,
        project=project,
        location=location,
        temperature=0,
        max_output_tokens=2048,
    )
    if embedding_model in {"mini_lm", "minilm"}:
        # Use the same MiniLM model as retrieval for RAGAS embedding-based metrics.
        # Cache the model weights locally so Cloud Shell doesn't re-download each run.
        from sentence_transformers import SentenceTransformer

        cache_dir = hf_cache_dir or Path(os.environ.get("HF_HOME", "") or ".").resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache_dir))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir / "transformers"))

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=str(cache_dir))

        class _MiniLMEmbeddings:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return model.encode(texts, show_progress_bar=False).tolist()

            def embed_query(self, text: str) -> list[float]:
                return model.encode([text], show_progress_bar=False).tolist()[0]

        embeddings = _MiniLMEmbeddings()
    else:
        embeddings = VertexAIEmbeddings(
            model_name=embedding_model,
            project=project,
            location=location,
        )

    # Newer RAGAS versions may want wrappers. Older versions accept LangChain directly.
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        return llm, embeddings


def run_ragas(dataset, metrics, llm, embeddings) -> pd.DataFrame:
    from ragas import evaluate

    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )

    # RAGAS result objects differ slightly by version.
    if hasattr(result, "to_pandas"):
        return result.to_pandas()
    if hasattr(result, "to_dataframe"):
        return result.to_dataframe()
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-xlsx", required=True, help="Path to finalized ragas_evaluation.xlsx")
    parser.add_argument("--output-dir", default="outputs/ragas_eval", help="Directory for outputs")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"))
    parser.add_argument("--judge-model", default="gemini-2.0-flash")
    parser.add_argument(
        "--embedding-model",
        default="mini_lm",
        help='RAGAS embedding model. Use "mini_lm" to match retrieval; otherwise pass a Vertex embedding model name (e.g. text-embedding-005).',
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional small test limit")
    parser.add_argument(
        "--thread-suffix",
        default="ragas_eval",
        help="Suffix appended to patient_id for LangGraph thread_id isolation",
    )
    parser.add_argument(
        "--checkpoint-sqlite",
        default=None,
        help="Optional path to a SQLite DB for persistent LangGraph checkpoints (enables resume). "
        "Defaults to <output-dir>/langgraph_checkpoints.sqlite when unset.",
    )
    args = parser.parse_args()

    if not args.project:
        raise ValueError("GCP project missing. Pass --project or set GOOGLE_CLOUD_PROJECT.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir = output_dir / ".hf_cache"
    checkpoint_path = Path(args.checkpoint_sqlite) if args.checkpoint_sqlite else (output_dir / "langgraph_checkpoints.sqlite")
    progress_path = output_dir / "progress.jsonl"

    raw = pd.read_excel(args.input_xlsx)

    # ── Run LangGraph once per (patient_id, turn) ────────────────────────────
    # The sheet contains 2 clinician rows per (patient_id, turn). We run the
    # pipeline once per patient-turn, then reuse its answer/contexts for both.
    required_cols = ["clinician", "patient_id", "turn", "query", "ground_truth"]
    missing_cols = [c for c in required_cols if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Excel: {missing_cols}")

    # Use persistent checkpoints so multi-turn state survives restarts and we can resume after 429s.
    from src.graph.graph import build_graph_with_sqlite
    graph = build_graph_with_sqlite(checkpoint_path)

    raw_work = raw.copy()
    raw_work["turn"] = raw_work["turn"].astype(int)
    raw_work["query"] = raw_work["query"].apply(clean_text)

    # Map (patient_id, turn) -> pipeline outputs
    pipeline_outputs: dict[tuple[str, int], dict[str, Any]] = {}

    # ── Load progress (if any) so we can skip completed turns ───────────────
    if progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    # Ignore malformed/truncated lines (e.g. interrupted write).
                    continue
                pid = clean_text(rec.get("patient_id", ""))
                turn = rec.get("turn", None)
                if not pid or turn is None:
                    continue
                try:
                    turn_i = int(turn)
                except Exception:
                    continue
                pipeline_outputs[(pid, turn_i)] = {
                    "pipeline_answer": clean_text(rec.get("pipeline_answer", "")),
                    "pipeline_contexts": rec.get("pipeline_contexts") or [],
                    "pipeline_sources": rec.get("pipeline_sources") or [],
                }

        print(f"[resume] loaded {len(pipeline_outputs)} completed patient-turn(s) from {progress_path}")

    def _append_progress(patient_id: str, turn: int, query: str, response: str, contexts_list: list[str], sources: list[str]) -> None:
        rec = {
            "patient_id": patient_id,
            "turn": int(turn),
            "query": query,
            "pipeline_answer": response,
            "pipeline_contexts": contexts_list,
            "pipeline_sources": sources,
        }
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for patient_id, g_patient in raw_work.groupby("patient_id", dropna=False):
        if pd.isna(patient_id) or clean_text(patient_id) == "":
            raise ValueError("Found missing patient_id in input Excel.")

        g_patient = g_patient.sort_values("turn")
        thread_id = f"{clean_text(patient_id)}:{args.thread_suffix}"

        # Run turns sequentially for this patient so follow-up behavior is preserved.
        for turn, g_turn in g_patient.groupby("turn"):
            pid = clean_text(patient_id)
            key = (pid, int(turn))
            if key in pipeline_outputs:
                # Already computed in a previous run; skip invoking the graph.
                continue

            queries = [q for q in g_turn["query"].tolist() if q]
            if not queries:
                raise ValueError(f"Empty query for patient_id={patient_id!r}, turn={turn!r}.")
            if len(set(queries)) != 1:
                raise ValueError(
                    f"Non-identical queries for patient_id={patient_id!r}, turn={turn!r}: {set(queries)}"
                )
            query = queries[0]

            # Provide a minimal input state. The MemorySaver checkpointer preserves
            # patient_record + chat_history across turns for the same thread_id.
            state = graph.invoke(
                {"query": query, "patient_id": clean_text(patient_id)},
                config={"configurable": {"thread_id": thread_id}},
            )

            response = clean_text(state.get("response", ""))
            clinical_hits = state.get("clinical_hits") or []
            contexts_list = [clean_text(h.get("document", "")) for h in clinical_hits if clean_text(h.get("document", ""))]

            # Keep a lightweight, analysis-friendly sources string too.
            sources = []
            for h in clinical_hits:
                meta = h.get("metadata", {}) or {}
                sources.append(
                    " | ".join(
                        [
                            clean_text(meta.get("organization", "")),
                            clean_text(meta.get("source_file", "")),
                            clean_text(meta.get("section_title", "")),
                        ]
                    ).strip(" |")
                )
            sources = [s for s in sources if s]

            pipeline_outputs[key] = {
                "pipeline_answer": response,
                "pipeline_contexts": contexts_list,
                "pipeline_sources": sources,
            }
            _append_progress(pid, int(turn), query, response, contexts_list, sources)

    # End-of-retrieval resume summary (helps after Vertex 429 interruptions).
    expected_keys = {
        (clean_text(r["patient_id"]), int(r["turn"]))
        for _, r in raw_work[["patient_id", "turn"]].drop_duplicates().iterrows()
        if clean_text(r["patient_id"])
    }
    completed_keys = set(pipeline_outputs.keys())
    missing_keys = sorted(expected_keys - completed_keys, key=lambda x: (x[0], x[1]))
    if missing_keys:
        preview = ", ".join([f"{pid}:turn{turn}" for pid, turn in missing_keys[:20]])
        suffix = "" if len(missing_keys) <= 20 else f" ... (+{len(missing_keys) - 20} more)"
        print(f"[resume] completed={len(completed_keys)}/{len(expected_keys)} patient-turn(s). Missing: {preview}{suffix}")
    else:
        print(f"[resume] completed all {len(expected_keys)} patient-turn(s).")

    # Attach pipeline outputs to every clinician row.
    raw_work["pipeline_answer"] = raw_work.apply(
        lambda r: pipeline_outputs[(clean_text(r["patient_id"]), int(r["turn"]))]["pipeline_answer"],
        axis=1,
    )
    raw_work["pipeline_contexts"] = raw_work.apply(
        lambda r: pipeline_outputs[(clean_text(r["patient_id"]), int(r["turn"]))]["pipeline_contexts"],
        axis=1,
    )
    raw_work["pipeline_sources"] = raw_work.apply(
        lambda r: pipeline_outputs[(clean_text(r["patient_id"]), int(r["turn"]))]["pipeline_sources"],
        axis=1,
    )

    # Build RAGAS input df (one row per clinician golden answer).
    ragas_raw = raw_work.copy()
    ragas_raw["answer"] = ragas_raw["pipeline_answer"]
    ragas_raw["contexts"] = ragas_raw["pipeline_contexts"]

    unique = build_ragas_input(ragas_raw)

    if args.limit:
        unique = unique.head(args.limit).copy()

    unique_export = unique.copy()
    unique_export["contexts_list"] = unique_export["contexts_list"].apply(lambda x: "\n\n---CONTEXT---\n\n".join(x))
    unique_export.to_csv(output_dir / "ragas_unique_input.csv", index=False)

    with_context = unique[unique["has_context"]].copy()
    no_context = unique[~unique["has_context"]].copy()

    print(f"Loaded clinician rows: {len(raw)}")
    print(f"Unique RAGAS cases: {len(unique)}")
    print(f"With context: {len(with_context)}")
    print(f"No context: {len(no_context)}")

    ragas_llm, ragas_embeddings = init_vertex_ragas_models(
        project=args.project,
        location=args.location,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        hf_cache_dir=hf_cache_dir,
    )

    # Import metrics. Names are stable across many RAGAS versions.
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    result_frames = []

    if len(with_context) > 0:
        print("Running full RAGAS metrics on rows with contexts...")
        ds_context = make_hf_dataset(with_context)
        full_scores = run_ragas(
            ds_context,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        full_scores.insert(0, "ragas_case_id", with_context["ragas_case_id"].tolist())
        result_frames.append(full_scores)

    if len(no_context) > 0:
        print("Running answer relevancy only on rows without contexts...")
        ds_no_context = make_hf_dataset(no_context)
        no_context_scores = run_ragas(
            ds_no_context,
            metrics=[answer_relevancy],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
        )
        no_context_scores.insert(0, "ragas_case_id", no_context["ragas_case_id"].tolist())
        no_context_scores["faithfulness"] = np.nan
        no_context_scores["context_precision"] = np.nan
        no_context_scores["context_recall"] = np.nan
        result_frames.append(no_context_scores)

    scores = pd.concat(result_frames, ignore_index=True) if result_frames else pd.DataFrame()
    scores.to_csv(output_dir / "ragas_unique_scores.csv", index=False)

    # Create merge key BEFORE merging with RAGAS scores
    def make_merge_key(question, answer, contexts_key, ground_truth):
        return (
            clean_text(question)
            + "||"
            + clean_text(answer)
            + "||"
            + clean_text(contexts_key)
            + "||"
            + clean_text(ground_truth)
        )

    unique["merge_key"] = unique.apply(
        lambda r: make_merge_key(
            r["question"],
            r["answer"],
            r["contexts_key"],
            r["reference"],
        ),
        axis=1,
    )

    # Merge scores onto unique cases.
    unique_scores = unique.merge(scores, on="ragas_case_id", how="left")
    unique_scores.to_csv(output_dir / "ragas_unique_results.csv", index=False)

    # Merge RAGAS case ids + scores back to all clinician rows.
    # Use a stable string key instead of relying on RAGAS output column names.
    def make_merge_key(question, answer, contexts_key, ground_truth):
        return (
            clean_text(question)
            + "||"
            + clean_text(answer)
            + "||"
            + clean_text(contexts_key)
            + "||"
            + clean_text(ground_truth)
        )

    raw_work = ragas_raw.copy()
    # Must match build_ragas_input() keying logic.
    raw_work["contexts_key"] = raw_work["contexts"].apply(parse_contexts).apply(lambda xs: "\n\n---CONTEXT---\n\n".join(xs or []))
    raw_work["merge_key"] = raw_work.apply(
        lambda r: make_merge_key(
            r["query"],
            r["answer"],
            r["contexts_key"],
            r["ground_truth"],
        ),
        axis=1,
    )

    exclude_cols = {
        "question",
        "answer",
        "contexts",
        "contexts_list",
        "reference",
        "ground_truth",
        "query",
    }
    
    cols_to_merge = [
        c for c in unique_scores.columns
        if c not in exclude_cols and c != "merge_key"
    ]

    merge_scores = unique_scores[["merge_key"] + cols_to_merge].copy()

    merged = raw_work.merge(
        merge_scores,
        on="merge_key",
        how="left",
    )

    merged = merged.drop(columns=["merge_key"])
    merged.to_csv(output_dir / "ragas_results_merged_with_clinicians.csv", index=False)

    # Simple summary.
    metric_summary_cols = [
        c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        if c in unique_scores.columns
    ]
    summary = unique_scores[metric_summary_cols].agg(["count", "mean", "median", "min", "max"]).T
    summary.to_csv(output_dir / "ragas_summary.csv")

    print("\nSaved outputs:")
    print(f"- {output_dir / 'ragas_unique_input.csv'}")
    print(f"- {output_dir / 'ragas_unique_scores.csv'}")
    print(f"- {output_dir / 'ragas_unique_results.csv'}")
    print(f"- {output_dir / 'ragas_results_merged_with_clinicians.csv'}")
    print(f"- {output_dir / 'ragas_summary.csv'}")


if __name__ == "__main__":
    main()
