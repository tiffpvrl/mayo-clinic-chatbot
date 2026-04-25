#!/usr/bin/env python3
"""
Run RAGAS evaluation for MayoChat clinician-reviewed dataset.

Input:
  - Excel file with 42 rows: 21 prompts x 2 clinicians
  - Required columns:
      clinician, query, answer, contexts, ground_truth
    plus clinician review columns:
      factual, factual_quality, accurate, accuracy_quality, relevant,
      relevance_quality, hallucination, hallucination_quality, harmful,
      harmfulness_quality

What this script does:
  1. Loads the clinician Excel file.
  2. Deduplicates to one unique RAGAS row per query/answer/context/ground_truth.
  3. Splits examples into:
       a) rows with retrieved contexts
       b) rows without retrieved contexts
  4. Runs full RAGAS metrics on rows with contexts:
       - faithfulness
       - answer_relevancy
       - context_precision
       - context_recall
  5. Runs answer_relevancy only on rows without contexts.
  6. Merges RAGAS scores back to all clinician rows.
  7. Exports CSV files for analysis.

Example:
python src/evaluation/run_ragas_eval.py \
  --input-xlsx data/ragas_evaluation.xlsx \
  --output-dir outputs/ragas_eval \
  --project YOUR_PROJECT_ID \
  --location us-central1 \
  --judge-model gemini-2.0-flash-001 \
  --embedding-model text-embedding-005
"""

from __future__ import annotations

import argparse
import ast
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

    # Deduplicate RAGAS inputs. Clinician labels are merged back later.
    unique_cols = ["question", "answer", "contexts", "reference"]
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


def init_vertex_ragas_models(project: str, location: str, judge_model: str, embedding_model: str):
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
    parser.add_argument("--judge-model", default="gemini-2.0-flash-001")
    parser.add_argument("--embedding-model", default="text-embedding-005")
    parser.add_argument("--limit", type=int, default=None, help="Optional small test limit")
    args = parser.parse_args()

    if not args.project:
        raise ValueError("GCP project missing. Pass --project or set GOOGLE_CLOUD_PROJECT.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.input_xlsx)
    unique = build_ragas_input(raw)

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
    def make_merge_key(question, answer, contexts, ground_truth):
        return (
            clean_text(question)
            + "||"
            + clean_text(answer)
            + "||"
            + clean_text(contexts)
            + "||"
            + clean_text(ground_truth)
        )

    unique["merge_key"] = unique.apply(
        lambda r: make_merge_key(
            r["question"],
            r["answer"],
            r["contexts"],
            r["reference"],
        ),
        axis=1,
    )

    # Merge scores onto unique cases.
    unique_scores = unique.merge(scores, on="ragas_case_id", how="left")
    unique_scores.to_csv(output_dir / "ragas_unique_results.csv", index=False)

    # Merge RAGAS case ids + scores back to all clinician rows.
    # Use a stable string key instead of relying on RAGAS output column names.
    def make_merge_key(question, answer, contexts, ground_truth):
        return (
            clean_text(question)
            + "||"
            + clean_text(answer)
            + "||"
            + clean_text(contexts)
            + "||"
            + clean_text(ground_truth)
        )

    raw_work = raw.copy()
    raw_work["merge_key"] = raw_work.apply(
        lambda r: make_merge_key(
            r["query"],
            r["answer"],
            r["contexts"],
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
