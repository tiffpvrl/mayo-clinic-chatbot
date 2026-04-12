#!/usr/bin/env python3
"""
Embedding-model retrieval evaluation for MayoChat.

Purpose
-------
Evaluate embedding models *only at the retrieval layer* against a small benchmark
derived from the golden references. This script does NOT call the chatbot model.
It tests whether the retriever surfaces the right knowledge-base chunks.

Expected input
--------------
A processed chunks file produced by your document processor, e.g.:
    processed_chunks.json

Each chunk should look roughly like:
{
  "id": "...",
  "content": "...",
  "metadata": {
      "section_title": "...",
      "source_file": "...",
      "document_type": "...",
      "tags": ["renal_disease", "timing", "med_class:anticoagulants", ...]
  }
}

What it outputs
---------------
1) per-model per-query retrieval results CSV
2) summary CSV with hit@k, mrr@k, avg relevant@k
3) optional cache of chunk embeddings to speed up reruns

Usage examples
--------------
python eval_embeddings.py \
  --chunks-path path/to/processed_chunks.json \
  --benchmark-path path/to/retrieval_benchmark.json \
  --models vertex_text_embedding_005 bge_base_en e5_base_v2 all_minilm_l6_v2 medcpt_biomedical \
  --top-k 5 \
  --output-dir outputs/embedding_eval

Notes
-----
- Keep the retriever settings fixed while comparing models.
- This script uses a curated retrieval benchmark of golden-reference-like cases.
- Scoring is intentionally lightweight and transparent:
    * hit@k: did at least one clearly relevant chunk appear?
    * mrr@k: was a relevant chunk ranked early?
    * avg_relevant@k: how many relevant chunks appeared in top-k on average?
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Optional imports loaded lazily
SentenceTransformer = None
vertexai = None
TextEmbeddingInput = None
TextEmbeddingModel = None
AutoTokenizer = None
AutoModel = None


# -----------------------------
# Model registry
# -----------------------------
MODEL_REGISTRY = {
    "bge_base_en": {
        "type": "hf",
        "hf_name": "BAAI/bge-base-en-v1.5",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "e5_base_v2": {
        "type": "hf",
        "hf_name": "intfloat/e5-base-v2",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
    },
    "all_minilm_l6_v2": {
        "type": "hf",
        "hf_name": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "biomedbert_hash_nano": {
        "type": "hf",
        "hf_name": "NeuML/biomedbert-hash-nano-embeddings",
        "query_prefix": "",
        "doc_prefix": "",
    },
    "vertex_text_embedding_005": {
        "type": "vertex",
        "vertex_name": "text-embedding-005",
    },
}


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class Chunk:
    chunk_id: str
    content: str
    source_file: str
    section_title: str
    document_type: str
    tags: List[str]


@dataclass
class BenchmarkCase:
    case_id: str
    gr_id: str
    patient_id: str
    query: str
    patient_context: Dict[str, Any]
    expected_any: List[Dict[str, Any]]
    notes: str = ""


# -----------------------------
# Utility helpers
# -----------------------------
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_get(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    val = d.get(key, default)
    return default if val is None else val


def cosine_similarity_matrix(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    # query_vec: (dim,)
    # doc_mat: (n, dim)
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_mat, axis=1)
    denom = np.maximum(query_norm * doc_norms, 1e-12)
    sims = (doc_mat @ query_vec) / denom
    return sims


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_of_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Loading chunks and benchmark
# -----------------------------
def load_chunks(chunks_path: Path) -> List[Chunk]:
    raw = load_json(chunks_path)
    chunks: List[Chunk] = []

    if not isinstance(raw, list):
        raise ValueError(f"Expected a list in {chunks_path}, got {type(raw)}")

    for i, item in enumerate(raw):
        metadata = item.get("metadata", {}) or {}
        chunk = Chunk(
            chunk_id=str(item.get("id", f"chunk_{i}")),
            content=str(item.get("content", "")).strip(),
            source_file=str(metadata.get("source_file", "")),
            section_title=str(metadata.get("section_title", "")),
            document_type=str(metadata.get("document_type", "")),
            tags=list(metadata.get("tags", []) or []),
        )
        if chunk.content:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(f"No non-empty chunks loaded from {chunks_path}")

    return chunks


def load_benchmark(benchmark_path: Path) -> List[BenchmarkCase]:
    raw = load_json(benchmark_path)
    cases: List[BenchmarkCase] = []
    for item in raw:
        cases.append(
            BenchmarkCase(
                case_id=item["case_id"],
                gr_id=item["gr_id"],
                patient_id=item["patient_id"],
                query=item["query"],
                patient_context=item["patient_context"],
                expected_any=item["expected_any"],
                notes=item.get("notes", ""),
            )
        )
    return cases


# -----------------------------
# Query construction
# -----------------------------
def build_patient_context_string(ctx: Dict[str, Any]) -> str:
    ordered_fields = [
        ("risk_tier", "Risk tier"),
        ("prep_agent", "Prep agent"),
        ("regimen", "Regimen"),
        ("procedure_time", "Procedure time"),
        ("comorbidities", "Comorbidities"),
        ("medications", "Medications"),
        ("dietary_restriction", "Dietary restriction"),
        ("prior_history", "Prior history"),
        ("prep_modifications", "Prep modifications"),
    ]
    parts = []
    for key, label in ordered_fields:
        value = safe_get(ctx, key, "")
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v)
        if value:
            parts.append(f"{label}: {value}")
    return "Patient context. " + ". ".join(parts) + "."


def build_query_text(case: BenchmarkCase) -> str:
    return f"{build_patient_context_string(case.patient_context)} Question: {case.query}"


# -----------------------------
# Relevance matching
# -----------------------------
def chunk_matches_rule(chunk: Chunk, rule: Dict[str, Any]) -> bool:
    """
    A chunk is relevant if it matches a rule.
    Supported rule fields:
      - keywords_any: list[str] => at least N of them appear in content/section/source
      - keywords_all: list[str] => all must appear
      - tags_any: list[str] => at least one tag appears
      - section_keywords_any: list[str] => one appears in section title
      - source_keywords_any: list[str] => one appears in source file
      - min_keyword_hits: int => for keywords_any
    """
    content_blob = normalize_text(
        " ".join([chunk.content, chunk.section_title, chunk.source_file, " ".join(chunk.tags)])
    )
    section_blob = normalize_text(chunk.section_title)
    source_blob = normalize_text(chunk.source_file)
    tag_set = {normalize_text(t) for t in chunk.tags}

    keywords_any = [normalize_text(x) for x in rule.get("keywords_any", [])]
    keywords_all = [normalize_text(x) for x in rule.get("keywords_all", [])]
    tags_any = [normalize_text(x) for x in rule.get("tags_any", [])]
    section_keywords_any = [normalize_text(x) for x in rule.get("section_keywords_any", [])]
    source_keywords_any = [normalize_text(x) for x in rule.get("source_keywords_any", [])]
    min_keyword_hits = int(rule.get("min_keyword_hits", 1))

    if keywords_all:
        if not all(k in content_blob for k in keywords_all):
            return False

    if keywords_any:
        hits = sum(1 for k in keywords_any if k in content_blob)
        if hits < min_keyword_hits:
            return False

    if tags_any:
        if not any(t in tag_set for t in tags_any):
            return False

    if section_keywords_any:
        if not any(k in section_blob for k in section_keywords_any):
            return False

    if source_keywords_any:
        if not any(k in source_blob for k in source_keywords_any):
            return False

    return True


def is_relevant_chunk(chunk: Chunk, case: BenchmarkCase) -> bool:
    return any(chunk_matches_rule(chunk, rule) for rule in case.expected_any)


# -----------------------------
# Embedding model wrappers
# -----------------------------
class BaseEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class HFEmbedder(BaseEmbedder):
    def __init__(self, hf_name: str, query_prefix: str = "", doc_prefix: str = "") -> None:
        global SentenceTransformer
        if SentenceTransformer is None:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer
            SentenceTransformer = _SentenceTransformer

        self.model = SentenceTransformer(hf_name)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        prefixed = [self.doc_prefix + t for t in texts]
        embs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        emb = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )[0]
        return emb.astype(np.float32)



class DualHFEmbedder(BaseEmbedder):
    """
    Separate Hugging Face encoders for queries and documents.
    Used for MedCPT-style biomedical retrieval models.
    """
    def __init__(self, query_hf_name: str, doc_hf_name: str) -> None:
        global AutoTokenizer, AutoModel
        if AutoTokenizer is None or AutoModel is None:
            from transformers import AutoTokenizer as _AutoTokenizer, AutoModel as _AutoModel
            AutoTokenizer = _AutoTokenizer
            AutoModel = _AutoModel

        self.query_tokenizer = AutoTokenizer.from_pretrained(query_hf_name)
        self.query_model = AutoModel.from_pretrained(query_hf_name)
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_hf_name)
        self.doc_model = AutoModel.from_pretrained(doc_hf_name)

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-9)
        return summed / denom

    def _encode(self, texts, tokenizer, model) -> np.ndarray:
        import torch
        vectors = []
        batch_size = 16
        model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i:i+batch_size])
                toks = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                out = model(**toks)
                pooled = self._mean_pool(out.last_hidden_state, toks["attention_mask"])
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                vectors.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(vectors)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, self.doc_tokenizer, self.doc_model)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], self.query_tokenizer, self.query_model)[0]

class VertexEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, project: str, location: str) -> None:
        global vertexai, TextEmbeddingInput, TextEmbeddingModel
        if vertexai is None:
            import vertexai as _vertexai
            from vertexai.language_models import (
                TextEmbeddingInput as _TextEmbeddingInput,
                TextEmbeddingModel as _TextEmbeddingModel,
            )
            vertexai = _vertexai
            TextEmbeddingInput = _TextEmbeddingInput
            TextEmbeddingModel = _TextEmbeddingModel

        vertexai.init(project=project, location=location)
        self.model = TextEmbeddingModel.from_pretrained(model_name)

    def _batch_embed(self, texts: Sequence[str], task_type: str) -> np.ndarray:
        vectors: List[np.ndarray] = []
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = [TextEmbeddingInput(text, task_type) for text in batch]
            outputs = self.model.get_embeddings(inputs)
            for o in outputs:
                vectors.append(np.asarray(o.values, dtype=np.float32))
        mat = np.vstack(vectors)
        # normalize for cosine retrieval
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.maximum(norms, 1e-12)
        return mat.astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._batch_embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        vec = self._batch_embed([text], task_type="RETRIEVAL_QUERY")[0]
        return vec.astype(np.float32)


def build_embedder(model_key: str, args: argparse.Namespace) -> BaseEmbedder:
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}")

    spec = MODEL_REGISTRY[model_key]
    if spec["type"] == "hf":
        return HFEmbedder(
            hf_name=spec["hf_name"],
            query_prefix=spec.get("query_prefix", ""),
            doc_prefix=spec.get("doc_prefix", ""),
        )
    if spec["type"] == "dual_hf":
        return DualHFEmbedder(
            query_hf_name=spec["query_hf_name"],
            doc_hf_name=spec["doc_hf_name"],
        )
    if spec["type"] == "vertex":
        project = args.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = args.gcp_location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            raise ValueError(
                "Vertex model requested but GCP project not provided. "
                "Use --gcp-project or set GOOGLE_CLOUD_PROJECT."
            )
        return VertexEmbedder(
            model_name=spec["vertex_name"],
            project=project,
            location=location,
        )
    raise ValueError(f"Unsupported model type: {spec['type']}")


# -----------------------------
# Caching
# -----------------------------
def cache_path_for_model(cache_dir: Path, model_key: str, chunks_path: Path, n_chunks: int) -> Path:
    fingerprint = sha1_of_text(f"{model_key}|{chunks_path.resolve()}|{n_chunks}")
    return cache_dir / f"{model_key}_{fingerprint}.pkl"


def load_or_compute_doc_embeddings(
    model_key: str,
    embedder: BaseEmbedder,
    chunk_texts: Sequence[str],
    chunks_path: Path,
    cache_dir: Path,
) -> np.ndarray:
    ensure_dir(cache_dir)
    cp = cache_path_for_model(cache_dir, model_key, chunks_path, len(chunk_texts))
    if cp.exists():
        with cp.open("rb") as f:
            return pickle.load(f)

    embs = embedder.embed_documents(chunk_texts)
    with cp.open("wb") as f:
        pickle.dump(embs, f)
    return embs


# -----------------------------
# Retrieval evaluation
# -----------------------------
def format_chunk_for_embedding(chunk: Chunk, include_metadata_text: bool) -> str:
    if not include_metadata_text:
        return chunk.content

    metadata_bits = []
    if chunk.section_title:
        metadata_bits.append(f"Section: {chunk.section_title}")
    if chunk.document_type:
        metadata_bits.append(f"Type: {chunk.document_type}")
    if chunk.tags:
        metadata_bits.append(f"Tags: {', '.join(chunk.tags)}")
    meta = " | ".join(metadata_bits)
    return f"{meta}\n\n{chunk.content}" if meta else chunk.content


def reciprocal_rank(top_chunks: Sequence[Chunk], case: BenchmarkCase) -> float:
    for rank, chunk in enumerate(top_chunks, start=1):
        if is_relevant_chunk(chunk, case):
            return 1.0 / rank
    return 0.0


def count_relevant(top_chunks: Sequence[Chunk], case: BenchmarkCase) -> int:
    return sum(1 for chunk in top_chunks if is_relevant_chunk(chunk, case))


def run_model_eval(
    model_key: str,
    chunks: List[Chunk],
    cases: List[BenchmarkCase],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"\n=== Evaluating model: {model_key} ===")
    embedder = build_embedder(model_key, args)

    chunk_texts = [format_chunk_for_embedding(c, args.include_metadata_text) for c in chunks]
    doc_mat = load_or_compute_doc_embeddings(
        model_key=model_key,
        embedder=embedder,
        chunk_texts=chunk_texts,
        chunks_path=Path(args.chunks_path),
        cache_dir=Path(args.output_dir) / "cache",
    )

    per_query_rows: List[Dict[str, Any]] = []

    hit_count = 0
    total_rr = 0.0
    total_relevant = 0

    for case in cases:
        query_text = build_query_text(case)
        q_vec = embedder.embed_query(query_text)
        sims = cosine_similarity_matrix(q_vec, doc_mat)
        top_idx = np.argsort(-sims)[: args.top_k]
        top_chunks = [chunks[i] for i in top_idx]
        top_scores = [float(sims[i]) for i in top_idx]

        hit = int(any(is_relevant_chunk(chunk, case) for chunk in top_chunks))
        rr = reciprocal_rank(top_chunks, case)
        n_rel = count_relevant(top_chunks, case)

        hit_count += hit
        total_rr += rr
        total_relevant += n_rel

        per_query_rows.append(
            {
                "model": model_key,
                "case_id": case.case_id,
                "gr_id": case.gr_id,
                "patient_id": case.patient_id,
                "query": case.query,
                "hit_at_k": hit,
                "rr_at_k": round(rr, 6),
                "relevant_count_at_k": n_rel,
                "top_chunk_ids": " | ".join(chunk.chunk_id for chunk in top_chunks),
                "top_sources": " | ".join(chunk.source_file for chunk in top_chunks),
                "top_sections": " | ".join(chunk.section_title for chunk in top_chunks),
                "top_scores": " | ".join(f"{s:.4f}" for s in top_scores),
            }
        )

    n_cases = len(cases)
    summary = {
        "model": model_key,
        "n_cases": n_cases,
        "top_k": args.top_k,
        "hit_at_k": round(hit_count / n_cases, 4),
        "mrr_at_k": round(total_rr / n_cases, 4),
        "avg_relevant_at_k": round(total_relevant / n_cases, 4),
    }
    return per_query_rows, summary


# -----------------------------
# Writing results
# -----------------------------
def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        raise ValueError(f"No rows to write to {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(summaries: List[Dict[str, Any]]) -> None:
    print("\n=== Summary ===")
    headers = ["model", "hit_at_k", "mrr_at_k", "avg_relevant_at_k", "n_cases", "top_k"]
    widths = {h: max(len(h), max(len(str(r[h])) for r in summaries)) for h in headers}
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep = "-+-".join("-" * widths[h] for h in headers)
    print(header_line)
    print(sep)
    for row in sorted(summaries, key=lambda x: (-x["hit_at_k"], -x["mrr_at_k"], -x["avg_relevant_at_k"])):
        print(" | ".join(str(row[h]).ljust(widths[h]) for h in headers))


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-path", required=True, help="Path to processed_chunks.json")
    parser.add_argument("--benchmark-path", required=True, help="Path to retrieval_benchmark.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["vertex_text_embedding_005", "bge_base_en", "e5_base_v2", "all_minilm_l6_v2", "medcpt_biomedical"],
        choices=list(MODEL_REGISTRY.keys()),
        help="Embedding models to compare",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Retrieve top-k chunks")
    parser.add_argument("--output-dir", default="outputs/embedding_eval", help="Output directory")
    parser.add_argument(
        "--include-metadata-text",
        action="store_true",
        help="Append section/type/tags to chunk text before embedding",
    )
    parser.add_argument("--gcp-project", default=None, help="GCP project for Vertex embeddings")
    parser.add_argument("--gcp-location", default="us-central1", help="GCP location for Vertex embeddings")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = load_chunks(Path(args.chunks_path))
    cases = load_benchmark(Path(args.benchmark_path))

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for model_key in args.models:
        rows, summary = run_model_eval(model_key, chunks, cases, args)
        all_rows.extend(rows)
        summaries.append(summary)

        model_rows_path = out_dir / f"{model_key}_per_query.csv"
        write_csv(model_rows_path, rows)

    summary_path = out_dir / "summary.csv"
    write_csv(summary_path, summaries)
    print_summary_table(summaries)

    all_rows_path = out_dir / "all_models_per_query.csv"
    write_csv(all_rows_path, all_rows)

    print(f"\nSaved:")
    print(f"  - {summary_path}")
    print(f"  - {all_rows_path}")
    print(f"  - {out_dir}/<model>_per_query.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())