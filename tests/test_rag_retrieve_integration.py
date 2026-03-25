"""Integration-style tests for retrieve() with mocked Chroma (research post-filter).

Skipped when numpy/chromadb/etc. are not installed (e.g. minimal CI); run in project venv.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def _install_lightweight_import_stubs() -> None:
    """Stub optional deps before importing rag (avoids HF download in tests)."""
    if "sentence_transformers" not in sys.modules:
        fake = types.ModuleType("sentence_transformers")
        mock_arr = MagicMock()
        mock_arr.tolist.return_value = [[0.0] * 384]
        mock_model = MagicMock()
        mock_model.encode.return_value = mock_arr
        fake.SentenceTransformer = MagicMock(return_value=mock_model)
        sys.modules["sentence_transformers"] = fake

    if "openai" not in sys.modules:
        o = types.ModuleType("openai")
        o.OpenAI = MagicMock
        sys.modules["openai"] = o

    if "chromadb" not in sys.modules:
        c = types.ModuleType("chromadb")
        mock_client = MagicMock()
        mock_coll = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_coll
        c.PersistentClient = MagicMock(return_value=mock_client)
        sys.modules["chromadb"] = c

    if "google.cloud.bigquery" not in sys.modules:
        bq = types.ModuleType("google.cloud.bigquery")
        bq.Client = MagicMock(return_value=MagicMock())
        cloud = types.ModuleType("google.cloud")
        cloud.bigquery = bq
        google = types.ModuleType("google")
        google.cloud = cloud
        sys.modules["google.cloud.bigquery"] = bq
        sys.modules["google.cloud"] = cloud
        sys.modules["google"] = google


rag = None
_install_lightweight_import_stubs()
if importlib.util.find_spec("numpy") is not None:
    try:
        import src.retrieval.rag as rag  # noqa: E402
    except ImportError:
        rag = None


@unittest.skipIf(rag is None, "requires numpy + rag import chain (use project venv)")
class TestRetrieveResearchFilter(unittest.TestCase):
    def _make_mock_collection(self, mock_results: dict) -> MagicMock:
        mock_collection = MagicMock()
        mock_collection.query.return_value = mock_results
        return mock_collection

    def test_drops_research_chunks_when_not_evidence_query(self) -> None:
        assert rag is not None
        mock_results = {
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [
                [
                    {"audience_tier": "research_education", "tags": ""},
                    {"audience_tier": "clinician_guideline", "tags": "diet"},
                    {"audience_tier": "", "tags": "foo,content_policy:research_background,bar"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
            "ids": [["id1", "id2", "id3"]],
        }
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.0] * 8]
        mock_collection = self._make_mock_collection(mock_results)

        with patch.object(rag, "embedder", mock_embedder), patch.object(rag, "clinical_collection", mock_collection):
            hits = rag.retrieve_clinical("When should I stop clear liquids?", top_k=2)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], "id2")

    def test_keeps_research_when_evidence_query(self) -> None:
        assert rag is not None
        mock_results = {
            "documents": [["d1", "d2"]],
            "metadatas": [[{"audience_tier": "research_education", "tags": ""}, {"audience_tier": "", "tags": ""}]],
            "distances": [[0.1, 0.2]],
            "ids": [["r1", "r2"]],
        }
        mock_embedder = MagicMock()
        mock_embedder.encode.return_value = [[0.0] * 8]
        mock_collection = self._make_mock_collection(mock_results)

        with patch.object(rag, "embedder", mock_embedder), patch.object(rag, "clinical_collection", mock_collection):
            hits = rag.retrieve_clinical("What does the meta-analysis say about bowel prep?", top_k=2)

        self.assertEqual(len(hits), 2)


if __name__ == "__main__":
    unittest.main()
