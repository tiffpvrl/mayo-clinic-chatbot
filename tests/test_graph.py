"""
tests/test_graph.py
Comprehensive unit and integration tests for the LangGraph orchestration layer,
including the judge guardrail, retry loop, and escalation path.

Coverage:
  - Routing functions (pure logic)
  - Every node function with mocked dependencies
  - Judge thresholds by risk tier
  - Retry loop: judge feedback re-injected into generation context
  - Escalation after max retries
  - retry_count reset between conversation turns
  - chat_history written only by finalize_node (not on retries)
  - Full graph invocation for each intent path

All external dependencies are stubbed via sys.modules before any project
code is imported — no GCP credentials or populated vector store required.

Run with:
    python -m pytest tests/test_graph.py -v
"""

from __future__ import annotations

import json
import operator
import sys
import unittest
from unittest.mock import MagicMock, patch

# ── Stub external packages before any project import ──────────────────────────
_STUB_MODULES = [
    "chromadb",
    "vertexai",
    "vertexai.generative_models",
    "google",
    "google.cloud",
    "google.cloud.bigquery",
    "google.cloud.aiplatform",
    "sentence_transformers",
    "openai",
]
for _mod in _STUB_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

# Stub src-level modules that run side-effectful code at import time
for _mod in [
    "src.risk.risk_model",
    "src.retrieval.chromadb_store",
    "src.retrieval.embedder",
    "src.patient_data.bigquery_client",
]:
    sys.modules.setdefault(_mod, MagicMock())

# ── Now safe to import project code ───────────────────────────────────────────
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.graph.graph import (  # noqa: E402
    _route_after_classify,
    _route_after_generate,
    _route_after_judge,
    _route_after_patient_data,
    build_graph,
)
from src.graph.nodes import (  # noqa: E402
    classify_query_node,
    escalate_node,
    fetch_patient_data_node,
    finalize_node,
    generate_response_node,
    judge_response_node,
    retrieve_rag_node,
    score_risk_node,
)
from src.graph.state import ChatState  # noqa: E402
from langgraph.graph import END  # noqa: E402


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    state: dict = {
        "query": "What can I eat before my colonoscopy?",
        "patient_id": "P001",
        "query_intent": None,
        "patient_record": None,
        "patient_context": "",
        "risk_tier": None,
        "risk_probability": None,
        "query_understanding": None,
        "query_where": None,
        "wants_research": False,
        "clinical_hits": [],
        "qa_hits": [],
        "conversation_hits": [],
        "clinical_context": "",
        "qa_context": "",
        "conversation_context": "",
        "combined_context": "",
        "response": "",
        "judge_score": None,
        "judge_reasoning": None,
        "retry_count": 0,
        "max_retries": 2,
        "escalated": False,
        "chat_history": [],
        "is_follow_up": False,
    }
    state.update(overrides)
    return state


def _llm_returning(text: str) -> MagicMock:
    m = MagicMock()
    m.return_value.generate_content.return_value.text = text
    return m


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Routing functions
# ══════════════════════════════════════════════════════════════════════════════

class TestRoutingFunctions(unittest.TestCase):

    # _route_after_classify ---------------------------------------------------

    def test_chitchat_skips_patient_data(self):
        self.assertEqual(_route_after_classify(_base_state(query_intent="chitchat")), "generate_response")

    def test_logistics_goes_to_patient_data(self):
        self.assertEqual(_route_after_classify(_base_state(query_intent="logistics")), "fetch_patient_data")

    def test_medical_goes_to_patient_data(self):
        self.assertEqual(_route_after_classify(_base_state(query_intent="medical")), "fetch_patient_data")

    def test_none_intent_defaults_to_patient_data(self):
        self.assertEqual(_route_after_classify(_base_state(query_intent=None)), "fetch_patient_data")

    # _route_after_patient_data -----------------------------------------------

    def test_logistics_goes_to_generate(self):
        self.assertEqual(_route_after_patient_data(_base_state(query_intent="logistics")), "generate_response")

    def test_medical_goes_to_score_risk(self):
        self.assertEqual(_route_after_patient_data(_base_state(query_intent="medical")), "score_risk")

    def test_non_logistics_defaults_to_score_risk(self):
        self.assertEqual(_route_after_patient_data(_base_state(query_intent="chitchat")), "score_risk")

    # _route_after_generate ---------------------------------------------------

    def test_medical_goes_to_judge(self):
        self.assertEqual(_route_after_generate(_base_state(query_intent="medical")), "judge_response")

    def test_chitchat_goes_to_finalize(self):
        self.assertEqual(_route_after_generate(_base_state(query_intent="chitchat")), "finalize")

    def test_logistics_goes_to_finalize(self):
        self.assertEqual(_route_after_generate(_base_state(query_intent="logistics")), "finalize")

    def test_none_intent_defaults_to_judge(self):
        self.assertEqual(_route_after_generate(_base_state(query_intent=None)), "judge_response")

    # _route_after_judge ------------------------------------------------------

    def test_score_above_threshold_goes_to_finalize(self):
        # Low risk threshold = 0.80; score 0.90 should pass
        s = _base_state(judge_score=0.90, risk_tier="Low", retry_count=0, max_retries=2)
        self.assertEqual(_route_after_judge(s), "finalize")

    def test_score_at_exactly_threshold_goes_to_finalize(self):
        s = _base_state(judge_score=0.85, risk_tier="Medium", retry_count=0, max_retries=2)
        self.assertEqual(_route_after_judge(s), "finalize")

    def test_score_below_threshold_with_retries_loops_back(self):
        # Medium threshold = 0.85; score 0.70; retry_count=1 < max_retries=2 → retry
        s = _base_state(judge_score=0.70, risk_tier="Medium", retry_count=1, max_retries=2)
        self.assertEqual(_route_after_judge(s), "generate_response")

    def test_score_below_threshold_exhausted_retries_escalates(self):
        # retry_count=2 == max_retries=2 → escalate
        s = _base_state(judge_score=0.60, risk_tier="High", retry_count=2, max_retries=2)
        self.assertEqual(_route_after_judge(s), "escalate")

    def test_high_risk_threshold_is_09(self):
        # High risk needs ≥ 0.90; score 0.89 should fail
        s = _base_state(judge_score=0.89, risk_tier="High", retry_count=0, max_retries=2)
        self.assertNotEqual(_route_after_judge(s), "finalize")

    def test_high_risk_threshold_09_just_passes(self):
        s = _base_state(judge_score=0.90, risk_tier="High", retry_count=0, max_retries=2)
        self.assertEqual(_route_after_judge(s), "finalize")

    def test_medium_risk_threshold_is_085(self):
        s = _base_state(judge_score=0.84, risk_tier="Medium", retry_count=2, max_retries=2)
        self.assertNotEqual(_route_after_judge(s), "finalize")

    def test_low_risk_threshold_is_08(self):
        s = _base_state(judge_score=0.80, risk_tier="Low", retry_count=0, max_retries=2)
        self.assertEqual(_route_after_judge(s), "finalize")

    def test_unknown_risk_uses_default_threshold(self):
        # Default is 0.85; score 0.84 should fail
        s = _base_state(judge_score=0.84, risk_tier=None, retry_count=2, max_retries=2)
        self.assertEqual(_route_after_judge(s), "escalate")

    def test_missing_judge_score_treated_as_zero(self):
        # None score → 0.0 → below any threshold → check retries
        s = _base_state(judge_score=None, risk_tier="Low", retry_count=2, max_retries=2)
        self.assertEqual(_route_after_judge(s), "escalate")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  classify_query_node
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyQueryNode(unittest.TestCase):

    def _run(self, llm_text, **overrides):
        with patch("src.graph.nodes.GenerativeModel", _llm_returning(llm_text)):
            return classify_query_node(_base_state(**overrides))

    def test_classifies_medical(self):
        self.assertEqual(self._run(json.dumps({"intent": "medical"}))["query_intent"], "medical")

    def test_classifies_logistics(self):
        self.assertEqual(self._run(json.dumps({"intent": "logistics"}))["query_intent"], "logistics")

    def test_classifies_chitchat(self):
        self.assertEqual(self._run(json.dumps({"intent": "chitchat"}))["query_intent"], "chitchat")

    def test_invalid_intent_defaults_to_medical(self):
        self.assertEqual(self._run(json.dumps({"intent": "gibberish"}))["query_intent"], "medical")

    def test_llm_error_defaults_to_medical(self):
        m = MagicMock()
        m.return_value.generate_content.side_effect = Exception("503")
        with patch("src.graph.nodes.GenerativeModel", m):
            result = classify_query_node(_base_state())
        self.assertEqual(result["query_intent"], "medical")

    def test_resets_retry_count_to_zero(self):
        result = self._run(json.dumps({"intent": "medical"}), retry_count=99)
        self.assertEqual(result["retry_count"], 0)

    def test_resets_escalated_to_false(self):
        result = self._run(json.dumps({"intent": "medical"}), escalated=True)
        self.assertFalse(result["escalated"])

    def test_is_follow_up_false_on_first_turn(self):
        result = self._run(json.dumps({"intent": "medical"}), chat_history=[])
        self.assertFalse(result["is_follow_up"])

    def test_is_follow_up_true_with_prior_history(self):
        history = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
        result = self._run(json.dumps({"intent": "medical"}), chat_history=history)
        self.assertTrue(result["is_follow_up"])


# ══════════════════════════════════════════════════════════════════════════════
# 3.  fetch_patient_data_node
# ══════════════════════════════════════════════════════════════════════════════

class TestFetchPatientDataNode(unittest.TestCase):

    def test_returns_record_and_context(self):
        with (
            patch("src.graph.nodes.get_patient_record", return_value={"patient_id": "P1"}),
            patch("src.graph.nodes.build_patient_context", return_value="CTX"),
        ):
            r = fetch_patient_data_node(_base_state())
        self.assertEqual(r["patient_record"], {"patient_id": "P1"})
        self.assertEqual(r["patient_context"], "CTX")

    def test_patient_not_found_skips_context_builder(self):
        with (
            patch("src.graph.nodes.get_patient_record", return_value=None),
            patch("src.graph.nodes.build_patient_context") as mock_ctx,
        ):
            r = fetch_patient_data_node(_base_state())
        self.assertIsNone(r["patient_record"])
        mock_ctx.assert_not_called()
        self.assertIn("No patient-specific data found", r["patient_context"])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  score_risk_node
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreRiskNode(unittest.TestCase):

    def test_capitalises_tier(self):
        with patch("src.graph.nodes.get_risk_score", return_value={"risk_tier": "high", "risk_probability": 0.91}):
            r = score_risk_node(_base_state(patient_record={"patient_id": "P1"}))
        self.assertEqual(r["risk_tier"], "High")

    def test_no_patient_record_returns_none(self):
        with patch("src.graph.nodes.get_risk_score") as m:
            r = score_risk_node(_base_state(patient_record=None))
        m.assert_not_called()
        self.assertIsNone(r["risk_tier"])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  retrieve_rag_node
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrieveRagNode(unittest.TestCase):

    def _run(self, **overrides):
        state = _base_state(
            patient_record={"patient_id": "P1"},
            patient_context="CTX",
            risk_tier="Medium",
            **overrides,
        )
        fake_und = {"is_diabetes_query": False, "wants_research": False}
        with (
            patch("src.graph.nodes.extract_query_understanding", return_value=fake_und),
            patch("src.graph.nodes.build_clinical_where", return_value=None),
            patch("src.graph.nodes.retrieve_clinical", return_value=[]) as rc,
            patch("src.graph.nodes.retrieve_qa", return_value=[]) as rq,
            patch("src.graph.nodes.retrieve_conversations", return_value=[]) as rconv,
            patch("src.graph.nodes.format_clinical_context", return_value="CLINICAL"),
            patch("src.graph.nodes.format_qa_context", return_value="QA"),
            patch("src.graph.nodes.format_conversation_context", return_value="CONV"),
        ):
            result = retrieve_rag_node(state)
            return result, {"rc": rc, "rq": rq, "rconv": rconv}

    def test_all_three_collections_retrieved(self):
        _, mocks = self._run()
        mocks["rc"].assert_called_once()
        mocks["rq"].assert_called_once()
        mocks["rconv"].assert_called_once()

    def test_combined_context_contains_all_sections(self):
        result, _ = self._run()
        ctx = result["combined_context"]
        for section in ("CTX", "CLINICAL", "QA", "CONV"):
            self.assertIn(section, ctx)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  generate_response_node
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateResponseNode(unittest.TestCase):

    def test_chitchat_uses_direct_llm(self):
        with (
            patch("src.graph.nodes.GenerativeModel", _llm_returning("Hello there!")),
            patch("src.graph.nodes.generate_response") as mock_gr,
        ):
            r = generate_response_node(_base_state(query_intent="chitchat"))
        mock_gr.assert_not_called()
        self.assertEqual(r["response"], "Hello there!")

    def test_logistics_passes_patient_context_only(self):
        captured = {}
        def _cap(query, context):
            captured["context"] = context
            return "9 AM"
        with patch("src.graph.nodes.generate_response", side_effect=_cap):
            generate_response_node(_base_state(
                query_intent="logistics",
                patient_context="PATIENT INFO",
                combined_context="FULL CTX WITH CLINICAL",
            ))
        self.assertIn("PATIENT INFO", captured["context"])
        self.assertNotIn("FULL CTX WITH CLINICAL", captured["context"])

    def test_medical_uses_combined_context(self):
        captured = {}
        def _cap(query, context):
            captured["context"] = context
            return "Take clear liquids."
        with patch("src.graph.nodes.generate_response", side_effect=_cap):
            generate_response_node(_base_state(
                query_intent="medical",
                combined_context="FULL CTX",
            ))
        self.assertIn("FULL CTX", captured["context"])

    def test_does_not_write_to_chat_history(self):
        with patch("src.graph.nodes.generate_response", return_value="answer"):
            r = generate_response_node(_base_state(query_intent="medical", combined_context="CTX"))
        self.assertNotIn("chat_history", r)

    def test_retry_prepends_judge_feedback_to_context(self):
        captured = {}
        def _cap(query, context):
            captured["context"] = context
            return "revised answer"
        with patch("src.graph.nodes.generate_response", side_effect=_cap):
            generate_response_node(_base_state(
                query_intent="medical",
                combined_context="BASE CTX",
                retry_count=1,
                judge_score=0.72,
                judge_reasoning="Response cited a drug not in the evidence.",
                risk_tier="Medium",
            ))
        ctx = captured["context"]
        self.assertIn("REVISION REQUEST", ctx)
        self.assertIn("0.72", ctx)
        self.assertIn("Response cited a drug not in the evidence.", ctx)
        self.assertIn("BASE CTX", ctx)

    def test_first_attempt_has_no_revision_prefix(self):
        captured = {}
        def _cap(query, context):
            captured["context"] = context
            return "original answer"
        with patch("src.graph.nodes.generate_response", side_effect=_cap):
            generate_response_node(_base_state(
                query_intent="medical",
                combined_context="BASE CTX",
                retry_count=0,
            ))
        self.assertNotIn("REVISION REQUEST", captured["context"])


# ══════════════════════════════════════════════════════════════════════════════
# 7.  judge_response_node
# ══════════════════════════════════════════════════════════════════════════════

class TestJudgeResponseNode(unittest.TestCase):

    def _run(self, llm_text, **overrides):
        base = {
            "response": "Take clear liquids.",
            "combined_context": "CLINICAL CTX",
            "risk_tier": "Medium",
            "retry_count": 0,
        }
        base.update(overrides)
        state = _base_state(**base)
        with patch("src.graph.nodes.GenerativeModel", _llm_returning(llm_text)):
            return judge_response_node(state)

    def test_high_score_does_not_increment_retry_count(self):
        r = self._run(json.dumps({"score": 0.95, "reasoning": "excellent"}))
        self.assertEqual(r.get("judge_score"), 0.95)
        self.assertNotIn("retry_count", r)  # not incremented on pass

    def test_low_score_increments_retry_count(self):
        r = self._run(json.dumps({"score": 0.50, "reasoning": "hallucinated"}))
        self.assertEqual(r["retry_count"], 1)

    def test_low_score_second_failure_increments_to_two(self):
        r = self._run(
            json.dumps({"score": 0.50, "reasoning": "still hallucinated"}),
            retry_count=1,
        )
        self.assertEqual(r["retry_count"], 2)

    def test_score_clamped_above_one(self):
        r = self._run(json.dumps({"score": 1.5, "reasoning": "ok"}))
        self.assertLessEqual(r["judge_score"], 1.0)

    def test_score_clamped_below_zero(self):
        r = self._run(json.dumps({"score": -0.3, "reasoning": "bad"}))
        self.assertGreaterEqual(r["judge_score"], 0.0)

    def test_fails_open_on_llm_error(self):
        m = MagicMock()
        m.return_value.generate_content.side_effect = Exception("timeout")
        state = _base_state(response="answer", combined_context="ctx", risk_tier="High")
        with patch("src.graph.nodes.GenerativeModel", m):
            r = judge_response_node(state)
        self.assertEqual(r["judge_score"], 1.0)  # fail open
        self.assertIn("Judge skipped", r["judge_reasoning"])

    def test_reasoning_stored_in_state(self):
        r = self._run(json.dumps({"score": 0.60, "reasoning": "Missing drug safety info."}))
        self.assertEqual(r["judge_reasoning"], "Missing drug safety info.")

    def test_markdown_json_parsed(self):
        r = self._run('```json\n{"score": 0.88, "reasoning": "good"}\n```')
        self.assertAlmostEqual(r["judge_score"], 0.88)

    def test_high_risk_uses_09_threshold_in_prompt(self):
        """Verify the prompt is built with the correct threshold for the tier."""
        captured_prompt = {}
        def fake_generate(prompt, **_):
            captured_prompt["text"] = prompt
            return MagicMock(text=json.dumps({"score": 0.91, "reasoning": "ok"}))

        m = MagicMock()
        m.return_value.generate_content.side_effect = fake_generate
        state = _base_state(
            response="answer",
            combined_context="ctx",
            risk_tier="High",
        )
        with patch("src.graph.nodes.GenerativeModel", m):
            judge_response_node(state)
        self.assertIn("0.90", captured_prompt["text"])


# ══════════════════════════════════════════════════════════════════════════════
# 8.  escalate_node
# ══════════════════════════════════════════════════════════════════════════════

class TestEscalateNode(unittest.TestCase):

    def test_sets_escalated_flag(self):
        r = escalate_node(_base_state(risk_tier="High", judge_score=0.55, retry_count=2))
        self.assertTrue(r["escalated"])

    def test_response_is_safe_escalation_message(self):
        r = escalate_node(_base_state())
        self.assertIn("care team", r["response"].lower())
        self.assertIn("follow up", r["response"].lower())

    def test_does_not_touch_judge_score(self):
        r = escalate_node(_base_state())
        self.assertNotIn("judge_score", r)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  finalize_node
# ══════════════════════════════════════════════════════════════════════════════

class TestFinalizeNode(unittest.TestCase):

    def test_appends_user_and_assistant_to_history(self):
        state = _base_state(query="Can I eat?", response="Take clear liquids.")
        r = finalize_node(state)
        msgs = r["chat_history"]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0], {"role": "user", "content": "Can I eat?"})
        self.assertEqual(msgs[1], {"role": "assistant", "content": "Take clear liquids."})

    def test_escalated_response_also_written_to_history(self):
        state = _base_state(query="Q?", response="Care team notified.", escalated=True)
        r = finalize_node(state)
        self.assertEqual(r["chat_history"][1]["content"], "Care team notified.")

    def test_does_not_overwrite_previous_history(self):
        """operator.add reducer accumulates; finalize only returns the NEW messages."""
        state = _base_state(query="Q?", response="A.")
        r = finalize_node(state)
        # finalize returns only the 2 new messages; LangGraph's reducer appends
        self.assertEqual(len(r["chat_history"]), 2)


# ══════════════════════════════════════════════════════════════════════════════
# 10. chat_history reducer
# ══════════════════════════════════════════════════════════════════════════════

class TestChatHistoryReducer(unittest.TestCase):

    def test_operator_add_appends(self):
        existing = [{"role": "user", "content": "Turn 1"}]
        update   = [{"role": "user", "content": "Turn 2"}, {"role": "assistant", "content": "Ans 2"}]
        self.assertEqual(operator.add(existing, update), existing + update)

    def test_empty_existing(self):
        self.assertEqual(operator.add([], [{"role": "user", "content": "Hi"}]),
                         [{"role": "user", "content": "Hi"}])

    def test_empty_update(self):
        existing = [{"role": "assistant", "content": "Hello"}]
        self.assertEqual(operator.add(existing, []), existing)


# ══════════════════════════════════════════════════════════════════════════════
# 11. Graph topology
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphTopology(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.graph = build_graph()

    def test_compiles_without_error(self):
        self.assertIsNotNone(self.graph)

    def test_all_nodes_present(self):
        expected = {
            "classify_query", "fetch_patient_data", "score_risk",
            "retrieve_rag", "generate_response", "judge_response",
            "escalate", "finalize",
        }
        self.assertTrue(expected.issubset(set(self.graph.nodes.keys())),
                        f"Missing: {expected - set(self.graph.nodes.keys())}")

    def test_checkpointer_attached(self):
        self.assertIsNotNone(self.graph.checkpointer)


# ══════════════════════════════════════════════════════════════════════════════
# 12. Full graph invocations
# ══════════════════════════════════════════════════════════════════════════════

class TestGraphFullInvocation(unittest.TestCase):
    """End-to-end graph.invoke() with all externals mocked."""

    def _common_patches(self, intent, judge_score=0.95):
        """Context manager stack covering all external calls."""
        fake_record = {"patient_id": "P001"}
        fake_und = {"is_diabetes_query": False, "wants_research": False}
        llm = MagicMock()
        llm.return_value.generate_content.return_value.text = json.dumps({
            "intent": intent,
            "score": judge_score,
            "reasoning": "ok",
        })
        return (
            patch("src.graph.nodes.GenerativeModel", llm),
            patch("src.graph.nodes.get_patient_record", return_value=fake_record),
            patch("src.graph.nodes.build_patient_context", return_value="CTX"),
            patch("src.graph.nodes.get_risk_score", return_value={"risk_tier": "low", "risk_probability": 0.1}),
            patch("src.graph.nodes.extract_query_understanding", return_value=fake_und),
            patch("src.graph.nodes.build_clinical_where", return_value=None),
            patch("src.graph.nodes.retrieve_clinical", return_value=[]),
            patch("src.graph.nodes.retrieve_qa", return_value=[]),
            patch("src.graph.nodes.retrieve_conversations", return_value=[]),
            patch("src.graph.nodes.format_clinical_context", return_value=""),
            patch("src.graph.nodes.format_qa_context", return_value=""),
            patch("src.graph.nodes.format_conversation_context", return_value=""),
            patch("src.graph.nodes.generate_response", return_value="Test answer."),
        )

    # ── Medical: passing judge ────────────────────────────────────────────────

    def test_medical_passing_judge_delivers_response(self):
        g = build_graph()
        config = {"configurable": {"thread_id": "med-pass"}}
        patches = self._common_patches("medical", judge_score=0.95)
        with patches[0], patches[1], patches[2], patches[3], patches[4], \
             patches[5], patches[6], patches[7], patches[8], patches[9], \
             patches[10], patches[11], patches[12]:
            r = g.invoke(
                {"query": "Can I take metformin?", "patient_id": "P001",
                 "chat_history": [], "retry_count": 0, "max_retries": 2, "escalated": False},
                config=config,
            )
        self.assertEqual(r["query_intent"], "medical")
        self.assertEqual(r["response"], "Test answer.")
        self.assertAlmostEqual(r["judge_score"], 0.95, places=2)
        self.assertFalse(r.get("escalated", False))
        # History written by finalize_node
        self.assertEqual(len(r["chat_history"]), 2)

    # ── Medical: retry then pass ──────────────────────────────────────────────

    def test_medical_retry_loop_then_pass(self):
        """First judge call fails; second passes after retry."""
        g = build_graph()
        config = {"configurable": {"thread_id": "med-retry-pass"}}

        llm = MagicMock()
        # generate_content is called three times:
        #   1. classify_query_node  → must return intent
        #   2. judge_response_node (attempt 1) → low score, triggers retry
        #   3. judge_response_node (attempt 2) → high score, passes
        llm.return_value.generate_content.side_effect = [
            MagicMock(text=json.dumps({"intent": "medical"})),
            MagicMock(text=json.dumps({"score": 0.60, "reasoning": "too vague"})),
            MagicMock(text=json.dumps({"score": 0.92, "reasoning": "good"})),
        ]

        with (
            patch("src.graph.nodes.GenerativeModel", llm),
            patch("src.graph.nodes.get_patient_record", return_value={"patient_id": "P001"}),
            patch("src.graph.nodes.build_patient_context", return_value="CTX"),
            patch("src.graph.nodes.get_risk_score", return_value={"risk_tier": "low", "risk_probability": 0.1}),
            patch("src.graph.nodes.extract_query_understanding", return_value={"is_diabetes_query": False, "wants_research": False}),
            patch("src.graph.nodes.build_clinical_where", return_value=None),
            patch("src.graph.nodes.retrieve_clinical", return_value=[]),
            patch("src.graph.nodes.retrieve_qa", return_value=[]),
            patch("src.graph.nodes.retrieve_conversations", return_value=[]),
            patch("src.graph.nodes.format_clinical_context", return_value=""),
            patch("src.graph.nodes.format_qa_context", return_value=""),
            patch("src.graph.nodes.format_conversation_context", return_value=""),
            patch("src.graph.nodes.generate_response", return_value="Revised answer."),
        ):
            r = g.invoke(
                {"query": "What prep do I use?", "patient_id": "P001",
                 "chat_history": [], "retry_count": 0, "max_retries": 2, "escalated": False},
                config=config,
            )

        self.assertFalse(r.get("escalated", False))
        self.assertEqual(r["response"], "Revised answer.")
        # retry_count was 1 after the first failed judge, then not incremented on pass
        self.assertEqual(r.get("retry_count", 0), 1)

    # ── Medical: all retries exhausted → escalation ───────────────────────────

    def test_medical_all_retries_exhausted_escalates(self):
        g = build_graph()
        config = {"configurable": {"thread_id": "med-escalate"}}

        llm = MagicMock()
        low_score_response = MagicMock(text=json.dumps({"score": 0.40, "reasoning": "still bad"}))
        # classify + 3 judge evaluations (all fail)
        llm.return_value.generate_content.side_effect = [low_score_response] * 10

        with (
            patch("src.graph.nodes.GenerativeModel", llm),
            patch("src.graph.nodes.get_patient_record", return_value={"patient_id": "P001"}),
            patch("src.graph.nodes.build_patient_context", return_value="CTX"),
            patch("src.graph.nodes.get_risk_score", return_value={"risk_tier": "low", "risk_probability": 0.1}),
            patch("src.graph.nodes.extract_query_understanding", return_value={"is_diabetes_query": False, "wants_research": False}),
            patch("src.graph.nodes.build_clinical_where", return_value=None),
            patch("src.graph.nodes.retrieve_clinical", return_value=[]),
            patch("src.graph.nodes.retrieve_qa", return_value=[]),
            patch("src.graph.nodes.retrieve_conversations", return_value=[]),
            patch("src.graph.nodes.format_clinical_context", return_value=""),
            patch("src.graph.nodes.format_qa_context", return_value=""),
            patch("src.graph.nodes.format_conversation_context", return_value=""),
            patch("src.graph.nodes.generate_response", return_value="Bad answer."),
        ):
            r = g.invoke(
                {"query": "What about my prep?", "patient_id": "P001",
                 "chat_history": [], "retry_count": 0, "max_retries": 2, "escalated": False},
                config=config,
            )

        self.assertTrue(r["escalated"])
        self.assertIn("care team", r["response"].lower())
        # History still written even on escalation
        self.assertEqual(len(r["chat_history"]), 2)

    # ── Logistics: judge skipped ──────────────────────────────────────────────

    def test_logistics_judge_not_called(self):
        g = build_graph()
        config = {"configurable": {"thread_id": "log-test"}}

        with (
            patch("src.graph.nodes.GenerativeModel") as llm,
            patch("src.graph.nodes.get_patient_record", return_value={"patient_id": "P001"}),
            patch("src.graph.nodes.build_patient_context", return_value="CTX"),
            patch("src.graph.nodes.get_risk_score") as mock_risk,
            patch("src.graph.nodes.retrieve_clinical") as mock_rc,
            patch("src.graph.nodes.generate_response", return_value="Your appointment is at 9 AM."),
        ):
            llm.return_value.generate_content.return_value.text = json.dumps({"intent": "logistics"})
            r = g.invoke(
                {"query": "What time is my procedure?", "patient_id": "P001",
                 "chat_history": [], "retry_count": 0, "max_retries": 2, "escalated": False},
                config=config,
            )
            mock_risk.assert_not_called()
            mock_rc.assert_not_called()

        self.assertIsNone(r.get("judge_score"))
        self.assertFalse(r.get("escalated", False))
        self.assertEqual(len(r["chat_history"]), 2)

    # ── Chitchat: everything skipped ─────────────────────────────────────────

    def test_chitchat_skips_all_except_generate_and_finalize(self):
        g = build_graph()
        config = {"configurable": {"thread_id": "chitchat-test"}}

        with (
            patch("src.graph.nodes.GenerativeModel") as llm,
            patch("src.graph.nodes.get_patient_record") as mock_bq,
            patch("src.graph.nodes.get_risk_score") as mock_risk,
            patch("src.graph.nodes.generate_response") as mock_gr,
        ):
            responses = [
                MagicMock(text=json.dumps({"intent": "chitchat"})),
                MagicMock(text="You're welcome!"),
            ]
            llm.return_value.generate_content.side_effect = responses
            r = g.invoke(
                {"query": "Thanks!", "patient_id": "P001",
                 "chat_history": [], "retry_count": 0, "max_retries": 2, "escalated": False},
                config=config,
            )
            mock_bq.assert_not_called()
            mock_risk.assert_not_called()
            mock_gr.assert_not_called()

        self.assertEqual(r["query_intent"], "chitchat")
        self.assertIsNone(r.get("judge_score"))
        self.assertEqual(len(r["chat_history"]), 2)

    # ── Multi-turn: retry_count resets between turns ──────────────────────────

    def test_retry_count_resets_on_new_turn(self):
        """A retry_count from turn N must not affect routing on turn N+1."""
        g = build_graph()
        config = {"configurable": {"thread_id": "retry-reset"}}

        # Turn 1: two failed judges then escalation
        llm = MagicMock()
        fail_resp = MagicMock(text=json.dumps({"score": 0.30, "reasoning": "bad"}))
        pass_resp = MagicMock(text=json.dumps({"score": 0.95, "reasoning": "ok"}))
        llm.return_value.generate_content.side_effect = [fail_resp] * 10

        base_patches = dict(
            get_patient_record={"return_value": {"patient_id": "P001"}},
            build_patient_context={"return_value": "CTX"},
            get_risk_score={"return_value": {"risk_tier": "low", "risk_probability": 0.1}},
            extract_query_understanding={"return_value": {"is_diabetes_query": False, "wants_research": False}},
            build_clinical_where={"return_value": None},
            retrieve_clinical={"return_value": []},
            retrieve_qa={"return_value": []},
            retrieve_conversations={"return_value": []},
            format_clinical_context={"return_value": ""},
            format_qa_context={"return_value": ""},
            format_conversation_context={"return_value": ""},
            generate_response={"return_value": "answer"},
        )

        with patch("src.graph.nodes.GenerativeModel", llm), \
             patch("src.graph.nodes.get_patient_record", **base_patches["get_patient_record"]), \
             patch("src.graph.nodes.build_patient_context", **base_patches["build_patient_context"]), \
             patch("src.graph.nodes.get_risk_score", **base_patches["get_risk_score"]), \
             patch("src.graph.nodes.extract_query_understanding", **base_patches["extract_query_understanding"]), \
             patch("src.graph.nodes.build_clinical_where", **base_patches["build_clinical_where"]), \
             patch("src.graph.nodes.retrieve_clinical", **base_patches["retrieve_clinical"]), \
             patch("src.graph.nodes.retrieve_qa", **base_patches["retrieve_qa"]), \
             patch("src.graph.nodes.retrieve_conversations", **base_patches["retrieve_conversations"]), \
             patch("src.graph.nodes.format_clinical_context", **base_patches["format_clinical_context"]), \
             patch("src.graph.nodes.format_qa_context", **base_patches["format_qa_context"]), \
             patch("src.graph.nodes.format_conversation_context", **base_patches["format_conversation_context"]), \
             patch("src.graph.nodes.generate_response", **base_patches["generate_response"]):

            # Turn 1 — escalates
            g.invoke(
                {"query": "Q1", "patient_id": "P001",
                 "chat_history": [], "retry_count": 0, "max_retries": 2, "escalated": False},
                config=config,
            )

            # Turn 2 — judge passes; retry_count must have been reset to 0 by classify
            llm.return_value.generate_content.side_effect = [pass_resp] * 10
            r2 = g.invoke(
                {"query": "Q2", "patient_id": "P001",
                 "chat_history": [], "retry_count": 0, "max_retries": 2, "escalated": False},
                config=config,
            )

        self.assertFalse(r2["escalated"])
        self.assertAlmostEqual(r2["judge_score"], 0.95, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
