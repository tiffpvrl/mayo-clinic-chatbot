# MayoChat — Colonoscopy Preparation Assistant

A patient-facing AI chatbot that delivers personalized colonoscopy preparation guidance. Built on a LangGraph stateful pipeline with BigQuery EHR integration, ChromaDB RAG, a joblib risk model, and a Gemini LLM judge guardrail.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [LangGraph Pipeline — Deep Dive](#langgraph-pipeline--deep-dive)
4. [Judge Guardrail](#judge-guardrail)
5. [Project Structure](#project-structure)
6. [Setup](#setup)
7. [Running the Server](#running-the-server)
8. [API Reference](#api-reference)
9. [Running Tests](#running-tests)

---

## Overview

MayoChat answers patient questions about their upcoming colonoscopy procedure. Every response is:

- **Personalized** — pulls the patient's EHR record from BigQuery (comorbidities, prep regimen, prior history) and adapts the answer accordingly.
- **Evidence-grounded** — retrieves relevant clinical guidelines, patient Q&A examples, and conversation flows from a ChromaDB vector store before generating a response.
- **Risk-aware** — a joblib classifier predicts the patient's risk of inadequate bowel prep. Higher-risk patients get more directive language and face a stricter quality bar before a response is delivered.
- **Quality-controlled** — a second Gemini LLM call scores every medical response on a 0–1 confidence scale. Responses that fall below the risk-tier threshold are regenerated with feedback, and escalated to the care team if retries are exhausted.

---

## Architecture

```
Patient HTTP Request
        │
        ▼
  FastAPI /chat
        │
        ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                    LangGraph StateGraph                     │
  │                                                             │
  │  START → classify_query                                     │
  │               │                                            │
  │     ┌─────────┼─────────┐                                  │
  │  chitchat  logistics  medical                               │
  │     │          │        │                                   │
  │     │    fetch_patient_data                                 │
  │     │          │        │                                   │
  │     │      (logistics) (medical)                            │
  │     │          │     score_risk                             │
  │     │          │        │                                   │
  │     │          │   retrieve_rag                             │
  │     │          │   (clinical + Q&A + conversations)         │
  │     │          │        │                                   │
  │     └──────────┴────────┘                                   │
  │                    generate_response  ◄──── (retry loop)    │
  │                          │                       │          │
  │              (medical)   │   (chitchat/logistics)│          │
  │                          │        │              │          │
  │                   judge_response  finalize       │          │
  │                          │                       │          │
  │             ┌────────────┼────────────┐          │          │
  │          pass         retry      exhausted       │          │
  │             │            └───────────────────────┘          │
  │             │                    escalate                   │
  │             └───────────► finalize ◄───────────────────────┘│
  │                               │                             │
  └───────────────────────────────┼─────────────────────────────┘
                                  │
                              HTTP Response
```

**Key design decisions:**

- `finalize_node` is the **sole writer** to `chat_history`. Retry intermediates never pollute the conversation log.
- `MemorySaver` checkpoints the full `ChatState` per `thread_id` (= `patient_id`), giving multi-turn memory across HTTP requests with no external session store.
- Chitchat skips BigQuery and RAG entirely. Logistics skips risk scoring and RAG. Only medical queries run the full pipeline.

---

## LangGraph Pipeline — Deep Dive

### State — `src/graph/state.py`

All nodes share a single `ChatState` TypedDict. LangGraph merges partial dicts returned by each node back into state.

```python
class ChatState(TypedDict):
    # Input
    query: str
    patient_id: str

    # Classification
    query_intent: str           # "medical" | "logistics" | "chitchat"
    is_follow_up: bool

    # Patient data
    patient_record: dict | None
    patient_context: str

    # Risk model
    risk_tier: str | None       # "Low" | "Medium" | "High"
    risk_probability: float | None

    # RAG retrieval
    query_understanding: dict
    combined_context: str
    clinical_hits: list
    qa_hits: list
    conversation_hits: list

    # Generation + judge
    response: str
    judge_score: float | None   # 0.0 – 1.0
    judge_reasoning: str | None
    retry_count: int
    max_retries: int
    escalated: bool

    # Multi-turn memory (operator.add reducer — appends, never overwrites)
    chat_history: List[Dict[str, str]]
```

`chat_history` uses `Annotated[List[...], operator.add]` so each finalize call appends two messages (user + assistant) rather than replacing the list. This is what makes multi-turn memory work across requests.

---

### Node 1 — `classify_query_node`

A lightweight Gemini call classifies the patient's message into one of three intents:

| Intent | Meaning | Pipeline path |
|---|---|---|
| `medical` | Medications, symptoms, prep instructions, side effects, clinical questions | Full pipeline (BigQuery → Risk → RAG → Generate → Judge) |
| `logistics` | Appointment timing, location, what to bring, check-in, parking | BigQuery → Generate (no RAG, no judge) |
| `chitchat` | Greetings, thanks, small talk | Generate only (no BigQuery, no RAG, no judge) |

Also resets `retry_count = 0` and `escalated = False` at the start of every new user turn so the judge loop from a previous turn doesn't bleed over.

---

### Node 2 — `fetch_patient_data_node`

Fetches the patient's full EHR record from BigQuery (`pre_procedure_data` dataset) via a parameterized query joining five tables:

- `Patients` — demographics, BMI, smoking, mobility, high-risk flag
- `Comorbidities` — ICD-10 codes, medications, diabetes, CKD, IBD, opioid use, GLP-1 agonists, etc.
- `Encounters` — procedure ID, colonoscopy datetime, indication, chief complaint
- `Prep_Details` — regimen type, prep agent, volume, adjuncts, diet protocol, symptom scores
- `Prior_Colonoscopy_History` — prior BBPS scores, prior prep adequacy, complication history

The result is passed to `build_patient_context()` which formats it into a structured string used in prompts.

> **Auth note:** The BigQuery client is created at module import time with `with_quota_project(None)` to strip the `x-goog-user-project` header. This prevents a `serviceusage.services.use` permission check that fails even for project owners in some GCP org configurations, and ensures the credentials are frozen before `vertexai.init()` can attach quota project metadata to the global auth state.

---

### Node 3 — `score_risk_node`

Runs a pre-trained joblib classifier (`src/risk/model_assets/`) on the patient record to predict the probability of **inadequate bowel prep**. Returns:

- `risk_tier`: `"Low"` / `"Medium"` / `"High"`
- `risk_probability`: float (0.0 – 1.0)

The tier is stored in state and used by both:
- `retrieve_rag_node` — to filter Q&A and conversation chunks by risk tier
- `judge_response_node` — to select the confidence threshold

---

### Node 4 — `retrieve_rag_node`

Runs the three-collection RAG pipeline against ChromaDB. A single `extract_query_understanding()` LLM call extracts structured intent (medication class, drug name, document type, procedure timing, whether the patient wants research) and is reused across all three retrievers.

| Collection | What it contains | Filter strategy |
|---|---|---|
| `clinical` | Guidelines, drug labels, patient instructions | Document-type filter + patient-specific tags (prep agent, procedure time, indication) |
| `qa` | Curated patient Q&A pairs | Risk tier + is_follow_up |
| `conversations` | Example dialogue flows | Risk tier + is_follow_up |

All retrieved chunks are formatted and combined into `combined_context`:

```
PATIENT-SPECIFIC CONTEXT
{patient_context}

CLINICAL KNOWLEDGE BASE
{clinical_context}

SIMILAR Q&A EXAMPLES
{qa_context}

SIMILAR CONVERSATION FLOWS
{conversation_context}
```

---

### Node 5 — `generate_response_node`

Calls `generate_response()` (in `src/llm/`) with context selected by intent:

- **chitchat** — direct Gemini call with no context, 1-2 sentence warm reply
- **logistics** — patient context only (no clinical RAG)
- **medical** — full `combined_context`

On retries (`retry_count > 0`), the judge's score and reasoning from the previous attempt are prepended as a `[REVISION REQUEST]` block so the model knows exactly what to fix:

```
[REVISION REQUEST — attempt 2]
Previous response scored 0.71 (required ≥ 0.90).
Judge feedback: Response mentioned ondansetron without it appearing in retrieved context.
Please revise the response to address the issues above.

PATIENT-SPECIFIC CONTEXT
...
```

This node does **not** write to `chat_history`. Only `finalize_node` does.

---

### Node 6 — `judge_response_node`

A second Gemini call scores the candidate response on **0.0 – 1.0** across four dimensions:

1. **Factual consistency** — every claim must be supported by retrieved evidence. Hallucinated drug names, dosages, or instructions are penalized heavily.
2. **Relevance** — does it directly answer the patient's question?
3. **Absence of unsupported claims** — no advice beyond what the context provides.
4. **Tone for risk tier** — Low=concise/reassuring, Medium=explicit/reinforcing, High=directive, emphasizes consequences, suggests care team contact when warranted.

**Thresholds by risk tier:**

| Risk tier | Required score |
|---|---|
| High | ≥ 0.90 |
| Medium | ≥ 0.85 |
| Low | ≥ 0.80 |

Returns `{"score": float, "reasoning": str}`. Fails open (score = 1.0) if the LLM call errors, so a judge failure never blocks a valid patient response.

---

### Node 7 — `escalate_node`

Triggered when retries are exhausted and score still hasn't cleared the threshold. Replaces the candidate response with a safe handoff message:

> *"I wasn't able to generate a confident enough answer to your question after multiple attempts. To make sure you receive accurate guidance, your care team has been notified and will follow up with you directly. Please do not make any changes to your preparation until you hear from them."*

Sets `escalated = True` in state so downstream systems can trigger a clinician notification.

---

### Node 8 — `finalize_node`

Terminal node for every path. Writes the accepted response (and the original user query) to `chat_history` via the `operator.add` reducer. Because this is the only writer to `chat_history`, retry intermediates never appear in the conversation log — only the final delivered response does.

---

### Routing Functions

```
classify_query
    chitchat → generate_response
    logistics/medical → fetch_patient_data

fetch_patient_data
    logistics → generate_response
    medical → score_risk

generate_response
    medical → judge_response
    chitchat/logistics → finalize

judge_response
    score ≥ threshold → finalize
    score < threshold, retries left → generate_response  (retry loop)
    score < threshold, max retries exhausted → escalate
```

---

## Judge Guardrail

The judge loop is what separates MayoChat from a simple LLM wrapper. For every medical response:

```
generate_response
    │
    ▼
judge_response ──── score ≥ threshold ──────────────► finalize
    │
    └── score < threshold
              │
              ├── retry_count < max_retries
              │       │
              │       ▼
              │   [inject judge reasoning into prompt]
              │       │
              │       ▼
              │   generate_response (attempt 2, then 3...)
              │
              └── retry_count ≥ max_retries
                      │
                      ▼
                  escalate → finalize
```

This means:
- A low-quality first attempt gets a second shot with specific feedback on what was wrong.
- A response that still can't clear the bar after 2 retries is never delivered — the patient gets a safe escalation message and the care team is notified.
- High-risk patients face a 0.90 threshold (vs 0.80 for low-risk), so the pipeline is more conservative where stakes are higher.

---

## Project Structure

```
mayo-clinic-chatbot/
├── main.py                          # FastAPI app + /chat endpoint
├── requirements.txt
│
├── src/
│   ├── config.py                    # Model names, top-K values, judge thresholds
│   │
│   ├── graph/                       # LangGraph orchestration (new)
│   │   ├── state.py                 # ChatState TypedDict
│   │   ├── nodes.py                 # All 8 node functions
│   │   └── graph.py                 # StateGraph construction + routing
│   │
│   ├── patient_data/
│   │   ├── bigquery_client.py       # BigQuery EHR fetch (module-level client)
│   │   └── patient_context.py       # Formats EHR record into prompt string
│   │
│   ├── retrieval/
│   │   ├── rag.py                   # Three-collection RAG pipeline
│   │   ├── filters.py               # ChromaDB where-clause builders
│   │   ├── chromadb_store.py        # Collection handles
│   │   └── embedder.py              # SentenceTransformer wrapper
│   │
│   ├── llm/
│   │   └── generate_response.py     # Gemini generation call
│   │
│   ├── risk/
│   │   ├── risk_model.py            # Joblib classifier wrapper
│   │   └── model_assets/            # Trained model + feature list
│   │
│   └── data_processing/
│       ├── document_processor.py    # Clinical guideline chunking
│       └── conversation_processor.py# Dialogue flow processing
│
└── tests/
    └── test_graph.py                # 69 unit tests (no GCP credentials needed)
```

---

## Setup

### Prerequisites

- Python 3.9+
- [Conda](https://docs.conda.io/) (recommended) or virtualenv
- GCP account with access to the `industrial-net-487818-h9` project
- Application Default Credentials configured:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project industrial-net-487818-h9
```

### Install dependencies

```bash
conda activate mayochat   # or your env of choice
pip install -r requirements.txt
pip install "numpy<2"     # torch compatibility
```

### Environment

No `.env` file required — GCP credentials come from ADC and the project ID is hardcoded in `src/config.py` and `src/patient_data/bigquery_client.py`.

---

## Running the Server

```bash
uvicorn main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) for the web UI, or hit the API directly.

---

## API Reference

### `POST /chat`

```json
{
  "patient_id": "P705309564",
  "query": "What should I avoid eating the day before my colonoscopy?"
}
```

**Response:**

```json
{
  "query": "What should I avoid eating the day before my colonoscopy?",
  "answer": "The day before your procedure, you will typically be asked to follow a clear liquid diet...",
  "debug": {
    "intent": "medical",
    "num_chunks": 3,
    "sources": [...],
    "context_preview": "PATIENT-SPECIFIC CONTEXT\n...",
    "judge_score": 1.0,
    "judge_reasoning": "Response accurately cites prep instructions and uses appropriate low-risk tone.",
    "retries": 0,
    "escalated": false
  }
}
```

The `debug` block shows the full audit trail — intent classification, how many RAG chunks were used, what the judge decided, and whether any retries or escalation occurred.

---

## Running Tests

Unit tests mock all GCP and LLM calls — no credentials needed:

```bash
pytest tests/test_graph.py -v
```

The test suite covers:
- All 4 routing functions including `None`-intent fallbacks
- Judge threshold boundaries for every risk tier (e.g. 0.89 fails High, 0.90 passes)
- Retry loop — judge increments `retry_count` only on failure
- Escalation after max retries
- `finalize_node` as sole writer to `chat_history`
- `retry_count` resets between conversation turns
- Full `graph.invoke()` for all 3 intent paths (medical, logistics, chitchat)
