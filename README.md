# MayoChat — Colonoscopy Preparation Assistant

A patient-facing AI chatbot that delivers personalized colonoscopy preparation guidance. Built on a LangGraph stateful pipeline with BigQuery EHR integration, ChromaDB RAG, a hybrid risk model, email notifications, and a Gemini LLM judge guardrail.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [LangGraph Pipeline](#langgraph-pipeline)
4. [Judge Guardrail](#judge-guardrail)
5. [Risk Scoring](#risk-scoring)
6. [RAG Retrieval](#rag-retrieval)
7. [Email Notifications](#email-notifications)
8. [Evaluation](#evaluation)
9. [Project Structure](#project-structure)
10. [Setup](#setup)
11. [Running the Server](#running-the-server)
12. [API Reference](#api-reference)
13. [Running Tests](#running-tests)

---

## Overview

MayoChat answers patient questions about their upcoming colonoscopy procedure. Every response is:

- **Personalized** — pulls the patient's EHR record from BigQuery (comorbidities, prep regimen, prior history) and adapts the answer accordingly.
- **Evidence-grounded** — retrieves relevant clinical guidelines, patient Q&A examples, and conversation flows from a ChromaDB vector store with diversity filtering and patient-aware query augmentation.
- **Risk-aware** — a hybrid model (logistic regression + clinical rule overrides) assigns each patient a risk tier for inadequate bowel prep. Higher-risk patients get more directive language and face a stricter quality bar.
- **Quality-controlled** — a Gemini LLM judge scores every medical and logistics response on a 0–1 confidence scale. Responses below the risk-tier threshold are regenerated with targeted feedback, and escalated to the care team if retries are exhausted.
- **Proactive** — SMTP email reminders (confirmation, 24-hour prep, prep start, procedure day) keep patients on track outside of the chat interface.

---

## Architecture

```
  Patient opens web UI
          │
          ▼
  ┌──────────────────┐     POST /validate-patient     ┌──────────────────┐
  │  Patient ID modal │ ───────────────────────────── ▶│  BigQuery lookup  │
  │  (verify record)  │◀ ───── patient summary ────── │                  │
  └────────┬─────────┘                                └──────────────────┘
           │ confirmed
           ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │                      LangGraph StateGraph                         │
  │                                                                   │
  │  START ──▶ classify_query                                         │
  │                  │                                                │
  │        ┌─────────┼──────────┐                                     │
  │     chitchat  logistics  medical                                  │
  │        │         │          │                                     │
  │        │    fetch_patient_data                                    │
  │        │         │          │                                     │
  │        │    (logistics)  (medical)                                │
  │        │         │      retrieve_rag                              │
  │        │         │      (clinical + Q&A + conversations)          │
  │        │         │          │                                     │
  │        └─────────┴──────────┘                                     │
  │                  │                                                │
  │          generate_response ◀───────────────────┐                  │
  │                  │                             │ retry loop       │
  │        ┌─────────┴──────────┐                  │                  │
  │   (med/logistics)      (chitchat)              │                  │
  │        │                    │                  │                  │
  │   judge_response        finalize ──▶ END       │                  │
  │        │                                       │                  │
  │   ┌────┼──────────┐                            │                  │
  │  pass  │     exhausted                         │                  │
  │   │  retry         │                           │                  │
  │   │    └───────────┼───────────────────────────┘                  │
  │   │            escalate                                           │
  │   │                │                                              │
  │   └───▶ finalize ◀─┘                                              │
  │            │                                                      │
  │           END                                                     │
  └────────────┼──────────────────────────────────────────────────────┘
               │
               ▼
       HTTP Response
  (answer + clinical sources)
```

**Key design decisions:**

- `finalize_node` is the **sole writer** to `chat_history`. Retry intermediates never pollute the conversation log.
- `MemorySaver` checkpoints the full `ChatState` per `thread_id` (= `patient_id`), giving multi-turn memory across HTTP requests with no external session store.
- Chitchat skips BigQuery and RAG entirely. Logistics skips RAG but still goes through the judge. Only medical queries run the full pipeline including retrieval.
- Risk tiers are **pre-computed** in BigQuery via a batch scoring pipeline, not calculated at request time.

---

## LangGraph Pipeline

### State — `src/graph/state.py`

All nodes share a single `ChatState` TypedDict. LangGraph merges partial dicts returned by each node back into state.

```python
class ChatState(TypedDict):
    # Input
    query: str
    patient_id: str

    # Classification
    query_intent: str               # "medical" | "logistics" | "chitchat"
    is_follow_up: bool

    # Patient data
    patient_record: dict | None
    patient_context: str

    # Risk (pre-computed, read from BigQuery)
    risk_tier: str | None           # "Low" | "Medium" | "High"
    risk_probability: float | None

    # RAG retrieval
    query_understanding: dict
    combined_context: str
    clinical_hits: list
    qa_hits: list
    conversation_hits: list
    evidence_tier: str              # "clinical" | "conversational_fallback" | "none"

    # Generation + judge
    response: str
    judge_score: float | None       # 0.0 – 1.0
    judge_reasoning: str | None
    retry_count: int
    max_retries: int
    escalated: bool

    # Multi-turn memory (operator.add reducer — appends, never overwrites)
    chat_history: List[Dict[str, str]]

    # Timing
    turn_start_time: float | None
```

`chat_history` uses `Annotated[List[...], operator.add]` so each finalize call appends two messages (user + assistant) rather than replacing the list. This is what makes multi-turn memory work across requests.

---

### Node 1 — `classify_query_node`

A lightweight Gemini call (`gemini-2.5-flash-lite`) classifies the patient's message into one of three intents:

| Intent | Meaning | Pipeline path |
|---|---|---|
| `medical` | Medications, symptoms, prep instructions, side effects, clinical questions | Full pipeline (BigQuery → RAG → Generate → Judge) |
| `logistics` | Appointment timing, location, what to bring, check-in, parking | BigQuery → Generate → Judge (no RAG) |
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

The result is passed to `build_patient_context()` which formats it into a structured string used in prompts. On follow-up turns, the record is already in the LangGraph checkpoint — BigQuery is skipped.

The node also reads pre-computed `risk_tier` and `predicted_inadequate_risk` directly from the patient record (see [Risk Scoring](#risk-scoring)).

---

### Node 3 — `retrieve_rag_node`

Runs the three-collection RAG pipeline against ChromaDB (see [RAG Retrieval](#rag-retrieval) for full details on filtering, diversity, and evidence tiers).

---

### Node 4 — `generate_response_node`

Calls `generate_response()` (in `src/llm/`) with context selected by intent:

- **chitchat** — direct Gemini call (`gemini-2.5-flash-lite`) with no context, 1-2 sentence warm reply
- **logistics** — patient context only (no clinical RAG)
- **medical** — full `combined_context` with evidence-tier-aware instructions

On retries (`retry_count > 0`), the judge's score and reasoning from the previous attempt are prepended as a `[REVISION REQUEST]` block so the model knows exactly what to fix:

```
[REVISION REQUEST — attempt 2]
Previous response scored 0.71 (required ≥ 0.90).
Judge feedback: Response mentioned ondansetron without it appearing in retrieved context.
Please revise the response to address the issues above.
```

Evidence tier notes are injected into the prompt:
- `clinical` — medication specificity guard (only apply instructions matching the patient's actual medications)
- `conversational_fallback` — acknowledge the limitation, recommend confirming with care team
- `none` — respond only from patient context, direct to care team

This node does **not** write to `chat_history`. Only `finalize_node` does.

---

### Node 5 — `judge_response_node`

A Gemini call (`gemini-2.5-flash`) scores the candidate response on **0.0 – 1.0** across four dimensions:

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

**Score ceilings by evidence tier** (applied after LLM scoring):

| Evidence tier | Max possible score |
|---|---|
| `clinical` | 1.0 |
| `conversational_fallback` | 0.85 |
| `none` | 0.70 |
| `logistics` | 1.0 (evaluated against patient record only) |

The judge now evaluates both **medical and logistics** responses. For logistics, it checks that dates, times, prep agent, and other facts match the patient record. Chitchat bypasses the judge entirely.

Fails open (score = 1.0) on LLM errors so a judge failure never blocks a valid patient response.

---

### Node 6 — `escalate_node`

Triggered when retries are exhausted and score still hasn't cleared the threshold. Replaces the candidate response with a safe handoff message and includes the care team phone number. Sets `escalated = True` in state so downstream systems can trigger a clinician notification.

---

### Node 7 — `finalize_node`

Terminal node for every path. Writes the accepted response (and the original user query) to `chat_history` via the `operator.add` reducer. Logs total turn latency. Because this is the only writer to `chat_history`, retry intermediates never appear in the conversation log.

---

### Routing Functions

```
classify_query
    chitchat → generate_response
    logistics/medical → fetch_patient_data

fetch_patient_data
    logistics → generate_response
    medical → retrieve_rag

generate_response
    medical/logistics → judge_response
    chitchat → finalize

judge_response
    score ≥ threshold → finalize
    score < threshold, retries left → generate_response  (retry loop)
    score < threshold, max retries exhausted → escalate
```

---

## Judge Guardrail

The judge loop is what separates MayoChat from a simple LLM wrapper. For every medical and logistics response:

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
- Evidence tier ceilings prevent over-confidence when clinical guidelines are absent — a response grounded only in conversational examples can never score above 0.85.

---

## Risk Scoring

Risk tiers are **pre-computed** via a batch pipeline (`src/risk/generate_risk_scores.py`) and stored in BigQuery's `Patient_Risk_Scores` table. The chatbot reads the stored tier at request time rather than scoring live.

### Hybrid model — `src/risk/scoring.py`

The scoring function combines two approaches:

1. **Rule-based overrides** — any of these conditions force `High` risk regardless of the model score:
   - Diabetes with complications (gastroparesis, nephropathy, retinopathy)
   - Regular opioid or GLP-1 agonist use
   - Prior inadequate prep or extended prep required
   - Prior colorectal surgery
   - Severe constipation or chronic laxative use
   - Paraplegia / quadriplegia (spinal cord injury)
   - Cirrhosis, Parkinson's disease, or dementia

2. **Logistic regression classifier** — a pre-trained joblib pipeline (`src/risk/model_assets/risk_pipeline.joblib`) predicts the probability of inadequate bowel prep from 11 features (BMI, constipation severity, gastroparesis, diabetes, hypertension, tobacco use, sex, mobility, age 65+, etc.). Patients not caught by rule overrides are assigned `Medium` (≥ 0.5 predicted risk) or `Low` (< 0.5).

### Batch scoring pipeline

`generate_risk_scores.py` queries all patients from BigQuery, engineers features to match the training notebook, runs both the rule-based and model-based scoring, and writes results back to BigQuery with `WRITE_TRUNCATE`.

---

## RAG Retrieval

### Three-collection architecture

| Collection | Contents | Top-K | Filter strategy |
|---|---|---|---|
| `clinical` | Guidelines, drug labels, patient instructions | 5 | Document-type + patient-specific tags (prep agent, procedure time, indication, comorbidities) |
| `qa` | Curated turn-level Q&A pairs | 2 | Risk tier + appointment time + is_follow_up |
| `conversations` | Full multi-turn dialogue flows | 2 | Risk tier + appointment time + demonstrates_multi_turn |

### Query understanding

A single `extract_query_understanding()` call (`gemini-2.5-flash-lite` with structured JSON output) extracts all query-level signals in one LLM round trip:
- `medication_class` — mapped from trade names, generics, or patient vernacular (e.g. "blood thinner" → anticoagulants)
- `drug_name` — specific bowel prep agent (SUPREP, GOLYTELY, MIRALAX, etc.)
- `document_type` — drug_label, clinical_guideline, or patient_instructions
- `procedure_timing` — morning or afternoon
- `is_diabetes_query` — triggers metabolic condition filters
- `wants_research` — controls whether research-background chunks are included or suppressed

### Patient-aware query augmentation

Before embedding, the raw query is prepended with a short patient summary (conditions, prep agent, procedure timing) so the query vector sits closer to chunks relevant to this specific patient, not just the question in the abstract.

### Diversity filtering

Each collection over-fetches by 3x, then applies a greedy cosine-similarity filter (threshold 0.95) to remove near-duplicate chunks before returning the final top-K. This ensures the LLM sees semantically distinct evidence rather than paraphrased copies of the same content.

### Union query strategy

For clinical retrieval, filtered results (patient_instructions / clinician_guideline) are fetched first. Unfiltered results only supplement when the filtered pass returns fewer than the target count. This prevents research-background chunks from displacing patient-facing content.

### Evidence tiers

The retrieval node classifies the quality of clinical evidence found:

| Tier | Condition | Effect |
|---|---|---|
| `clinical` | Clinical guideline chunks found | Normal path — full RAG context |
| `conversational_fallback` | No clinical hits, but Q&A/conversation hits exist | Dialogue examples promoted to factual reference; judge score capped at 0.85 |
| `none` | Nothing useful found in any collection | Response limited to patient context; judge score capped at 0.70 |

### Post-processing

Retrieved hits are filtered and reordered:
- Contact/admin-only chunks are dropped
- Research-background chunks are suppressed unless the patient explicitly asked for clinical evidence
- Preferred sources (Mayo Clinic, FDA, DailyMed) are ranked first
- Outside-hospital chunks are flagged with a source caution

### Clinical source display

For medical responses, the UI shows a collapsible "View clinical sources" panel. `source_display.py` builds patient-friendly citation rows (source name, section title, snippet) from clinical hits, using pre-computed labels from chunk metadata with heuristic fallbacks.

---

## Email Notifications

MayoChat sends prep reminder emails via SMTP (`src/notifications/`).

### Email templates — `email_templates.py`

Four template types, each personalized with the patient's schedule and prep details:

| Type | When | Content |
|---|---|---|
| `confirmation` | At patient verification | Full appointment details, prep instructions, medication reminder |
| `prep_24h` | 24 hours before prep start | Prep start time, colonoscopy time, prep agent, diet instructions |
| `prep_start` | At prep start time | Step-by-step prep details, medication reminder, emergency contacts |
| `procedure_day` | Day of colonoscopy | Appointment time, transportation reminder |

### Demo schedule — `schedule_utils.py`

For demo purposes, the original procedure timeline from BigQuery is shifted forward so the colonoscopy is always 3 days in the future. The relative gaps between prep start, prep end, and procedure time are preserved.

### SMTP service — `email_service.py`

Sends emails via Gmail SMTP (TLS on port 587). Configured through environment variables `SMTP_EMAIL` and `SMTP_PASSWORD`. Sending is gated by the `EMAIL_ENABLED` env var — when disabled, API endpoints return the email payload as a preview without sending.

---

## Evaluation

### RAGAS evaluation — `src/evaluation/run_ragas_eval.py`

End-to-end evaluation of the full LangGraph pipeline against a clinician-reviewed dataset.

**Input:** Excel file with 42 rows (7 patients × 3 turns × 2 clinicians), containing queries and clinician-written golden answers.

**Process:**
1. Runs the full LangGraph pipeline once per (patient_id, turn), preserving multi-turn state via SQLite checkpointing (enables resume after rate limiting or process restarts)
2. Builds RAGAS inputs: question, pipeline answer, retrieved contexts, clinician golden answer
3. Deduplicates to unique RAGAS cases
4. Runs RAGAS metrics on rows with contexts: **faithfulness**, **answer_relevancy**, **context_precision**, **context_recall**
5. Runs answer_relevancy only on rows without contexts
6. Merges scores back to all clinician rows

**Output:** CSV files in `outputs/ragas_eval/` — unique inputs, unique scores, merged results with clinician labels, and a summary with count/mean/median/min/max per metric.

```bash
PYTHONPATH="$PWD" python src/evaluation/run_ragas_eval.py \
  --input-xlsx src/evaluation/ragas_evaluation.xlsx \
  --output-dir outputs/ragas_eval \
  --project industrial-net-487818-h9 \
  --judge-model gemini-2.0-flash \
  --embedding-model mini_lm
```

### Embedding model evaluation — `src/evaluation/eval_embeddings.py`

Compares embedding models at the retrieval layer only (no LLM calls). Tests whether the retriever surfaces the right knowledge-base chunks for a curated benchmark.

**Models supported:** all-MiniLM-L6-v2, BGE-base-en, E5-base-v2, BioMedBERT, Vertex text-embedding-005

**Metrics:** hit@K, MRR@K, average relevant@K

```bash
python src/evaluation/eval_embeddings.py \
  --chunks-path src/data_processing/patient_kb/processed_chunks/clinical_processed_chunks.json \
  --benchmark-path src/evaluation/retrieval_benchmark.json \
  --models all_minilm_l6_v2 bge_base_en e5_base_v2 \
  --top-k 5 \
  --output-dir outputs/embedding_eval_clinical_open_source
```

### Inter-annotator agreement — `outputs/generation_iaa_eval/`

Clinician inter-annotator agreement analysis on generation quality dimensions (factual accuracy, relevance, hallucination, harmfulness).

---

## Project Structure

```
mayo-clinic-chatbot/
├── main.py                              # FastAPI app, web UI, all API endpoints
├── requirements.txt
├── Dockerfile                           # Python 3.11 container for deployment
│
├── src/
│   ├── config.py                        # Model names, top-K, thresholds, email settings
│   │
│   ├── graph/                           # LangGraph orchestration
│   │   ├── state.py                     # ChatState TypedDict (all shared pipeline state)
│   │   ├── nodes.py                     # All 7 node functions
│   │   └── graph.py                     # StateGraph construction, routing, MemorySaver
│   │
│   ├── patient_data/
│   │   ├── bigquery_client.py           # BigQuery EHR fetch (5-table join)
│   │   └── patient_context.py           # Formats EHR record into prompt string
│   │
│   ├── retrieval/
│   │   ├── rag.py                       # Three-collection RAG with diversity filtering
│   │   ├── filters.py                   # Query + patient ChromaDB where-clause builders
│   │   ├── research_filters.py          # Research-background chunk detection
│   │   ├── chromadb_store.py            # Collection handles
│   │   ├── embedder.py                  # SentenceTransformer wrapper (MiniLM)
│   │   ├── source_display.py            # Patient-facing clinical source citations
│   │   ├── patient_source_label.py      # Source name heuristics (DailyMed, Mayo, etc.)
│   │   └── index_kb.py                  # Knowledge base indexing script
│   │
│   ├── llm/
│   │   ├── generate_response.py         # Gemini generation with few-shot examples
│   │   └── chatbot_fewshot_examples.md  # Tone and structure examples for the LLM
│   │
│   ├── risk/
│   │   ├── scoring.py                   # Hybrid risk model (rule-based + logistic regression)
│   │   ├── generate_risk_scores.py      # Batch scoring pipeline → BigQuery
│   │   └── model_assets/                # Trained pipeline (risk_pipeline.joblib) + features
│   │
│   ├── notifications/
│   │   ├── email_service.py             # SMTP email sender (Gmail TLS)
│   │   ├── email_templates.py           # 4 email templates (confirmation, 24h, start, day-of)
│   │   └── schedule_utils.py            # Demo schedule shifting for live demos
│   │
│   ├── evaluation/
│   │   ├── run_ragas_eval.py            # RAGAS pipeline (faithfulness, relevancy, precision, recall)
│   │   ├── eval_embeddings.py           # Embedding model retrieval benchmark
│   │   ├── ragas_evaluation.xlsx        # Clinician-reviewed evaluation dataset
│   │   ├── retrieval_benchmark.json     # Curated retrieval test cases
│   │   └── golden_references.md         # Golden reference answers
│   │
│   └── data_processing/
│       ├── document_processor.py        # Clinical guideline chunking + tagging
│       ├── conversation_processor.py    # Dialogue flow processing
│       ├── drugscrape.py                # Drug label data collection
│       └── patient_kb/                  # Processed chunks, PDFs, conversation data
│
├── tests/
│   ├── test_graph.py                    # LangGraph pipeline unit tests
│   ├── test_rag_retrieval.py            # RAG retrieval tests
│   ├── test_rag_gcp.py                  # GCP integration tests
│   ├── test_rag_retrieve_integration.py # End-to-end retrieval tests
│   ├── test_pdf_routing.py              # PDF document routing tests
│   └── test_source_display.py           # Source citation display tests
│
└── outputs/
    ├── ragas_eval/                      # RAGAS evaluation results
    ├── ragas_eval_ARCHIVED/             # Previous RAGAS runs
    ├── embedding_eval_clinical_open_source/  # Embedding model comparison results
    └── generation_iaa_eval/             # Clinician inter-annotator agreement
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

For RAGAS evaluation, install additional dependencies:

```bash
pip install -r ragas-requirements.txt
```

### Environment variables

GCP credentials come from ADC and the project ID is hardcoded in `src/config.py`.

Optional `.env` variables:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_MODEL` | `gemini-2.5-flash` | Primary model (generation + judge) |
| `ROUTING_LLM_MODEL` | `gemini-2.5-flash-lite` | Classification + chitchat |
| `STRUCTURED_LLM_MODEL` | `gemini-2.5-flash-lite` | Filter extraction (structured JSON) |
| `EMAIL_ENABLED` | `false` | Set `true` to send real emails |
| `SMTP_EMAIL` | `mayochatbot1@gmail.com` | Gmail sender address |
| `SMTP_PASSWORD` | (empty) | Gmail app password |
| `DEFAULT_PATIENT_EMAIL` | `mayochatbot1@gmail.com` | Fallback recipient |

---

## Running the Server

```bash
uvicorn main:app --reload --port 8000
```

Then open [http://localhost:8000](http://localhost:8000) for the web UI, or hit the API directly.

The web UI flow:
1. A modal prompts for the patient ID
2. The record is fetched from BigQuery and displayed for verification
3. On confirmation, the chat interface unlocks with a personalized greeting
4. Each response shows the answer and a collapsible "View clinical sources" panel for medical queries

---

## API Reference

### `POST /validate-patient`

Looks up a patient record from BigQuery and returns a summary for identity verification.

```json
{ "patient_id": "P705309564" }
```

**Response:**

```json
{
  "valid": true,
  "summary": {
    "patient_id": "P705309564",
    "patient_name": "Jane Doe",
    "sex_at_birth": "Female",
    "colonoscopy_datetime": "2026-05-10 08:00:00",
    "prep_agent": "SUPREP",
    ...
  }
}
```

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
  "clinical_sources": [
    {
      "source_name": "Mayo Clinic",
      "title": "Clear Liquid Diet Instructions",
      "snippet": "A clear liquid diet includes water, broth, gelatin..."
    }
  ],
  "debug": {
    "intent": "medical",
    "num_chunks": 3,
    "sources": [...],
    "context_preview": "PATIENT-SPECIFIC CONTEXT\n...",
    "judge_score": 0.95,
    "judge_reasoning": "Response accurately cites prep instructions and uses appropriate low-risk tone.",
    "retries": 0,
    "escalated": false
  }
}
```

### `POST /preview-prep-reminder`

Returns the email payload without sending.

```json
{ "patient_id": "P705309564", "email": "patient@example.com" }
```

### `POST /send-prep-reminder`

Sends the prep confirmation email (requires `EMAIL_ENABLED=true`).

```json
{ "patient_id": "P705309564", "email": "patient@example.com" }
```

---

## Running Tests

Unit tests mock all GCP and LLM calls — no credentials needed:

```bash
pytest tests/ -v
```

The test suite (6 files, ~1400 lines) covers:
- All routing functions including `None`-intent fallbacks
- Judge threshold boundaries for every risk tier (e.g. 0.89 fails High, 0.90 passes)
- Retry loop — judge increments `retry_count` only on failure
- Escalation after max retries
- `finalize_node` as sole writer to `chat_history`
- `retry_count` resets between conversation turns
- Full `graph.invoke()` for all 3 intent paths (medical, logistics, chitchat)
- RAG retrieval and diversity filtering
- PDF document routing
- Patient-facing source citation display
