# RAG policy: research chunks and comorbidities

This file documents how to use chunk metadata from `clinical_processed_chunks.json` for retrieval and LLM prompting. The pipeline sets these fields on PDF-derived chunks:

- **`audience_tier`**: `patient_care` | `clinician_guideline` | `research_education`
- **`source_category`**: coarse provenance (e.g. `society_guideline`, `hospital`, `trial`, `meta_analysis`, `patient_handout`, `other`)
- **`content_use_policy`**: e.g. `research_background` when `audience_tier` is `research_education`
- **Tags**: research PDFs also get `content_policy:research_background` and `audience:research_education`

## Research vs patient-care content

- **Default prep Q&A** (what to eat, timing, logistics): prefer chunks with `audience_tier` in (`patient_care`, `clinician_guideline`) and drug-label chunks with colonoscopy prep labeling. **Downrank or exclude** `audience_tier=research_education` unless the user asks about studies, trials, meta-analyses, or evidence quality.
- **Educational / evidence questions**: trial and meta-analysis chunks are appropriate; still avoid presenting a single study as standard of care.

## Comorbidities and safety

- When the user mentions **high-risk** contexts (e.g. anticoagulation, pregnancy, IBD, bariatric surgery, advanced CKD), responses should include a **disclaimer** to **confirm with their clinician** and must not treat one trial or secondary source as definitive care instructions.
- Chunk tags such as `ibd`, `crohns_disease`, `bariatric_surgery`, `anticoagulation_context`, `pregnancy_lactation`, `renal_disease` help retrieval; they do not replace clinical judgment.

## Overrides

Edit `pdf_manifest.json` to force `audience_tier` / `source_type` for a given filename stem when heuristics are wrong.

## Chroma indexing and retrieval

- After regenerating `clinical_processed_chunks.json`, run `python -m src.retrieval.index_kb` from the repo root so Chroma stores `audience_tier`, `content_use_policy`, and `source_category` (see `src/retrieval/chromadb_store.py`).
- `src/retrieval/rag.py` **post-filters** trial/meta-analysis chunks (`audience_tier=research_education` or tag `content_policy:research_background`) out of default search results unless the query appears to ask for studies or evidence (`query_requests_research_evidence`). Retrieved context lines include these fields when present.
- Chroma metadata includes **`source_file`** (PDF/JSON basename) for outside-hospital / contact heuristics in `postprocess_hits`; re-index after upgrading `chromadb_store`.
