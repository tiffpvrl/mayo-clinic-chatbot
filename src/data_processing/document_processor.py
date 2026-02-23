"""
Document Processing & Semantic Chunking for Bowel Prep Patient Knowledge Base

Strategies:
1. Document-Type Aware Chunking - Clinical guidelines (by section), drug labels (by FDA sections),
   patient instructions (by steps/days)
2. Contextual Tagging - Medical conditions, medication classes, topics for filtered retrieval
3. Metadata Preservation - Source, doc type, publication year, section hierarchy
4. Semantic Boundaries - Respect paragraph/section boundaries, optional overlap for context
5. Parent-Child References - Enable hierarchical retrieval in RAG
"""

from pathlib import Path
import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator
from enum import Enum

# Optional: use PyPDF2 or pypdf for PDF extraction
try:
    from PyPDF2 import PdfReader
except ImportError:
    try:
        from pypdf import PdfReader
    except ImportError:
        PdfReader = None


class DocumentType(str, Enum):
    """Classification of documents in the patient knowledge base."""
    CLINICAL_GUIDELINE = "clinical_guideline"
    DRUG_LABEL = "drug_label"
    PATIENT_INSTRUCTIONS = "patient_instructions"
    UNKNOWN = "unknown"


# Standard FDA drug label section keys (for contextual tagging & chunking)
DRUG_LABEL_SECTIONS = {
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings",
    "warnings_and_precautions",
    "adverse_reactions",
    "drug_interactions",
    "special_populations",
    "patient_counseling_information",
}

# Medical conditions relevant to bowel prep (for contextual tagging)
RELEVANT_CONDITIONS = {
    "renal_disease", "kidney_disease", "CKD", "ESRD",
    "diabetes", "G6PD_deficiency", "phenylketonuria", "PKU",
    "ulcerative_colitis", "ileus", "bowel_obstruction", " gastric_retention",
    "heart_failure", "arrhythmia", "hypertension",
    "geriatric", "elderly", "pediatric", "children",
    "pregnancy", "lactation", "breastfeeding",
    "anticoagulation", "antiplatelet",
}

# Medication classes (for contextual tagging)
MEDICATION_CLASSES = {
    "anticoagulants": ["warfarin", "xarelto", "rivaroxaban", "eliquis", "apixaban", "pradaxa", "dabigatran", "lovenox", "enoxaparin"],
    "antiplatelet": ["clopidogrel", "plavix", "ticagrelor", "brilinta", "prasugrel", "effient", "aspirin"],
    "diuretics": ["furosemide", "hydrochlorothiazide", "HCTZ", "spironolactone"],
    "ace_inhibitors": ["lisinopril", "enalapril", "ramipril", "ACE-inhibitor", "ACE inhibitor"],
    "sglt2_inhibitors": ["invokana", "canagliflozin", "farxiga", "dapagliflozin", "jardiance", "empagliflozin"],
    "nsaids": ["ibuprofen", "naproxen", "aleve", "naprelan", "naprosyn", "anaprox"],
}

# Topic tags for filtered retrieval
TOPIC_TAGS = {
    "dosing", "timing", "diet", "clear_liquids", "side_effects", "contraindications",
    "drug_interactions", "medication_management", "split_dose", "same_day",
    "transportation", "preparation_steps", "what_to_expect", "when_to_contact_provider",
}


@dataclass
class ChunkMetadata:
    """Metadata preserved with each chunk for RAG retrieval."""
    source_file: str
    document_type: DocumentType
    section_title: str = ""
    section_hierarchy: list[str] = field(default_factory=list)  # e.g., ["Before Colonoscopy", "Diet"]
    publication_year: str | None = None
    organization: str | None = None  # e.g., "USMSTF", "UCSF", "FDA"
    drug_name: str | None = None  # For drug labels
    page_number: int | None = None
    chunk_index: int = 0
    total_chunks: int = 0
    tags: set[str] = field(default_factory=set)  # medical_conditions, medication_classes, topics
    parent_chunk_id: str | None = None  # For hierarchical retrieval


@dataclass
class ProcessedChunk:
    """A semantically coherent chunk ready for embedding and RAG."""
    id: str
    content: str
    metadata: ChunkMetadata


def detect_document_type(file_path: Path, content_preview: str = "") -> DocumentType:
    """Classify document type from path patterns and content cues."""
    path_str = str(file_path).lower()
    name = file_path.stem.lower()

    # Drug labels: JSON with sections or path containing drug_labels
    if "drug_label" in path_str or "dailymed" in path_str or "openfda" in path_str:
        return DocumentType.DRUG_LABEL
    if file_path.suffix == ".json" and ("consolidated" in path_str or "dailymed_json" in path_str or "openfda" in path_str):
        return DocumentType.DRUG_LABEL

    # Clinical guidelines: taskforce, consensus, surveillance, quality indicators
    guideline_keywords = ["taskforce", "task_force", "consensus", "guideline", "surveillance", "quality_indicators"]
    if any(kw in name for kw in guideline_keywords):
        return DocumentType.CLINICAL_GUIDELINE

    # Patient instructions: colonoscopy instructions, prep booklet, extended prep
    instruction_keywords = ["instructions", "booklet", "prep", "colonoscopy", "2 day", "extended"]
    if any(kw in name for kw in instruction_keywords):
        return DocumentType.PATIENT_INSTRUCTIONS

    # Content-based fallback
    if "indications and usage" in content_preview.lower() or "dosage and administration" in content_preview.lower():
        return DocumentType.DRUG_LABEL
    if "we recommend" in content_preview.lower() or "consensus" in content_preview.lower():
        return DocumentType.CLINICAL_GUIDELINE
    if "days before" in content_preview.lower() or "stop:" in content_preview.lower() or "ok/approved" in content_preview.lower():
        return DocumentType.PATIENT_INSTRUCTIONS

    return DocumentType.UNKNOWN


def extract_publication_year(path: Path, content: str) -> str | None:
    """Extract publication year from filename or content."""
    # From filename: e.g., 2020_taskforce, 2025_taskforce_bowelprep
    match = re.search(r"(20\d{2})", path.stem)
    if match:
        return match.group(1)
    match = re.search(r"\b(20\d{2})\b", content[:2000])
    if match:
        return match.group(1)
    return None


def extract_organization(path: Path, content: str) -> str | None:
    """Extract organization/source from content or path."""
    content_lower = content[:3000].lower()
    orgs = ["USMSTF", "UCSF", "ACG", "ASGE", "AGA", "FDA", "Mayo", "Multi-Society"]
    for org in orgs:
        if org.lower() in content_lower or org in content[:3000]:
            return org
    if "ucsf" in str(path).lower():
        return "UCSF"
    return None


def tag_content(content: str) -> set[str]:
    """Extract contextual tags (conditions, medication classes, topics) from content."""
    tags = set()
    text_lower = content.lower()

    # Medical conditions
    condition_patterns = [
        (r"\b(renal|kidney|ckd|esrd)\b", "renal_disease"),
        (r"\b(diabete|g6pd|phenylketonuria|pku)\b", "metabolic_conditions"),
        (r"\b(ulcerative colitis|ileus|obstruction)\b", "gi_conditions"),
        (r"\b(pregnancy|pregnant|lactation|breastfeed)\b", "pregnancy_lactation"),
        (r"\b(geriatric|elderly|age 60|over 60)\b", "geriatric"),
        (r"\b(pediatric|children|child)\b", "pediatric"),
    ]
    for pattern, tag in condition_patterns:
        if re.search(pattern, text_lower):
            tags.add(tag)

    # Medication classes
    for med_class, drug_list in MEDICATION_CLASSES.items():
        if any(drug in text_lower for drug in drug_list):
            tags.add(f"med_class:{med_class}")

    # Topics
    topic_patterns = [
        (r"\b(dosage|dose|dosing)\b", "dosing"),
        (r"\b(timing|when to take|hours before)\b", "timing"),
        (r"\b(diet|food|clear liquid|low.?fiber|low.?residue)\b", "diet"),
        (r"\b(side effect|adverse|nausea|vomiting)\b", "side_effects"),
        (r"\b(contraindication|do not use)\b", "contraindications"),
        (r"\b(drug interaction|medication)\b", "drug_interactions"),
        (r"\b(split.?dose|split dose)\b", "split_dose"),
        (r"\b(same.?day|same day)\b", "same_day"),
        (r"\b(transport|ride home|driver)\b", "transportation"),
        (r"\b(contact|call|provider|healthcare)\b", "when_to_contact_provider"),
    ]
    for pattern, tag in topic_patterns:
        if re.search(pattern, text_lower):
            tags.add(tag)

    return tags


# ---------------------------------------------------------------------------
# Chunking Strategies by Document Type
# ---------------------------------------------------------------------------


def chunk_drug_label(
    doc: dict,
    source_path: Path,
) -> list[ProcessedChunk]:
    """
    Chunk drug labels by standardized FDA sections.
    Each section becomes one chunk; very long sections are split by paragraphs.
    """
    chunks = []
    drug_name = doc.get("drug_name", "Unknown")
    doc_type = DocumentType.DRUG_LABEL
    sections = doc.get("sections", {})
    set_id = doc.get("set_id", "")

    base_metadata = ChunkMetadata(
        source_file=str(source_path),
        document_type=doc_type,
        drug_name=drug_name,
        organization=doc.get("source", "DailyMed"),
    )

    for idx, (section_key, section_data) in enumerate(sections.items()):
        if not isinstance(section_data, dict):
            continue
        title = section_data.get("title", section_key.replace("_", " ").title())
        content = section_data.get("content", "")
        if not content or not content.strip():
            continue

        # Split very long sections by double newlines (paragraphs), max ~800 chars per sub-chunk
        sub_contents = _split_long_section(content, max_chars=800)

        for sub_idx, sub_content in enumerate(sub_contents):
            chunk_id = f"drug_{drug_name}_{set_id[:8]}_{section_key}_{sub_idx}" if set_id else f"drug_{drug_name}_{section_key}_{sub_idx}"
            tags = tag_content(sub_content)

            meta = ChunkMetadata(
                source_file=base_metadata.source_file,
                document_type=doc_type,
                section_title=title,
                section_hierarchy=[drug_name, title],
                drug_name=drug_name,
                organization=base_metadata.organization,
                chunk_index=len(chunks),
                tags=tags,
            )
            chunks.append(ProcessedChunk(id=chunk_id, content=sub_content.strip(), metadata=meta))

    # Update total_chunks
    for c in chunks:
        c.metadata.total_chunks = len(chunks)

    return chunks


def chunk_clinical_guideline(
    content: str,
    source_path: Path,
) -> list[ProcessedChunk]:
    """
    Chunk clinical guidelines by section headers.
    Looks for patterns like INTRODUCTION, METHODS, RECOMMENDATIONS, Table N, etc.
    """
    chunks = []
    doc_type = DocumentType.CLINICAL_GUIDELINE
    year = extract_publication_year(source_path, content)
    org = extract_organization(source_path, content)

    # Section header pattern: ALL CAPS lines, numbered sections (1. 2. 3.), or "Table N"
    section_pattern = re.compile(
        r"^(?:(?:Table\s+\d+[.:]?\s*)|(?:[A-Z][A-Z\s\-]+(?:\d+[.:])?)|(?:\d+\.\s+[A-Z][A-Za-z\s]+))\s*$",
        re.MULTILINE,
    )

    # Simpler: split on lines that look like headers (ALL CAPS, or "1. Title")
    lines = content.split("\n")
    current_section_title = "Introduction"
    current_content: list[str] = []
    section_titles: list[str] = []

    def flush_section(title: str, text: str):
        if not text.strip():
            return
        sub_chunks = _split_long_section(text, max_chars=600)
        for i, sub in enumerate(sub_chunks):
            chunk_id = f"guideline_{source_path.stem}_{len(chunks):03d}"
            tags = tag_content(sub)
            meta = ChunkMetadata(
                source_file=str(source_path),
                document_type=doc_type,
                section_title=title,
                section_hierarchy=section_titles.copy(),
                publication_year=year,
                organization=org,
                chunk_index=len(chunks),
                tags=tags,
            )
            chunks.append(ProcessedChunk(id=chunk_id, content=sub.strip(), metadata=meta))

    for line in lines:
        stripped = line.strip()
        # Header heuristic: ALL CAPS (at least 4 chars), or "1. SECTION NAME"
        is_header = (
            len(stripped) >= 4
            and stripped.isupper()
            and len(stripped) < 80
            and not stripped.endswith(".")
        ) or bool(re.match(r"^\d+\.\s+[A-Z]", stripped))

        if is_header and current_content:
            flush_section(current_section_title, "\n".join(current_content))
            current_section_title = stripped[:100]
            section_titles = [current_section_title]
            current_content = []
        elif is_header:
            current_section_title = stripped[:100]
            if not section_titles or section_titles[-1] != current_section_title:
                section_titles.append(current_section_title)
            current_content = []
        else:
            current_content.append(line)

    if current_content:
        flush_section(current_section_title, "\n".join(current_content))

    for c in chunks:
        c.metadata.total_chunks = len(chunks)

    return chunks


def chunk_patient_instructions(
    content: str,
    source_path: Path,
) -> list[ProcessedChunk]:
    """
    Chunk patient instructions by time phases and steps.
    E.g., "3 Days Before", "2 Days Before", "1 Day Before", "Day Of", medication lists, etc.
    """
    chunks = []
    doc_type = DocumentType.PATIENT_INSTRUCTIONS
    year = extract_publication_year(source_path, content)
    org = extract_organization(source_path, content)

    # Phase patterns: "3 Days Before", "Two nights before", "One Day Before", "6-8 Hours Before"
    phase_pattern = re.compile(
        r"^(\d+\s*Days?\s*Before|\d+[-–]\d+\s*Hours?\s*Before|"
        r"One\s+Day\s+Before|Two\s+Nights?\s*Before|Three\s+Nights?\s*Before|"
        r"The\s+Day\s+(Before|Of)|Before\s+Your\s+Colonoscopy|"
        r"STOP:\s*|Ok/Approved\s*to\s*Take:|\d+\s*Days?\s*Before\s*Your)",
        re.IGNORECASE | re.MULTILINE,
    )

    sections: list[tuple[str, str]] = []
    last_end = 0
    for m in phase_pattern.finditer(content):
        if m.start() > last_end:
            # Content before first section
            before = content[last_end:m.start()].strip()
            if before:
                sections.append(("Introduction / General", before))
        # Extract section title (first line of match)
        line_start = content.rfind("\n", 0, m.start()) + 1
        line_end = content.find("\n", m.end())
        if line_end == -1:
            line_end = len(content)
        title = content[line_start:line_end].strip()[:80]
        # Content up to next section
        next_m = phase_pattern.search(content, m.end())
        end = next_m.start() if next_m else len(content)
        section_content = content[m.start():end].strip()
        if section_content:
            sections.append((title or m.group(0), section_content))
        last_end = end

    if last_end < len(content):
        rest = content[last_end:].strip()
        if rest:
            sections.append(("Additional Information", rest))

    # If no phases found, fall back to paragraph-based chunking
    if not sections:
        sections = [("Full Document", content)]

    for idx, (title, text) in enumerate(sections):
        sub_chunks = _split_long_section(text, max_chars=500)
        for sub_idx, sub in enumerate(sub_chunks):
            chunk_id = f"patient_{source_path.stem}_{idx}_{sub_idx}"
            tags = tag_content(sub)
            meta = ChunkMetadata(
                source_file=str(source_path),
                document_type=doc_type,
                section_title=title,
                section_hierarchy=[title],
                publication_year=year,
                organization=org,
                chunk_index=len(chunks),
                tags=tags,
            )
            chunks.append(ProcessedChunk(id=chunk_id, content=sub.strip(), metadata=meta))

    for c in chunks:
        c.metadata.total_chunks = len(chunks)

    return chunks


def _split_long_section(text: str, max_chars: int = 600) -> list[str]:
    """Split long text by paragraphs; avoid splitting mid-sentence."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    parts = []
    paragraphs = re.split(r"\n\s*\n", text)
    current = []
    current_len = 0

    for p in paragraphs:
        p_stripped = p.strip()
        if not p_stripped:
            continue
        if current_len + len(p_stripped) + 2 <= max_chars:
            current.append(p_stripped)
            current_len += len(p_stripped) + 2
        else:
            if current:
                parts.append("\n\n".join(current))
            if len(p_stripped) > max_chars:
                # Split long paragraph by sentences
                sentences = re.split(r"(?<=[.!?])\s+", p_stripped)
                current = []
                current_len = 0
                for s in sentences:
                    if current_len + len(s) + 1 <= max_chars:
                        current.append(s)
                        current_len += len(s) + 1
                    else:
                        if current:
                            parts.append(" ".join(current))
                        current = [s]
                        current_len = len(s) + 1
            else:
                current = [p_stripped]
                current_len = len(p_stripped) + 2

    if current:
        parts.append("\n\n".join(current))

    return parts


# ---------------------------------------------------------------------------
# Document Loading & Main Processing
# ---------------------------------------------------------------------------


def load_pdf_text(path: Path) -> str:
    """Extract text from PDF using PyPDF2 or pypdf."""
    if PdfReader is None:
        raise ImportError(
            "PDF processing requires PyPDF2 or pypdf. Install with: pip install PyPDF2"
        )
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_drug_label_json(path: Path) -> list[dict]:
    """Load drug label(s) from JSON. Handles both single-doc and consolidated arrays."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return [data]


def process_document(path: Path, base_dir: Path) -> list[ProcessedChunk]:
    """
    Process a single document and return list of chunks.
    Dispatches to appropriate chunker based on detected document type.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        docs = load_drug_label_json(path)
        if not docs:
            return []
        all_chunks = []
        for doc in docs:
            chunks = chunk_drug_label(doc, path)
            all_chunks.extend(chunks)
        return all_chunks

    if suffix == ".pdf":
        content = load_pdf_text(path)
    else:
        content = path.read_text(encoding="utf-8", errors="replace")

    doc_type = detect_document_type(path, content[:3000])

    if doc_type == DocumentType.CLINICAL_GUIDELINE:
        return chunk_clinical_guideline(content, path)
    if doc_type == DocumentType.PATIENT_INSTRUCTIONS:
        return chunk_patient_instructions(content, path)
    if doc_type == DocumentType.DRUG_LABEL:
        # PDF drug label - treat as patient-ish for now
        return chunk_patient_instructions(content, path)

    # Fallback: semantic split by paragraphs
    return chunk_clinical_guideline(content, path)


def process_patient_kb(
    kb_dir: Path | str = "patient_kb",
    output_path: Path | str | None = "patient_kb/processed_chunks.json",
) -> list[ProcessedChunk]:
    """
    Process entire patient knowledge base and optionally save chunks to JSON.
    Scans pdf_assets, drug_labels/processed, and drug_labels dailymed/openfda JSONs.
    """
    kb_dir = Path(kb_dir)
    all_chunks: list[ProcessedChunk] = []

    # PDFs in pdf_assets
    pdf_dir = kb_dir / "pdf_assets"
    if pdf_dir.exists():
        for f in pdf_dir.glob("*.pdf"):
            try:
                chunks = process_document(f, kb_dir)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Warning: Failed to process {f}: {e}")

    # Consolidated drug labels
    consolidated = kb_dir / "drug_labels" / "processed" / "consolidated_drug_labels.json"
    if consolidated.exists():
        try:
            chunks = process_document(consolidated, kb_dir)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Warning: Failed to process {consolidated}: {e}")

    # Individual drug label JSONs
    for subdir in ["dailymed_json", "openfda_json"]:
        d = kb_dir / "drug_labels" / subdir
        if d.exists():
            for f in d.glob("*.json"):
                try:
                    chunks = process_document(f, kb_dir)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"Warning: Failed to process {f}: {e}")

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {
                "id": c.id,
                "content": c.content,
                "metadata": {
                    "source_file": c.metadata.source_file,
                    "document_type": c.metadata.document_type.value,
                    "section_title": c.metadata.section_title,
                    "section_hierarchy": c.metadata.section_hierarchy,
                    "publication_year": c.metadata.publication_year,
                    "organization": c.metadata.organization,
                    "drug_name": c.metadata.drug_name,
                    "page_number": c.metadata.page_number,
                    "chunk_index": c.metadata.chunk_index,
                    "total_chunks": c.metadata.total_chunks,
                    "tags": list(c.metadata.tags),
                },
            }
            for c in all_chunks
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    kb = Path("src/data_processing/patient_kb")
    out = Path("src/data_processing/patient_kb/processed_chunks.json")
    # print(f"kb path: {}")

    if len(sys.argv) > 1:
        kb = Path(sys.argv[1])
    if len(sys.argv) > 2:
        out = Path(sys.argv[2])

    chunks = process_patient_kb(kb_dir=kb, output_path=out)
    print(f"Processed {len(chunks)} chunks from patient knowledge base")
    print(f"Saved to {out}")
    print("\nSample chunk:")
    if chunks:
        c = chunks[0]
        print(f"  ID: {c.id}")
        print(f"  Type: {c.metadata.document_type.value}")
        print(f"  Section: {c.metadata.section_title}")
        print(f"  Tags: {c.metadata.tags}")
        print(f"  Content preview: {c.content[:200]}...")
