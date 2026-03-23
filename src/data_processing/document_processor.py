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


# Audience / provenance for PDF chunks (see pdf_manifest.json + infer_default_pdf_manifest_entry)
AUDIENCE_TIER_PATIENT_CARE = "patient_care"
AUDIENCE_TIER_CLINICIAN_GUIDELINE = "clinician_guideline"
AUDIENCE_TIER_RESEARCH_EDUCATION = "research_education"

# Cached manifest by resolved base_dir
_pdf_manifest_cache: dict[str, dict[str, Any]] = {}


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
    "patient_information",
    "precautions",
}

# Map alternate SPL section keys to a canonical name (DailyMed vs OpenFDA drift)
DRUG_SECTION_KEY_ALIASES: dict[str, str] = {
    "warnings": "warnings_and_precautions",
}


def _normalize_drug_label_sections(sections: dict[str, Any]) -> dict[str, Any]:
    """Merge aliased keys (e.g. warnings -> warnings_and_precautions) for consistent RAG sections."""
    out: dict[str, Any] = {}
    for key, val in sections.items():
        if not isinstance(val, dict):
            continue
        canon = DRUG_SECTION_KEY_ALIASES.get(key, key)
        if canon in out:
            prev = out[canon]
            if isinstance(prev, dict):
                oc = prev.get("content", "")
                nc = val.get("content", "")
                out[canon] = {
                    "title": prev.get("title") or val.get("title") or canon.replace("_", " ").title(),
                    "content": f"{oc}\n\n{nc}".strip(),
                }
        else:
            out[canon] = dict(val)
    return out

# Medical conditions relevant to bowel prep (for contextual tagging)
RELEVANT_CONDITIONS = {
    "renal_disease",
    "kidney_disease",
    "CKD",
    "ESRD",
    "diabetes",
    "G6PD_deficiency",
    "phenylketonuria",
    "PKU",
    "ulcerative_colitis",
    "Crohns_disease",
    "IBD",
    "ileus",
    "bowel_obstruction",
    "gastric_retention",
    "bariatric_surgery",
    "heart_failure",
    "arrhythmia",
    "hypertension",
    "geriatric",
    "elderly",
    "pediatric",
    "children",
    "pregnancy",
    "lactation",
    "breastfeeding",
    "anticoagulation",
    "antiplatelet",
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

# Topic / routing tags for filtered retrieval
# NOTE: Some tags are simple topics (e.g., "diet"), others act more like facets
# and are emitted with prefixes at runtime (e.g., "procedure_time:morning").
TOPIC_TAGS = {
    # Core topics
    "dosing",
    "timing",
    "diet",
    "clear_liquids",
    "side_effects",
    "contraindications",
    "drug_interactions",
    "medication_management",
    "split_dose",
    "same_day",
    "transportation",
    "preparation_steps",
    "what_to_expect",
    "when_to_contact_provider",
    # Patient risk / history
    "constipation_history",
    "prior_poor_prep",
    "dehydration_risk",
    "electrolyte_risk",
    "phenylalanine_content",
    "phosphate_risk",
    "seizure_risk",
    # Procedure / indication context
    "quality_indicator",
    "surveillance_interval",
    "scoring_system",
    # Faceted families (actual emitted tags may use prefixes, e.g. procedure_time:morning)
    "procedure_time",
    "setting",
    "indication",
    "regimen_pattern",
    "diet_pattern",
    "holds",
    "diabetes_meds",
    "adjunct",
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
    # Drug label provenance / RAG routing (from SPL + prep_agents rag_metadata)
    label_source: str | None = None  # "DailyMed" | "OpenFDA"
    canonical_section_key: str | None = None  # After aliasing, e.g. warnings_and_precautions
    labeled_for_colonoscopy_prep: bool | None = None
    rag_product_note: str | None = None
    # PDF provenance (pdf_manifest.json + filename inference)
    audience_tier: str | None = None  # patient_care | clinician_guideline | research_education
    source_category: str | None = None  # society_guideline | hospital | trial | meta_analysis | patient_handout | other
    content_use_policy: str | None = None  # e.g. research_background for trials/meta-analyses


@dataclass
class ProcessedChunk:
    """A semantically coherent chunk ready for embedding and RAG."""
    id: str
    content: str
    metadata: ChunkMetadata


def _stem_for_routing(file_path: Path) -> str:
    """Lowercase filename stem for routing (no extension)."""
    return file_path.stem.lower()


def _prep_token_match(stem: str) -> bool:
    """True if stem suggests bowel prep (token-level), not arbitrary 'prep' substrings."""
    if re.search(r"\bbowel\s*prep\b", stem) or "bowelprep" in stem:
        return True
    if re.search(r"\bprep\b", stem) and any(
        x in stem for x in ("bowel", "colon", "ucsf", "2day", "2-day", "split", "guide", "boston")
    ):
        return True
    if "_prep" in stem or stem.endswith("prepguide") or "prepguide" in stem:
        return True
    if re.match(r"^typeofbowelprep\d*$", stem):
        return True
    return False


def _colonoscopy_instruction_cue(stem: str) -> bool:
    """Colonoscopy in filename is a patient-instruction cue only with instructional context."""
    if "colonoscopy" not in stem:
        return False
    instructional = (
        "instruction",
        "instructions",
        "prep",
        "booklet",
        "2day",
        "2-day",
        "ucsf",
        "patient",
        "bowel",
    )
    return any(x in stem for x in instructional) or _prep_token_match(stem)


def _society_or_hospital_stem(stem: str) -> bool:
    tokens = (
        "usmstf",
        "asge",
        "mayoclinic",
        "massgeneral",
        "clevelandclinic",
        "ucsf",
        "bostonbowelprep",
    )
    return any(t in stem for t in tokens)


def _research_or_quality_denylist_stem(stem: str) -> bool:
    """Stems that must not be classified as PATIENT_INSTRUCTIONS from filename alone."""
    if stem.startswith("clinicaltrial") or stem.startswith("metaanalysis"):
        return True
    if re.match(r"^study\d*$", stem):
        return True
    patterns = (
        "colonoscopyquality",
        "typeofbowelprep",
        "adenoma",
        "missedwork",
    )
    return any(p in stem for p in patterns)


def detect_document_type(file_path: Path, content_preview: str = "") -> DocumentType:
    """Classify document type from path patterns and content cues."""
    path_str = str(file_path).lower()
    stem = _stem_for_routing(file_path)
    preview = content_preview.lower()

    # Drug labels: JSON with sections or path containing drug_labels
    if "drug_label" in path_str or "dailymed" in path_str or "openfda" in path_str:
        return DocumentType.DRUG_LABEL
    if file_path.suffix == ".json" and ("consolidated" in path_str or "dailymed_json" in path_str or "openfda" in path_str):
        return DocumentType.DRUG_LABEL

    # Clinical guidelines: taskforce, consensus, surveillance, quality indicators
    guideline_keywords = [
        "taskforce",
        "task_force",
        "consensus",
        "guideline",
        "surveillance",
        "quality_indicators",
        "qualityindicator",
    ]
    if any(kw in stem for kw in guideline_keywords):
        return DocumentType.CLINICAL_GUIDELINE

    # Research / quality stems → guideline (not patient prep handout)
    if _research_or_quality_denylist_stem(stem):
        return DocumentType.CLINICAL_GUIDELINE

    # Society / hospital filename hints → guideline
    if _society_or_hospital_stem(stem):
        return DocumentType.CLINICAL_GUIDELINE

    deny_instruction_from_name = _research_or_quality_denylist_stem(stem)

    # Patient instructions: stricter prep/colonoscopy matching
    if not deny_instruction_from_name:
        if _prep_token_match(stem):
            return DocumentType.PATIENT_INSTRUCTIONS
        if any(
            kw in stem
            for kw in ("instruction", "instructions", "booklet", "handout", "patient")
        ):
            return DocumentType.PATIENT_INSTRUCTIONS
        if _colonoscopy_instruction_cue(stem):
            return DocumentType.PATIENT_INSTRUCTIONS
        if "bowel" in stem and ("prep" in stem or "preparation" in stem):
            return DocumentType.PATIENT_INSTRUCTIONS
        if "2day" in stem or "2_day" in stem or "extended" in stem:
            return DocumentType.PATIENT_INSTRUCTIONS

    # Content-based fallback
    if "indications and usage" in preview or "dosage and administration" in preview:
        return DocumentType.DRUG_LABEL
    if "we recommend" in preview or "consensus" in preview:
        return DocumentType.CLINICAL_GUIDELINE
    if "days before" in preview or "stop:" in preview or "ok/approved" in preview:
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


def reset_pdf_manifest_cache() -> None:
    """Clear cached pdf_manifest.json loads (for tests or multiple KB roots)."""
    _pdf_manifest_cache.clear()


def load_pdf_manifest(base_dir: Path) -> dict[str, Any]:
    """Load patient_kb/pdf_manifest.json once per base_dir (cached)."""
    key = str(Path(base_dir).resolve())
    if key in _pdf_manifest_cache:
        return _pdf_manifest_cache[key]
    manifest_path = Path(base_dir) / "pdf_manifest.json"
    data: dict[str, Any] = {"version": 1, "files": {}}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = {**data, **loaded}
            if not isinstance(data.get("files"), dict):
                data["files"] = {}
    _pdf_manifest_cache[key] = data
    return data


def infer_default_pdf_manifest_entry(stem: str) -> dict[str, Any]:
    """Default audience/source when a stem is absent from pdf_manifest.json files map."""
    s = stem.lower()
    if s.startswith("clinicaltrial") or s.startswith("metaanalysis") or re.match(r"^study\d*$", s):
        return {"audience_tier": AUDIENCE_TIER_RESEARCH_EDUCATION, "source_type": "trial_or_meta"}
    if any(x in s for x in ("usmstf", "asge")):
        return {"audience_tier": AUDIENCE_TIER_CLINICIAN_GUIDELINE, "source_type": "society_guideline"}
    if any(x in s for x in ("mayoclinic", "massgeneral", "clevelandclinic", "ucsf", "bostonbowelprep")):
        return {"audience_tier": AUDIENCE_TIER_CLINICIAN_GUIDELINE, "source_type": "hospital"}
    if "colonoscopyquality" in s or "typeofbowelprep" in s:
        return {"audience_tier": AUDIENCE_TIER_CLINICIAN_GUIDELINE, "source_type": "educational"}
    if any(x in s for x in ("pregnant", "crohns", "ulcerative", "bariatric")):
        return {"audience_tier": AUDIENCE_TIER_PATIENT_CARE, "source_type": "patient_handout"}
    return {"audience_tier": AUDIENCE_TIER_CLINICIAN_GUIDELINE, "source_type": "other"}


def _normalize_source_category(stem: str, source_type: str) -> str:
    """Map manifest source_type to a stable source_category for export."""
    st = (source_type or "other").lower()
    if st == "trial_or_meta":
        sl = stem.lower()
        if sl.startswith("metaanalysis"):
            return "meta_analysis"
        if sl.startswith("clinicaltrial") or re.match(r"^study\d*$", sl):
            return "trial"
        return "other"
    mapping = {
        "society_guideline": "society_guideline",
        "hospital": "hospital",
        "trial": "trial",
        "meta_analysis": "meta_analysis",
        "patient_handout": "patient_handout",
        "educational": "other",
        "other": "other",
    }
    return mapping.get(st, "other")


def resolve_pdf_manifest_entry(stem: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Merge pdf_manifest.json overrides with infer_default_pdf_manifest_entry."""
    files = manifest.get("files") or {}
    override = files.get(stem.lower())
    if not isinstance(override, dict):
        override = {}
    defaults = infer_default_pdf_manifest_entry(stem)
    merged = {**defaults, **override}
    tier = merged.get("audience_tier")
    if tier not in (AUDIENCE_TIER_PATIENT_CARE, AUDIENCE_TIER_CLINICIAN_GUIDELINE, AUDIENCE_TIER_RESEARCH_EDUCATION):
        merged["audience_tier"] = defaults["audience_tier"]
    return merged


def apply_pdf_manifest_to_metadata(meta: ChunkMetadata, stem: str, pdf_entry: dict[str, Any]) -> None:
    """Set audience/source fields and deterministic tags for research-background PDFs."""
    tier = pdf_entry.get("audience_tier") or AUDIENCE_TIER_CLINICIAN_GUIDELINE
    if tier not in (AUDIENCE_TIER_PATIENT_CARE, AUDIENCE_TIER_CLINICIAN_GUIDELINE, AUDIENCE_TIER_RESEARCH_EDUCATION):
        tier = AUDIENCE_TIER_CLINICIAN_GUIDELINE
    meta.audience_tier = tier
    meta.source_category = _normalize_source_category(stem, str(pdf_entry.get("source_type") or "other"))
    if tier == AUDIENCE_TIER_RESEARCH_EDUCATION:
        meta.content_use_policy = "research_background"
        meta.tags.add("content_policy:research_background")
        meta.tags.add("audience:research_education")


def tag_content(content: str) -> set[str]:
    """Extract contextual tags (conditions, medication classes, topics) from content."""
    tags = set()
    text_lower = content.lower()

    # Medical conditions / risk profiles
    condition_patterns = [
        (r"\b(renal|kidney|ckd|esrd)\b", "renal_disease"),
        (r"\b(diabete|g6pd|phenylketonuria|pku)\b", "metabolic_conditions"),
        (r"\b(ulcerative colitis|ileus|obstruction)\b", "gi_conditions"),
        (r"\b(crohn'?s|crohn's disease)\b", "crohns_disease"),
        (r"\binflammatory bowel disease\b|\bibd\b", "ibd"),
        (r"\b(pregnancy|pregnant|lactation|breastfeed)\b", "pregnancy_lactation"),
        (r"\b(bariatric|gastric bypass|gastric sleeve|roux[- ]en[- ]y|duodenal switch)\b", "bariatric_surgery"),
        (r"\b(warfarin|anticoagulant|anticoagulation|coumadin|apixaban|rivaroxaban|dabigatran|heparin|enoxaparin|clopidogrel|antiplatelet)\b", "anticoagulation_context"),
        (r"\b(geriatric|elderly|age 60|over 60)\b", "geriatric"),
        (r"\b(pediatric|children|child)\b", "pediatric"),
        (r"\b(heart failure|congestive heart failure|chf)\b", "heart_failure"),
        (r"\b(cirrhosis|cirrhotic|liver disease)\b", "cirrhosis"),
        (r"\b(chronic constipation|constipation)\b", "constipation_history"),
        (r"\b(previous|prior)\s+(poor|inadequate)\s+(prep|preparation|bowel preparation)\b", "prior_poor_prep"),
        (r"\b(frail|frailty|limited mobility|walker|wheelchair)\b", "mobility_frailty"),
        (r"\b(sleep apnea|osa|cpap)\b", "sleep_apnea_obesity"),
    ]
    for pattern, tag in condition_patterns:
        if re.search(pattern, text_lower):
            tags.add(tag)

    # Medication classes
    for med_class, drug_list in MEDICATION_CLASSES.items():
        if any(drug in text_lower for drug in drug_list):
            tags.add(f"med_class:{med_class}")

    # Topics / routing facets
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
        # Risk / safety
        (r"\b(dehydration|dehydrate)\b", "dehydration_risk"),
        (r"\b(electrolyte|hyponatremia|hypokalemia|hyperphosphatemia)\b", "electrolyte_risk"),
        (r"\b(phenylalanine|aspartame|pku)\b", "phenylalanine_content"),
        (r"\b(sodium phosphate|phosphate nephropathy)\b", "phosphate_risk"),
        (r"\b(seizure|tonic[- ]clonic)\b", "seizure_risk"),
        # Guideline / quality
        (r"\b(quality indicator|quality metrics|adequate bowel preparation)\b", "quality_indicator"),
        (r"\b(Boston Bowel Preparation Scale|BBPS)\b", "scoring_system"),
        (r"\bsurveillance interval\b", "surveillance_interval"),
        (r"\b(1[- ]year|3[- ]year|5[- ]year|10[- ]year)\b.*\b(surveillance|follow[- ]?up)\b", "surveillance_interval"),
        # Procedure context
        (r"\b(morning colonoscopy|morning appointment)\b", "procedure_time:morning"),
        (r"\b(afternoon colonoscopy|afternoon appointment)\b", "procedure_time:afternoon"),
        (r"\b(inpatient)\b", "setting:inpatient"),
        (r"\b(outpatient|ambulatory)\b", "setting:outpatient"),
        (r"\b(screening colonoscopy)\b", "indication:screening"),
        (r"\b(surveillance colonoscopy|post[- ]polypectomy|postpolypectomy|post resection|after colorectal cancer)\b", "indication:surveillance"),
        (r"\b(diagnostic colonoscopy)\b", "indication:diagnostic"),
        (r"\b(first colonoscopy|index colonoscopy)\b", "first_colonoscopy"),
        (r"\b(inflammatory bowel disease|ibd)\b.*\b(surveillance|colonoscopy)\b", "ibd_surveillance"),
        (r"\b(colorectal cancer resection|after crc resection|post[- ]crc)\b", "surveillance_after_crc"),
        # Regimen & diet patterns
        (r"\b(two[- ]day prep|2[- ]day prep|2 day prep)\b", "regimen_pattern:2_day"),
        (r"\b(clear liquid diet\b|\bclear liquids for 2 days\b)\b", "diet_pattern:clear_liquid_multi_day"),
        (r"\b(low[- ]residue diet|low[- ]fiber diet)\b", "diet_pattern:low_residue"),
        (r"\b4\s*(liter|l)\b", "volume_type:high"),
        (r"\b2\s*(liter|l)\b", "volume_type:low"),
        # Medication holding & diabetes regimens
        (r"\b(iron supplement|iron tablet|ferrous)\b", "holds:iron"),
        (r"\b(nsaid|nonsteroidal anti[- ]inflammatory)\b", "holds:nsaids"),
        (r"\b(herbal supplement|herbal medicine|ginkgo|ginseng|st\.?\s*john['’]s wort)\b", "holds:herbal_supplements"),
        (r"\b(insulin)\b", "diabetes_meds:insulin"),
        (r"\b(metformin|sulfonylurea|glipizide|glyburide|dpp-4|sglt2)\b", "diabetes_meds:oral_agents"),
        # Adjunctive agents
        (r"\b(simethicone)\b", "adjunct:simethicone"),
        (r"\b(ondansetron|zofran|antiemetic)\b", "adjunct:antiemetic"),
    ]
    for pattern, tag in topic_patterns:
        if re.search(pattern, text_lower):
            tags.add(tag)

    # Derived regimen patterns from simpler tags
    if "split_dose" in tags:
        tags.add("regimen_pattern:split_dose")
    if "same_day" in tags:
        tags.add("regimen_pattern:same_day")

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
    sections = _normalize_drug_label_sections(doc.get("sections", {}))
    set_id = doc.get("set_id", "") or ""
    label_source = doc.get("source", "DailyMed")
    rag = doc.get("rag_metadata") or {}
    labeled_prep = rag.get("labeled_for_colonoscopy_prep")
    if labeled_prep is None:
        labeled_prep = True
    rag_note = rag.get("product_note")

    id_slug = set_id[:8] if len(set_id) >= 8 else (set_id or label_source.replace(" ", "")[:8] or "label")

    base_metadata = ChunkMetadata(
        source_file=str(source_path),
        document_type=doc_type,
        drug_name=drug_name,
        organization=label_source,
        label_source=label_source,
        labeled_for_colonoscopy_prep=labeled_prep,
        rag_product_note=rag_note,
    )

    for section_key, section_data in sections.items():
        if not isinstance(section_data, dict):
            continue
        title = section_data.get("title", section_key.replace("_", " ").title())
        content = section_data.get("content", "")
        if not content or not content.strip():
            continue

        # Split very long sections by double newlines (paragraphs), max ~800 chars per sub-chunk
        sub_contents = _split_long_section(content, max_chars=800)

        for sub_idx, sub_content in enumerate(sub_contents):
            chunk_id = f"drug_{drug_name}_{id_slug}_{section_key}_{sub_idx}"
            tags = tag_content(sub_content)
            if labeled_prep is False:
                tags.add("product_context:generic_peg_not_prep_specific")
            tags.add(f"label_source:{label_source}")

            meta = ChunkMetadata(
                source_file=base_metadata.source_file,
                document_type=doc_type,
                section_title=title,
                section_hierarchy=[drug_name, title],
                drug_name=drug_name,
                organization=base_metadata.organization,
                chunk_index=len(chunks),
                tags=tags,
                label_source=label_source,
                canonical_section_key=section_key,
                labeled_for_colonoscopy_prep=labeled_prep,
                rag_product_note=rag_note,
            )
            chunks.append(ProcessedChunk(id=chunk_id, content=sub_content.strip(), metadata=meta))

    # Update total_chunks
    for c in chunks:
        c.metadata.total_chunks = len(chunks)

    return chunks


def _strip_references_section(content: str) -> str:
    """
    Remove the REFERENCES / BIBLIOGRAPHY section from clinical guideline content.
    Citation lists add noise and can confuse the RAG chatbot.
    """
    ref_pattern = re.compile(
        r"\n\s*(REFERENCES|References|BIBLIOGRAPHY|Bibliography|REFERENCES AND NOTES)\s*\n",
        re.IGNORECASE,
    )
    match = ref_pattern.search(content)
    if match:
        return content[: match.start()].rstrip()
    return content


def chunk_clinical_guideline(
    content: str,
    source_path: Path,
    pdf_entry: dict[str, Any] | None = None,
) -> list[ProcessedChunk]:
    """
    Chunk clinical guidelines by section headers.
    Looks for patterns like INTRODUCTION, METHODS, RECOMMENDATIONS, Table N, etc.
    Excludes the REFERENCES section to avoid confusing the chatbot.
    """
    content = _strip_references_section(content)

    chunks = []
    doc_type = DocumentType.CLINICAL_GUIDELINE
    year = extract_publication_year(source_path, content)
    org = extract_organization(source_path, content)

    # Section header pattern: ALL CAPS lines, numbered sections (1. 2. 3.), or "Table N"
    section_pattern = re.compile(
        r"^(?:(?:Table\s+\d+[.:]?\s*)|(?:[A-Z][A-Z\s\-]+(?:\d+[.:])?)|(?:\d+\.\s+[A-Z][A-Za-z\s]+))\s*$",
        re.MULTILINE,
    )

    # Simpler: split on lines that look like headers (ALL CAPS, or "1. Title", or "Table N")
    lines = content.split("\n")
    current_section_title = "Introduction"
    current_content: list[str] = []
    section_titles: list[str] = []
    _table_header_re = re.compile(r"^TABLE\s+\d+", re.IGNORECASE)

    def is_table_section(title: str) -> bool:
        return bool(_table_header_re.match(title.strip()))

    def flush_section(title: str, text: str):
        if not text.strip():
            return
        # Keep tables in a single chunk so row/column relationships aren't split
        if is_table_section(title):
            max_chars = 4000
        else:
            max_chars = 600
        sub_chunks = _split_long_section(text, max_chars=max_chars)
        for i, sub in enumerate(sub_chunks):
            chunk_id = f"guideline_{source_path.stem}_{len(chunks):03d}"
            tags = tag_content(sub)
            if is_table_section(title):
                tags.add("content_type:table")
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
            if pdf_entry:
                apply_pdf_manifest_to_metadata(meta, source_path.stem, pdf_entry)
            chunks.append(ProcessedChunk(id=chunk_id, content=sub.strip(), metadata=meta))

    for line in lines:
        stripped = line.strip()
        # Header heuristic: ALL CAPS, "1. SECTION NAME", or "Table N" / "TABLE N. Title"
        is_header = (
            len(stripped) >= 4
            and stripped.isupper()
            and len(stripped) < 80
            and not stripped.endswith(".")
        ) or bool(re.match(r"^\d+\.\s+[A-Z]", stripped)) or bool(_table_header_re.match(stripped))

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
    pdf_entry: dict[str, Any] | None = None,
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
            if pdf_entry:
                apply_pdf_manifest_to_metadata(meta, source_path.stem, pdf_entry)
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


def _clean_pdf_page_text(page_text: str) -> str:
    """
    Heuristically remove headers, footers, and page numbers from a single PDF page.
    This targets things like journal headers, URLs, and standalone page numbers.
    """
    if not page_text:
        return ""

    cleaned_lines: list[str] = []
    for line in page_text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue

        # Standalone page numbers or simple ranges (e.g., "719" or "719-721")
        if re.fullmatch(r"\d+(-\d+)?", stripped):
            continue

        # Common header/footer patterns: URLs, volume/issue, journal names
        if re.search(r"https?://\S+", stripped) or re.search(r"www\.[\w\.-]+", stripped, re.IGNORECASE):
            continue
        if re.search(r"\b(volume|vol\.|no\.|issue)\b", stripped, re.IGNORECASE):
            continue
        if re.search(r"\bGASTROINTESTINAL ENDOSCOPY\b", stripped, re.IGNORECASE):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def load_pdf_text(path: Path) -> str:
    """Extract text from PDF using PyPDF2 or pypdf."""
    if PdfReader is None:
        raise ImportError(
            "PDF processing requires PyPDF2 or pypdf. Install with: pip install PyPDF2"
        )
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        pages.append(_clean_pdf_page_text(raw))
    return "\n\n".join(p for p in pages if p)


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

    pdf_entry = None
    if suffix == ".pdf":
        manifest = load_pdf_manifest(base_dir)
        pdf_entry = resolve_pdf_manifest_entry(path.stem, manifest)

    if doc_type == DocumentType.CLINICAL_GUIDELINE:
        return chunk_clinical_guideline(content, path, pdf_entry=pdf_entry)
    if doc_type == DocumentType.PATIENT_INSTRUCTIONS:
        return chunk_patient_instructions(content, path, pdf_entry=pdf_entry)
    if doc_type == DocumentType.DRUG_LABEL:
        # PDF drug label - treat as patient-ish for now
        return chunk_patient_instructions(content, path, pdf_entry=pdf_entry)

    # Fallback: semantic split by paragraphs
    return chunk_clinical_guideline(content, path, pdf_entry=pdf_entry)


def build_processing_summary(chunks: list[ProcessedChunk]) -> dict[str, Any]:
    """
    Build a summary dict from processed chunks for reporting and debugging.
    Matches the structure of processing_summary.json.
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "chunks_by_document_type": {},
            "chunks_by_source_file": {},
            "chunks_by_drug": {},
            "chunks_by_audience_tier": {},
            "chunks_by_content_use_policy": {},
            "tag_frequencies": {},
            "content_length": {"min": 0, "max": 0, "avg": 0.0},
            "unique_source_files": 0,
        }

    by_doc_type: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_drug: dict[str, int] = {}
    by_audience: dict[str, int] = {}
    by_policy: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    lengths: list[int] = []

    for c in chunks:
        dt = c.metadata.document_type.value
        by_doc_type[dt] = by_doc_type.get(dt, 0) + 1

        source_name = Path(c.metadata.source_file).name
        by_source[source_name] = by_source.get(source_name, 0) + 1

        if c.metadata.drug_name:
            by_drug[c.metadata.drug_name] = by_drug.get(c.metadata.drug_name, 0) + 1

        if c.metadata.audience_tier:
            by_audience[c.metadata.audience_tier] = by_audience.get(c.metadata.audience_tier, 0) + 1
        if c.metadata.content_use_policy:
            pol = c.metadata.content_use_policy
            by_policy[pol] = by_policy.get(pol, 0) + 1

        for tag in c.metadata.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        lengths.append(len(c.content))

    n = len(lengths)
    return {
        "total_chunks": n,
        "chunks_by_document_type": dict(sorted(by_doc_type.items())),
        "chunks_by_source_file": dict(sorted(by_source.items(), key=lambda x: -x[1])),
        "chunks_by_drug": dict(sorted(by_drug.items(), key=lambda x: -x[1])),
        "chunks_by_audience_tier": dict(sorted(by_audience.items())),
        "chunks_by_content_use_policy": dict(sorted(by_policy.items())),
        "tag_frequencies": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
        "content_length": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": round(sum(lengths) / n, 1),
        },
        "unique_source_files": len(by_source),
    }


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
                    "label_source": c.metadata.label_source,
                    "canonical_section_key": c.metadata.canonical_section_key,
                    "labeled_for_colonoscopy_prep": c.metadata.labeled_for_colonoscopy_prep,
                    "rag_product_note": c.metadata.rag_product_note,
                    "audience_tier": c.metadata.audience_tier,
                    "source_category": c.metadata.source_category,
                    "content_use_policy": c.metadata.content_use_policy,
                },
            }
            for c in all_chunks
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

        # Write processing summary alongside chunks (same directory)
        summary_path = output_path.parent / "processing_summary.json"
        summary = build_processing_summary(all_chunks)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Summary saved to: {summary_path}")

    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    kb = Path("src/data_processing/patient_kb")
    out = Path("src/data_processing/patient_kb/processed_chunks/processed_chunks.json")

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