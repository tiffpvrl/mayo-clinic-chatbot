"""
Patient context builder.

Transforms pre-procedure raw patient data into a
readable context block that is included in the LLM prompt.
"""

from typing import Dict, Any, List
from datetime import datetime
from src.risk.risk_model import get_risk_score

COMORBIDITY_LABELS = {
    "diabetes": "Diabetes",
    "on_diabetes_medication": "Diabetes medication use",
    "obesity_bmi_ge_30": "Obesity (BMI >= 30)",
    "chronic_constipation": "Chronic constipation",
    "cirrhosis": "Cirrhosis",
    "prior_abdominal_pelvic_surgery": "Prior abdominal/pelvic surgery",
    "dementia": "Dementia",
    "history_of_stroke": "History of stroke",
    "tricyclic_antidepressant_use": "Tricyclic antidepressant use",
    "anticholinergic_use": "Anticholinergic use",
    "hypertension": "Hypertension",
    "hyperlipidemia": "Hyperlipidemia",
    "heart_failure": "Heart failure",
    "ibd_diagnosis": "Inflammatory bowel disease",
    "family_history_crc": "Family history of colorectal cancer",
    "prior_adenoma_or_polyps": "Prior adenoma or polyps",
    "diabetic_gastroparesis": "Diabetic gastroparesis",
    "diabetic_nephropathy": "Diabetic nephropathy",
    "diabetic_retinopathy": "Diabetic retinopathy",
    "gastroparesis": "Gastroparesis",
    "chronic_laxative_use": "Chronic laxative use",
    "parkinsons_disease": "Parkinson's disease",
    "spinal_cord_injury": "Spinal cord injury",
    "neurogenic_bowel": "Neurogenic bowel",
    "prior_colorectal_surgery": "Prior colorectal surgery",
}


def is_positive(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False

def format_value(value: Any, default: str = "Unknown") -> str:
    if value is None:
        return default

    if isinstance(value, str) and value.strip() == "":
        return default

    if isinstance(value, datetime):
        return value.isoformat(timespec="minutes")

    return str(value)

def summarize_comorbidities(patient: Dict[str, Any]) -> List[str]:
    conditions = []

    for column, label in COMORBIDITY_LABELS.items():
        if is_positive(patient.get(column)):
            conditions.append(label)

    constipation_severity = patient.get("constipation_severity")
    if constipation_severity:
        conditions.append(f"Constipation severity: {constipation_severity}")

    ckd_stage = patient.get("ckd_stage")
    if ckd_stage not in (None, "", 0, "0"):
        conditions.append(f"CKD stage: {ckd_stage}")

    sci_type = patient.get("sci_type")
    if sci_type not in (None, "") and is_positive(patient.get("spinal_cord_injury")):
        conditions.append(f"SCI type: {sci_type}")

    opioid_type = patient.get("opioid_type")
    opioid_drug = patient.get("opioid_specific_drugs")
    if is_positive(patient.get("opioid_use_regular")) or is_positive(patient.get("medication_class_opioid")):
        info = " ".join(
            filter(None, [
                str(opioid_type) if opioid_type else None,
                str(opioid_drug) if opioid_drug else None
            ])
        )
        conditions.append(f"Chronic opioid use {info}".strip())

    glp1_drugs = patient.get("glp1_agonist_specific_drugs")
    glp1_indication = patient.get("glp1_indication")
    if glp1_drugs and is_positive(patient.get("medication_class_glp1_agonist")):
        if glp1_indication:
            conditions.append(f"GLP-1 agonist ({glp1_indication}): {glp1_drugs}")
        else:
            conditions.append(f"GLP-1 agonist: {glp1_drugs}")

    laxative_subclass = patient.get("laxative_subclass")
    laxative_drugs = patient.get("laxative_specific_drugs")
    if is_positive(patient.get("medication_class_laxative")):
        if laxative_subclass and laxative_drugs:
            conditions.append(f"Laxative use ({laxative_subclass}): {laxative_drugs}")
        elif laxative_subclass:
            conditions.append(f"Laxative use ({laxative_subclass})")
        elif laxative_drugs:
            conditions.append(f"Laxative use: {laxative_drugs}")

    return conditions

def build_patient_context(patient: Dict[str, Any]) -> str:
    if not patient:
        return "No patient-specific data found."

    risk = get_risk_score(patient)

    comorbidities = summarize_comorbidities(patient)
    comorbidity_block = "\n".join(f"- {item}" for item in comorbidities) if comorbidities else "- None reported"

    current_medications = format_value(patient.get("current_medications"), "None listed")

    encounter_lines = [
        f"- Procedure ID: {format_value(patient.get('procedure_id'))}",
        f"- Colonoscopy indication: {format_value(patient.get('colonoscopy_indication'))}",
        f"- Chief complaint: {format_value(patient.get('chief_complaint'))}",
        f"- Referral source: {format_value(patient.get('referral_source'))}",
        f"- Primary ICD-10 code: {format_value(patient.get('primary_icd10_code'))}",
        f"- Bowel prep start: {format_value(patient.get('bowel_prep_start_datetime'))}",
        f"- Bowel prep end: {format_value(patient.get('bowel_prep_end_datetime'))}",
        f"- Colonoscopy datetime: {format_value(patient.get('colonoscopy_datetime'))}",
    ]

    prep_lines = [
        f"- Regimen type: {format_value(patient.get('regimen_type'))}",
        f"- Prep agent: {format_value(patient.get('prep_agent'))}",
        f"- Diet protocol: {format_value(patient.get('diet_protocol'))}",
        f"- Estimated compliance (%): {format_value(patient.get('estimated_compliance_pct'))}",
        f"- Number of bowel movements: {format_value(patient.get('number_of_bowel_movements'))}",
        f"- Stool character at end of prep: {format_value(patient.get('stool_character_end_of_prep'))}",
    ]
    volume = patient.get("volume_liters")
    if volume is not None:
        prep_lines.append(f"- Prep volume: {volume} L")
    if patient.get("adjuncts") not in (None, "", "None"):
        prep_lines.append(f"- Adjuncts: {patient['adjuncts']}")
    if patient.get("dietary_restriction") not in (None, "", "None"):
        prep_lines.append(f"- Dietary restriction: {patient['dietary_restriction']}")
    if patient.get("prep_modifications") not in (None, "", "None"):
        prep_lines.append(f"- Prep modifications: {patient['prep_modifications']}")

    if is_positive(patient.get("nausea_during_prep")):
        prep_lines.append("- Nausea during prep")
    if is_positive(patient.get("vomiting_during_prep")):
        prep_lines.append("- Vomiting during prep")
    if is_positive(patient.get("abdominal_cramps_during_prep")):
        prep_lines.append("- Abdominal cramps during prep")
    if is_positive(patient.get("dizziness_during_prep")):
        prep_lines.append("- Dizziness during prep")

    prior_colonoscopy_lines = []

    if is_positive(patient.get("prior_colonoscopy_flag")):
        prior_colonoscopy_lines.append("- Prior colonoscopy: Yes")

    if patient.get("prior_colonoscopy_date"):
        prior_colonoscopy_lines.append(
            f"- Prior colonoscopy date: {format_value(patient['prior_colonoscopy_date'])}"
        )

    if patient.get("years_since_prior_colonoscopy") is not None:
        prior_colonoscopy_lines.append(
            f"- Years since prior colonoscopy: {patient['years_since_prior_colonoscopy']}"
        )

    if patient.get("prior_bbps_total") is not None:
        prior_colonoscopy_lines.append(
            f"- Prior BBPS total: {patient['prior_bbps_total']}"
        )

    if patient.get("prior_prep_adequate_flag") is not None:
        adequate = "Yes" if is_positive(patient.get("prior_prep_adequate_flag")) else "No"
        prior_colonoscopy_lines.append(f"- Prior prep adequate: {adequate}")

    if is_positive(patient.get("prior_extended_prep_required")):
        prior_colonoscopy_lines.append("- Prior extended prep required")

    if patient.get("prior_prep_attempts_count") is not None:
        prior_colonoscopy_lines.append(
            f"- Prior prep attempts: {patient['prior_prep_attempts_count']}"
        )

    prior_section = ""
    if prior_colonoscopy_lines:
        prior_section = f"""
PRIOR COLONOSCOPY HISTORY
{chr(10).join(prior_colonoscopy_lines)}
        """

    return f"""
RISK ASSESSMENT
- Risk tier for inadequate bowel prep: {risk['risk_tier']}

PATIENT PROFILE
- Age: {format_value(patient.get("age_at_colonoscopy"))}
- Sex at birth: {format_value(patient.get("sex_at_birth"))}
- Gender identity: {format_value(patient.get("gender_identity"))}
- Preferred language: {format_value(patient.get("preferred_language"))}
- BMI: {format_value(patient.get("bmi"))}
- Smoking status: {format_value(patient.get("smoking_status"))}
- Alcohol use: {format_value(patient.get("alcohol_use"))}
- Mobility status: {format_value(patient.get("mobility_status"))}
- High risk flag: {format_value(patient.get("high_risk_flag"))}

COMORBIDITIES AND RELEVANT HISTORY
{comorbidity_block}

CURRENT MEDICATIONS
- {current_medications}

ENCOUNTER DETAILS
{chr(10).join(encounter_lines)}

PREP DETAILS
{chr(10).join(prep_lines)}
{prior_section}
""".strip()