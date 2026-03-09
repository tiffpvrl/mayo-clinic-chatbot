"""
Patient context builder.

Transforms pre-procedure raw patient data into a
readable context block that is included in the LLM prompt.
"""

from typing import Dict, Any, List


COMORBIDITY_LABELS = {
    "diabetes": "Diabetes",
    "on_diabetes_medication": "Diabetes medication use",
    "obesity_bmi_ge_30": "Obesity (BMI >= 30)",
    "chronic_constipation": "Chronic constipation",
    "cirrhosis": "Cirrhosis",
    "prior_abdominal_pelvic_surgery": "Prior abdominal/pelvic surgery",
    "dementia": "Dementia",
    "history_of_stroke": "History of stroke",
    "opioid_use_regular": "Regular opioid use",
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
    "medication_class_opioid": "Opioid medication",
    "medication_class_glp1_agonist": "GLP-1 agonist medication",
    "medication_class_laxative": "Laxative medication",
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
        conditions.append(f"CKD stage {ckd_stage}")

    sci_type = patient.get("sci_type")
    if sci_type not in (None, "") and is_positive(patient.get("spinal_cord_injury")):
        conditions.append(f"SCI type: {sci_type}")

    opioid_type = patient.get("opioid_type")
    opioid_drug = patient.get("opioid_specific_drugs")
    if is_positive(patient.get("opioid_use_regular")) or is_positive(patient.get("medication_class_opioid")):
        info = " ".join(filter(None, [opioid_type, opioid_drug]))
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

    comorbidities = summarize_comorbidities(patient)
    comorbidity_block = "\n".join(f"- {c}" for c in comorbidities) if comorbidities else "None reported"
    current_medications = patient.get("current_medications") or "None listed"

    return f"""
PATIENT PROFILE
Age: {patient.get("age_at_colonoscopy")}
Sex: {patient.get("sex_at_birth")}
Gender Identity: {patient.get("gender_identity")}
BMI: {patient.get("bmi")}
Smoking Status: {patient.get("smoking_status")}
Alcohol Use: {patient.get("alcohol_use")}
Mobility status: {patient.get("mobility_status")}
High risk flag: {patient.get("high_risk_flag")}

COMORBIDITIES
{comorbidity_block}

CURRENT MEDICATIONS
{current_medications}
""".strip()