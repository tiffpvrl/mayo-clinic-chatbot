"""
BigQuery access layer.

Fetches patient-level data using patient_id and returns a structured
record used to personalize responses.
"""

import google.auth
from google.cloud import bigquery
from typing import Optional, Dict, Any

# Create the client once at import time — before vertexai.init() runs in main.py
# (main.py imports this module on line 17, vertexai.init() fires on line 21).
# Building the client early means the credentials object is frozen before vertexai
# can attach a quota_project to the global auth state, preventing the
# x-goog-user-project header that triggers serviceusage.services.use checks.
_creds, _ = google.auth.default()
_creds = _creds.with_quota_project(None)
_BQ_CLIENT = bigquery.Client(project="industrial-net-487818-h9", credentials=_creds)

_QUERY = """
    SELECT
        p.patient_id,
        p.patient_name,
        p.sex_at_birth,
        p.gender_identity,
        p.age_at_colonoscopy,
        p.preferred_language,
        p.bmi,
        p.smoking_status,
        p.alcohol_use,
        p.mobility_status,
        p.high_risk_flag,

        c.comorbidity_icd10_codes,
        c.comorbidity_descriptions,
        c.current_medications,
        c.diabetes,
        c.on_diabetes_medication,
        c.obesity_bmi_ge_30,
        c.chronic_constipation,
        c.cirrhosis,
        c.prior_abdominal_pelvic_surgery,
        c.dementia,
        c.history_of_stroke,
        c.opioid_use_regular,
        c.tricyclic_antidepressant_use,
        c.anticholinergic_use,
        c.hypertension,
        c.hyperlipidemia,
        c.ckd_stage,
        c.heart_failure,
        c.ibd_diagnosis,
        c.family_history_crc,
        c.prior_adenoma_or_polyps,
        c.diabetic_gastroparesis,
        c.diabetic_nephropathy,
        c.diabetic_retinopathy,
        c.gastroparesis,
        c.constipation_severity,
        c.chronic_laxative_use,
        c.parkinsons_disease,
        c.spinal_cord_injury,
        c.sci_type,
        c.neurogenic_bowel,
        c.prior_colorectal_surgery,
        c.medication_class_opioid,
        c.opioid_type,
        c.opioid_specific_drugs,
        c.medication_class_glp1_agonist,
        c.glp1_indication,
        c.glp1_agonist_specific_drugs,
        c.medication_class_laxative,
        c.laxative_subclass,
        c.laxative_specific_drugs,

        e.procedure_id,
        e.referral_source,
        e.bowel_prep_start_datetime,
        e.bowel_prep_end_datetime,
        e.colonoscopy_datetime,
        e.colonoscopy_indication,
        e.primary_icd10_code,
        e.chief_complaint,

        pd.regimen_type,
        pd.prep_agent,
        pd.volume_liters,
        pd.adjuncts,
        pd.diet_protocol,
        pd.dietary_restriction,
        pd.prep_modifications,
        pd.number_of_bowel_movements,
        pd.stool_character_end_of_prep,
        pd.nausea_during_prep,
        pd.vomiting_during_prep,
        pd.abdominal_cramps_during_prep,
        pd.dizziness_during_prep,

        h.prior_colonoscopy_flag,
        h.prior_colonoscopy_date,
        h.prior_bbps_total,
        h.prior_bbps_right,
        h.prior_bbps_transverse,
        h.prior_bbps_left,
        h.prior_prep_adequate_flag,
        h.prior_prep_regimen,
        h.prior_complications,
        h.years_since_prior_colonoscopy,
        h.prior_prep_inadequate_verbatim,
        h.prior_extended_prep_required,
        h.prior_prep_attempts_count,

        rs.risk_tier,
        rs.predicted_inadequate_risk,
        rs.model_version,
        rs.scored_at
    FROM `industrial-net-487818-h9.pre_procedure_data.Patients` AS p
    LEFT JOIN `industrial-net-487818-h9.pre_procedure_data.Comorbidities` AS c
        ON p.patient_id = c.patient_id
    LEFT JOIN `industrial-net-487818-h9.pre_procedure_data.Encounters` AS e
        ON p.patient_id = e.patient_id
    LEFT JOIN `industrial-net-487818-h9.pre_procedure_data.Prep_Details` AS pd
        ON e.procedure_id = pd.procedure_id
    LEFT JOIN `industrial-net-487818-h9.pre_procedure_data.Prior_Colonoscopy_History` AS h
        ON p.patient_id = h.patient_id
    LEFT JOIN `industrial-net-487818-h9.pre_procedure_data.Patient_Risk_Scores` AS rs
        ON e.procedure_id = rs.procedure_id
    WHERE p.patient_id = @patient_id
"""


def get_patient_record(patient_id: str) -> Optional[Dict[str, Any]]:
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("patient_id", "STRING", patient_id)
        ]
    )

    rows = list(_BQ_CLIENT.query(_QUERY, job_config=job_config).result())

    if not rows:
        return None

    return dict(rows[0].items())
