"""
BigQuery access layer.

Fetches patient-level data using patient_id and returns a structured
record used to personalize responses.
"""

from google.cloud import bigquery
from typing import Optional, Dict, Any

client = bigquery.Client()

def get_patient_record(patient_id: str) -> Optional[Dict[str, Any]]:
    query = """
    SELECT
      p.patient_id,
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
      c.laxative_specific_drugs
    FROM `industrial-net-487818-h9.pre_procedure_data.Patients` p
    LEFT JOIN `industrial-net-487818-h9.pre_procedure_data.Comorbidities` c
      ON p.patient_id = c.patient_id
    WHERE p.patient_id = @patient_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("patient_id", "STRING", patient_id)
        ]
    )

    rows = list(client.query(query, job_config=job_config).result())

    if not rows:
        return None

    return dict(rows[0].items())