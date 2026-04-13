from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from google.cloud import bigquery

PROJECT_ID = "industrial-net-487818-h9"
PREPROC_DATASET = "pre_procedure_data"
OUTPUT_TABLE = f"{PROJECT_ID}.{PREPROC_DATASET}.Patient_Risk_Scores"

MODEL_DIR = Path(__file__).resolve().parent / "model_assets"

SOURCE_QUERY = f"""
SELECT
  p.patient_id,
  e.procedure_id,
  p.sex_at_birth,
  p.age_at_colonoscopy,
  p.bmi,
  p.mobility_status,
  p.smoking_status,
  p.insurance_type,

  c.chronic_constipation,
  c.constipation_severity,
  c.gastroparesis,
  c.neurogenic_bowel,
  c.parkinsons_disease,
  c.diabetes,
  c.medication_class_opioid,
  c.medication_class_glp1_agonist,
  c.chronic_laxative_use,
  c.dementia,
  c.diabetic_gastroparesis,
  c.diabetic_nephropathy,
  c.diabetic_retinopathy,
  c.on_diabetes_medication,
  c.opioid_specific_drugs,
  c.glp1_agonist_specific_drugs,
  c.laxative_specific_drugs,
  c.cirrhosis,
  c.tricyclic_antidepressant_use,
  c.prior_colorectal_surgery,
  c.hypertension,

  h.prior_prep_inadequate_verbatim,
  h.prior_prep_attempts_count
FROM `{PROJECT_ID}.{PREPROC_DATASET}.Patients` AS p
LEFT JOIN `{PROJECT_ID}.{PREPROC_DATASET}.Comorbidities` AS c
  ON p.patient_id = c.patient_id
LEFT JOIN `{PROJECT_ID}.{PREPROC_DATASET}.Encounters` AS e
  ON p.patient_id = e.patient_id
LEFT JOIN `{PROJECT_ID}.{PREPROC_DATASET}.Prior_Colonoscopy_History` AS h
  ON p.patient_id = h.patient_id
WHERE e.procedure_id IS NOT NULL
"""

# These are the raw columns the notebook drops after engineering.
DROP_AFTER_ENGINEERING = [
    "mobility_status",
    "on_diabetes_medication",
    "opioid_specific_drugs",
    "glp1_agonist_specific_drugs",
    "laxative_specific_drugs",
    "chronic_constipation",
    "age_at_colonoscopy",
    "diabetic_gastroparesis",
    "diabetic_nephropathy",
    "diabetic_retinopathy",
    "smoking_status",
    "insurance_type",
]


def _normalize_binary(value: Any) -> int:
    """Robustly coerce common binary encodings to 0/1."""
    if pd.isna(value) or value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "present", "positive"}:
            return 1
        if v in {"0", "false", "no", "n", "none", ""}:
            return 0
    return 0


def _normalize_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.fillna(default)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror the notebook feature engineering exactly enough for production scoring.
    """
    df = df.copy()

    # Explicit .copy() above avoids the notebook's SettingWithCopyWarning pattern.
    df["tobacco_use"] = (df["smoking_status"] == "Current").astype(int)
    df["medicaid"] = (df["insurance_type"] == "Medicaid").astype(int)
    df["medicare"] = (df["insurance_type"] == "Medicare").astype(int)

    df = pd.get_dummies(df, columns=["sex_at_birth"], drop_first=True)

    # Ensure expected dummy exists even if current batch contains only one category.
    if "sex_at_birth_Male" not in df.columns:
        df["sex_at_birth_Male"] = 0

    df["mobility_independent"] = (df["mobility_status"] == "Independent").astype(int)

    df["constipation_severity"] = (
        df["constipation_severity"]
        .map({"Mild": 1, "Moderate": 2, "Severe": 3})
        .fillna(0)
        .astype(float)
    )

    df["diabetes_complications"] = (
        df["diabetic_gastroparesis"].apply(_normalize_binary)
        | df["diabetic_nephropathy"].apply(_normalize_binary)
        | df["diabetic_retinopathy"].apply(_normalize_binary)
    ).astype(int)

    df["age_65_plus"] = (_normalize_numeric(df["age_at_colonoscopy"]) >= 65).astype(int)

    # Coerce likely model features to numeric/binary safely.
    likely_binary_cols = [
        "chronic_constipation",
        "gastroparesis",
        "neurogenic_bowel",
        "parkinsons_disease",
        "diabetes",
        "medication_class_opioid",
        "medication_class_glp1_agonist",
        "chronic_laxative_use",
        "dementia",
        "diabetic_gastroparesis",
        "diabetic_nephropathy",
        "diabetic_retinopathy",
        "on_diabetes_medication",
        "prior_prep_inadequate_verbatim",
        "cirrhosis",
        "tricyclic_antidepressant_use",
        "prior_colorectal_surgery",
        "hypertension",
    ]
    for col in likely_binary_cols:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_binary)

    if "prior_prep_attempts_count" in df.columns:
        df["prior_prep_attempts_count"] = _normalize_numeric(df["prior_prep_attempts_count"], default=0)

    if "bmi" in df.columns:
        df["bmi"] = _normalize_numeric(df["bmi"], default=0.0)

    # Notebook converts bool columns to int.
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    df = df.drop(columns=[c for c in DROP_AFTER_ENGINEERING if c in df.columns], errors="ignore")

    return df


def assign_risk(prob):
    if prob > 0.50:
        return "High"
    elif prob > 0.45:
        return "Medium"
    return "Low"


def main() -> None:
    model_path = MODEL_DIR / "risk_pipeline.joblib"
    features_path = MODEL_DIR / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing feature columns file: {features_path}")

    pipeline = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_columns = json.load(f)

    client = bigquery.Client(project=PROJECT_ID)
    raw_df = client.query(SOURCE_QUERY).to_dataframe()

    if raw_df.empty:
        print("No rows returned from BigQuery.")
        return

    id_cols = raw_df[["patient_id", "procedure_id"]].copy()
    feature_df = engineer_features(raw_df)

    # Add any missing expected features as zero-filled columns.
    for col in feature_columns:
        if col not in feature_df.columns:
            feature_df[col] = 0

    # Keep only the exact order used by the trained model.
    X = feature_df[feature_columns].copy()

    # Final numeric cleanup.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    # Notebook predicts probability of ADEQUATE prep, then converts to inadequate risk as 1 - p.
    predicted_adequate = pipeline.predict_proba(X)[:, 1]
    risk_proba = 1 - predicted_adequate
    risk_tiers = [assign_risk(p) for p in risk_proba]

    output_df = id_cols.copy()
    output_df["predicted_adequate_prob"] = predicted_adequate
    output_df["predicted_inadequate_risk"] = risk_proba
    output_df["risk_tier"] = risk_tiers
    output_df["model_version"] = "logreg_v1"
    output_df["scored_at"] = datetime.now(timezone.utc)

    # Keep one row per procedure_id.
    output_df = output_df.drop_duplicates(subset=["procedure_id"])

    job_config = bigquery.LoadJobConfig(rate_risk_scores.py
        write_disposition="WRITE_TRUNCATE",
        schema=[
            bigquery.SchemaField("patient_id", "STRING"),
            bigquery.SchemaField("procedure_id", "STRING"),
            bigquery.SchemaField("predicted_adequate_prob", "FLOAT"),
            bigquery.SchemaField("predicted_inadequate_risk", "FLOAT"),
            bigquery.SchemaField("risk_tier", "STRING"),
            bigquery.SchemaField("model_version", "STRING"),
            bigquery.SchemaField("scored_at", "TIMESTAMP"),
        ],
    )

    load_job = client.load_table_from_dataframe(
        output_df,
        OUTPUT_TABLE,
        job_config=job_config,
    )
    load_job.result()

    print(f"Wrote {len(output_df)} rows to {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()