import json
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model_assets"

pipeline = joblib.load(MODEL_DIR / "risk_pipeline.joblib")

with open(MODEL_DIR / "feature_columns.json", "r") as f:
    FEATURE_COLUMNS = json.load(f)

with open(MODEL_DIR / "risk_thresholds.json", "r") as f:
    THRESHOLDS = json.load(f)

def _safe_int(value):
    if pd.isna(value) or value is None:
        return 0
    return int(value)


def _safe_float(value):
    if pd.isna(value) or value is None:
        return 0.0
    return float(value)


def _map_constipation_severity(value):
    if pd.isna(value) or value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    value_str = str(value).strip().lower()
    mapping = {
        "mild": 1.0,
        "moderate": 2.0,
        "severe": 3.0,
    }
    return mapping.get(value_str, 0.0)


def _notnull_flag(value):
    if value is None:
        return 0
    if isinstance(value, str) and value.strip() == "":
        return 0
    if pd.isna(value):
        return 0
    return 1


def build_risk_features(patient_record: dict) -> pd.DataFrame:
    row = {}

    # Base variables kept after your feature engineering
    row["bmi"] = _safe_float(patient_record.get("bmi"))
    row["constipation_severity"] = _map_constipation_severity(
        patient_record.get("constipation_severity")
    )
    row["gastroparesis"] = _safe_int(patient_record.get("gastroparesis"))
    row["neurogenic_bowel"] = _safe_int(patient_record.get("neurogenic_bowel"))
    row["parkinsons_disease"] = _safe_int(patient_record.get("parkinsons_disease"))
    row["spinal_cord_injury"] = _safe_int(patient_record.get("spinal_cord_injury"))
    row["diabetes"] = _safe_int(patient_record.get("diabetes"))
    row["medication_class_opioid"] = _safe_int(patient_record.get("medication_class_opioid"))
    row["anticholinergic_use"] = _safe_int(patient_record.get("anticholinergic_use"))
    row["medication_class_glp1_agonist"] = _safe_int(patient_record.get("medication_class_glp1_agonist"))
    row["chronic_laxative_use"] = _safe_int(patient_record.get("chronic_laxative_use"))
    row["dementia"] = _safe_int(patient_record.get("dementia"))
    row["history_of_stroke"] = _safe_int(patient_record.get("history_of_stroke"))
    row["prior_prep_inadequate_verbatim"] = _safe_int(patient_record.get("prior_prep_inadequate_verbatim"))
    row["prior_prep_attempts_count"] = _safe_float(patient_record.get("prior_prep_attempts_count"))

    # Derived features from training
    row["mobility_independent"] = int(patient_record.get("mobility_status") == "Independent")

    row["diabetes_complications"] = int(
        _safe_int(patient_record.get("diabetic_gastroparesis")) == 1
        or _safe_int(patient_record.get("diabetic_nephropathy")) == 1
        or _safe_int(patient_record.get("diabetic_retinopathy")) == 1
    )

    row["medication_count"] = (
        _notnull_flag(patient_record.get("on_diabetes_medication"))
        + _notnull_flag(patient_record.get("opioid_specific_drugs"))
        + _notnull_flag(patient_record.get("glp1_agonist_specific_drugs"))
        + _notnull_flag(patient_record.get("laxative_specific_drugs"))
    )

    row["age_65_plus"] = int(_safe_float(patient_record.get("age_at_colonoscopy")) >= 65)

    # One-hot encoded fields from training with drop_first=True
    # These exact names must match what ended up in FEATURE_COLUMNS
    sex = patient_record.get("sex_at_birth")
    diet = patient_record.get("diet_protocol")

    row["sex_at_birth_Male"] = int(sex == "Male")

    row["diet_protocol_Clear liquids 24h"] = int(diet == "Clear liquids 24h")
    row["diet_protocol_2-day low-residue diet + clear liquids"] = int(
        diet == "2-day low-residue diet + clear liquids"
    )

    df = pd.DataFrame([row])

    # Add any missing columns expected by trained model
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Keep exact training column order
    df = df[FEATURE_COLUMNS]

    # Convert any remaining booleans to int
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype(int)

    return df

def get_risk_score(patient_record: dict) -> dict:
    X = build_risk_features(patient_record)

    # if your model predicts adequate_prep_flag = 1 for adequate
    prob_adequate = pipeline.predict_proba(X)[0][1]
    prob_inadequate = 1 - prob_adequate

    low_max = THRESHOLDS["low_max"]
    medium_max = THRESHOLDS["medium_max"]

    if prob_inadequate <= low_max:
        risk_tier = "low"
    elif prob_inadequate <= medium_max:
        risk_tier = "medium"
    else:
        risk_tier = "high"

    return {
        "risk_probability": round(float(prob_inadequate), 4),
        "risk_tier": risk_tier,
    }