from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent / "model_assets"
PIPELINE_PATH = MODEL_DIR / "risk_pipeline.joblib"
FEATURES_PATH = MODEL_DIR / "feature_columns.json"

MEDIUM_THRESHOLD = 0.5

pipeline = joblib.load(PIPELINE_PATH)

with open(FEATURES_PATH, "r") as f:
    FEATURE_COLUMNS = json.load(f)


def _to_int(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x).strip().lower()
    return 1 if s in {"1", "true", "yes", "y"} else 0


def _rule_based_high_risk(patient: Dict[str, Any]) -> int:
    diabetes_with_complications = int(
        _to_int(patient.get("diabetes")) == 1 and (
            _to_int(patient.get("diabetic_gastroparesis")) == 1 or
            _to_int(patient.get("diabetic_nephropathy")) == 1 or
            _to_int(patient.get("diabetic_retinopathy")) == 1
        )
    )

    regular_opioid_use_flag = _to_int(patient.get("medication_class_opioid"))
    glp1_use_flag = _to_int(patient.get("medication_class_glp1_agonist"))

    prior_bad_prep_flag = int(
        _to_int(patient.get("prior_prep_inadequate_verbatim")) == 1 or
        _to_int(patient.get("prior_extended_prep_required")) == 1
    )

    colon_surgery_flag = _to_int(patient.get("prior_colorectal_surgery"))

    severe_constipation_flag = int(
        (
            _to_int(patient.get("chronic_constipation")) == 1 and
            str(patient.get("constipation_severity", "")).strip().lower() == "severe"
        ) or _to_int(patient.get("chronic_laxative_use")) == 1
    )

    paraplegia_quadriplegia_flag = int(
        _to_int(patient.get("spinal_cord_injury")) == 1 and
        str(patient.get("sci_type", "")).strip().lower() in {"paraplegia", "quadriplegia"}
    )

    cirrhosis_flag = _to_int(patient.get("cirrhosis"))
    parkinsons_flag = _to_int(patient.get("parkinsons_disease"))
    dementia_flag = _to_int(patient.get("dementia"))

    return int(any([
        diabetes_with_complications,
        regular_opioid_use_flag,
        glp1_use_flag,
        prior_bad_prep_flag,
        colon_surgery_flag,
        severe_constipation_flag,
        paraplegia_quadriplegia_flag,
        cirrhosis_flag,
        parkinsons_flag,
        dementia_flag,
    ]))


def _build_model_features(patient: Dict[str, Any]) -> pd.DataFrame:
    sex = str(patient.get("sex_at_birth", "")).strip().lower()
    constipation_raw = str(patient.get("constipation_severity", "")).strip().lower()

    constipation_map = {
        "mild": 1,
        "moderate": 2,
        "severe": 3,
    }

    row = {
        "bmi": float(patient.get("bmi") or 0),
        "constipation_severity": constipation_map.get(constipation_raw, 0),
        "gastroparesis": _to_int(patient.get("gastroparesis")),
        "neurogenic_bowel": _to_int(patient.get("neurogenic_bowel")),
        "diabetes": _to_int(patient.get("diabetes")),
        "tricyclic_antidepressant_use": _to_int(patient.get("tricyclic_antidepressant_use")),
        "hypertension": _to_int(patient.get("hypertension")),
        "tobacco_use": int(str(patient.get("smoking_status", "")).strip().lower() == "current"),
        "sex_at_birth_Male": int(sex == "male"),
        "mobility_independent": int(str(patient.get("mobility_status", "")).strip().lower() == "independent"),
        "age_65_plus": int(float(patient.get("age_at_colonoscopy") or 0) >= 65),
    }

    row = {col: row.get(col, 0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def get_risk_score(patient: Dict[str, Any]) -> Dict[str, Any]:
    X = _build_model_features(patient)
    adequate_prob = float(pipeline.predict_proba(X)[0, 1])
    inadequate_risk = 1 - adequate_prob

    if _rule_based_high_risk(patient) == 1:
        risk_tier = "High"
    else:
        risk_tier = "Medium" if inadequate_risk >= MEDIUM_THRESHOLD else "Low"

    return {
        "predicted_adequate_prob": adequate_prob,
        "predicted_inadequate_risk": inadequate_risk,
        "risk_tier": risk_tier,
    }