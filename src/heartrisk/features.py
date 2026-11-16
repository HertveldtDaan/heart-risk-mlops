"""Feature contract and cleaning rules for the CDC BRFSS 2022 heart dataset.

Single source of truth for:
- which columns the model uses (and which are deliberately left out)
- the allowed category values (in a meaningful order)
- the cleaning rules that must be identical in training and in the API
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "HadHeartAttack"

# Sensitive attribute: NOT a model feature, but kept in the evaluation data
# so we can check whether the model performs equally well for every group.
AUDIT_COLUMNS = ["RaceEthnicityCategory"]

YES_NO = ["No", "Yes"]

# Ordered categories: the order is used by the OrdinalEncoder, so "Poor" < "Fair" < ...
CATEGORICAL_FEATURES: dict[str, list[str]] = {
    "Sex": ["Female", "Male"],
    "AgeCategory": [
        "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
        "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
        "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
        "Age 80 or older",
    ],
    "GeneralHealth": ["Poor", "Fair", "Good", "Very good", "Excellent"],
    "LastCheckupTime": [
        "Within past year (anytime less than 12 months ago)",
        "Within past 2 years (1 year but less than 2 years ago)",
        "Within past 5 years (2 years but less than 5 years ago)",
        "5 or more years ago",
    ],
    "PhysicalActivities": YES_NO,
    "RemovedTeeth": ["None of them", "1 to 5", "6 or more, but not all", "All"],
    "HadStroke": YES_NO,
    "HadAsthma": YES_NO,
    "HadSkinCancer": YES_NO,
    "HadCOPD": YES_NO,
    "HadDepressiveDisorder": YES_NO,
    "HadKidneyDisease": YES_NO,
    "HadArthritis": YES_NO,
    "HadDiabetes": [
        "No",
        "No, pre-diabetes or borderline diabetes",
        "Yes, but only during pregnancy (female)",
        "Yes",
    ],
    "DeafOrHardOfHearing": YES_NO,
    "BlindOrVisionDifficulty": YES_NO,
    "DifficultyConcentrating": YES_NO,
    "DifficultyWalking": YES_NO,
    "DifficultyDressingBathing": YES_NO,
    "DifficultyErrands": YES_NO,
    "SmokerStatus": [
        "Never smoked",
        "Former smoker",
        "Current smoker - now smokes some days",
        "Current smoker - now smokes every day",
    ],
    "ECigaretteUsage": [
        "Never used e-cigarettes in my entire life",
        "Not at all (right now)",
        "Use them some days",
        "Use them every day",
    ],
    "AlcoholDrinkers": YES_NO,
}

NUMERIC_FEATURES = ["PhysicalHealthDays", "MentalHealthDays", "SleepHours", "BMI"]

FEATURES = list(CATEGORICAL_FEATURES) + NUMERIC_FEATURES

# Columns we deliberately do NOT use, with the reason (documented in the report).
EXCLUDED_COLUMNS = {
    "HadAngina": "target leakage: angina is coronary heart disease itself / often follows a heart attack",
    "ChestScan": "target leakage: chest scans are mostly done because of heart problems",
    "State": "US-specific, not usable for a Belgian/European client",
    "HeightInMeters": "only used to fill in a missing BMI",
    "WeightInKilograms": "only used to fill in a missing BMI",
    "HIVTesting": "not a plausible heart risk factor",
    "FluVaxLast12": "proxy for healthcare usage, not a risk factor",
    "PneumoVaxEver": "proxy for healthcare usage/age, not a risk factor",
    "TetanusLast10Tdap": "not a plausible heart risk factor",
    "HighRiskLastYear": "about HIV risk behaviour, not heart risk",
    "CovidPos": "not a stable risk factor, many missing values",
    "RaceEthnicityCategory": "sensitive attribute: only used to audit fairness",
}

# Plausible ranges: values outside are treated as missing (NaN).
PLAUSIBLE_RANGES = {
    "PhysicalHealthDays": (0, 30),
    "MentalHealthDays": (0, 30),
    "SleepHours": (1, 18),
    "BMI": (12, 80),
}


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning rules shared by training and serving.

    - Fill a missing BMI from height and weight when both are known
    - Values outside a plausible range become NaN (the model handles NaN natively)
    - Unknown category values become NaN (e.g. a typo in an API request)
    Returns only the model feature columns, in a fixed order.
    """
    df = df.copy()

    if "BMI" not in df.columns:
        df["BMI"] = np.nan
    if {"HeightInMeters", "WeightInKilograms"}.issubset(df.columns):
        height = pd.to_numeric(df["HeightInMeters"], errors="coerce")
        weight = pd.to_numeric(df["WeightInKilograms"], errors="coerce")
        computed = weight / height.pow(2)
        df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce").fillna(computed)

    for col in NUMERIC_FEATURES:
        raw = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        values = pd.to_numeric(raw, errors="coerce").astype("float64")
        low, high = PLAUSIBLE_RANGES[col]
        df[col] = values.where(values.between(low, high))

    for col, categories in CATEGORICAL_FEATURES.items():
        values = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
        df[col] = values.where(values.isin(categories)).astype("object")

    return df[FEATURES]


def encode_target(series: pd.Series) -> pd.Series:
    """'Yes' -> 1, 'No' -> 0 (rows with a missing target are removed before)."""
    return series.map({"Yes": 1, "No": 0}).astype("int64")


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full cleaning of the raw CSV for training. Returns (clean_df, stats)."""
    stats = {"rows_raw": int(len(raw))}

    df = raw.drop_duplicates()
    stats["duplicates_removed"] = int(len(raw) - len(df))

    df = df[df[TARGET].isin(["Yes", "No"])]
    stats["missing_target_removed"] = int(len(raw) - stats["duplicates_removed"] - len(df))

    features = prepare_features(df)
    audit = df[AUDIT_COLUMNS].astype("object")
    clean = pd.concat([features, audit], axis=1)
    clean[TARGET] = encode_target(df[TARGET])
    clean = clean.reset_index(drop=True)

    stats["rows_clean"] = int(len(clean))
    stats["positive_rate"] = float(clean[TARGET].mean())
    stats["missing_cells_pct"] = float(clean[FEATURES].isna().mean().mean() * 100)
    return clean, stats
