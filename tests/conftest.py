"""Test fixtures: a tiny model trained on synthetic data, so CI needs no real data or Azure access."""
import json
import sys
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "api")]

from heartrisk.features import CATEGORICAL_FEATURES, FEATURES, TARGET  # noqa: E402
from heartrisk.modeling import SKOPS_TRUSTED_TYPES, build_lgbm_pipeline  # noqa: E402


def synthetic_raw(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Random data in the raw CSV format; heart attacks are more likely for older smokers."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(cats, n) for col, cats in CATEGORICAL_FEATURES.items()})
    df["PhysicalHealthDays"] = rng.integers(0, 31, n).astype(float)
    df["MentalHealthDays"] = rng.integers(0, 31, n).astype(float)
    df["SleepHours"] = rng.integers(3, 11, n).astype(float)
    df["HeightInMeters"] = rng.normal(1.7, 0.1, n)
    df["WeightInKilograms"] = rng.normal(80, 15, n)
    df["BMI"] = np.where(rng.random(n) < 0.1, np.nan, df["WeightInKilograms"] / df["HeightInMeters"] ** 2)
    df["RaceEthnicityCategory"] = rng.choice(["White only, Non-Hispanic", "Hispanic", "Black only, Non-Hispanic"], n)
    age_idx = df["AgeCategory"].map({c: i for i, c in enumerate(CATEGORICAL_FEATURES["AgeCategory"])})
    logit = -4 + 0.25 * age_idx + 1.5 * (df["SmokerStatus"] != "Never smoked") + rng.normal(0, 0.5, n)
    df[TARGET] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return synthetic_raw()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, raw_df) -> Path:
    from heartrisk.features import clean_dataset

    clean, _ = clean_dataset(raw_df)
    pipeline = build_lgbm_pipeline(n_estimators=30).fit(clean[FEATURES], clean[TARGET])
    out = tmp_path_factory.mktemp("model") / "heart-risk"
    mlflow.sklearn.save_model(pipeline, str(out), skops_trusted_types=SKOPS_TRUSTED_TYPES)
    (out / "release.json").write_text(json.dumps({
        "model_name": "heart-risk", "threshold": 0.3, "git_commit": "test",
        "data": {"data_sha256": "abc"}, "validation_metrics": {}, "features": FEATURES,
    }), encoding="utf-8")
    return out


@pytest.fixture()
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MODEL_VERSION", "7")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.delenv("POSTGRES_HOST", raising=False)
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        yield c
