"""Loads the MLflow model bundle downloaded from the Azure ML registry and makes predictions."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlflow.sklearn
import pandas as pd

from heartrisk.features import prepare_features
from heartrisk.modeling import explain

DISCLAIMER = ("This is a statistical risk estimate based on survey data, not a medical diagnosis. "
              "Always consult a doctor.")


class ModelService:
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.pipeline = mlflow.sklearn.load_model(str(self.model_dir))
        self.release = json.loads((self.model_dir / "release.json").read_text(encoding="utf-8"))
        evaluation_file = self.model_dir / "evaluation.json"
        self.evaluation = (json.loads(evaluation_file.read_text(encoding="utf-8"))
                           if evaluation_file.exists() else {})
        self.name = self.release.get("model_name", "heart-risk")
        # The registry version is only known after registration -> injected at image build time
        self.version = os.getenv("MODEL_VERSION", "dev")
        self.threshold = float(self.release["threshold"])

    def risk_level(self, probability: float) -> str:
        if probability < self.threshold:
            return "low"
        if probability < min(0.9, 3 * self.threshold):
            return "elevated"
        return "high"

    def predict(self, answers: dict, top_n: int = 5) -> dict:
        features = prepare_features(pd.DataFrame([answers]))
        probability = float(self.pipeline.predict_proba(features)[0, 1])

        shap_row = explain(self.pipeline, features).iloc[0].drop("base_value")
        top = shap_row.reindex(shap_row.abs().sort_values(ascending=False).index)[:top_n]
        factors = []
        for feature, impact in top.items():
            value = features.iloc[0][feature]
            if isinstance(value, float) and math.isnan(value):
                value = None
            elif isinstance(value, float):
                value = round(value, 2)
            factors.append({
                "feature": feature,
                "value": value,
                "impact": round(float(impact), 4),
                "direction": "increases risk" if impact > 0 else "decreases risk",
            })

        return {
            "risk_probability": round(probability, 4),
            "above_threshold": probability >= self.threshold,
            "risk_level": self.risk_level(probability),
            "top_factors": factors,
        }
