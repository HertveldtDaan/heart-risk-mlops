"""Model building, metrics and explanations (used by train, evaluate and the API)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from heartrisk.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

CATEGORICAL = list(CATEGORICAL_FEATURES)

# MLflow 3 saves sklearn models with skops (no pickle -> loading can't execute arbitrary code).
# skops only loads types that are explicitly trusted; these are the LightGBM internals.
SKOPS_TRUSTED_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]


def build_lgbm_pipeline(
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42,
) -> Pipeline:
    """Preprocessing + LightGBM in ONE sklearn pipeline.

    The encoder uses the ordered categories from the feature contract, so every
    column stays one feature (needed for readable SHAP values per question).
    Missing values stay NaN: LightGBM learns the best direction for them itself.
    """
    encoder = OrdinalEncoder(
        categories=[CATEGORICAL_FEATURES[c] for c in CATEGORICAL],
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
        encoded_missing_value=np.nan,
    )
    preprocess = ColumnTransformer(
        [("cat", encoder, CATEGORICAL), ("num", "passthrough", NUMERIC_FEATURES)],
        remainder="drop",  # audit columns (e.g. race) are never used by the model
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    model = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def build_baseline_pipeline(seed: int = 42) -> Pipeline:
    """Simple, explainable baseline. The real model must beat this one (quality gate)."""
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    preprocess = ColumnTransformer(
        [("cat", categorical, CATEGORICAL), ("num", numeric, NUMERIC_FEATURES)],
        remainder="drop",
    )
    model = LogisticRegression(max_iter=1000, random_state=seed)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def find_threshold(y_true, proba, beta: float = 2.0) -> float:
    """Pick the decision threshold that maximises the F-beta score.

    beta=2 weighs recall twice as much as precision: for a screening app,
    missing a person at risk is worse than an extra (unnecessary) check-up.
    """
    best_t, best_score = 0.5, -1.0
    for t in np.arange(0.02, 0.9, 0.01):
        score = fbeta_score(y_true, proba >= t, beta=beta, zero_division=0)
        if score > best_score:
            best_t, best_score = float(t), score
    return round(best_t, 2)


def compute_metrics(y_true, proba, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    pred = np.asarray(proba) >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "positive_rate_predicted": float(pred.mean()),
    }


def group_metrics(y_true, proba, threshold: float, groups: pd.Series) -> pd.DataFrame:
    """Per-group metrics for the fairness audit (same idea as fairlearn's MetricFrame)."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba), "g": groups.values})
    frame["g"] = frame["g"].fillna("Unknown")
    rows = []
    for group, part in frame.groupby("g"):
        pred = part["p"] >= threshold
        has_both = part["y"].nunique() == 2
        rows.append({
            "group": str(group),
            "count": int(len(part)),
            "actual_positive_rate": float(part["y"].mean()),
            "selection_rate": float(pred.mean()),
            "recall": float(recall_score(part["y"], pred, zero_division=0)),
            "precision": float(precision_score(part["y"], pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(part["y"], part["p"])) if has_both else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def age_group(age_category: pd.Series) -> pd.Series:
    """Collapse the 13 age categories into 3 groups for the fairness audit."""
    order = CATEGORICAL_FEATURES["AgeCategory"]
    idx = age_category.map({c: i for i, c in enumerate(order)})
    return pd.cut(idx, bins=[-1, 5, 9, 12], labels=["18-44", "45-64", "65+"]).astype("object")


def explain(pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """SHAP values per feature (in log-odds), via LightGBM's built-in TreeSHAP.

    Positive value = this answer pushes the risk UP, negative = pushes it DOWN.
    The last column 'base_value' is the average model output.
    """
    transformed = pipeline.named_steps["preprocess"].transform(features)
    contrib = pipeline.named_steps["model"].predict(transformed, pred_contrib=True)
    columns = list(transformed.columns) + ["base_value"]
    return pd.DataFrame(contrib, columns=columns, index=features.index)


def global_importance(pipeline: Pipeline, features: pd.DataFrame) -> dict[str, float]:
    """Mean |SHAP| per feature: which questions matter most for the model overall."""
    shap_values = explain(pipeline, features).drop(columns="base_value")
    importance = shap_values.abs().mean().sort_values(ascending=False)
    return {k: round(float(v), 4) for k, v in importance.items()}
