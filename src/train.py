"""Component 2 - Training.

Trains a logistic regression BASELINE and the LightGBM CANDIDATE on the train
split. The validation split is used for early stopping and to choose the
decision threshold; the test split is never touched here (it is reserved for
the quality gate in the evaluate step).

Output: an MLflow model folder (sklearn flavour) + release.json with all
lineage info (git commit, data checksum, seed, params, metrics, versions).
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from lightgbm import early_stopping

from heartrisk.features import AUDIT_COLUMNS, CATEGORICAL_FEATURES, EXCLUDED_COLUMNS, FEATURES, TARGET
from heartrisk.modeling import (
    SKOPS_TRUSTED_TYPES,
    build_baseline_pipeline,
    build_lgbm_pipeline,
    compute_metrics,
    find_threshold,
)

MODEL_NAME = "heart-risk"

# Exact versions the model needs at load time (also used by the API image and RAI dashboard)
PIP_REQUIREMENTS = [
    f"scikit-learn=={sklearn.__version__}",
    f"lightgbm=={lightgbm.__version__}",
    f"pandas=={pd.__version__}",
    f"mlflow=={mlflow.__version__}",
    "skops==0.16.0",
]


def load_split(folder: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = pd.read_csv(os.path.join(folder, "data.csv"))
    meta = json.loads(Path(folder, "metadata.json").read_text(encoding="utf-8"))
    return df.drop(columns=[TARGET]), df[TARGET], meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--model_output", type=str, required=True)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--git_sha", type=str, default="local")
    args = parser.parse_args()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    X_train, y_train, data_meta = load_split(args.train_data)
    X_val, y_val, _ = load_split(args.val_data)
    print(f"Train {X_train.shape}, val {X_val.shape}, positive rate {y_train.mean():.3%}")

    params = {k: getattr(args, k) for k in ("n_estimators", "learning_rate", "num_leaves", "seed")}
    mlflow.log_params({**params, "git_sha": args.git_sha, "model_type": "LGBMClassifier"})

    # 1. Baseline
    baseline = build_baseline_pipeline(seed=args.seed).fit(X_train, y_train)
    base_proba = baseline.predict_proba(X_val)[:, 1]
    base_threshold = find_threshold(y_val, base_proba)
    base_metrics = compute_metrics(y_val, base_proba, base_threshold)
    print("Baseline (val):", base_metrics)

    # 2. Candidate: LightGBM with early stopping on the validation set
    model = build_lgbm_pipeline(args.n_estimators, args.learning_rate, args.num_leaves, args.seed)
    preprocess = model.named_steps["preprocess"].fit(X_train)
    model.named_steps["model"].fit(
        preprocess.transform(X_train), y_train,
        eval_set=[(preprocess.transform(X_val), y_val)],
        eval_metric="average_precision",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    best_iter = int(model.named_steps["model"].best_iteration_ or args.n_estimators)
    proba = model.predict_proba(X_val)[:, 1]
    threshold = find_threshold(y_val, proba)
    val_metrics = compute_metrics(y_val, proba, threshold)
    print(f"LightGBM (val, best_iteration={best_iter}, threshold={threshold}):", val_metrics)

    mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
    mlflow.log_metrics({f"baseline_val_{k}": v for k, v in base_metrics.items()})
    mlflow.log_metrics({"threshold": threshold, "best_iteration": best_iter})

    # 3. Save as MLflow model (the output folder itself becomes the model folder)
    out = Path(args.model_output)
    out.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, str(out), pip_requirements=PIP_REQUIREMENTS,
                              skops_trusted_types=SKOPS_TRUSTED_TYPES)

    active = mlflow.active_run()
    release = {
        "model_name": MODEL_NAME,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": args.git_sha,
        "azureml_run_id": os.environ.get("AZUREML_RUN_ID", "local"),
        "mlflow_run_id": active.info.run_id if active else None,
        "data": data_meta,
        "params": {**params, "best_iteration": best_iter},
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "baseline_validation_metrics": base_metrics,
        "features": FEATURES,
        "categories": CATEGORICAL_FEATURES,
        "audit_columns": AUDIT_COLUMNS,
        "excluded_columns": EXCLUDED_COLUMNS,
        "library_versions": {r.split("==")[0]: r.split("==")[1] for r in PIP_REQUIREMENTS},
    }
    (out / "release.json").write_text(json.dumps(release, indent=2), encoding="utf-8")
    print(f"Model + release.json saved to {out}")


if __name__ == "__main__":
    main()
