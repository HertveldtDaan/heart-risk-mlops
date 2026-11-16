"""Component 3 - Evaluation + quality gate.

Evaluates the trained model ONCE on the held-out test set, audits fairness
per sex / age group / ethnicity and computes the global SHAP importance.

Quality gate: the model is only copied to the `approved_model` output when
  - test ROC-AUC >= min_roc_auc
  - test recall   >= min_recall
  - it beats the logistic regression baseline on validation PR-AUC
If the gate fails the step exits with an error, so the register step never
runs: a bad model can never reach the model registry.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from heartrisk.features import TARGET
from heartrisk.modeling import age_group, compute_metrics, global_importance, group_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_input", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--min_roc_auc", type=float, default=0.80)
    parser.add_argument("--min_recall", type=float, default=0.60)
    parser.add_argument("--approved_model", type=str, required=True)
    args = parser.parse_args()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    model = mlflow.sklearn.load_model(args.model_input)
    release = json.loads(Path(args.model_input, "release.json").read_text(encoding="utf-8"))
    threshold = release["threshold"]

    test = pd.read_csv(os.path.join(args.test_data, "data.csv"))
    X_test, y_test = test.drop(columns=[TARGET]), test[TARGET]
    proba = model.predict_proba(X_test)[:, 1]

    test_metrics = compute_metrics(y_test, proba, threshold)
    print("Test metrics:", json.dumps(test_metrics, indent=2))
    mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

    # Fairness audit: does the model work equally well for every group?
    audits = {
        "sex": X_test["Sex"],
        "age_group": age_group(X_test["AgeCategory"]),
        "ethnicity": X_test["RaceEthnicityCategory"],
    }
    fairness = {}
    for name, groups in audits.items():
        table = group_metrics(y_test, proba, threshold, groups)
        print(f"\nFairness by {name}:\n{table.to_string(index=False)}")
        fairness[name] = {
            "groups": table.to_dict(orient="records"),
            "recall_difference": float(table["recall"].max() - table["recall"].min()),
        }
        mlflow.log_metric(f"fairness_recall_diff_{name}", fairness[name]["recall_difference"])
        mlflow.log_table(table, artifact_file=f"fairness/{name}.json")

    sample = X_test.sample(n=min(5000, len(X_test)), random_state=42)
    importance = global_importance(model, sample)
    print("\nGlobal SHAP importance (top 10):", list(importance.items())[:10])
    mlflow.log_dict(importance, "shap_global_importance.json")

    # Quality gate
    baseline_pr_auc = release["baseline_validation_metrics"]["pr_auc"]
    checks = {
        f"test_roc_auc >= {args.min_roc_auc}": test_metrics["roc_auc"] >= args.min_roc_auc,
        f"test_recall >= {args.min_recall}": test_metrics["recall"] >= args.min_recall,
        "beats baseline (val pr_auc)": release["validation_metrics"]["pr_auc"] >= baseline_pr_auc,
    }
    passed = all(checks.values())
    print("\nQuality gate:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    mlflow.log_metric("quality_gate_passed", int(passed))

    evaluation = {
        "test_metrics": test_metrics,
        "test_rows": int(len(test)),
        "quality_gate": {"passed": passed, "checks": checks},
        "fairness": fairness,
        "shap_global_importance": importance,
    }

    if not passed:
        print("\nQuality gate FAILED: the model will not be registered.")
        sys.exit(1)

    # Gate passed: copy the model (+ evaluation report) to the output for registration
    shutil.copytree(args.model_input, args.approved_model, dirs_exist_ok=True)
    Path(args.approved_model, "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(f"\nQuality gate PASSED: model copied to {args.approved_model}")


if __name__ == "__main__":
    main()
