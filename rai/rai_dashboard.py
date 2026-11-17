"""Create the Responsible AI dashboard for a registered heart-risk model (Azure ML SDK v2).

Same registry components as the "Create Responsible AI dashboard" wizard in the
Studio (see lab 2.1): explanations, error analysis, causal analysis and
(optionally) counterfactuals. The dashboard appears on the model's
"Responsible AI" tab in Azure ML Studio.

The RAI components can't handle missing values and are meant for a few
thousand rows, so we build a complete-case stratified sample with the SAME
cleaning code as the training pipeline (heartrisk.features).

Usage (after `az login`, or in GitHub Actions after azure/login):
  python rai/rai_dashboard.py --model-version 3 --data-asset heart-disease-2022:1
  python rai/rai_dashboard.py --model-version 3 --raw-csv data/raw/heart_2022_with_nans.csv
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from azure.ai.ml import Input, MLClient, dsl
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from heartrisk.features import AUDIT_COLUMNS, CATEGORICAL_FEATURES, FEATURES, TARGET, clean_dataset  # noqa: E402

MODEL_NAME = "heart-risk"
MLTABLE = """\
type: mltable
paths:
  - file: ./data.csv
transformations:
  - read_delimited:
      delimiter: ','
      encoding: 'utf8'
      header: all_files_same_headers
"""


def build_samples(raw_csv: str, n_train: int, n_test: int, seed: int = 42):
    clean, _ = clean_dataset(pd.read_csv(raw_csv))
    clean = clean[FEATURES + AUDIT_COLUMNS + [TARGET]].dropna()
    train, test = train_test_split(clean, test_size=0.2, stratify=clean[TARGET], random_state=seed)
    train = train.groupby(TARGET, group_keys=False).sample(frac=n_train / len(train), random_state=seed)
    test = test.groupby(TARGET, group_keys=False).sample(frac=n_test / len(test), random_state=seed)
    return train, test


def download_data_asset(ml_client: MLClient, asset: str) -> str:
    """Download a uri_file data asset (name:version) to a temp folder and return the CSV path."""
    from azure.ai.ml._artifacts._artifact_utilities import download_artifact_from_aml_uri

    name, version = asset.split(":")
    data = ml_client.data.get(name=name, version=version)
    target = Path(tempfile.mkdtemp())
    download_artifact_from_aml_uri(uri=data.path, destination=str(target),
                                   datastore_operation=ml_client.datastores)
    csv = next(target.rglob("*.csv"))
    print(f"Downloaded data asset {asset} -> {csv} ({csv.stat().st_size / 1e6:.0f} MB)")
    return str(csv)


def register_mltable(ml_client: MLClient, df: pd.DataFrame, name: str) -> Data:
    folder = Path(tempfile.mkdtemp()) / name
    folder.mkdir(parents=True)
    df.to_csv(folder / "data.csv", index=False)
    (folder / "MLTable").write_text(MLTABLE, encoding="utf-8")
    data = ml_client.data.create_or_update(
        Data(name=name, path=str(folder), type=AssetTypes.MLTABLE,
             description=f"Complete-case sample ({len(df)} rows) for the RAI dashboard")
    )
    print(f"Registered data asset {data.name}:{data.version} ({len(df)} rows)")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-asset", help="Azure ML data asset as name:version, e.g. heart-disease-2022:1")
    source.add_argument("--raw-csv", help="Local copy of the raw CSV")
    parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=True,
                        help="--no-submit: only prepare the samples (dry run)")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--counterfactuals", action="store_true", help="Also compute counterfactuals (slow)")
    parser.add_argument("--compute", default="heart-cluster")
    parser.add_argument("--subscription", default=os.getenv("AZURE_SUBSCRIPTION_ID"))
    parser.add_argument("--resource-group", default=os.getenv("AZURE_RESOURCE_GROUP"))
    parser.add_argument("--workspace", default=os.getenv("AZURE_ML_WORKSPACE"))
    args = parser.parse_args()

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, args.subscription, args.resource_group, args.workspace)
    registry = MLClient(credential, args.subscription, args.resource_group, registry_name="azureml")

    raw_csv = args.raw_csv or download_data_asset(ml_client, args.data_asset)
    train, test = build_samples(raw_csv, args.n_train, args.n_test)
    print(f"RAI samples: train {train.shape}, test {test.shape}, "
          f"positive rate {train['HadHeartAttack'].mean():.3f} / {test['HadHeartAttack'].mean():.3f}")
    if not args.submit:
        return
    train_data = register_mltable(ml_client, train, "heart-rai-train")
    test_data = register_mltable(ml_client, test, "heart-rai-test")

    constructor = registry.components.get(name="rai_tabular_insight_constructor", label="latest")
    version = constructor.version
    rai_explanation = registry.components.get(name="rai_tabular_explanation", version=version)
    rai_erroranalysis = registry.components.get(name="rai_tabular_erroranalysis", version=version)
    rai_causal = registry.components.get(name="rai_tabular_causal", version=version)
    rai_counterfactual = registry.components.get(name="rai_tabular_counterfactual", version=version)
    rai_gather = registry.components.get(name="rai_tabular_insight_gather", version=version)
    print("RAI component version:", version)

    model_id = f"{MODEL_NAME}:{args.model_version}"
    categorical = list(CATEGORICAL_FEATURES) + AUDIT_COLUMNS

    @dsl.pipeline(compute=args.compute, experiment_name="heart-risk-rai",
                  description=f"Responsible AI dashboard for {model_id}")
    def rai_pipeline(train_dataset, test_dataset):
        create = constructor(
            title=f"RAI dashboard {model_id}",
            task_type="classification",
            model_info=model_id,
            model_input=Input(type=AssetTypes.MLFLOW_MODEL, path=f"azureml:{model_id}"),
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            target_column_name=TARGET,
            categorical_column_names=json.dumps(categorical),
            classes="[0, 1]",
            use_model_dependency=True,  # install the model's own pinned requirements (lightgbm, skops, ...)
        )
        create.set_limits(timeout=3600)
        dashboard = create.outputs.rai_insights_dashboard

        explain = rai_explanation(comment="Global + local feature importance", rai_insights_dashboard=dashboard)
        errors = rai_erroranalysis(rai_insights_dashboard=dashboard,
                                   filter_features=json.dumps(["AgeCategory", "Sex"]))
        # "What if" a person slept more, lost weight or started exercising? (lifestyle = changeable)
        causal = rai_causal(rai_insights_dashboard=dashboard,
                            treatment_features=json.dumps(["SleepHours", "BMI", "PhysicalActivities"]))
        insights = {
            "insight_1": explain.outputs.explanation,
            "insight_2": errors.outputs.error_analysis,
            "insight_3": causal.outputs.causal,
        }
        if args.counterfactuals:
            cf = rai_counterfactual(rai_insights_dashboard=dashboard, total_cfs=5, desired_class="opposite")
            insights["insight_4"] = cf.outputs.counterfactual

        gather = rai_gather(constructor=dashboard, **insights)
        gather.outputs.dashboard.mode = "upload"
        gather.outputs.ux_json.mode = "upload"

    job = rai_pipeline(
        train_dataset=Input(type=AssetTypes.MLTABLE, path=f"azureml:{train_data.name}:{train_data.version}",
                            mode="download"),
        test_dataset=Input(type=AssetTypes.MLTABLE, path=f"azureml:{test_data.name}:{test_data.version}",
                           mode="download"),
    )
    job.display_name = f"rai-dashboard-{MODEL_NAME}-v{args.model_version}"
    # The RAI components still use the legacy "path" input type, which the newer SDK rejects client-side
    submitted = ml_client.jobs.create_or_update(job, skip_validation=True)
    print("Submitted:", submitted.name)
    print("Studio URL:", submitted.studio_url)


if __name__ == "__main__":
    main()
