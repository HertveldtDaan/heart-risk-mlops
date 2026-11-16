"""Component 1 - Data preparation.

Reads the raw BRFSS 2022 CSV (Azure ML data asset), cleans it, and writes a
stratified train / validation / test split. Every output folder contains a
data.csv plus an MLTable file, so it can also be used by the Responsible AI
dashboard. Lineage info (data checksum, row counts) is logged with MLflow and
written to metadata.json so the training step can store it in the model.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

from heartrisk.features import TARGET, clean_dataset

MLTABLE = """\
$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json
type: mltable
paths:
  - file: ./data.csv
transformations:
  - read_delimited:
      delimiter: ','
      encoding: 'utf8'
      header: all_files_same_headers
"""


def resolve_csv(path: str) -> Path:
    """The input can be a file (uri_file) or a folder (uri_folder) with one CSV."""
    p = Path(path)
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV file found in {p}")
        return csvs[0]
    return p


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_split(df: pd.DataFrame, folder: str, metadata: dict) -> None:
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, "data.csv"), index=False)
    Path(folder, "MLTable").write_text(MLTABLE, encoding="utf-8")
    Path(folder, "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--sample_fraction", type=float, default=1.0,
                        help="Use e.g. 0.1 for fast development runs; 1.0 for the real model")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    args = parser.parse_args()
    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    csv_path = resolve_csv(args.raw_data)
    data_sha256 = sha256_of(csv_path)
    raw = pd.read_csv(csv_path)
    print(f"Raw data: {raw.shape} from {csv_path.name} (sha256 {data_sha256[:12]}...)")

    clean, stats = clean_dataset(raw)
    print("Cleaning stats:", stats)

    if args.sample_fraction < 1.0:
        clean, _ = train_test_split(clean, train_size=args.sample_fraction,
                                    stratify=clean[TARGET], random_state=args.seed)
        print(f"Sampled {args.sample_fraction:.0%} -> {len(clean)} rows")

    # Stratified split: every split keeps the same ~5.6% heart attack rate
    train_val, test = train_test_split(clean, test_size=args.test_size,
                                       stratify=clean[TARGET], random_state=args.seed)
    relative_val = args.val_size / (1 - args.test_size)
    train, val = train_test_split(train_val, test_size=relative_val,
                                  stratify=train_val[TARGET], random_state=args.seed)

    metadata = {
        "source_file": csv_path.name,
        "data_sha256": data_sha256,
        "seed": args.seed,
        "sample_fraction": args.sample_fraction,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        **stats,
    }
    write_split(train, args.train_data, metadata)
    write_split(val, args.val_data, metadata)
    write_split(test, args.test_data, metadata)

    mlflow.log_params({"data_sha256": data_sha256[:16], "sample_fraction": args.sample_fraction,
                       "seed": args.seed, "test_size": args.test_size, "val_size": args.val_size})
    mlflow.log_metrics({**{k: v for k, v in stats.items()},
                        "rows_train": len(train), "rows_val": len(val), "rows_test": len(test)})
    print("Split sizes:", metadata["split"])


if __name__ == "__main__":
    main()
