import numpy as np
import pandas as pd

from heartrisk.features import FEATURES, TARGET, clean_dataset, prepare_features


def test_bmi_is_computed_from_height_and_weight():
    df = pd.DataFrame([{"BMI": np.nan, "HeightInMeters": 2.0, "WeightInKilograms": 100.0}])
    assert prepare_features(df)["BMI"].iloc[0] == 25.0


def test_existing_bmi_is_kept():
    df = pd.DataFrame([{"BMI": 30.0, "HeightInMeters": 2.0, "WeightInKilograms": 100.0}])
    assert prepare_features(df)["BMI"].iloc[0] == 30.0


def test_implausible_values_become_missing():
    df = pd.DataFrame([{"SleepHours": 24.0, "BMI": 99.0, "PhysicalHealthDays": 5.0}])
    out = prepare_features(df)
    assert np.isnan(out["SleepHours"].iloc[0])
    assert np.isnan(out["BMI"].iloc[0])
    assert out["PhysicalHealthDays"].iloc[0] == 5.0


def test_unknown_category_becomes_missing():
    out = prepare_features(pd.DataFrame([{"Sex": "Mael", "GeneralHealth": "Good"}]))
    assert pd.isna(out["Sex"].iloc[0])
    assert out["GeneralHealth"].iloc[0] == "Good"


def test_output_has_exactly_the_model_features_in_order():
    out = prepare_features(pd.DataFrame([{"Sex": "Male", "State": "Ohio", "HadAngina": "Yes"}]))
    assert list(out.columns) == FEATURES
    assert "HadAngina" not in out.columns  # leakage column must never reach the model


def test_clean_dataset_removes_duplicates_and_missing_target(raw_df):
    raw = pd.concat([raw_df, raw_df.head(5)], ignore_index=True)
    raw.loc[10, TARGET] = np.nan  # a row that is not one of the duplicates
    clean, stats = clean_dataset(raw)
    assert stats["duplicates_removed"] == 5
    assert stats["missing_target_removed"] == 1
    assert set(clean[TARGET].unique()) <= {0, 1}
    assert "RaceEthnicityCategory" in clean.columns  # audit column kept for the fairness check
