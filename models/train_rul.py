
from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit


# -------------------------------------------------------
# AERO-GUARD NEXUS - C-MAPSS FD001 RUL baseline
# All dataset records are NASA simulated engine data.
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cmapss"
OUTPUT = ROOT / "models"

TRAIN_FILE = DATA / "train_FD001.txt"
TEST_FILE = DATA / "test_FD001.txt"
RUL_FILE = DATA / "RUL_FD001.txt"

WINDOW = 20
RUL_CAP = 125
RANDOM_STATE = 42

SETTING_COLS = [
    "setting_1", "setting_2", "setting_3"
]

SENSOR_COLS = [
    f"sensor_{i}" for i in range(1, 22)
]

COLUMNS = ["unit", "cycle"] + SETTING_COLS + SENSOR_COLS


def read_cmapss(path):
    """Read and validate a whitespace-separated C-MAPSS file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Check that the FD001 files are inside data/cmapss."
        )

    df = pd.read_csv(path, sep=r"\s+", header=None)

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name}: expected {len(COLUMNS)} columns, "
            f"but found {df.shape[1]}."
        )

    df.columns = COLUMNS
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    if df.isna().any().any():
        raise ValueError(f"Unexpected missing values in {path.name}.")

    return df.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)


def make_features(df):
    """
    Build features using only current and past sensor values.
    Rolling windows never use future readings.
    """
    base_cols = ["cycle"] + SETTING_COLS + SENSOR_COLS

    base = df[base_cols].reset_index(drop=True)

    grouped = df.groupby("unit", sort=False)[SENSOR_COLS]

    rolling_mean = (
        grouped
        .rolling(window=WINDOW, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    rolling_std = (
        grouped
        .rolling(window=WINDOW, min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0.0)
    )

    rolling_mean.columns = [
        f"{col}_mean_{WINDOW}" for col in SENSOR_COLS
    ]

    rolling_std.columns = [
        f"{col}_std_{WINDOW}" for col in SENSOR_COLS
    ]

    return pd.concat(
        [
            base,
            rolling_mean.reset_index(drop=True),
            rolling_std.reset_index(drop=True),
        ],
        axis=1,
    )


def regression_metrics(actual, predicted):
    """Calculate MAE and RMSE."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(
            np.sqrt(mean_squared_error(actual, predicted))
        ),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AERO-GUARD NEXUS | RUL BASELINE TRAINING")
    print("=" * 60)

    print("\n[1/6] Reading NASA C-MAPSS FD001 files...")

    train_df = read_cmapss(TRAIN_FILE)
    test_df = read_cmapss(TEST_FILE)

    rul_df = pd.read_csv(RUL_FILE, sep=r"\s+", header=None)
    true_test_rul = rul_df.iloc[:, 0].to_numpy(dtype=float)

    train_units = train_df["unit"].nunique()
    test_units = test_df["unit"].nunique()

    if len(true_test_rul) != test_units:
        raise ValueError(
            f"Test RUL labels: {len(true_test_rul)}, "
            f"but test engines: {test_units}."
        )

    print(f"Training rows: {len(train_df):,}")
    print(f"Training engines: {train_units}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Test engines: {test_units}")

    print("\n[2/6] Creating training RUL labels...")

    max_cycle = train_df.groupby("unit")["cycle"].transform("max")

    # Standard piecewise-linear target used in many C-MAPSS baselines.
    train_rul = (max_cycle - train_df["cycle"]).clip(
        upper=RUL_CAP
    )

    X = make_features(train_df)
    y = train_rul.to_numpy(dtype=float)
    groups = train_df["unit"].to_numpy()

    print(f"Features per row: {X.shape[1]}")
    print(f"RUL target cap: {RUL_CAP} cycles")

    print("\n[3/6] Performing engine-wise validation...")

    # All observations from an engine stay in one split.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    fit_idx, val_idx = next(
        splitter.split(X, y, groups=groups)
    )

    model_params = {
        "max_iter": 160,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "random_state": RANDOM_STATE,
    }

    model = HistGradientBoostingRegressor(**model_params)

    model.fit(X.iloc[fit_idx], y[fit_idx])

    val_pred = np.maximum(
        0.0, model.predict(X.iloc[val_idx])
    )

    validation = regression_metrics(y[val_idx], val_pred)

    print(
        f"Validation engines: "
        f"{len(np.unique(groups[val_idx]))}"
    )
    print(f"Validation MAE:  {validation['mae']:.2f} cycles")
    print(f"Validation RMSE: {validation['rmse']:.2f} cycles")

    print("\n[4/6] Training final model on all training engines...")

    model.fit(X, y)

    print("\n[5/6] Evaluating official FD001 test endpoints...")

    test_features = make_features(test_df)

    # Use only the latest observed cycle for each test engine.
    last_indices = (
        test_df.groupby("unit", sort=True)["cycle"]
        .idxmax()
        .to_numpy()
    )

    X_test = test_features.loc[last_indices].reset_index(drop=True)

    test_unit_ids = (
        test_df.loc[last_indices, "unit"]
        .to_numpy(dtype=int)
    )

    test_pred = np.maximum(
        0.0, model.predict(X_test)
    )

    test_metrics = regression_metrics(
        true_test_rul, test_pred
    )

    print(f"Test MAE:  {test_metrics['mae']:.2f} cycles")
    print(f"Test RMSE: {test_metrics['rmse']:.2f} cycles")

    print("\n[6/6] Saving local model and evaluation results...")

    model_path = OUTPUT / "rul_baseline.joblib"
    metrics_path = OUTPUT / "rul_metrics.json"
    predictions_path = OUTPUT / "rul_test_predictions.csv"
    metadata_path = OUTPUT / "rul_metadata.json"

    joblib.dump(model, model_path)

    predictions = pd.DataFrame({
        "unit": test_unit_ids,
        "actual_rul": true_test_rul,
        "predicted_rul": np.round(test_pred, 2),
        "absolute_error": np.round(
            np.abs(true_test_rul - test_pred), 2
        ),
    })

    predictions.to_csv(predictions_path, index=False)

    metrics = {
        "dataset": "NASA C-MAPSS FD001",
        "data_type": "simulated_engine_benchmark",
        "validation": validation,
        "official_test": test_metrics,
        "training_rows": int(len(train_df)),
        "training_engines": int(train_units),
        "test_engines": int(test_units),
        "feature_count": int(X.shape[1]),
        "validation_method": "GroupShuffleSplit by engine ID",
        "rul_target_cap": RUL_CAP,
    }

    metadata = {
        "model": "HistGradientBoostingRegressor",
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": "NASA C-MAPSS FD001",
        "rolling_window_cycles": WINDOW,
        "feature_columns": list(X.columns),
        "training_target": (
            "max observed training cycle - current cycle, "
            f"capped at {RUL_CAP}"
        ),
        "model_parameters": model_params,
    }

    metrics_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved files:")
    print(f"  Model:       {model_path}")
    print(f"  Metrics:     {metrics_path}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Metadata:    {metadata_path}")

    print("\nBaseline training and evaluation completed.")
    print(
        "Note: results apply to the C-MAPSS benchmark, "
        "not certified aircraft maintenance."
    )


if __name__ == "__main__":
    main()
