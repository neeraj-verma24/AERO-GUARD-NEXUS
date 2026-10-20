
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from train_rul import (
    OUTPUT,
    RUL_CAP,
    RUL_FILE,
    TEST_FILE,
    TRAIN_FILE,
    make_features,
    read_cmapss,
    regression_metrics,
)


def main():
    print("=" * 64)
    print("AERO-GUARD NEXUS | RUL TARGET-CAP EXPERIMENT")
    print("=" * 64)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    production_path = OUTPUT / "rul_baseline.joblib"

    if not production_path.exists():
        raise FileNotFoundError(
            "Existing baseline model not found. "
            "Run python models\\train_rul.py first."
        )

    print("\n[1/5] Loading training and test data...")

    train_df = read_cmapss(TRAIN_FILE)
    test_df = read_cmapss(TEST_FILE)

    rul_df = pd.read_csv(
        RUL_FILE, sep=r"\s+", header=None
    )
    actual_rul = rul_df.iloc[:, 0].to_numpy(dtype=float)

    print(f"Training engines: {train_df['unit'].nunique()}")
    print(f"Test engines: {test_df['unit'].nunique()}")

    print("\n[2/5] Preparing equivalent features...")

    X_train = make_features(train_df)
    X_test_all = make_features(test_df)

    last_indices = (
        test_df.groupby("unit", sort=True)["cycle"]
        .idxmax()
        .to_numpy()
    )

    X_test = (
        X_test_all.loc[last_indices]
        .reset_index(drop=True)
    )

    test_rows = (
        test_df.loc[last_indices, ["unit", "cycle"]]
        .reset_index(drop=True)
    )

    if len(actual_rul) != len(X_test):
        raise ValueError(
            "Test engine and RUL label counts do not match."
        )

    max_cycle = train_df.groupby("unit")[
        "cycle"
    ].transform("max")

    raw_rul = (
        max_cycle - train_df["cycle"]
    ).to_numpy(dtype=float)

    capped_rul = np.minimum(raw_rul, RUL_CAP)

    print(f"Features: {X_train.shape[1]}")
    print(f"Capped target: maximum {RUL_CAP} cycles")
    print(
        "Uncapped target maximum: "
        f"{raw_rul.max():.0f} cycles"
    )
    print(
        "Official test labels above cap: "
        f"{int((actual_rul > RUL_CAP).sum())}"
    )

    # Keep the current production model unchanged.
    print("\n[3/5] Evaluating existing capped-target model...")

    existing_model = joblib.load(production_path)
    existing_pred = np.maximum(
        0.0, existing_model.predict(X_test)
    )

    capped_metrics = regression_metrics(
        actual_rul, existing_pred
    )

    print(
        f"Capped model MAE: "
        f"{capped_metrics['mae']:.2f} cycles"
    )
    print(
        f"Capped model RMSE: "
        f"{capped_metrics['rmse']:.2f} cycles"
    )

    print("\n[4/5] Training uncapped-target experiment...")

    params = {
        "max_iter": 160,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "random_state": 42,
    }

    alternative_model = HistGradientBoostingRegressor(
        **params
    )

    alternative_model.fit(X_train, raw_rul)

    alternative_pred = np.maximum(
        0.0, alternative_model.predict(X_test)
    )

    uncapped_metrics = regression_metrics(
        actual_rul, alternative_pred
    )

    print(
        f"Uncapped model MAE: "
        f"{uncapped_metrics['mae']:.2f} cycles"
    )
    print(
        f"Uncapped model RMSE: "
        f"{uncapped_metrics['rmse']:.2f} cycles"
    )

    print("\n[5/5] Saving the isolated experiment...")

    alternative_path = OUTPUT / "rul_uncapped_experiment.joblib"
    report_path = OUTPUT / "rul_target_cap_experiment.json"
    predictions_path = OUTPUT / "rul_target_cap_predictions.csv"

    joblib.dump(alternative_model, alternative_path)

    comparison = {
        "dataset": "NASA C-MAPSS FD001",
        "evaluation": "same official test labels",
        "test_engines": int(len(actual_rul)),
        "current_model": {
            "target": f"RUL capped at {RUL_CAP} cycles",
            **capped_metrics,
        },
        "alternative_model": {
            "target": "uncapped training RUL",
            **uncapped_metrics,
        },
        "delta_uncapped_minus_capped": {
            "mae_cycles": round(
                uncapped_metrics["mae"] - capped_metrics["mae"],
                4,
            ),
            "rmse_cycles": round(
                uncapped_metrics["rmse"] - capped_metrics["rmse"],
                4,
            ),
        },
        "note": (
            "Exploratory experiment using a NASA simulated "
            "benchmark; not validation for real aircraft."
        ),
    }

    report_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    predictions = pd.DataFrame({
        "unit": test_rows["unit"].to_numpy(),
        "last_observed_cycle": test_rows["cycle"].to_numpy(),
        "actual_rul_cycles": actual_rul,
        "capped_model_rul": np.round(existing_pred, 2),
        "uncapped_model_rul": np.round(alternative_pred, 2),
        "capped_absolute_error": np.round(
            np.abs(actual_rul - existing_pred), 2
        ),
        "uncapped_absolute_error": np.round(
            np.abs(actual_rul - alternative_pred), 2
        ),
    })

    predictions.to_csv(predictions_path, index=False)

    print("\nSaved alternative model:")
    print(alternative_path)

    print("Saved comparison report:")
    print(report_path)

    print("Saved per-engine predictions:")
    print(predictions_path)

    print("\nExperiment completed.")
    print(
        "The existing production model was not overwritten."
    )


if __name__ == "__main__":
    main()
