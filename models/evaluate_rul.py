
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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
    model_path = OUTPUT / "rul_baseline.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            "Trained model not found. Run "
            "python models\\train_rul.py first."
        )

    print("=" * 64)
    print("AERO-GUARD NEXUS | RUL BASELINE COMPARISON")
    print("=" * 64)

    # Load the same model and official FD001 evaluation files.
    model = joblib.load(model_path)
    train_df = read_cmapss(TRAIN_FILE)
    test_df = read_cmapss(TEST_FILE)

    rul_df = pd.read_csv(
        RUL_FILE, sep=r"\s+", header=None
    )
    actual = rul_df.iloc[:, 0].to_numpy(dtype=float)

    # Extract the final observed cycle for every test engine.
    latest_indices = (
        test_df.groupby("unit", sort=True)["cycle"]
        .idxmax()
        .to_numpy()
    )

    test_features = make_features(test_df)
    X_test = (
        test_features.loc[latest_indices]
        .reset_index(drop=True)
    )

    latest_rows = (
        test_df.loc[latest_indices, ["unit", "cycle"]]
        .reset_index(drop=True)
    )

    if len(actual) != len(X_test):
        raise ValueError(
            f"Test label count {len(actual)} differs from "
            f"test engine count {len(X_test)}."
        )

    # AERO-GUARD trained model predictions.
    ml_prediction = np.maximum(
        0.0, model.predict(X_test)
    )

    # Simple baseline 1: mean capped RUL label from training rows.
    max_cycle = train_df.groupby("unit")["cycle"].transform("max")
    training_rul = (max_cycle - train_df["cycle"]).clip(
        upper=RUL_CAP
    )
    mean_baseline_value = float(training_rul.mean())

    mean_prediction = np.full(
        len(actual), mean_baseline_value
    )

    # Simple baseline 2: always predict the target cap.
    cap_prediction = np.full(
        len(actual), float(RUL_CAP)
    )

    methods = {
        "AERO-GUARD ML": ml_prediction,
        "Training-mean baseline": mean_prediction,
        "Fixed-125 baseline": cap_prediction,
    }

    results = {}

    print("\nEvaluation: one final observed cycle per test engine")
    print(f"Dataset: NASA C-MAPSS FD001")
    print(f"Number of test engines: {len(actual)}")
    print(f"Training-label mean: {mean_baseline_value:.2f} cycles")

    for name, prediction in methods.items():
        metrics = regression_metrics(actual, prediction)
        results[name] = metrics

        print(f"\n{name}")
        print(f"  MAE:  {metrics['mae']:.2f} cycles")
        print(f"  RMSE: {metrics['rmse']:.2f} cycles")

    # Compute model improvement relative to both simple baselines.
    ml_metrics = results["AERO-GUARD ML"]

    improvements = {}

    for baseline_name in [
        "Training-mean baseline",
        "Fixed-125 baseline",
    ]:
        baseline = results[baseline_name]

        mae_gain = (
            (baseline["mae"] - ml_metrics["mae"])
            / baseline["mae"] * 100
            if baseline["mae"] > 0
            else None
        )

        rmse_gain = (
            (baseline["rmse"] - ml_metrics["rmse"])
            / baseline["rmse"] * 100
            if baseline["rmse"] > 0
            else None
        )

        improvements[baseline_name] = {
            "mae_reduction_percent": (
                round(mae_gain, 2)
                if mae_gain is not None else None
            ),
            "rmse_reduction_percent": (
                round(rmse_gain, 2)
                if rmse_gain is not None else None
            ),
        }

    # Save individual test-engine predictions and errors.
    report = pd.DataFrame({
        "unit": latest_rows["unit"].to_numpy(),
        "last_observed_cycle": latest_rows["cycle"].to_numpy(),
        "actual_rul_cycles": actual,
        "ml_predicted_rul_cycles": np.round(ml_prediction, 2),
        "ml_absolute_error_cycles": np.round(
            np.abs(actual - ml_prediction), 2
        ),
        "mean_baseline_prediction": np.round(
            mean_prediction, 2
        ),
        "fixed_125_prediction": np.round(
            cap_prediction, 2
        ),
    })

    report_path = OUTPUT / "rul_error_analysis.csv"
    comparison_path = OUTPUT / "rul_model_comparison.json"

    report.to_csv(report_path, index=False)

    comparison = {
        "dataset": "NASA C-MAPSS FD001",
        "evaluation_type": "official test RUL labels",
        "test_engines": int(len(actual)),
        "primary_metric": "MAE in cycles; lower is better",
        "results": results,
        "improvement_vs_baselines": improvements,
        "limitations": [
            "NASA simulated benchmark only",
            "Not validated for real military aircraft",
            "Simple baselines are comparison references, "
            "not state-of-the-art models",
        ],
    }

    comparison_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    worst = report.nlargest(
        5, "ml_absolute_error_cycles"
    )

    print("\nLargest five absolute ML prediction errors:")
    print(
        worst[
            [
                "unit",
                "actual_rul_cycles",
                "ml_predicted_rul_cycles",
                "ml_absolute_error_cycles",
            ]
        ].to_string(index=False)
    )

    print("\nSaved reports:")
    print(f"  {report_path}")
    print(f"  {comparison_path}")

    print(
        "\nComparison complete. Lower MAE/RMSE indicates "
        "smaller prediction error on this evaluation set."
    )


if __name__ == "__main__":
    main()
