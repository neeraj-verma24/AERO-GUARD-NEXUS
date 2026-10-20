
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor

from train_rul import (
    OUTPUT,
    RUL_FILE,
    TEST_FILE,
    TRAIN_FILE,
    make_features,
    read_cmapss,
    regression_metrics,
)


TARGET_CAP = 200


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("AERO-GUARD NEXUS | CAP-200 CANDIDATE MODEL")
    print("=" * 64)

    baseline_path = OUTPUT / "rul_baseline.joblib"

    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Existing baseline not found: {baseline_path}"
        )

    print("\n[1/5] Loading data...")

    train_df = read_cmapss(TRAIN_FILE)
    test_df = read_cmapss(TEST_FILE)

    rul_df = pd.read_csv(
        RUL_FILE,
        sep=r"\s+",
        header=None,
    )
    actual_rul = rul_df.iloc[:, 0].to_numpy(dtype=float)

    print(f"Training engines: {train_df['unit'].nunique()}")
    print(f"Test engines: {test_df['unit'].nunique()}")

    print("\n[2/5] Creating features and cap-200 labels...")

    X_train = make_features(train_df)

    max_cycle = train_df.groupby("unit")[
        "cycle"
    ].transform("max")

    training_rul = (
        max_cycle - train_df["cycle"]
    ).clip(upper=TARGET_CAP).to_numpy(dtype=float)

    X_test_all = make_features(test_df)

    latest_indices = (
        test_df.groupby("unit", sort=True)["cycle"]
        .idxmax()
        .to_numpy()
    )

    X_test = (
        X_test_all.loc[latest_indices]
        .reset_index(drop=True)
    )

    test_rows = (
        test_df.loc[latest_indices, ["unit", "cycle"]]
        .reset_index(drop=True)
    )

    if len(actual_rul) != len(X_test):
        raise ValueError(
            "Test RUL label count does not match test-engine count."
        )

    print(f"Features: {X_train.shape[1]}")
    print(f"Target cap: {TARGET_CAP} cycles")

    print("\n[3/5] Training candidate model...")

    candidate = HistGradientBoostingRegressor(
        max_iter=160,
        learning_rate=0.08,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    candidate.fit(X_train, training_rul)

    candidate_prediction = np.maximum(
        0.0,
        candidate.predict(X_test),
    )

    print("\n[4/5] Comparing on FD001 final-cycle labels...")

    baseline = joblib.load(baseline_path)

    baseline_prediction = np.maximum(
        0.0,
        baseline.predict(X_test),
    )

    baseline_metrics = regression_metrics(
        actual_rul,
        baseline_prediction,
    )

    candidate_metrics = regression_metrics(
        actual_rul,
        candidate_prediction,
    )

    print("\nExisting cap-125 baseline:")
    print(f"  MAE:  {baseline_metrics['mae']:.2f} cycles")
    print(f"  RMSE: {baseline_metrics['rmse']:.2f} cycles")

    print("\nNew cap-200 candidate:")
    print(f"  MAE:  {candidate_metrics['mae']:.2f} cycles")
    print(f"  RMSE: {candidate_metrics['rmse']:.2f} cycles")

    print("\n[5/5] Saving candidate separately...")

    model_path = OUTPUT / "rul_cap200_candidate.joblib"
    metrics_path = OUTPUT / "rul_cap200_candidate_metrics.json"
    predictions_path = OUTPUT / "rul_cap200_candidate_predictions.csv"

    joblib.dump(candidate, model_path)

    predictions_df = pd.DataFrame({
        "unit": test_rows["unit"].to_numpy(),
        "last_observed_cycle": test_rows["cycle"].to_numpy(),
        "actual_rul_cycles": actual_rul,
        "baseline_cap125_prediction": np.round(
            baseline_prediction, 2
        ),
        "candidate_cap200_prediction": np.round(
            candidate_prediction, 2
        ),
        "baseline_absolute_error": np.round(
            np.abs(actual_rul - baseline_prediction), 2
        ),
        "candidate_absolute_error": np.round(
            np.abs(actual_rul - candidate_prediction), 2
        ),
    })

    predictions_df.to_csv(
        predictions_path,
        index=False,
    )

    report = {
        "dataset": "NASA C-MAPSS FD001",
        "evaluation": "100 final-cycle test-engine observations",
        "candidate_target_cap": TARGET_CAP,
        "baseline": {
            "model": "Existing cap-125 model",
            **baseline_metrics,
        },
        "candidate": {
            "model": "HistGradientBoostingRegressor",
            "target_cap": TARGET_CAP,
            **candidate_metrics,
        },
        "delta_candidate_minus_baseline": {
            "mae_cycles": round(
                candidate_metrics["mae"] - baseline_metrics["mae"],
                4,
            ),
            "rmse_cycles": round(
                candidate_metrics["rmse"] - baseline_metrics["rmse"],
                4,
            ),
        },
        "caveat": (
            "The FD001 test labels have already been used for exploratory "
            "model comparisons. Treat this as an exploratory comparison, "
            "not a completely untouched final evaluation."
        ),
        "deployment_decision": "candidate_only_existing_model_unchanged",
    }

    metrics_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nCandidate model: {model_path}")
    print(f"Metrics report: {metrics_path}")
    print(f"Per-engine report: {predictions_path}")

    print("\nExisting rul_baseline.joblib was not modified.")
    print("Candidate experiment completed.")


if __name__ == "__main__":
    main()
