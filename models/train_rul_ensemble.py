
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
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("AERO-GUARD NEXUS | BOOTSTRAP RUL ENSEMBLE")
    print("=" * 64)

    baseline_path = OUTPUT / "rul_baseline.joblib"

    if not baseline_path.is_file():
        raise FileNotFoundError(
            "Existing baseline model not found. "
            "Train it first using python models\\train_rul.py."
        )

    print("\n[1/5] Loading dataset...")

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

    print("\n[2/5] Preparing training and test features...")

    X_train = make_features(train_df)

    max_cycle = train_df.groupby("unit")[
        "cycle"
    ].transform("max")

    training_rul = (
        max_cycle - train_df["cycle"]
    ).clip(upper=RUL_CAP).to_numpy(dtype=float)

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

    latest_rows = (
        test_df.loc[latest_indices, ["unit", "cycle"]]
        .reset_index(drop=True)
    )

    if len(actual_rul) != len(X_test):
        raise ValueError(
            "Test RUL label count does not match test engines."
        )

    print(f"Features per engine: {X_test.shape[1]}")

    print("\n[3/5] Training five engine-bootstrap models...")

    rng = np.random.default_rng(2026)

    unique_units = np.sort(
        train_df["unit"].unique()
    )

    unit_values = train_df["unit"].to_numpy()

    # Keep every row from a sampled engine together.
    unit_positions = {
        int(unit): np.flatnonzero(unit_values == unit)
        for unit in unique_units
    }

    ensemble = []

    parameters = {
        "max_iter": 160,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
    }

    for member_index in range(5):
        sampled_units = rng.choice(
            unique_units,
            size=len(unique_units),
            replace=True,
        )

        bootstrap_indices = np.concatenate([
            unit_positions[int(unit)]
            for unit in sampled_units
        ])

        model = HistGradientBoostingRegressor(
            **parameters,
            random_state=100 + member_index,
        )

        model.fit(
            X_train.iloc[bootstrap_indices],
            training_rul[bootstrap_indices],
        )

        ensemble.append(model)

        print(
            f"  Member {member_index + 1}/5 trained "
            f"using {len(sampled_units)} sampled engine IDs"
        )

    print("\n[4/5] Measuring ensemble disagreement...")

    member_predictions = np.vstack([
        np.maximum(0.0, model.predict(X_test))
        for model in ensemble
    ])

    ensemble_mean = member_predictions.mean(axis=0)
    ensemble_std = member_predictions.std(axis=0, ddof=1)

    lower_quantile = np.percentile(
        member_predictions, 10, axis=0
    )

    upper_quantile = np.percentile(
        member_predictions, 90, axis=0
    )

    # Reference model remains unchanged.
    baseline = joblib.load(baseline_path)

    baseline_prediction = np.maximum(
        0.0, baseline.predict(X_test)
    )

    baseline_metrics = regression_metrics(
        actual_rul, baseline_prediction
    )

    ensemble_metrics = regression_metrics(
        actual_rul, ensemble_mean
    )

    print("\nExisting cap-125 model:")
    print(f"  MAE:  {baseline_metrics['mae']:.2f} cycles")
    print(f"  RMSE: {baseline_metrics['rmse']:.2f} cycles")

    print("\nEnsemble mean:")
    print(f"  MAE:  {ensemble_metrics['mae']:.2f} cycles")
    print(f"  RMSE: {ensemble_metrics['rmse']:.2f} cycles")

    print(
        "\nMean member disagreement (standard deviation): "
        f"{ensemble_std.mean():.2f} cycles"
    )

    print(
        "Median member disagreement: "
        f"{np.median(ensemble_std):.2f} cycles"
    )

    print("\n[5/5] Saving the experimental ensemble...")

    ensemble_path = OUTPUT / "rul_ensemble.joblib"
    report_path = OUTPUT / "rul_ensemble_metrics.json"
    predictions_path = OUTPUT / "rul_ensemble_predictions.csv"

    bundle = {
        "models": ensemble,
        "target_cap": RUL_CAP,
        "feature_columns": list(X_train.columns),
        "dataset": "NASA C-MAPSS FD001",
        "random_seed": 2026,
        "ensemble_size": len(ensemble),
    }

    joblib.dump(bundle, ensemble_path)

    predictions = pd.DataFrame({
        "unit": latest_rows["unit"].to_numpy(),
        "last_observed_cycle": latest_rows["cycle"].to_numpy(),
        "actual_rul_cycles": actual_rul,
        "baseline_rul_cycles": np.round(
            baseline_prediction, 2
        ),
        "ensemble_mean_rul_cycles": np.round(
            ensemble_mean, 2
        ),
        "ensemble_std_cycles": np.round(
            ensemble_std, 2
        ),
        "member_p10_cycles": np.round(
            lower_quantile, 2
        ),
        "member_p90_cycles": np.round(
            upper_quantile, 2
        ),
    })

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    report = {
        "dataset": "NASA C-MAPSS FD001",
        "evaluation": "100 final-cycle test-engine observations",
        "ensemble_size": len(ensemble),
        "target_cap": RUL_CAP,
        "baseline": baseline_metrics,
        "ensemble_mean": ensemble_metrics,
        "mean_member_std_cycles": round(
            float(ensemble_std.mean()), 4
        ),
        "median_member_std_cycles": round(
            float(np.median(ensemble_std)), 4
        ),
        "note": (
            "Member spread measures model disagreement only. "
            "It is not a calibrated confidence interval or failure probability. "
            "FD001 test labels have been used in previous exploratory comparisons."
        ),
    }

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved ensemble: {ensemble_path}")
    print(f"Saved metrics: {report_path}")
    print(f"Saved predictions: {predictions_path}")

    print(
        "\nExisting rul_baseline.joblib was not modified."
    )


if __name__ == "__main__":
    main()
