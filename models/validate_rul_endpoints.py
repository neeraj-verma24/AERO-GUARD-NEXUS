
import json

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

from train_rul import (
    OUTPUT,
    TRAIN_FILE,
    make_features,
    read_cmapss,
)


CAPS = [75, 100, 125, 150, 200, None]
RANDOM_STATE = 42
CUTOFFS_PER_ENGINE = 5


def calculate_metrics(actual, predicted):
    return {
        "mae_cycles": round(
            float(mean_absolute_error(actual, predicted)), 4
        ),
        "rmse_cycles": round(
            float(np.sqrt(mean_squared_error(actual, predicted))), 4
        ),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("AERO-GUARD NEXUS | ENDPOINT-STYLE VALIDATION")
    print("=" * 64)

    all_data = read_cmapss(TRAIN_FILE)
    units = np.sort(all_data["unit"].unique())

    # Split by engine ID, not by individual rows.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    fit_idx, heldout_idx = next(
        splitter.split(units, groups=units)
    )

    fit_units = units[fit_idx]
    heldout_units = units[heldout_idx]

    fit_df = all_data[
        all_data["unit"].isin(fit_units)
    ].copy()

    heldout_df = all_data[
        all_data["unit"].isin(heldout_units)
    ].copy()

    print(f"Training engines: {len(fit_units)}")
    print(f"Held-out engines: {len(heldout_units)}")

    print("\n[1/4] Preparing training features...")

    X_train = make_features(fit_df)

    full_life = fit_df.groupby("unit")[
        "cycle"
    ].transform("max")

    raw_train_rul = (
        full_life - fit_df["cycle"]
    ).to_numpy(dtype=float)

    # Generate pseudo-test endpoints using only held-out trajectories.
    # A target RUL is chosen first; only observations up to its
    # corresponding cutoff are presented to the model.
    print("\n[2/4] Generating held-out partial-history endpoints...")

    rng = np.random.default_rng(RANDOM_STATE)

    val_feature_rows = []
    val_labels = []
    val_metadata = []

    for unit in heldout_units:
        engine = (
            heldout_df[heldout_df["unit"] == unit]
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        life_end = int(engine["cycle"].max())
        largest_target = min(150, life_end - 1)

        if largest_target < 1:
            continue

        possible_targets = np.arange(
            1, largest_target + 1
        )

        number_to_sample = min(
            CUTOFFS_PER_ENGINE,
            len(possible_targets),
        )

        chosen_targets = rng.choice(
            possible_targets,
            size=number_to_sample,
            replace=False,
        )

        for target_rul in chosen_targets:
            cutoff_cycle = life_end - int(target_rul)

            history = engine[
                engine["cycle"] <= cutoff_cycle
            ].copy()

            if history.empty:
                continue

            # Features use only the observed history up to this cutoff.
            feature_row = make_features(history).iloc[-1]

            val_feature_rows.append(feature_row)
            val_labels.append(float(target_rul))

            val_metadata.append({
                "unit": int(unit),
                "cutoff_cycle": cutoff_cycle,
                "actual_remaining_cycles": int(target_rul),
            })

    X_val = pd.DataFrame(
        val_feature_rows,
        columns=X_train.columns,
    ).reset_index(drop=True)

    y_val = np.asarray(val_labels, dtype=float)

    if len(y_val) == 0:
        raise RuntimeError("No validation endpoints were generated.")

    if not np.isfinite(X_val.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found in validation features.")

    print(f"Endpoint observations: {len(y_val)}")
    print(f"Validation engines: {len(heldout_units)}")

    print(
        "Cutoff sampling: random remaining life from 1 to "
        "at most 150 cycles."
    )
    print(
        "Note: this is a synthetic censoring protocol, "
        "not the official NASA test distribution."
    )

    print("\n[3/4] Comparing target caps...")

    params = {
        "max_iter": 160,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "random_state": RANDOM_STATE,
    }

    results = []
    prediction_columns = {
        "unit": [item["unit"] for item in val_metadata],
        "cutoff_cycle": [
            item["cutoff_cycle"] for item in val_metadata
        ],
        "actual_remaining_cycles": y_val,
    }

    for cap in CAPS:
        if cap is None:
            target = raw_train_rul.copy()
            label = "Uncapped"
        else:
            target = np.minimum(
                raw_train_rul,
                float(cap),
            )
            label = str(cap)

        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, target)

        prediction = np.maximum(
            0.0,
            model.predict(X_val),
        )

        metrics = calculate_metrics(y_val, prediction)

        results.append({
            "target_cap": label,
            **metrics,
        })

        prediction_columns[
            f"prediction_cap_{label}"
        ] = np.round(prediction, 2)

        print(
            f"\nCap {label}: "
            f"MAE={metrics['mae_cycles']:.2f}, "
            f"RMSE={metrics['rmse_cycles']:.2f}"
        )

    print("\n[4/4] Saving validation results...")

    results_df = (
        pd.DataFrame(results)
        .sort_values("mae_cycles")
        .reset_index(drop=True)
    )

    metrics_path = OUTPUT / "endpoint_validation.json"
    predictions_path = OUTPUT / "endpoint_validation_predictions.csv"

    results_df.to_csv(
        OUTPUT / "endpoint_validation_summary.csv",
        index=False,
    )

    pd.DataFrame(prediction_columns).to_csv(
        predictions_path,
        index=False,
    )

    report = {
        "dataset": "NASA C-MAPSS FD001 training trajectories",
        "random_state": RANDOM_STATE,
        "training_engine_count": int(len(fit_units)),
        "heldout_engine_count": int(len(heldout_units)),
        "endpoint_observation_count": int(len(y_val)),
        "cutoffs_per_engine": CUTOFFS_PER_ENGINE,
        "selection_metric": "MAE in remaining cycles",
        "results_sorted_by_mae": results_df.to_dict(
            orient="records"
        ),
        "limitations": [
            "Synthetic cutoff distribution, not official test distribution",
            "Multiple endpoints per held-out engine are correlated",
            "Results are for model selection, not aircraft validation",
        ],
    }

    metrics_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nResults sorted by validation MAE:")
    print(results_df.to_string(index=False))

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {predictions_path}")
    print("\nExisting production model remains unchanged.")


if __name__ == "__main__":
    main()
