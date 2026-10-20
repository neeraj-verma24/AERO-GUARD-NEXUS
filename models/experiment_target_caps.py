
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


CANDIDATE_CAPS = [75, 100, 125, 150, 200, None]


def main():
    print("=" * 64)
    print("AERO-GUARD NEXUS | TRAINING-ONLY TARGET CAP EXPERIMENT")
    print("=" * 64)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = read_cmapss(TRAIN_FILE)

    # Ground-truth remaining cycles for completed training trajectories.
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    raw_rul = (max_cycle - df["cycle"]).to_numpy(dtype=float)

    X = make_features(df)
    groups = df["unit"].to_numpy()

    # Every observation from an engine stays in one split.
    split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, val_idx = next(
        split.split(X, raw_rul, groups=groups)
    )

    y_train_raw = raw_rul[train_idx]
    y_val_actual = raw_rul[val_idx]

    print(f"Training engines: {len(np.unique(groups[train_idx]))}")
    print(f"Validation engines: {len(np.unique(groups[val_idx]))}")
    print(f"Validation observations: {len(val_idx)}")
    print("Metric target: uncapped remaining cycles")
    print(
        "Note: validation includes all observed cycles of held-out "
        "engines, not just one final cutoff per engine."
    )

    results = []

    params = {
        "max_iter": 160,
        "learning_rate": 0.08,
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "random_state": 42,
    }

    for cap in CANDIDATE_CAPS:
        if cap is None:
            y_train = y_train_raw.copy()
            label = "Uncapped"
        else:
            y_train = np.minimum(y_train_raw, cap)
            label = str(cap)

        model = HistGradientBoostingRegressor(**params)
        model.fit(X.iloc[train_idx], y_train)

        predictions = np.maximum(
            0.0,
            model.predict(X.iloc[val_idx]),
        )

        mae = mean_absolute_error(
            y_val_actual, predictions
        )
        rmse = np.sqrt(
            mean_squared_error(y_val_actual, predictions)
        )

        result = {
            "target_cap": label,
            "mae_cycles": round(float(mae), 4),
            "rmse_cycles": round(float(rmse), 4),
            "validation_engines": int(
                len(np.unique(groups[val_idx]))
            ),
            "validation_rows": int(len(val_idx)),
        }

        results.append(result)

        print(
            f"\nTarget cap: {label}"
            f"\n  Validation MAE:  {mae:.2f} cycles"
            f"\n  Validation RMSE: {rmse:.2f} cycles"
        )

    report = pd.DataFrame(results).sort_values(
        "mae_cycles"
    ).reset_index(drop=True)

    csv_path = OUTPUT / "target_cap_validation.csv"
    json_path = OUTPUT / "target_cap_validation.json"

    report.to_csv(csv_path, index=False)

    json_path.write_text(
        json.dumps(
            {
                "dataset": "NASA C-MAPSS FD001 training trajectories",
                "split": "GroupShuffleSplit by engine ID",
                "random_state": 42,
                "selection_metric": "MAE against uncapped RUL",
                "results_sorted_by_validation_mae": (
                    report.to_dict(orient="records")
                ),
                "note": (
                    "Validation rows are correlated observations from "
                    "held-out engines. This is a model-selection experiment, "
                    "not an official test-set result or aircraft validation."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nResults sorted by validation MAE:")
    print(report.to_string(index=False))

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")

    print(
        "\nNo existing model was overwritten. "
        "Review the validation results before selecting a cap."
    )


if __name__ == "__main__":
    main()
