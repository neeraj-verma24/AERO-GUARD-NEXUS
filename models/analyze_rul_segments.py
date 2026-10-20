
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    ROOT / "models" / "rul_target_cap_predictions.csv"
)

OUTPUT_FILE = (
    ROOT / "models" / "rul_cap_segment_analysis.csv"
)


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Experiment predictions not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = [
        "actual_rul_cycles",
        "capped_model_rul",
        "uncapped_model_rul",
    ]

    missing = [column for column in required if column not in df]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df[required].isna().any().any():
        raise ValueError("Missing values found in evaluation data.")

    # Actual test RUL determines the evaluation subgroup.
    df["rul_band"] = pd.cut(
        df["actual_rul_cycles"],
        bins=[-np.inf, 30, 125, np.inf],
        labels=[
            "Low RUL (<=30)",
            "Medium RUL (31-125)",
            "High RUL (>125)",
        ],
        right=True,
    )

    model_columns = [
        ("capped_model_rul", "Capped-target model"),
        ("uncapped_model_rul", "Uncapped-target model"),
    ]

    results = []

    print("=" * 64)
    print("AERO-GUARD NEXUS | RUL ERROR BY TEST-RUL SEGMENT")
    print("=" * 64)

    for band in df["rul_band"].cat.categories:
        group = df[df["rul_band"] == band]

        if group.empty:
            print(f"\n{band}: no engines in this group")
            continue

        actual = group["actual_rul_cycles"].to_numpy(dtype=float)

        print(f"\n{band} | Engines: {len(group)}")

        for column, model_name in model_columns:
            predicted = group[column].to_numpy(dtype=float)

            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(
                mean_squared_error(actual, predicted)
            )

            results.append({
                "rul_band": str(band),
                "model": model_name,
                "engine_count": int(len(group)),
                "mae_cycles": round(float(mae), 4),
                "rmse_cycles": round(float(rmse), 4),
            })

            print(
                f"  {model_name}: "
                f"MAE={mae:.2f}, RMSE={rmse:.2f}"
            )

    output_df = pd.DataFrame(results)
    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved segment report: {OUTPUT_FILE}")
    print("Analysis complete.")


if __name__ == "__main__":
    main()
