from pathlib import Path

from models.train_rul import TEST_FILE, read_cmapss


TRAIN_FILE = TEST_FILE.with_name("train_FD001.txt")
LOW_QUANTILE = 0.005
HIGH_QUANTILE = 0.995


def analyze_latest_sensor_screens():
    """Screen latest FD001 test readings against training-data quantiles.

    This is an exploratory, per-sensor distribution screen. It is not a
    calibrated fault classifier, causal diagnosis, or failure probability.
    """
    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(f"NASA training dataset not found: {TRAIN_FILE}")
    if not TEST_FILE.is_file():
        raise FileNotFoundError(f"NASA test dataset not found: {TEST_FILE}")

    train_df = read_cmapss(TRAIN_FILE)
    test_df = read_cmapss(TEST_FILE)

    numeric_columns = list(train_df.select_dtypes(include="number").columns)
    excluded = {"unit", "cycle"}
    sensor_columns = [
        column for column in numeric_columns
        if column.lower() not in excluded
        and "setting" not in column.lower()
        and column in test_df.columns
    ]

    if not sensor_columns:
        raise ValueError(
            "Could not identify numeric sensor columns in NASA C-MAPSS data."
        )

    lower_bounds = train_df[sensor_columns].quantile(LOW_QUANTILE)
    upper_bounds = train_df[sensor_columns].quantile(HIGH_QUANTILE)

    latest_indices = (
        test_df.groupby("unit", sort=True)["cycle"].idxmax().to_numpy()
    )
    latest_rows = test_df.loc[latest_indices].sort_values("unit")

    results = {}
    for _, row in latest_rows.iterrows():
        flagged = []
        available = 0

        for column in sensor_columns:
            raw_value = row[column]
            if raw_value is None:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if value != value:  # NaN check
                continue

            available += 1
            lower = float(lower_bounds[column])
            upper = float(upper_bounds[column])
            if value < lower or value > upper:
                flagged.append({
                    "sensor": str(column),
                    "observed_value": round(value, 5),
                    "training_lower_quantile": round(lower, 5),
                    "training_upper_quantile": round(upper, 5),
                    "direction": "LOW" if value < lower else "HIGH",
                })

        unit = int(row["unit"])
        results[unit] = {
            "engine_unit": unit,
            "engine_id": f"FD001-{unit:03d}",
            "last_observed_cycle": int(row["cycle"]),
            "status": "SCREEN_POSITIVE" if flagged else "NO_THRESHOLD_EXCEEDANCE",
            "sensor_count_evaluated": available,
            "flagged_sensor_count": len(flagged),
            "flagged_sensors": flagged,
            "method": "Training-distribution quantile screen",
            "training_quantiles": {
                "lower": LOW_QUANTILE,
                "upper": HIGH_QUANTILE,
            },
            "limitation": (
                "Exploratory univariate screening only. A flagged reading is not "
                "a confirmed fault; no flag does not establish sensor health."
            ),
        }

    return results
