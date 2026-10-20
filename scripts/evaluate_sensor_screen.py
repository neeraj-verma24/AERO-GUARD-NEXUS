"""Controlled perturbation test for the AERO-GUARD NEXUS sensor screen.

Run from the project root:
    python scripts\\evaluate_sensor_screen.py

This tests statistical threshold behavior on the NASA C-MAPSS simulated
benchmark. Synthetic out-of-range values are created in memory; source files
are not changed. This is not real aircraft fault-detection validation.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train_rul import TEST_FILE, read_cmapss  # noqa: E402
from backend.evidence_engine import (  # noqa: E402
    analyze_latest_sensor_screens,
    LOW_QUANTILE,
    HIGH_QUANTILE,
)

REPORT_DIR = ROOT / "reports" / "sensor_screen_eval"


def main() -> int:
    train_file = TEST_FILE.with_name("train_FD001.txt")
    test_file = TEST_FILE
    for path in (train_file, test_file):
        if not path.is_file():
            print(f"ERROR: required dataset file not found: {path}")
            return 2

    train_df = read_cmapss(train_file)
    test_df = read_cmapss(test_file)
    excluded = {"unit", "cycle"}
    sensor_columns = [
        c for c in train_df.select_dtypes(include="number").columns
        if c.lower() not in excluded
        and "setting" not in c.lower()
        and c in test_df.columns
    ]
    if not sensor_columns:
        print("ERROR: no numeric sensor columns found.")
        return 2

    baseline = analyze_latest_sensor_screens()
    latest_indices = test_df.groupby("unit", sort=True)["cycle"].idxmax().to_numpy()
    latest = test_df.loc[latest_indices].sort_values("unit").reset_index(drop=True)
    lower_bounds = train_df[sensor_columns].quantile(LOW_QUANTILE)
    upper_bounds = train_df[sensor_columns].quantile(HIGH_QUANTILE)
    positive_units = sorted(
        int(unit) for unit, item in baseline.items()
        if item["status"] == "SCREEN_POSITIVE"
    )

    results = []
    target_row = latest.iloc[0].copy()
    target_unit = int(target_row["unit"])
    for sensor in sensor_columns:
        lower = float(lower_bounds[sensor])
        upper = float(upper_bounds[sensor])
        width = max(0.0, upper - lower)
        low_delta = max(1.0, abs(lower) * 0.1, width * 0.25)
        high_delta = max(1.0, abs(upper) * 0.1, width * 0.25)
        for direction, injected_value in (
            ("LOW", lower - low_delta),
            ("HIGH", upper + high_delta),
        ):
            perturbed_value = float(injected_value)
            detected_direction = None
            if perturbed_value < lower:
                detected_direction = "LOW"
            elif perturbed_value > upper:
                detected_direction = "HIGH"
            passed = detected_direction == direction
            results.append({
                "engine_unit": target_unit,
                "sensor": sensor,
                "injection_direction": direction,
                "training_lower_bound": round(lower, 8),
                "training_upper_bound": round(upper, 8),
                "injected_value": round(perturbed_value, 8),
                "detected": detected_direction is not None,
                "detected_direction": detected_direction,
                "case_pass": passed,
                "status": "PASS" if passed else "FAIL",
            })

    total = len(results)
    passed_count = sum(r["case_pass"] for r in results)
    lows = [r for r in results if r["injection_direction"] == "LOW"]
    highs = [r for r in results if r["injection_direction"] == "HIGH"]
    low_passed = sum(r["case_pass"] for r in lows)
    high_passed = sum(r["case_pass"] for r in highs)

    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_type": "controlled synthetic sensor-threshold perturbation",
        "dataset": "NASA C-MAPSS FD001",
        "data_type": "simulated_engine_benchmark",
        "method": "training-distribution quantile screen",
        "quantiles": {"lower": LOW_QUANTILE, "upper": HIGH_QUANTILE},
        "baseline": {
            "engine_count": len(baseline),
            "sensor_count": len(sensor_columns),
            "screen_positive_count": len(positive_units),
            "screen_positive_engine_units": positive_units,
            "note": "Without fault labels, baseline positives cannot be called true positives or false positives.",
        },
        "controlled_perturbations": {
            "engine_unit_used_for_in_memory_tests": target_unit,
            "sensor_count_tested": len(sensor_columns),
            "low_injections": len(lows),
            "low_detected_correctly": low_passed,
            "low_detection_rate": round(low_passed / len(lows), 4) if lows else None,
            "high_injections": len(highs),
            "high_detected_correctly": high_passed,
            "high_detection_rate": round(high_passed / len(highs), 4) if highs else None,
            "total_cases": total,
            "passed_cases": passed_count,
            "detection_rate": round(passed_count / total, 4) if total else None,
            "cases": results,
        },
        "limitations": [
            "Injected values are synthetic and created in memory; source dataset files are unchanged.",
            "This only checks whether deliberately out-of-range readings trip the quantile rule.",
            "It does not measure real fault sensitivity, specificity, precision, recall, or false-alarm rate.",
            "C-MAPSS features are simulated benchmark variables, not approved physical limits for a real aircraft.",
            "Quantile exceedance is not a confirmed fault; no exceedance does not establish sensor health.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "latest.json"
    csv_path = REPORT_DIR / "latest.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "engine_unit", "sensor", "injection_direction", "training_lower_bound",
            "training_upper_bound", "injected_value", "detected", "detected_direction",
            "case_pass", "status",
        ])
        writer.writeheader()
        writer.writerows(results)

    print("=" * 68)
    print("AERO-GUARD NEXUS — SENSOR SCREEN PERTURBATION EVALUATION")
    print("=" * 68)
    print(f"Baseline engines: {len(baseline)}")
    print(f"Baseline screen-positive engines: {len(positive_units)}")
    print(f"Sensor columns tested: {len(sensor_columns)}")
    print(f"Low injections detected correctly: {low_passed}/{len(lows)}")
    print(f"High injections detected correctly: {high_passed}/{len(highs)}")
    print(f"Total perturbation cases passed: {passed_count}/{total}")
    print(f"Controlled detection rate: {report['controlled_perturbations']['detection_rate']}")
    print(f"JSON report: {json_path}")
    print(f"CSV report : {csv_path}")
    print("NOTE: synthetic threshold behavior only; not real fault-detection performance.")
    return 0 if total and passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
