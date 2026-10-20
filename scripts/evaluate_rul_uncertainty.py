"""Exploratory RUL ensemble uncertainty evaluation for AERO-GUARD NEXUS.

Run from repository root:
    python scripts\\evaluate_rul_uncertainty.py

Reads saved FD001 prediction CSV only; writes JSON/CSV reports. Does not modify
models or project database. This is exploratory analysis, not safety validation.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "models" / "rul_ensemble_predictions.csv"
OUT_DIR = ROOT / "reports" / "rul_uncertainty_eval"
LOW_RUL_REVIEW_CYCLES = 30.0
HIGH_DISAGREEMENT_REVIEW_CYCLES = 8.0
REQUIRED = {
    "unit", "actual_rul_cycles", "baseline_rul_cycles",
    "ensemble_mean_rul_cycles", "ensemble_std_cycles",
    "member_p10_cycles", "member_p90_cycles",
}


def safe_float(value):
    try:
        value = float(value)
        return round(value, 6) if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else None


def metrics_for(frame, prediction_col):
    errors = frame[prediction_col].to_numpy(dtype=float) - frame["actual_rul_cycles"].to_numpy(dtype=float)
    abs_errors = np.abs(errors)
    return {
        "mae_cycles": safe_float(np.mean(abs_errors)),
        "rmse_cycles": safe_float(rmse(errors)),
        "mean_signed_error_cycles": safe_float(np.mean(errors)),
        "median_absolute_error_cycles": safe_float(np.median(abs_errors)),
    }


def main():
    if not INPUT.is_file():
        print(f"ERROR: Required prediction file not found: {INPUT}")
        print("Make sure models/rul_ensemble_predictions.csv exists.")
        return 2

    try:
        df = pd.read_csv(INPUT)
    except Exception as exc:
        print(f"ERROR: Could not read input CSV: {type(exc).__name__}: {exc}")
        return 2

    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        print("ERROR: Required columns are missing:", ", ".join(missing))
        print("Columns found:", list(df.columns))
        return 2

    numeric_cols = sorted(REQUIRED - {"unit"})
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=list(REQUIRED)).copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=list(REQUIRED)).copy()
    df = df[(df["ensemble_std_cycles"] >= 0) & (df["member_p90_cycles"] >= df["member_p10_cycles"])].copy()
    if df.empty:
        print("ERROR: No valid prediction rows remain after numeric and range checks.")
        return 2

    df["baseline_abs_error_cycles"] = (df["baseline_rul_cycles"] - df["actual_rul_cycles"]).abs()
    df["ensemble_abs_error_cycles"] = (df["ensemble_mean_rul_cycles"] - df["actual_rul_cycles"]).abs()
    df["ensemble_signed_error_cycles"] = df["ensemble_mean_rul_cycles"] - df["actual_rul_cycles"]
    df["interval_width_cycles"] = df["member_p90_cycles"] - df["member_p10_cycles"]
    df["member_p10_p90_contains_actual"] = (
        (df["actual_rul_cycles"] >= df["member_p10_cycles"])
        & (df["actual_rul_cycles"] <= df["member_p90_cycles"])
    )

    # This reproduces current prototype review rules, not validated safety thresholds.
    df["low_rul_trigger"] = df["ensemble_mean_rul_cycles"] <= LOW_RUL_REVIEW_CYCLES
    df["high_disagreement_trigger"] = df["ensemble_std_cycles"] >= HIGH_DISAGREEMENT_REVIEW_CYCLES
    df["prototype_review_trigger"] = df["low_rul_trigger"] | df["high_disagreement_trigger"]
    df["actual_low_rul_label"] = df["actual_rul_cycles"] <= LOW_RUL_REVIEW_CYCLES

    pearson = df["ensemble_std_cycles"].corr(df["ensemble_abs_error_cycles"], method="pearson")
    spearman = df["ensemble_std_cycles"].corr(df["ensemble_abs_error_cycles"], method="spearman")

    bins = [-np.inf, 2.0, 5.0, HIGH_DISAGREEMENT_REVIEW_CYCLES, np.inf]
    labels = ["<2 cycles", "2 to <5 cycles", "5 to <8 cycles", ">=8 cycles"]
    df["disagreement_band"] = pd.cut(df["ensemble_std_cycles"], bins=bins, labels=labels, right=False)
    band_rows = []
    for label in labels:
        part = df[df["disagreement_band"] == label]
        band_rows.append({
            "band": label,
            "count": int(len(part)),
            "mean_disagreement_cycles": safe_float(part["ensemble_std_cycles"].mean()) if len(part) else None,
            "mae_cycles": safe_float(part["ensemble_abs_error_cycles"].mean()) if len(part) else None,
            "rmse_cycles": safe_float(rmse(part["ensemble_signed_error_cycles"])) if len(part) else None,
            "median_absolute_error_cycles": safe_float(part["ensemble_abs_error_cycles"].median()) if len(part) else None,
        })

    actual_low = df["actual_low_rul_label"]
    review = df["prototype_review_trigger"]
    actual_low_count = int(actual_low.sum())
    actual_low_captured = int((review & actual_low).sum())
    review_count = int(review.sum())
    baseline_metrics = metrics_for(df, "baseline_rul_cycles")
    ensemble_metrics = metrics_for(df, "ensemble_mean_rul_cycles")
    coverage = safe_float(df["member_p10_p90_contains_actual"].mean())

    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_type": "exploratory model-error and ensemble-disagreement analysis",
        "input_file": str(INPUT),
        "dataset": "NASA C-MAPSS FD001",
        "data_type": "simulated_engine_benchmark",
        "valid_rows": int(len(df)),
        "rows_dropped_for_missing_or_invalid_numeric_data": int(before - len(df)),
        "metrics": {
            "baseline": baseline_metrics,
            "ensemble_mean": ensemble_metrics,
            "ensemble_mae_improvement_over_baseline_cycles": safe_float(baseline_metrics["mae_cycles"] - ensemble_metrics["mae_cycles"]),
            "ensemble_rmse_change_vs_baseline_cycles_positive_means_worse": safe_float(ensemble_metrics["rmse_cycles"] - baseline_metrics["rmse_cycles"]),
            "absolute_error_vs_disagreement_pearson_correlation": safe_float(pearson),
            "absolute_error_vs_disagreement_spearman_correlation": safe_float(spearman),
            "member_p10_p90_empirical_test_label_coverage": coverage,
            "member_p10_p90_mean_width_cycles": safe_float(df["interval_width_cycles"].mean()),
            "member_p10_p90_median_width_cycles": safe_float(df["interval_width_cycles"].median()),
            "member_p10_p90_inside_count": int(df["member_p10_p90_contains_actual"].sum()),
            "member_p10_p90_outside_count": int((~df["member_p10_p90_contains_actual"]).sum()),
        },
        "prototype_review_threshold_diagnostics": {
            "low_rul_review_threshold_cycles": LOW_RUL_REVIEW_CYCLES,
            "high_disagreement_review_threshold_cycles": HIGH_DISAGREEMENT_REVIEW_CYCLES,
            "reviewed_count": review_count,
            "review_rate": safe_float(review.mean()),
            "low_rul_trigger_count": int(df["low_rul_trigger"].sum()),
            "high_disagreement_trigger_count": int(df["high_disagreement_trigger"].sum()),
            "actual_rul_le_30_count": actual_low_count,
            "actual_low_rul_captured_by_any_prototype_trigger": actual_low_captured,
            "diagnostic_capture_rate_of_actual_rul_le_30": safe_float(actual_low_captured / actual_low_count) if actual_low_count else None,
            "review_flags_where_actual_rul_gt_30_count": int((review & ~actual_low).sum()),
            "note": "Diagnostic only on previously explored FD001 test labels. Review flags are not safety classifications, and thresholds are not validated operational limits.",
        },
        "disagreement_bands": band_rows,
        "interpretation_limits": [
            "Ensemble standard deviation is model disagreement, not a calibrated confidence interval or failure probability.",
            "Member p10/p90 are quantiles across only five ensemble models; empirical label coverage is descriptive and not proof of statistical calibration.",
            "FD001 test labels were used in prior exploratory model comparisons; these numbers are not an untouched final benchmark.",
            "NASA C-MAPSS FD001 is a simulated benchmark, not real military aircraft data.",
            "Low-RUL/disagreement thresholds are prototype review rules, not approved maintenance or airworthiness limits.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "latest.json"
    csv_path = OUT_DIR / "latest.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out = df.copy()
    out["member_p10_p90_contains_actual"] = out["member_p10_p90_contains_actual"].astype(bool)
    out["low_rul_trigger"] = out["low_rul_trigger"].astype(bool)
    out["high_disagreement_trigger"] = out["high_disagreement_trigger"].astype(bool)
    out["prototype_review_trigger"] = out["prototype_review_trigger"].astype(bool)
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("AERO-GUARD NEXUS — RUL UNCERTAINTY & EVIDENCE-GATE EVALUATION")
    print("=" * 70)
    print(f"Valid FD001 test rows: {len(df)}")
    print(f"Baseline MAE/RMSE: {baseline_metrics['mae_cycles']} / {baseline_metrics['rmse_cycles']}")
    print(f"Ensemble MAE/RMSE: {ensemble_metrics['mae_cycles']} / {ensemble_metrics['rmse_cycles']}")
    print(f"MAE improvement (cycles): {report['metrics']['ensemble_mae_improvement_over_baseline_cycles']}")
    print(f"RMSE change, positive means worse: {report['metrics']['ensemble_rmse_change_vs_baseline_cycles_positive_means_worse']}")
    print(f"Disagreement vs absolute error: Pearson={safe_float(pearson)}, Spearman={safe_float(spearman)}")
    print(f"Empirical member p10-p90 label coverage: {coverage}; mean interval width={report['metrics']['member_p10_p90_mean_width_cycles']} cycles")
    print(f"Prototype review flags: {review_count}/{len(df)}")
    print("Disagreement-band MAE:")
    for row in band_rows:
        print(f"  {row['band']}: n={row['count']}, MAE={row['mae_cycles']}")
    print(f"\nJSON report: {json_path}")
    print(f"CSV report : {csv_path}")
    print("WARNING: Exploratory benchmark analysis only; not a calibrated uncertainty or aircraft safety assessment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
