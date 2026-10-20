"""Engine-wise held-out validation for AERO-GUARD NEXUS RUL uncertainty.

Run from project root:
    python scripts\\evaluate_rul_validation_uncertainty.py

Uses only NASA C-MAPSS FD001 training trajectories. A group split holds out whole
engine IDs. Models train in memory and are not saved. The official FD001 test labels
are not loaded or used. This is an exploratory validation, not safety certification.
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train_rul import RUL_CAP, TRAIN_FILE, make_features, read_cmapss  # noqa: E402

SEED = 5487
VALIDATION_FRACTION = 0.20
ENSEMBLE_SIZE = 5
LOW_RUL_REVIEW_CYCLES = 30.0
HIGH_DISAGREEMENT_REVIEW_CYCLES = 8.0
OUTPUT_DIR = ROOT / "reports" / "rul_uncertainty_eval"


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "mae_cycles": round(float(np.mean(np.abs(error))), 4),
        "rmse_cycles": round(float(np.sqrt(np.mean(error ** 2))), 4),
        "mean_error_signed_cycles": round(float(np.mean(error)), 4),
        "n_observations": int(len(actual)),
    }


def _corr(a, b, method="pearson"):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    if method == "spearman":
        a = pd.Series(a).rank(method="average").to_numpy()
        b = pd.Series(b).rank(method="average").to_numpy()
    return round(float(np.corrcoef(a, b)[0, 1]), 4)


def main():
    if not TRAIN_FILE.is_file():
        print(f"ERROR: NASA training data not found: {TRAIN_FILE}")
        return 2

    print("[1/5] Loading training trajectories only...")
    frame = read_cmapss(TRAIN_FILE)
    X = make_features(frame)
    if len(X) != len(frame):
        print(f"ERROR: feature rows ({len(X)}) != source rows ({len(frame)}).")
        return 2

    last_cycle = frame.groupby("unit")["cycle"].transform("max")
    y = (last_cycle - frame["cycle"]).clip(upper=RUL_CAP).to_numpy(dtype=float)
    groups = frame["unit"].to_numpy()
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        print("ERROR: feature matrix contains non-finite values; investigate feature engineering.")
        return 2

    print("[2/5] Splitting by engine ID (no unit appears in both folds)...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_FRACTION, random_state=SEED)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))
    train_units = sorted(set(int(v) for v in groups[train_idx]))
    val_units = sorted(set(int(v) for v in groups[val_idx]))
    if set(train_units) & set(val_units):
        print("ERROR: engine leakage detected between training and validation folds.")
        return 2

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    groups_train = groups[train_idx]
    X_val = X.iloc[val_idx].reset_index(drop=True)
    y_val = y[val_idx]
    groups_val = groups[val_idx]

    print(f"  Train engines: {len(train_units)}; validation engines: {len(val_units)}")
    print(f"  Training rows: {len(train_idx)}; validation rows: {len(val_idx)}")
    print("[3/5] Training five bootstrap models in memory...")
    rng = np.random.default_rng(SEED)
    unique_train_units = np.array(train_units, dtype=int)
    positions = {int(unit): np.flatnonzero(groups_train == unit) for unit in unique_train_units}
    params = {"max_iter": 160, "learning_rate": 0.08, "max_leaf_nodes": 15, "l2_regularization": 1.0}
    models = []
    member_predictions = []
    for member_index in range(ENSEMBLE_SIZE):
        sampled_units = rng.choice(unique_train_units, size=len(unique_train_units), replace=True)
        sample_idx = np.concatenate([positions[int(unit)] for unit in sampled_units])
        model = HistGradientBoostingRegressor(**params, random_state=100 + member_index)
        model.fit(X_train.iloc[sample_idx], y_train[sample_idx])
        models.append(model)
        member_predictions.append(np.maximum(0.0, model.predict(X_val)))
        print(f"  Model {member_index + 1}/{ENSEMBLE_SIZE} trained")

    print("[4/5] Computing held-out validation metrics...")
    member_pred = np.vstack(member_predictions)
    mean_pred = member_pred.mean(axis=0)
    disagreement = member_pred.std(axis=0, ddof=1)
    p10 = np.percentile(member_pred, 10, axis=0)
    p90 = np.percentile(member_pred, 90, axis=0)
    abs_error = np.abs(mean_pred - y_val)
    covered = (y_val >= p10) & (y_val <= p90)
    widths = p90 - p10
    review_low = mean_pred <= LOW_RUL_REVIEW_CYCLES
    review_disagreement = disagreement >= HIGH_DISAGREEMENT_REVIEW_CYCLES
    review_any = review_low | review_disagreement

    rows = pd.DataFrame({
        "unit": groups_val.astype(int),
        "actual_rul_cycles": y_val,
        "ensemble_mean_rul_cycles": mean_pred,
        "absolute_error_cycles": abs_error,
        "ensemble_std_cycles": disagreement,
        "member_p10_cycles": p10,
        "member_p90_cycles": p90,
        "interval_width_cycles": widths,
        "p10_p90_contains_actual": covered,
        "low_rul_review_flag": review_low,
        "high_disagreement_review_flag": review_disagreement,
        "any_review_flag": review_any,
    })
    per_engine = rows.groupby("unit", as_index=False).agg(
        row_count=("unit", "size"),
        mean_absolute_error_cycles=("absolute_error_cycles", "mean"),
        mean_disagreement_cycles=("ensemble_std_cycles", "mean"),
        empirical_coverage=("p10_p90_contains_actual", "mean"),
        mean_interval_width_cycles=("interval_width_cycles", "mean"),
        review_row_fraction=("any_review_flag", "mean"),
    )

    band_summary = []
    bands = [("<2", -np.inf, 2.0), ("2 to <5", 2.0, 5.0), ("5 to <8", 5.0, 8.0), (">=8", 8.0, np.inf)]
    for label, low, high in bands:
        mask = (disagreement >= low) & (disagreement < high)
        band_summary.append({
            "band_cycles": label,
            "n_rows": int(mask.sum()),
            "mae_cycles": round(float(abs_error[mask].mean()), 4) if mask.any() else None,
            "mean_actual_rul_cycles": round(float(y_val[mask].mean()), 4) if mask.any() else None,
        })

    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_type": "engine-wise held-out validation split of NASA C-MAPSS FD001 training trajectories",
        "dataset_scope": "simulated benchmark only; official test labels are not loaded or used",
        "split": {
            "method": "GroupShuffleSplit by engine ID",
            "seed": SEED,
            "validation_fraction_requested": VALIDATION_FRACTION,
            "train_engine_count": len(train_units),
            "validation_engine_count": len(val_units),
            "train_engine_ids": train_units,
            "validation_engine_ids": val_units,
            "train_row_count": int(len(train_idx)),
            "validation_row_count": int(len(val_idx)),
            "engine_overlap_count": len(set(train_units) & set(val_units)),
            "target_cap_cycles": RUL_CAP,
        },
        "model": {"kind": "HistGradientBoostingRegressor engine-bootstrap ensemble", "ensemble_size": ENSEMBLE_SIZE, "parameters": params},
        "point_prediction_metrics": _metrics(y_val, mean_pred),
        "uncertainty_metrics": {
            "p10_p90_empirical_coverage": round(float(covered.mean()), 4),
            "nominal_interval_level": 0.80,
            "mean_interval_width_cycles": round(float(widths.mean()), 4),
            "median_interval_width_cycles": round(float(np.median(widths)), 4),
            "mean_member_disagreement_cycles": round(float(disagreement.mean()), 4),
            "median_member_disagreement_cycles": round(float(np.median(disagreement)), 4),
            "pearson_disagreement_vs_absolute_error": _corr(disagreement, abs_error, "pearson"),
            "spearman_disagreement_vs_absolute_error": _corr(disagreement, abs_error, "spearman"),
            "disagreement_bands": band_summary,
        },
        "review_threshold_analysis": {
            "low_rul_review_threshold_cycles": LOW_RUL_REVIEW_CYCLES,
            "high_disagreement_review_threshold_cycles": HIGH_DISAGREEMENT_REVIEW_CYCLES,
            "low_rul_flag_rows": int(review_low.sum()),
            "high_disagreement_flag_rows": int(review_disagreement.sum()),
            "any_review_flag_rows": int(review_any.sum()),
            "any_review_flag_row_fraction": round(float(review_any.mean()), 4),
            "flagged_row_mae_cycles": round(float(abs_error[review_any].mean()), 4) if review_any.any() else None,
            "unflagged_row_mae_cycles": round(float(abs_error[~review_any].mean()), 4) if (~review_any).any() else None,
            "note": "Rows within each engine are correlated; row-level review coverage is descriptive, not an independent-event rate.",
        },
        "per_engine_summary": per_engine.to_dict(orient="records"),
        "limitations": [
            "This is a newly seeded held-out split of the training trajectories, not the previously reported validation split.",
            "Training trajectories end at failure; row-level validation distribution differs from censored final-cycle test observations.",
            "Ensemble member p10-p90 percentiles are not automatically calibrated prediction intervals.",
            "Coverage is empirical on this held-out split only and is not a guarantee of future coverage.",
            "No real aircraft validation or maintenance-clearance decision is supported by this report.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "validation_split.json"
    csv_path = OUTPUT_DIR / "validation_split_rows.csv"
    engine_csv_path = OUTPUT_DIR / "validation_split_by_engine.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows.to_csv(csv_path, index=False, encoding="utf-8-sig")
    per_engine.to_csv(engine_csv_path, index=False, encoding="utf-8-sig")

    print("[5/5] Reports saved (no models or source data modified).")
    print("=" * 68)
    print("AERO-GUARD NEXUS — HELD-OUT RUL UNCERTAINTY VALIDATION")
    print("=" * 68)
    print(f"Train engines / validation engines: {len(train_units)} / {len(val_units)}")
    print(f"Validation rows: {len(val_idx)}")
    print(f"Point metrics MAE / RMSE: {report['point_prediction_metrics']['mae_cycles']} / {report['point_prediction_metrics']['rmse_cycles']} cycles")
    print(f"Empirical p10-p90 coverage: {report['uncertainty_metrics']['p10_p90_empirical_coverage']}")
    print(f"Mean interval width: {report['uncertainty_metrics']['mean_interval_width_cycles']} cycles")
    print(f"Disagreement vs abs error Pearson / Spearman: {report['uncertainty_metrics']['pearson_disagreement_vs_absolute_error']} / {report['uncertainty_metrics']['spearman_disagreement_vs_absolute_error']}")
    print(f"Review-flagged rows: {report['review_threshold_analysis']['any_review_flag_rows']} / {len(val_idx)}")
    print(f"Engine overlap between train and validation: {len(set(train_units) & set(val_units))}")
    print(f"JSON report: {json_path}")
    print(f"Row CSV: {csv_path}")
    print(f"Per-engine CSV: {engine_csv_path}")
    print("NOTE: exploratory C-MAPSS validation only, not aircraft safety validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
