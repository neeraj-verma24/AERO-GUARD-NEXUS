"""Step 65: group-split conformal RUL interval evaluation for C-MAPSS FD001.

Run from project root:
    python scripts\\evaluate_rul_group_conformal.py

The script splits whole engines into independent train/calibration/evaluation sets.
It trains five bootstrap HistGradientBoostingRegressor members in memory, calibrates
an absolute-residual score based on each calibration engine's maximum row error, and
evaluates simultaneous full-trajectory coverage on held-out engines.

It does not load official FD001 test labels, save models, alter application state,
change physical-rule configuration, or assert aircraft applicability/safety.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.train_rul import RUL_CAP, TRAIN_FILE, make_features, read_cmapss  # noqa: E402

SEED = 6501
ENSEMBLE_SIZE = 5
TARGET_COVERAGE = 0.80
TRAIN_ENGINE_FRACTION = 0.60
CALIBRATION_ENGINE_FRACTION = 0.20
OUTPUT_DIR = ROOT / "reports" / "rul_uncertainty_eval"
MODEL_PARAMS = {
    "max_iter": 160,
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
}


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae_cycles": round(float(np.mean(np.abs(error))), 4),
        "rmse_cycles": round(float(np.sqrt(np.mean(error ** 2))), 4),
        "mean_signed_error_cycles": round(float(np.mean(error)), 4),
        "row_count": int(error.size),
    }


def _finite_matrix(frame: pd.DataFrame, label: str) -> bool:
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        print(f"ERROR: {label} feature matrix has NaN or infinite values.")
        return False
    return True


def _conformal_order_statistic(scores: np.ndarray, target_coverage: float) -> tuple[float, int]:
    """Finite-sample split-conformal order statistic; rank is 1-based."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be non-empty and finite")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    rank = int(math.ceil((scores.size + 1) * target_coverage))
    if rank > scores.size:
        return float("inf"), rank
    return float(np.sort(scores)[rank - 1]), rank


def main() -> int:
    if not TRAIN_FILE.is_file():
        print(f"ERROR: NASA C-MAPSS training file not found: {TRAIN_FILE}")
        print("Place train_FD001.txt in the project's configured data/cmapss folder.")
        return 2

    print("AERO-GUARD NEXUS — STEP 65 GROUP-SPLIT CONFORMAL RUL EVALUATION")
    print("Scope: NASA C-MAPSS FD001 training trajectories only; no official test labels.")
    print("Mode: trains models in memory; writes new evaluation reports only.")
    print("[1/6] Loading data and creating RUL features...")
    frame = read_cmapss(TRAIN_FILE)
    X = make_features(frame)
    if len(X) != len(frame):
        print(f"ERROR: feature rows ({len(X)}) do not match source rows ({len(frame)}).")
        return 2
    last_cycle = frame.groupby("unit")["cycle"].transform("max")
    y = (last_cycle - frame["cycle"]).clip(upper=RUL_CAP).to_numpy(dtype=float)
    groups = frame["unit"].to_numpy(dtype=int)
    if not np.isfinite(y).all() or not _finite_matrix(X, "full"):
        return 2

    units = np.array(sorted(np.unique(groups)), dtype=int)
    if len(units) < 25:
        print(f"ERROR: expected at least 25 distinct engines for group split; found {len(units)}.")
        return 2

    print("[2/6] Creating deterministic disjoint engine-level folds...")
    rng = np.random.default_rng(SEED)
    shuffled_units = units.copy()
    rng.shuffle(shuffled_units)
    n_train = int(len(units) * TRAIN_ENGINE_FRACTION)
    n_cal = int(len(units) * CALIBRATION_ENGINE_FRACTION)
    n_train = max(1, n_train)
    n_cal = max(1, n_cal)
    train_units = sorted(int(x) for x in shuffled_units[:n_train])
    calibration_units = sorted(int(x) for x in shuffled_units[n_train:n_train + n_cal])
    evaluation_units = sorted(int(x) for x in shuffled_units[n_train + n_cal:])
    if not calibration_units or not evaluation_units:
        print("ERROR: empty calibration or evaluation engine fold.")
        return 2
    assert not (set(train_units) & set(calibration_units))
    assert not (set(train_units) & set(evaluation_units))
    assert not (set(calibration_units) & set(evaluation_units))

    train_idx = np.flatnonzero(np.isin(groups, train_units))
    calibration_idx = np.flatnonzero(np.isin(groups, calibration_units))
    evaluation_idx = np.flatnonzero(np.isin(groups, evaluation_units))
    X_train, y_train, groups_train = X.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx]
    X_cal, y_cal, groups_cal = X.iloc[calibration_idx].reset_index(drop=True), y[calibration_idx], groups[calibration_idx]
    X_eval, y_eval, groups_eval = X.iloc[evaluation_idx].reset_index(drop=True), y[evaluation_idx], groups[evaluation_idx]
    for name, matrix in (("train", X_train), ("calibration", X_cal), ("evaluation", X_eval)):
        if not _finite_matrix(matrix, name):
            return 2

    print(f"  Engines: train={len(train_units)}, calibration={len(calibration_units)}, evaluation={len(evaluation_units)}")
    print(f"  Rows: train={len(train_idx)}, calibration={len(calibration_idx)}, evaluation={len(evaluation_idx)}")
    print("[3/6] Training bootstrap ensemble in memory...")
    rng_boot = np.random.default_rng(SEED + 1)
    train_positions = {int(unit): np.flatnonzero(groups_train == unit) for unit in train_units}
    cal_member_predictions: list[np.ndarray] = []
    eval_member_predictions: list[np.ndarray] = []
    for member_idx in range(ENSEMBLE_SIZE):
        sampled_units = rng_boot.choice(np.asarray(train_units, dtype=int), size=len(train_units), replace=True)
        sampled_rows = np.concatenate([train_positions[int(unit)] for unit in sampled_units])
        model = HistGradientBoostingRegressor(**MODEL_PARAMS, random_state=100 + member_idx)
        model.fit(X_train.iloc[sampled_rows], y_train[sampled_rows])
        cal_pred = np.clip(model.predict(X_cal), 0.0, float(RUL_CAP))
        eval_pred = np.clip(model.predict(X_eval), 0.0, float(RUL_CAP))
        cal_member_predictions.append(cal_pred)
        eval_member_predictions.append(eval_pred)
        print(f"  Model {member_idx + 1}/{ENSEMBLE_SIZE} trained")

    cal_members = np.vstack(cal_member_predictions)
    eval_members = np.vstack(eval_member_predictions)
    cal_mean = cal_members.mean(axis=0)
    eval_mean = eval_members.mean(axis=0)
    cal_member_p10 = np.percentile(cal_members, 10, axis=0)
    cal_member_p90 = np.percentile(cal_members, 90, axis=0)
    eval_member_p10 = np.percentile(eval_members, 10, axis=0)
    eval_member_p90 = np.percentile(eval_members, 90, axis=0)

    print("[4/6] Calibrating on held-out engines using maximum trajectory error...")
    cal_frame = pd.DataFrame({
        "unit": groups_cal,
        "actual_rul_cycles": y_cal,
        "ensemble_mean_rul_cycles": cal_mean,
        "absolute_error_cycles": np.abs(cal_mean - y_cal),
        "member_p10_cycles": cal_member_p10,
        "member_p90_cycles": cal_member_p90,
    })
    cal_engine_scores = cal_frame.groupby("unit")["absolute_error_cycles"].max().sort_index()
    q, rank = _conformal_order_statistic(cal_engine_scores.to_numpy(), TARGET_COVERAGE)

    print("[5/6] Evaluating calibrated interval on untouched evaluation engines...")
    low = np.maximum(0.0, eval_mean - q)
    high = np.minimum(float(RUL_CAP), eval_mean + q)
    row_covered = (y_eval >= low) & (y_eval <= high)
    raw_member_covered = (y_eval >= eval_member_p10) & (y_eval <= eval_member_p90)
    eval_frame = pd.DataFrame({
        "unit": groups_eval.astype(int),
        "actual_rul_cycles": y_eval,
        "ensemble_mean_rul_cycles": eval_mean,
        "absolute_error_cycles": np.abs(eval_mean - y_eval),
        "member_p10_cycles": eval_member_p10,
        "member_p90_cycles": eval_member_p90,
        "member_p10_p90_width_cycles": eval_member_p90 - eval_member_p10,
        "member_p10_p90_contains_actual": raw_member_covered,
        "conformal_target_coverage": TARGET_COVERAGE,
        "conformal_radius_cycles": q if np.isfinite(q) else None,
        "conformal_lower_cycles": low,
        "conformal_upper_cycles": high,
        "conformal_interval_width_cycles": high - low,
        "conformal_contains_actual": row_covered,
    })
    per_engine = eval_frame.groupby("unit", as_index=False).agg(
        row_count=("unit", "size"),
        mean_absolute_error_cycles=("absolute_error_cycles", "mean"),
        conformal_row_coverage=("conformal_contains_actual", "mean"),
        conformal_full_trajectory_covered=("conformal_contains_actual", "all"),
        mean_conformal_width_cycles=("conformal_interval_width_cycles", "mean"),
        raw_member_p10_p90_row_coverage=("member_p10_p90_contains_actual", "mean"),
    )
    eval_engine_coverage = float(per_engine["conformal_full_trajectory_covered"].mean())
    conformal_row_coverage = float(row_covered.mean())
    raw_row_coverage = float(raw_member_covered.mean())
    width_mean = float(eval_frame["conformal_interval_width_cycles"].mean())

    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "step": 65,
        "evaluation_type": "group-split split-conformal interval evaluation using per-engine maximum absolute residual",
        "dataset": "NASA C-MAPSS FD001",
        "dataset_scope": "simulated benchmark; only train_FD001.txt used; official FD001 test labels not loaded",
        "target": "RUL clipped at the project's RUL_CAP, matching existing training helper",
        "seed": SEED,
        "target_coverage": TARGET_COVERAGE,
        "folds": {
            "method": "deterministic split by distinct engine ID before model training/calibration/evaluation",
            "train_engine_count": len(train_units),
            "calibration_engine_count": len(calibration_units),
            "evaluation_engine_count": len(evaluation_units),
            "train_row_count": int(len(train_idx)),
            "calibration_row_count": int(len(calibration_idx)),
            "evaluation_row_count": int(len(evaluation_idx)),
            "train_engine_ids": train_units,
            "calibration_engine_ids": calibration_units,
            "evaluation_engine_ids": evaluation_units,
            "pairwise_engine_overlap_count": 0,
        },
        "model": {
            "kind": "HistGradientBoostingRegressor bootstrap ensemble",
            "ensemble_size": ENSEMBLE_SIZE,
            "parameters": MODEL_PARAMS,
            "trained_in_memory": True,
            "model_files_written": False,
        },
        "calibration": {
            "score": "maximum absolute residual across all rows of each calibration engine",
            "engine_score_count": int(cal_engine_scores.size),
            "finite_sample_order_statistic_rank_1_based": rank,
            "conformal_radius_cycles": q if np.isfinite(q) else None,
            "infinite_radius_required": bool(not np.isfinite(q)),
            "scope": "engine-level simultaneous trajectory band, conditional on exchangeability of engine trajectories",
        },
        "evaluation_metrics": {
            "point_prediction": _metrics(y_eval, eval_mean),
            "conformal_row_coverage": round(conformal_row_coverage, 4),
            "conformal_full_trajectory_engine_coverage": round(eval_engine_coverage, 4),
            "target_engine_coverage": TARGET_COVERAGE,
            "mean_conformal_interval_width_cycles": round(width_mean, 4),
            "raw_member_p10_p90_row_coverage": round(raw_row_coverage, 4),
            "raw_member_p10_p90_nominal_level": 0.80,
            "raw_member_p10_p90_mean_width_cycles": round(float((eval_member_p90 - eval_member_p10).mean()), 4),
        },
        "interpretation": {
            "empirical_target_met_on_evaluation_split": bool(eval_engine_coverage >= TARGET_COVERAGE),
            "label": "EVALUATION_ONLY_NOT_A_SAFETY_CLAIM",
            "note": "Observed coverage on this one split is an estimate, not a guarantee of future coverage. Small engine counts make uncertainty large.",
        },
        "limitations": [
            "Calibration/evaluation use uncensored full degradation trajectories from the simulated training split; the official test distribution is not assessed here.",
            "Conformal guarantee assumptions require exchangeable independent engine-level examples; real fleet/domain shift is untested.",
            "Each calibration score is a maximum over correlated rows within one engine, to target simultaneous trajectory coverage rather than treating rows as independent.",
            "A wide conformal radius may be statistically honest but operationally not useful; report width together with coverage.",
            "This does not establish calibrated uncertainty for real aircraft, valid engineering thresholds, maintenance instructions, or maintenance clearance.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "step65_group_conformal.json"
    rows_path = OUTPUT_DIR / "step65_group_conformal_rows.csv"
    engine_path = OUTPUT_DIR / "step65_group_conformal_by_engine.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    eval_frame.to_csv(rows_path, index=False, encoding="utf-8-sig")
    per_engine.to_csv(engine_path, index=False, encoding="utf-8-sig")

    print("[6/6] Reports saved. No model, dataset, database, or application configuration was changed.")
    print("=" * 72)
    print("AERO-GUARD NEXUS — STEP 65 GROUP-SPLIT CONFORMAL EVALUATION")
    print("=" * 72)
    print(f"Engines train/calibration/evaluation: {len(train_units)}/{len(calibration_units)}/{len(evaluation_units)}")
    print(f"Calibration radius: {q:.4f} cycles" if np.isfinite(q) else "Calibration radius: infinity (insufficient calibration sample for requested coverage)")
    print(f"Evaluation point MAE / RMSE: {report['evaluation_metrics']['point_prediction']['mae_cycles']} / {report['evaluation_metrics']['point_prediction']['rmse_cycles']} cycles")
    print(f"Conformal row coverage: {report['evaluation_metrics']['conformal_row_coverage']}")
    print(f"Conformal full-trajectory engine coverage: {report['evaluation_metrics']['conformal_full_trajectory_engine_coverage']} (target {TARGET_COVERAGE})")
    print(f"Mean conformal interval width: {report['evaluation_metrics']['mean_conformal_interval_width_cycles']} cycles")
    print(f"Raw member p10-p90 row coverage: {report['evaluation_metrics']['raw_member_p10_p90_row_coverage']} (nominal 0.8)")
    print(f"JSON report: {json_path}")
    print(f"Rows CSV: {rows_path}")
    print(f"Per-engine CSV: {engine_path}")
    print("WARNING: one simulated-data split only; this is not aircraft safety validation or a maintenance authorization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
