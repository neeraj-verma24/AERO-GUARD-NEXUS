"""Step 66: repeated engine-group split conformal RUL evaluation for C-MAPSS FD001.

Run from project root:
    python scripts\\evaluate_rul_group_conformal_multi_split.py

Repeats Step 65's train/calibration/evaluation method over several deterministic
engine-level splits to estimate split sensitivity. Models are trained in memory.
Only fresh reports are written; existing model files, source data, the database,
application configuration and physics-rule registry are not modified.

This is simulated benchmark analysis, not aircraft maintenance validation.
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

STEP = 66
BASE_SEED = 6600
REPEATS = 5
ENSEMBLE_SIZE = 5
TARGET_COVERAGE = 0.80
TRAIN_ENGINE_FRACTION = 0.60
CALIBRATION_ENGINE_FRACTION = 0.20
MODEL_PARAMS = {
    "max_iter": 160,
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
}
OUTPUT_DIR = ROOT / "reports" / "rul_uncertainty_eval" / "step66_multi_split_conformal"


def finite_sample_order_statistic(scores: np.ndarray, target_coverage: float) -> tuple[float, int]:
    """One-based finite-sample split-conformal rank over calibration engines."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Calibration scores must be non-empty and finite")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target_coverage must be between 0 and 1")
    rank = int(math.ceil((scores.size + 1) * target_coverage))
    if rank > scores.size:
        return float("inf"), rank
    return float(np.sort(scores)[rank - 1]), rank


def metric_pair(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    return {
        "mae_cycles": round(float(np.mean(np.abs(error))), 4),
        "rmse_cycles": round(float(np.sqrt(np.mean(error ** 2))), 4),
        "mean_signed_error_cycles": round(float(np.mean(error)), 4),
        "row_count": int(error.size),
    }


def finite_matrix(frame: pd.DataFrame, label: str) -> bool:
    try:
        values = frame.to_numpy(dtype=float)
    except (TypeError, ValueError):
        print(f"ERROR: {label} feature matrix cannot be converted to finite numeric values.")
        return False
    if not np.isfinite(values).all():
        print(f"ERROR: {label} feature matrix contains NaN or infinite values.")
        return False
    return True


def main() -> int:
    if not TRAIN_FILE.is_file():
        print(f"ERROR: NASA C-MAPSS training file not found: {TRAIN_FILE}")
        print("Place train_FD001.txt in the project's configured data/cmapss folder.")
        return 2

    print("AERO-GUARD NEXUS — STEP 66 MULTI-SPLIT GROUP CONFORMAL RUL EVALUATION")
    print(f"Scope: NASA C-MAPSS FD001 train trajectories only; repeats={REPEATS}; target={TARGET_COVERAGE:.0%}")
    print("Mode: engine-group splits; bootstrap ensemble in memory; report files only.")
    print("Note: repeated splits reuse engines across repetitions; aggregate results are descriptive, not independent trials.")
    print("[1/5] Loading data and creating features...")

    frame = read_cmapss(TRAIN_FILE)
    X = make_features(frame)
    if len(X) != len(frame):
        print(f"ERROR: feature rows ({len(X)}) do not match source rows ({len(frame)}).")
        return 2
    last_cycle = frame.groupby("unit")["cycle"].transform("max")
    y = (last_cycle - frame["cycle"]).clip(upper=RUL_CAP).to_numpy(dtype=float)
    groups = frame["unit"].to_numpy(dtype=int)
    if not np.isfinite(y).all() or not finite_matrix(X, "full"):
        return 2

    units = np.array(sorted(np.unique(groups)), dtype=int)
    if len(units) < 25:
        print(f"ERROR: expected at least 25 distinct engines for group split; found {len(units)}.")
        return 2

    n_train = max(1, int(len(units) * TRAIN_ENGINE_FRACTION))
    n_cal = max(1, int(len(units) * CALIBRATION_ENGINE_FRACTION))
    if n_train + n_cal >= len(units):
        print("ERROR: split configuration leaves no evaluation engines.")
        return 2

    row_reports: list[pd.DataFrame] = []
    engine_reports: list[pd.DataFrame] = []
    split_summaries: list[dict] = []

    for rep in range(REPEATS):
        split_seed = BASE_SEED + rep
        print(f"[2/5] Split {rep + 1}/{REPEATS}: forming disjoint engine folds (seed={split_seed})...")
        rng_split = np.random.default_rng(split_seed)
        shuffled = units.copy()
        rng_split.shuffle(shuffled)
        train_units = sorted(int(x) for x in shuffled[:n_train])
        calibration_units = sorted(int(x) for x in shuffled[n_train:n_train + n_cal])
        evaluation_units = sorted(int(x) for x in shuffled[n_train + n_cal:])

        sets = (set(train_units), set(calibration_units), set(evaluation_units))
        if (sets[0] & sets[1]) or (sets[0] & sets[2]) or (sets[1] & sets[2]):
            print("ERROR: engine overlap detected across train/calibration/evaluation in one split.")
            return 2

        train_idx = np.flatnonzero(np.isin(groups, train_units))
        cal_idx = np.flatnonzero(np.isin(groups, calibration_units))
        eval_idx = np.flatnonzero(np.isin(groups, evaluation_units))
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_cal = X.iloc[cal_idx].reset_index(drop=True)
        X_eval = X.iloc[eval_idx].reset_index(drop=True)
        y_train, y_cal, y_eval = y[train_idx], y[cal_idx], y[eval_idx]
        groups_train, groups_cal, groups_eval = groups[train_idx], groups[cal_idx], groups[eval_idx]
        for label, matrix in (("train", X_train), ("calibration", X_cal), ("evaluation", X_eval)):
            if not finite_matrix(matrix, f"split {rep + 1} {label}"):
                return 2

        print(f"[3/5] Split {rep + 1}/{REPEATS}: training {ENSEMBLE_SIZE} bootstrap members in memory...")
        rng_boot = np.random.default_rng(split_seed + 1000)
        train_positions = {int(unit): np.flatnonzero(groups_train == unit) for unit in train_units}
        cal_predictions: list[np.ndarray] = []
        eval_predictions: list[np.ndarray] = []
        for member_idx in range(ENSEMBLE_SIZE):
            sampled_units = rng_boot.choice(np.asarray(train_units, dtype=int), size=len(train_units), replace=True)
            sampled_rows = np.concatenate([train_positions[int(unit)] for unit in sampled_units])
            model = HistGradientBoostingRegressor(
                **MODEL_PARAMS,
                random_state=10000 + rep * 100 + member_idx,
            )
            model.fit(X_train.iloc[sampled_rows], y_train[sampled_rows])
            cal_predictions.append(np.clip(model.predict(X_cal), 0.0, float(RUL_CAP)))
            eval_predictions.append(np.clip(model.predict(X_eval), 0.0, float(RUL_CAP)))
            print(f"  Split {rep + 1} model {member_idx + 1}/{ENSEMBLE_SIZE} trained")

        cal_members = np.vstack(cal_predictions)
        eval_members = np.vstack(eval_predictions)
        cal_mean = cal_members.mean(axis=0)
        eval_mean = eval_members.mean(axis=0)
        cal_p10 = np.percentile(cal_members, 10, axis=0)
        cal_p90 = np.percentile(cal_members, 90, axis=0)
        eval_p10 = np.percentile(eval_members, 10, axis=0)
        eval_p90 = np.percentile(eval_members, 90, axis=0)

        print(f"[4/5] Split {rep + 1}/{REPEATS}: calibrating per-engine max residual and evaluating...")
        cal_frame = pd.DataFrame({
            "unit": groups_cal,
            "absolute_error_cycles": np.abs(cal_mean - y_cal),
            "member_p10_cycles": cal_p10,
            "member_p90_cycles": cal_p90,
        })
        cal_scores = cal_frame.groupby("unit")["absolute_error_cycles"].max().sort_index().to_numpy()
        radius, rank = finite_sample_order_statistic(cal_scores, TARGET_COVERAGE)

        low = np.maximum(0.0, eval_mean - radius)
        high = np.minimum(float(RUL_CAP), eval_mean + radius)
        conformal_covered = (y_eval >= low) & (y_eval <= high)
        raw_covered = (y_eval >= eval_p10) & (y_eval <= eval_p90)
        eval_rows = pd.DataFrame({
            "repeat": rep + 1,
            "split_seed": split_seed,
            "unit": groups_eval.astype(int),
            "actual_rul_cycles": y_eval,
            "ensemble_mean_rul_cycles": eval_mean,
            "absolute_error_cycles": np.abs(eval_mean - y_eval),
            "member_p10_cycles": eval_p10,
            "member_p90_cycles": eval_p90,
            "member_p10_p90_width_cycles": eval_p90 - eval_p10,
            "member_p10_p90_contains_actual": raw_covered,
            "conformal_target_coverage": TARGET_COVERAGE,
            "conformal_radius_cycles": radius if np.isfinite(radius) else None,
            "conformal_lower_cycles": low,
            "conformal_upper_cycles": high,
            "conformal_interval_width_cycles": high - low,
            "conformal_contains_actual": conformal_covered,
        })
        eval_engines = eval_rows.groupby(["repeat", "split_seed", "unit"], as_index=False).agg(
            row_count=("unit", "size"),
            mean_absolute_error_cycles=("absolute_error_cycles", "mean"),
            conformal_row_coverage=("conformal_contains_actual", "mean"),
            conformal_full_trajectory_covered=("conformal_contains_actual", "all"),
            mean_conformal_width_cycles=("conformal_interval_width_cycles", "mean"),
            raw_member_p10_p90_row_coverage=("member_p10_p90_contains_actual", "mean"),
        )
        engine_coverage = float(eval_engines["conformal_full_trajectory_covered"].mean())
        row_coverage = float(conformal_covered.mean())
        raw_coverage = float(raw_covered.mean())
        mean_width = float((high - low).mean())
        point_metrics = metric_pair(y_eval, eval_mean)
        summary = {
            "repeat": rep + 1,
            "split_seed": split_seed,
            "train_engine_count": len(train_units),
            "calibration_engine_count": len(calibration_units),
            "evaluation_engine_count": len(evaluation_units),
            "train_row_count": int(len(train_idx)),
            "calibration_row_count": int(len(cal_idx)),
            "evaluation_row_count": int(len(eval_idx)),
            "calibration_order_statistic_rank_1_based": rank,
            "conformal_radius_cycles": round(float(radius), 4) if np.isfinite(radius) else None,
            "conformal_row_coverage": round(row_coverage, 4),
            "conformal_full_trajectory_engine_coverage": round(engine_coverage, 4),
            "target_engine_coverage": TARGET_COVERAGE,
            "mean_conformal_interval_width_cycles": round(mean_width, 4),
            "raw_member_p10_p90_row_coverage": round(raw_coverage, 4),
            "raw_member_p10_p90_mean_width_cycles": round(float((eval_p90 - eval_p10).mean()), 4),
            "point_mae_cycles": point_metrics["mae_cycles"],
            "point_rmse_cycles": point_metrics["rmse_cycles"],
            "engine_ids": {
                "train": train_units,
                "calibration": calibration_units,
                "evaluation": evaluation_units,
            },
            "split_target_met": bool(engine_coverage >= TARGET_COVERAGE),
        }
        row_reports.append(eval_rows)
        engine_reports.append(eval_engines)
        split_summaries.append(summary)
        print(
            f"  Coverage engines={engine_coverage:.3f} (target {TARGET_COVERAGE:.2f}); "
            f"row={row_coverage:.3f}; width={mean_width:.2f}; radius={radius:.2f}; "
            f"MAE={point_metrics['mae_cycles']:.3f}"
        )

    print("[5/5] Writing aggregate and per-split reports...")
    rows_all = pd.concat(row_reports, ignore_index=True)
    engines_all = pd.concat(engine_reports, ignore_index=True)
    summaries_frame = pd.DataFrame([{k: v for k, v in item.items() if k != "engine_ids"} for item in split_summaries])

    numeric_fields = [
        "conformal_radius_cycles",
        "conformal_row_coverage",
        "conformal_full_trajectory_engine_coverage",
        "mean_conformal_interval_width_cycles",
        "raw_member_p10_p90_row_coverage",
        "raw_member_p10_p90_mean_width_cycles",
        "point_mae_cycles",
        "point_rmse_cycles",
    ]
    aggregate = {}
    for field in numeric_fields:
        values = pd.to_numeric(summaries_frame[field], errors="coerce").dropna().to_numpy(dtype=float)
        aggregate[field] = {
            "mean_across_repeats": round(float(np.mean(values)), 4) if values.size else None,
            "std_across_repeats_population": round(float(np.std(values)), 4) if values.size else None,
            "min_across_repeats": round(float(np.min(values)), 4) if values.size else None,
            "max_across_repeats": round(float(np.max(values)), 4) if values.size else None,
        }

    per_repeat_target_count = sum(bool(x["split_target_met"]) for x in split_summaries)
    # This pooled result deliberately repeats engines evaluated under different splits; it is descriptive only.
    pooled_row_coverage = float(rows_all["conformal_contains_actual"].astype(bool).mean())
    pooled_engine_coverage = float(engines_all["conformal_full_trajectory_covered"].astype(bool).mean())
    report = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "step": STEP,
        "evaluation_type": "repeated engine-group split conformal RUL evaluation",
        "dataset": "NASA C-MAPSS FD001",
        "dataset_scope": "simulated benchmark; only train_FD001.txt used; official test labels not loaded",
        "target": "RUL clipped at the configured RUL_CAP, matching the project's training helper",
        "configuration": {
            "base_seed": BASE_SEED,
            "repeat_count": REPEATS,
            "ensemble_size_per_repeat": ENSEMBLE_SIZE,
            "target_engine_coverage": TARGET_COVERAGE,
            "train_engine_fraction": TRAIN_ENGINE_FRACTION,
            "calibration_engine_fraction": CALIBRATION_ENGINE_FRACTION,
            "evaluation_engine_fraction": round(1.0 - TRAIN_ENGINE_FRACTION - CALIBRATION_ENGINE_FRACTION, 4),
            "model_kind": "HistGradientBoostingRegressor bootstrap ensemble",
            "model_parameters": MODEL_PARAMS,
            "trained_in_memory": True,
            "model_files_written": False,
        },
        "repeats": split_summaries,
        "aggregate_split_metrics": aggregate,
        "descriptive_pooled_metrics_non_independent": {
            "pooled_row_coverage": round(pooled_row_coverage, 4),
            "pooled_engine_coverage": round(pooled_engine_coverage, 4),
            "warning": "Repeated splits reuse the same engines; pooled rows/engines are not independent observations and must not be interpreted as independent-trial confidence evidence.",
        },
        "summary": {
            "splits_meeting_or_exceeding_target_engine_coverage": per_repeat_target_count,
            "total_splits": REPEATS,
            "target_engine_coverage": TARGET_COVERAGE,
            "all_splits_meet_target": bool(per_repeat_target_count == REPEATS),
            "interpretation": "Split-sensitivity diagnostic only. Five random partitions of one simulated dataset do not establish robust coverage under fleet/domain shift or certify a model for maintenance decisions.",
        },
        "limitations": [
            "Only the simulated NASA C-MAPSS FD001 training file is used; the official FD001 test labels are not loaded.",
            "Each split separates whole engines across train, calibration and evaluation folds; rows from a single engine never cross folds within a split.",
            "The same finite set of engines is reused across random splits, so results across repeats are correlated and are not independent validation trials.",
            "Conformal coverage assumes exchangeable engine-level trajectories; distribution shift, real-fleet behavior and censoring differences are untested.",
            "Coverage must be reported together with interval width; overly wide intervals may be of limited practical use.",
            "No aircraft engineering rules, physical limits, maintenance approvals or operational clearance are inferred.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "step66_multi_split_conformal.json"
    rows_path = OUTPUT_DIR / "step66_rows.csv"
    engines_path = OUTPUT_DIR / "step66_by_engine.csv"
    splits_path = OUTPUT_DIR / "step66_by_split.csv"
    md_path = OUTPUT_DIR / "step66_summary.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    rows_all.to_csv(rows_path, index=False, encoding="utf-8-sig")
    engines_all.to_csv(engines_path, index=False, encoding="utf-8-sig")
    summaries_frame.drop(columns=[c for c in ["engine_ids"] if c in summaries_frame.columns], errors="ignore").to_csv(splits_path, index=False, encoding="utf-8-sig")

    md_lines = [
        "# AERO-GUARD NEXUS — Step 66 Multi-Split Conformal RUL Evaluation",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Dataset: `{report['dataset']}` (simulated benchmark; train file only)",
        f"- Repeats: **{REPEATS}**; target full-trajectory coverage: **{TARGET_COVERAGE:.0%}**",
        f"- Splits meeting target: **{per_repeat_target_count}/{REPEATS}**",
        "- Official test labels loaded: **No**",
        "- Existing model files changed: **No**",
        "",
        "## Per-split results",
        "",
        "| Split | Seed | Engine coverage | Row coverage | Mean width (cycles) | Radius (cycles) | MAE | RMSE | Target met |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for item in split_summaries:
        md_lines.append(
            f"| {item['repeat']} | {item['split_seed']} | {item['conformal_full_trajectory_engine_coverage']:.3f} | "
            f"{item['conformal_row_coverage']:.3f} | {item['mean_conformal_interval_width_cycles']:.2f} | "
            f"{item['conformal_radius_cycles'] if item['conformal_radius_cycles'] is not None else 'infinity'} | "
            f"{item['point_mae_cycles']:.3f} | {item['point_rmse_cycles']:.3f} | "
            f"{'Yes' if item['split_target_met'] else 'No'} |"
        )
    md_lines.extend(["", "## Mean across splits", "", "| Metric | Mean | Population SD | Min | Max |", "|---|---:|---:|---:|---:|"])
    for field in numeric_fields:
        stats = aggregate[field]
        md_lines.append(
            f"| {field} | {stats['mean_across_repeats']} | {stats['std_across_repeats_population']} | "
            f"{stats['min_across_repeats']} | {stats['max_across_repeats']} |"
        )
    md_lines.extend([
        "",
        "## Limitations",
        "",
        "Repeated partitions reuse the same finite set of engines and are correlated. This is split-sensitivity evidence, not independent-trial confidence evidence.",
        "Only the simulated C-MAPSS FD001 training trajectories were used. This does not validate real aircraft behavior, engineering limits, maintenance instructions or operational clearance.",
    ])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("=" * 76)
    print("STEP 66 SUMMARY")
    print("=" * 76)
    print(f"Splits meeting/exceeding target engine coverage: {per_repeat_target_count}/{REPEATS}")
    for field, label in [
        ("conformal_full_trajectory_engine_coverage", "Mean full-trajectory engine coverage"),
        ("conformal_row_coverage", "Mean row coverage"),
        ("mean_conformal_interval_width_cycles", "Mean conformal interval width (cycles)"),
        ("conformal_radius_cycles", "Mean calibration radius (cycles)"),
        ("point_mae_cycles", "Mean point MAE (cycles)"),
        ("point_rmse_cycles", "Mean point RMSE (cycles)"),
        ("raw_member_p10_p90_row_coverage", "Mean raw member p10-p90 row coverage"),
    ]:
        stats = aggregate[field]
        print(f"{label}: {stats['mean_across_repeats']} (range {stats['min_across_repeats']}–{stats['max_across_repeats']})")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Rows CSV: {rows_path}")
    print(f"Per-engine CSV: {engines_path}")
    print(f"Per-split CSV: {splits_path}")
    print("WARNING: descriptive repeated-split evidence only; not aircraft safety validation or maintenance authorization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
