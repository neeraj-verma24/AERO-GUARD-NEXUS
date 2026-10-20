"""AERO-GUARD NEXUS Step 52: physics/data sanity checks for C-MAPSS.

Important scope boundary:
- Provides data-integrity sanity checks and an engineering-rule registry interface.
- Does NOT invent sensor bounds or call statistical quantiles "physical limits".
- No approved sensor engineering-limit rules are shipped by default.
- C-MAPSS sensor units/real-aircraft applicability are not established by this module.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any

from models.train_rul import TEST_FILE, read_cmapss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = TEST_FILE.with_name("train_FD001.txt")
RULES_FILE = PROJECT_ROOT / "data" / "physics_rules.json"


def _numeric_columns(frame):
    return [str(c) for c in frame.select_dtypes(include="number").columns]


def _sensor_columns(frame):
    excluded = {"unit", "cycle"}
    return [c for c in _numeric_columns(frame)
            if c.lower() not in excluded and "setting" not in c.lower()]


def _load_rules(sensor_columns):
    """Load optional local engineering-limit rules; missing config means no limits."""
    if not RULES_FILE.is_file():
        return {
            "status": "NO_APPROVED_PHYSICAL_RULES_CONFIGURED",
            "path": str(RULES_FILE),
            "rule_count": 0,
            "rules": [],
            "errors": [],
        }
    try:
        raw = json.loads(RULES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "RULES_CONFIG_INVALID", "path": str(RULES_FILE),
                "rule_count": 0, "rules": [], "errors": [str(exc)]}

    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or not isinstance(raw.get("rules"), list):
        return {"status": "RULES_CONFIG_INVALID", "path": str(RULES_FILE),
                "rule_count": 0, "rules": [],
                "errors": ["Expected schema_version=1 and a rules array."]}

    accepted, errors = [], []
    seen = set()
    required_text = ("units", "reference", "applicability", "reviewed_by")
    for idx, rule in enumerate(raw["rules"]):
        if not isinstance(rule, dict):
            errors.append(f"rules[{idx}] must be an object")
            continue
        sensor = str(rule.get("sensor", "")).strip()
        if sensor not in sensor_columns:
            errors.append(f"rules[{idx}].sensor must match an available sensor column")
            continue
        if sensor in seen:
            errors.append(f"duplicate rule for sensor {sensor}")
            continue
        seen.add(sensor)
        if str(rule.get("approval_status", "")).upper() != "APPROVED":
            errors.append(f"rules[{idx}] for {sensor} not enabled: approval_status must be APPROVED")
            continue
        if any(not str(rule.get(field, "")).strip() for field in required_text):
            errors.append(f"rules[{idx}] for {sensor} missing units/reference/applicability/reviewed_by")
            continue
        try:
            low = float(rule["minimum"])
            high = float(rule["maximum"])
            if not math.isfinite(low) or not math.isfinite(high) or low > high:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            errors.append(f"rules[{idx}] for {sensor} needs finite minimum <= maximum")
            continue
        accepted.append({
            "sensor": sensor, "minimum": low, "maximum": high,
            "units": str(rule["units"]).strip(),
            "reference": str(rule["reference"]).strip(),
            "applicability": str(rule["applicability"]).strip(),
            "reviewed_by": str(rule["reviewed_by"]).strip(),
            # Even a local APPROVED flag is only user-declared; it is not independently authenticated.
            "approval_status": "USER_DECLARED_APPROVED",
        })
    status = "RULES_CONFIGURED_USER_DECLARED" if accepted else (
        "NO_USABLE_PHYSICAL_RULES" if raw["rules"] else "NO_APPROVED_PHYSICAL_RULES_CONFIGURED"
    )
    return {"status": status, "path": str(RULES_FILE), "rule_count": len(accepted),
            "rules": accepted, "errors": errors,
            "note": "Rules and references are locally user-supplied and are not independently authenticated by this prototype."}


def get_physics_rule_status():
    """Show discoverable sensor columns and whether documented bounds are available."""
    if not TRAIN_FILE.is_file() or not TEST_FILE.is_file():
        return {"status": "DATASET_UNAVAILABLE", "train_file_exists": TRAIN_FILE.is_file(),
                "test_file_exists": TEST_FILE.is_file(), "rules_file": str(RULES_FILE)}
    train = read_cmapss(TRAIN_FILE)
    sensors = _sensor_columns(train)
    rules = _load_rules(sensors)
    return {
        "dataset": "NASA C-MAPSS FD001",
        "data_type": "simulated_engine_benchmark",
        "sensor_columns": sensors,
        "sensor_column_count": len(sensors),
        "engineering_rules": rules,
        "statistical_screen_is_separate": True,
        "note": "No sensor-specific physical bounds are inferred from training quantiles. C-MAPSS sensor values must not be presented as real aircraft physical measurements without a documented mapping and units.",
    }


def analyze_latest_physics_checks():
    """Run deterministic data sanity checks and any configured rule checks."""
    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(f"NASA training dataset not found: {TRAIN_FILE}")
    if not TEST_FILE.is_file():
        raise FileNotFoundError(f"NASA test dataset not found: {TEST_FILE}")

    train = read_cmapss(TRAIN_FILE)
    test = read_cmapss(TEST_FILE)
    sensors = _sensor_columns(train)
    if not sensors:
        raise ValueError("No numeric sensor columns identified in training data.")
    if not {"unit", "cycle"}.issubset(set(test.columns)):
        raise ValueError("Expected unit and cycle columns in C-MAPSS test data.")

    rules_config = _load_rules(sensors)
    train_columns = set(train.columns)
    test_columns = set(test.columns)
    applicable_rules = rules_config.get("rules", [])
    latest_indices = test.groupby("unit", sort=True)["cycle"].idxmax().to_numpy()
    latest_rows = test.loc[latest_indices].sort_values("unit")

    # Dataset-level structural checks. These are integrity checks, not engine physics limits.
    problems = []
    if test["unit"].isna().any():
        problems.append("UNIT_ID_MISSING")
    if test["cycle"].isna().any():
        problems.append("CYCLE_MISSING")
    numeric_test = test.select_dtypes(include="number")
    nonfinite_columns = []
    for column in numeric_test.columns:
        values = numeric_test[column].to_numpy(dtype=float, copy=False)
        if not all(math.isfinite(float(v)) for v in values):
            nonfinite_columns.append(str(column))
    if nonfinite_columns:
        problems.append("NONFINITE_NUMERIC_VALUES")
    duplicate_cycles = int(test.duplicated(subset=["unit", "cycle"]).sum())
    if duplicate_cycles:
        problems.append("DUPLICATE_UNIT_CYCLE_ROWS")
    invalid_cycle_count = int((test["cycle"] <= 0).fillna(False).sum())
    if invalid_cycle_count:
        problems.append("NONPOSITIVE_CYCLE_VALUES")

    expected_numeric_columns = _numeric_columns(train)
    missing_columns = sorted(set(expected_numeric_columns) - test_columns)
    extra_columns = sorted(test_columns - train_columns)
    if missing_columns:
        problems.append("EXPECTED_COLUMNS_MISSING_FROM_TEST")
    if extra_columns:
        problems.append("UNEXPECTED_COLUMNS_IN_TEST")

    per_unit = []
    for _, row in latest_rows.iterrows():
        unit = int(row["unit"])
        cycle = int(row["cycle"])
        checks = []
        missing_values = []
        nonfinite_values = []
        rule_violations = []
        for sensor in sensors:
            value = row.get(sensor)
            try:
                fvalue = float(value)
            except (TypeError, ValueError):
                missing_values.append(sensor)
                continue
            if not math.isfinite(fvalue):
                nonfinite_values.append(sensor)
                continue
        checks.append({"rule_id": "DATA_FINITE_SENSOR_VALUES", "category": "DATA_INTEGRITY",
                       "status": "FAIL" if nonfinite_values else ("REVIEW" if missing_values else "PASS"),
                       "missing_sensors": missing_values, "nonfinite_sensors": nonfinite_values,
                       "basis": "Numeric values must be available and finite for a reliable calculation."})
        checks.append({"rule_id": "POSITIVE_CYCLE_INDEX", "category": "DATA_INTEGRITY",
                       "status": "PASS" if cycle > 0 else "FAIL", "observed_cycle": cycle,
                       "basis": "C-MAPSS cycle index is expected to be a positive integer; benchmark data convention, not an aircraft physics limit."})
        if applicable_rules:
            for rule in applicable_rules:
                try:
                    value = float(row[rule["sensor"]])
                except (TypeError, ValueError):
                    result = "REVIEW"
                    observed = None
                else:
                    observed = value
                    result = "FAIL" if not math.isfinite(value) or value < rule["minimum"] or value > rule["maximum"] else "PASS"
                if result == "FAIL":
                    rule_violations.append(rule["sensor"])
                checks.append({"rule_id": f"USER_RULE_{rule['sensor']}", "category": "USER_DECLARED_ENGINEERING_BOUND",
                    "status": result, "sensor": rule["sensor"], "observed_value": observed,
                    "minimum": rule["minimum"], "maximum": rule["maximum"], "units": rule["units"],
                    "reference": rule["reference"], "applicability": rule["applicability"],
                    "reviewed_by": rule["reviewed_by"], "rule_provenance": "USER_DECLARED_NOT_AUTHENTICATED"})
        failure_count = sum(1 for check in checks if check["status"] == "FAIL")
        review_count = sum(1 for check in checks if check["status"] == "REVIEW")
        per_unit.append({"engine_unit": unit, "engine_id": f"FD001-{unit:03d}",
            "last_observed_cycle": cycle, "checks": checks,
            "failed_check_count": failure_count, "review_check_count": review_count,
            "status": "SANITY_CHECK_FAILED" if failure_count else ("REVIEW_REQUIRED" if review_count else "DATA_SANITY_CHECKS_PASSED"),
            "interpretation": "Data sanity outcomes only; not a fault diagnosis, serviceability statement, or maintenance release."})

    failed_units = sum(1 for item in per_unit if item["failed_check_count"])
    review_units = sum(1 for item in per_unit if item["review_check_count"])
    return {
        "run_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).replace(microsecond=0).isoformat(),
        "dataset": "NASA C-MAPSS FD001", "data_type": "simulated_engine_benchmark",
        "engine_count": len(per_unit), "sensor_count": len(sensors),
        "dataset_integrity": {"status": "ISSUES_FOUND" if problems else "BASIC_CHECKS_PASSED",
            "issues": problems, "duplicate_unit_cycle_rows": duplicate_cycles,
            "nonpositive_cycle_count": invalid_cycle_count, "nonfinite_columns": nonfinite_columns,
            "missing_expected_columns": missing_columns, "unexpected_columns": extra_columns},
        "rule_configuration_status": rules_config["status"],
        "configured_rule_count": len(applicable_rules),
        "engines_with_failed_checks": failed_units,
        "engines_requiring_review": review_units,
        "results": per_unit,
        "limitations": [
            "Built-in checks are data-integrity sanity checks, not physical bounds for engine sensors.",
            "No sensor-specific physical limits are configured unless provided with units, applicability, source reference, and reviewer metadata.",
            "Local rule metadata is user-declared and not independently authenticated.",
            "NASA C-MAPSS FD001 is a simulated benchmark and is not mapped to a real aircraft fleet.",
            "Passing checks does not establish sensor health, engine safety, or maintenance release.",
        ],
    }
