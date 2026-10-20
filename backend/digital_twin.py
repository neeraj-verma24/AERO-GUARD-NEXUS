"""AERO-GUARD NEXUS Step 47: auditable what-if scheduling comparison.

This is an operational planning scenario tool, not a physics-based engine twin
or an approved aircraft-maintenance scheduler. It reads one snapshot of open
work orders, solves each scenario independently with OR-Tools CP-SAT, and saves
comparison results separately from maintenance_schedule_runs. It never updates
work orders or claims that spare parts, shifts, skills, or safety constraints
have been verified.
"""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import math
import uuid

from ortools.sat.python import cp_model

from backend.database import get_connection

from backend.parts_engine import (
    assess_mapped_inventory,
    build_parts_clearance_gate,
    record_parts_decision_audit,
)


DEFAULT_SCENARIOS = [
    {
        "name": "Baseline",
        "technician_capacity": 2,
        "horizon_days": 7,
        "duration_multiplier": 1.0,
        "priority_weights": {"High": 100, "Medium": 20, "Low": 5},
    },
    {
        "name": "Extra technician",
        "technician_capacity": 3,
        "horizon_days": 7,
        "duration_multiplier": 1.0,
        "priority_weights": {"High": 100, "Medium": 20, "Low": 5},
    },
    {
        "name": "Longer horizon",
        "technician_capacity": 2,
        "horizon_days": 14,
        "duration_multiplier": 1.0,
        "priority_weights": {"High": 100, "Medium": 20, "Low": 5},
    },
    {
        "name": "Conservative duration (+25%)",
        "technician_capacity": 2,
        "horizon_days": 7,
        "duration_multiplier": 1.25,
        "priority_weights": {"High": 100, "Medium": 20, "Low": 5},
    },
]


def _ensure_tables(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS digital_twin_comparison_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            work_order_count INTEGER NOT NULL,
            scenario_count INTEGER NOT NULL,
            results_json TEXT NOT NULL,
            assumptions_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_digital_twin_runs_created
            ON digital_twin_comparison_runs(created_at);
        """
    )


def _duration_hours(task, priority):
    text = (task or "").lower()
    if any(word in text for word in ("repair", "replace", "replacement")):
        return 8, "prototype task-keyword rule: repair/replacement = 8h"
    if any(word in text for word in ("inspection", "inspect")):
        return 4, "prototype task-keyword rule: inspection = 4h"
    if any(word in text for word in ("verification", "verify", "test")):
        return 3, "prototype task-keyword rule: verification/test = 3h"
    if any(word in text for word in ("assessment", "routine")):
        return 2, "prototype task-keyword rule: assessment/routine = 2h"
    defaults = {"High": 4, "Medium": 3, "Low": 2}
    return defaults.get(priority, 3), "prototype priority-based fallback duration"


def _normalise_scenarios(scenarios):
    if scenarios is None:
        scenarios = [dict(s) for s in DEFAULT_SCENARIOS]
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("scenarios must be a non-empty list")
    if len(scenarios) > 10:
        raise ValueError("A maximum of 10 scenarios is allowed per comparison")

    normalised = []
    names = set()
    for index, raw in enumerate(scenarios, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Scenario {index} must be an object")
        name = str(raw.get("name") or f"Scenario {index}").strip()
        if not name:
            raise ValueError(f"Scenario {index} needs a non-empty name")
        if name.lower() in names:
            raise ValueError("Scenario names must be unique (case-insensitive)")
        names.add(name.lower())

        try:
            capacity = int(raw.get("technician_capacity", 2))
            horizon = int(raw.get("horizon_days", 7))
            multiplier = float(raw.get("duration_multiplier", 1.0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Scenario '{name}' contains an invalid numeric value") from exc
        if not 1 <= capacity <= 20:
            raise ValueError(f"Scenario '{name}': technician_capacity must be 1..20")
        if not 1 <= horizon <= 30:
            raise ValueError(f"Scenario '{name}': horizon_days must be 1..30")
        if not math.isfinite(multiplier) or not 0.5 <= multiplier <= 3.0:
            raise ValueError(f"Scenario '{name}': duration_multiplier must be 0.5..3.0")

        weights = raw.get("priority_weights") or {"High": 100, "Medium": 20, "Low": 5}
        if not isinstance(weights, dict):
            raise ValueError(f"Scenario '{name}': priority_weights must be an object")
        clean_weights = {}
        for priority, default in (("High", 100), ("Medium", 20), ("Low", 5)):
            try:
                weight = int(weights.get(priority, default))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Scenario '{name}': invalid weight for {priority}") from exc
            if not 1 <= weight <= 10000:
                raise ValueError(f"Scenario '{name}': weights must be between 1 and 10000")
            clean_weights[priority] = weight

        normalised.append({
            "name": name,
            "technician_capacity": capacity,
            "horizon_days": horizon,
            "duration_multiplier": multiplier,
            "priority_weights": clean_weights,
        })
    return normalised


def _solve_one_scenario(scenario, work_orders, created_dt):
    horizon_hours = scenario["horizon_days"] * 24
    prepared = []
    for row in work_orders:
        base_duration, duration_basis = _duration_hours(row.get("task"), row.get("priority"))
        duration = max(1, int(math.ceil(base_duration * scenario["duration_multiplier"])))
        item = dict(row)
        item["duration_hours"] = duration
        item["base_duration_hours"] = base_duration
        item["duration_basis"] = duration_basis
        item["priority_weight"] = scenario["priority_weights"].get(item.get("priority"), 10)
        prepared.append(item)

    scheduled = []
    unscheduled = []
    solver_status = "NO_PENDING_WORK_ORDERS"
    makespan_hours = 0

    if prepared:
        model = cp_model.CpModel()
        starts, ends, presences, intervals = [], [], [], []
        for index, item in enumerate(prepared):
            duration = item["duration_hours"]
            start = model.new_int_var(0, horizon_hours, f"start_{index}")
            end = model.new_int_var(0, horizon_hours, f"end_{index}")
            presence = model.new_bool_var(f"scheduled_{index}")
            interval = model.new_optional_interval_var(start, duration, end, presence, f"work_{index}")
            model.add(end <= horizon_hours).only_enforce_if(presence)
            model.add(start == 0).only_enforce_if(presence.Not())
            starts.append(start)
            ends.append(end)
            presences.append(presence)
            intervals.append(interval)

        model.add_cumulative(intervals, [1] * len(intervals), scenario["technician_capacity"])
        weight_sum = sum(item["priority_weight"] for item in prepared)
        scheduled_value = sum(presences[i] * prepared[i]["priority_weight"] for i in range(len(prepared)))
        weighted_start_cost = sum(starts[i] * prepared[i]["priority_weight"] for i in range(len(prepared)))
        priority_scale = horizon_hours * weight_sum + 1
        model.maximize(scheduled_value * priority_scale - weighted_start_cost)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        solver.parameters.num_search_workers = 8
        status = solver.solve(model)
        if status == cp_model.OPTIMAL:
            solver_status = "OPTIMAL"
        elif status == cp_model.FEASIBLE:
            solver_status = "FEASIBLE"
        else:
            solver_status = solver.status_name(status)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            for index, item in enumerate(prepared):
                if solver.value(presences[index]):
                    start_offset = int(solver.value(starts[index]))
                    end_offset = int(solver.value(ends[index]))
                    scheduled.append({
                        "work_order_id": str(item["id"]),
                        "asset_id": item.get("asset_id"),
                        "task": item["task"],
                        "priority": item["priority"],
                        "original_status": item["status"],
                        "duration_hours": item["duration_hours"],
                        "base_duration_hours": item["base_duration_hours"],
                        "start_offset_hours": start_offset,
                        "end_offset_hours": end_offset,
                        "planned_start_utc": (created_dt + timedelta(hours=start_offset)).isoformat(),
                        "planned_end_utc": (created_dt + timedelta(hours=end_offset)).isoformat(),
                        "duration_basis": item["duration_basis"],
                        "schedule_status": "PROPOSED_PENDING_VERIFICATION",
                    })
                else:
                    unscheduled.append({
                        "work_order_id": str(item["id"]),
                        "asset_id": item.get("asset_id"),
                        "task": item["task"],
                        "priority": item["priority"],
                        "duration_hours": item["duration_hours"],
                        "base_duration_hours": item["base_duration_hours"],
                        "schedule_status": "UNSCHEDULED",
                        "reason": "Not selected within the scenario horizon/capacity by the solver",
                    })
            if scheduled:
                makespan_hours = max(item["end_offset_hours"] for item in scheduled)
        else:
            for item in prepared:
                unscheduled.append({
                    "work_order_id": str(item["id"]),
                    "asset_id": item.get("asset_id"),
                    "task": item["task"],
                    "priority": item["priority"],
                    "duration_hours": item["duration_hours"],
                    "base_duration_hours": item["base_duration_hours"],
                    "schedule_status": "UNSCHEDULED",
                    "reason": f"Solver did not produce a feasible solution ({solver_status})",
                })

    scheduled.sort(key=lambda item: (item["start_offset_hours"], item["work_order_id"]))
    unscheduled.sort(key=lambda item: (item["priority"], item["work_order_id"]))
    all_weight = sum(scenario["priority_weights"].get(row.get("priority"), 10) for row in work_orders)
    scheduled_weight = sum(scenario["priority_weights"].get(item["priority"], 10) for item in scheduled)
    return {
        "name": scenario["name"],
        "parameters": scenario,
        "solver": "OR-Tools CP-SAT",
        "solver_status": solver_status,
        "task_count": len(prepared),
        "scheduled_count": len(scheduled),
        "unscheduled_count": len(unscheduled),
        "priority_weighted_coverage_percent": round(100 * scheduled_weight / all_weight, 2) if all_weight else 100.0,
        "makespan_hours": makespan_hours,
        "scheduled_items": scheduled,
        "unscheduled_items": unscheduled,
        "spare_part_assessment": "NOT_VERIFIED",
    }


def compare_scenarios(scenarios=None):
    """Run several independent what-if plans on the same work-order snapshot.

    STEP61_PARTS_CLEARANCE_GATE: scenarios include a fail-closed parts-gate snapshot.

    Persists one comparison record in its own table. It does not write schedule
    records to maintenance_schedule_runs or modify work_orders/assets/spares.
    """
    scenario_list = _normalise_scenarios(scenarios)
    # STEP61_PARTS_CLEARANCE_GATE: a snapshot for comparison only, never a clearance.
    parts_assessment = assess_mapped_inventory()
    parts_clearance_gate = build_parts_clearance_gate(parts_assessment)
    run_id = str(uuid.uuid4())
    created_dt = datetime.now(timezone.utc).replace(microsecond=0)
    created_at = created_dt.isoformat()

    with closing(get_connection()) as connection:
        _ensure_tables(connection)
        work_orders = [dict(row) for row in connection.execute(
            """
            SELECT w.id, w.asset_id, w.task, w.priority, w.status, w.created_at
            FROM work_orders AS w
            WHERE LOWER(TRIM(w.status)) NOT IN ('completed', 'closed')
            ORDER BY CASE w.priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END,
                     w.created_at, w.id
            """
        ).fetchall()]
        spare_count = connection.execute("SELECT COUNT(*) FROM spare_parts").fetchone()[0]

        results = [
            _solve_one_scenario(scenario, work_orders, created_dt)
            for scenario in scenario_list
        ]
        for scenario_result in results:
            scenario_result["spare_part_assessment"] = parts_assessment.get("assessment_status", "UNKNOWN")
            scenario_result["parts_clearance_gate"] = parts_clearance_gate
            scenario_result["operational_clearance"] = False
        baseline = results[0]
        for result in results:
            result["delta_vs_first_scenario"] = {
                "scheduled_count": result["scheduled_count"] - baseline["scheduled_count"],
                "makespan_hours": result["makespan_hours"] - baseline["makespan_hours"],
                "priority_weighted_coverage_percentage_points": round(
                    result["priority_weighted_coverage_percent"] - baseline["priority_weighted_coverage_percent"], 2
                ),
            }

        assumptions = {
            "purpose": "operational what-if scheduling comparison; not a physics-based aircraft digital twin",
            "same_work_order_snapshot_for_all_scenarios": True,
            "open_work_order_definition": "status not completed/closed",
            "duration_estimation": "prototype task-keyword rules; scenario duration multiplier applied and rounded up to whole hours",
            "shift_calendar": "NOT_MODELED",
            "technician_skills_and_certifications": "NOT_MODELED",
            "task_precedence": "NOT_MODELED",
            "aircraft_or_engine_crosswalk": "NOT_AVAILABLE",
            "spare_part_assessment": parts_assessment.get("assessment_status", "UNKNOWN"),
            "parts_assessment_snapshot": parts_assessment,
            "parts_clearance_gate": parts_clearance_gate,
            "operational_clearance": False,
            "stock_reservations_created": False,
            "spare_sku_count_in_snapshot": int(spare_count),
            "safety_validation": "NOT_PERFORMED",
            "changes_to_operational_records": "NONE; only a separate comparison snapshot is saved",
            "interpretation": "Solver results are conditional planning estimates and require qualified human review.",
        }
        connection.execute(
            """INSERT INTO digital_twin_comparison_runs
               (run_id, created_at, work_order_count, scenario_count, results_json, assumptions_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, created_at, len(work_orders), len(results), json.dumps(results), json.dumps(assumptions)),
        )
        connection.commit()

    parts_decision_audit = record_parts_decision_audit(
        context="DIGITAL_TWIN_COMPARE",
        run_id=run_id,
        assessment=parts_assessment,
        gate=parts_clearance_gate,
    )

    return {
        "run_id": run_id,
        "created_at": created_at,
        "work_order_count": len(work_orders),
        "scenario_count": len(results),
        "parts_assessment": parts_assessment,
        "parts_clearance_gate": parts_clearance_gate,
        "operational_clearance": False,
        "parts_decision_audit_id": parts_decision_audit["audit_id"],
        "comparison_metric_note": "Compare scheduled_count, priority_weighted_coverage_percent, and makespan_hours; faster is not automatically safer or better.",
        "assumptions": assumptions,
        "scenarios": results,
        "note": "What-if prototype only. It does not model approved instructions, shifts, skill/certification limits, task precedence, verified parts, or validated aircraft physics.",
    }


def _decode_comparison(row):
    if row is None:
        return None
    result = dict(row)
    result["scenarios"] = json.loads(result.pop("results_json"))
    result["assumptions"] = json.loads(result.pop("assumptions_json"))
    result["parts_assessment"] = result["assumptions"].get(
        "parts_assessment_snapshot", {"assessment_status": "LEGACY_RUN_NO_PARTS_ASSESSMENT_SNAPSHOT"}
    )
    result["parts_clearance_gate"] = result["assumptions"].get(
        "parts_clearance_gate",
        {
            "gate_status": "LEGACY_RUN_NO_SAVED_PARTS_GATE",
            "operational_clearance": False,
            "scheduling_clearance": False,
            "blockers": ["LEGACY_RUN_HAS_NO_PARTS_GATE_SNAPSHOT"],
        },
    )
    result["operational_clearance"] = False
    return result


def get_latest_scenario_comparison():
    with closing(get_connection()) as connection:
        _ensure_tables(connection)
        row = connection.execute(
            "SELECT * FROM digital_twin_comparison_runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return _decode_comparison(row)


def get_scenario_comparison_history(limit=20):
    limit = max(1, min(int(limit), 200))
    with closing(get_connection()) as connection:
        _ensure_tables(connection)
        rows = connection.execute(
            "SELECT run_id, created_at, work_order_count, scenario_count FROM digital_twin_comparison_runs "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
