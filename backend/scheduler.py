"""Prototype OR-Tools maintenance scheduler for AERO-GUARD NEXUS.

Important: output is a planning prototype, not an approved aircraft-maintenance
schedule. Durations and team capacity are configurable assumptions. Since the
current work_orders schema contains no bill-of-materials/task-to-spare mapping,
spare availability is reported as NOT_VERIFIED and never assumed satisfied.
"""

from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
import uuid

from ortools.sat.python import cp_model

from backend.database import get_connection

from backend.parts_engine import (
    assess_mapped_inventory,
    build_parts_clearance_gate,
    record_parts_decision_audit,
)


def _duration_hours(task, priority):
    """Return a transparent prototype duration estimate in whole hours."""
    task_text = (task or "").lower()
    if any(word in task_text for word in ("repair", "replace", "replacement")):
        return 8, "prototype task-keyword rule: repair/replacement = 8h"
    if any(word in task_text for word in ("inspection", "inspect")):
        return 4, "prototype task-keyword rule: inspection = 4h"
    if any(word in task_text for word in ("verification", "verify", "test")):
        return 3, "prototype task-keyword rule: verification/test = 3h"
    if any(word in task_text for word in ("assessment", "routine")):
        return 2, "prototype task-keyword rule: assessment/routine = 2h"
    defaults = {"High": 4, "Medium": 3, "Low": 2}
    return defaults.get(priority, 3), "prototype priority-based fallback duration"


def _ensure_schedule_tables(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS maintenance_schedule_runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            solver_status TEXT NOT NULL,
            horizon_days INTEGER NOT NULL,
            technician_capacity INTEGER NOT NULL,
            task_count INTEGER NOT NULL,
            scheduled_count INTEGER NOT NULL,
            unscheduled_count INTEGER NOT NULL,
            makespan_hours INTEGER,
            assumptions_json TEXT NOT NULL,
            spare_assessment_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS maintenance_schedule_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            work_order_id TEXT NOT NULL,
            asset_id TEXT NOT NULL,
            task TEXT NOT NULL,
            priority TEXT NOT NULL,
            original_status TEXT NOT NULL,
            schedule_status TEXT NOT NULL,
            duration_hours INTEGER NOT NULL,
            start_offset_hours INTEGER,
            end_offset_hours INTEGER,
            planned_start_utc TEXT,
            planned_end_utc TEXT,
            schedule_note TEXT NOT NULL,
            duration_basis TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES maintenance_schedule_runs(run_id),
            UNIQUE (run_id, work_order_id)
        );

        CREATE INDEX IF NOT EXISTS idx_schedule_items_run
            ON maintenance_schedule_items(run_id, start_offset_hours);
        CREATE INDEX IF NOT EXISTS idx_schedule_runs_created
            ON maintenance_schedule_runs(created_at);
        """
    )


def _decode_run(row, items):
    if row is None:
        return None
    result = dict(row)
    result["assumptions"] = json.loads(result.pop("assumptions_json"))
    result["spare_part_assessment"] = json.loads(
        result.pop("spare_assessment_json")
    )
    result["items"] = [dict(item) for item in items]
    result["parts_clearance_gate"] = result["spare_part_assessment"].get(
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


def generate_and_save_schedule(technician_capacity=2, horizon_days=7):
    """Schedule open work orders with CP-SAT and append an immutable run snapshot.

    STEP61_PARTS_CLEARANCE_GATE: all outputs stay proposals; no operational clearance is granted."""
    if not 1 <= int(technician_capacity) <= 20:
        raise ValueError("technician_capacity must be between 1 and 20")
    if not 1 <= int(horizon_days) <= 30:
        raise ValueError("horizon_days must be between 1 and 30")

    run_id = str(uuid.uuid4())
    created_dt = datetime.now(timezone.utc).replace(microsecond=0)
    created_at = created_dt.isoformat()
    horizon_hours = int(horizon_days) * 24

    # STEP61_PARTS_CLEARANCE_GATE: capture the actual mapped-demand assessment.
    parts_assessment = assess_mapped_inventory()
    parts_clearance_gate = build_parts_clearance_gate(parts_assessment)

    with closing(get_connection()) as connection:
        _ensure_schedule_tables(connection)
        work_orders = [dict(row) for row in connection.execute(
            """
            SELECT w.id, w.asset_id, w.task, w.priority, w.status, w.created_at
            FROM work_orders AS w
            WHERE LOWER(TRIM(w.status)) NOT IN ('completed', 'closed')
            ORDER BY
                CASE w.priority
                    WHEN 'High' THEN 1
                    WHEN 'Medium' THEN 2
                    ELSE 3
                END,
                w.created_at,
                w.id
            """
        ).fetchall()]
        spare_rows = [dict(row) for row in connection.execute(
            "SELECT code, name, stock, required FROM spare_parts ORDER BY code"
        ).fetchall()]

        prepared = []
        for row in work_orders:
            duration, basis = _duration_hours(row["task"], row["priority"])
            row["duration_hours"] = duration
            row["duration_basis"] = basis
            row["priority_weight"] = {"High": 100, "Medium": 20, "Low": 5}.get(
                row["priority"], 10
            )
            prepared.append(row)

        assumptions = {
            "solver": "OR-Tools CP-SAT",
            "default_technician_capacity": int(technician_capacity),
            "horizon_days": int(horizon_days),
            "time_unit": "continuous hours from run creation (UTC)",
            "priority_weights": {"High": 100, "Medium": 20, "Low": 5},
            "duration_estimation": "task-keyword prototype rules, priority fallback",
            "shift_calendar": "NOT_MODELED",
            "technician_skills_and_certifications": "NOT_MODELED",
            "task_precedence": "NOT_MODELED",
            "aircraft_or_engine_crosswalk": "NOT_AVAILABLE",
            "safety_validation": "NOT_PERFORMED",
            "parts_gate_policy": "FAIL_CLOSED; operational clearance is never granted by this prototype",
        }
        spare_assessment = {
            "status": parts_assessment.get("assessment_status", "UNKNOWN"),
            "assessment_status": parts_assessment.get("assessment_status", "UNKNOWN"),
            "parts_clearance_gate": parts_clearance_gate,
            "open_work_order_count": parts_assessment.get("open_work_order_count", 0),
            "open_work_orders_with_mappings": parts_assessment.get("open_work_orders_with_mappings", 0),
            "open_work_orders_without_mappings": parts_assessment.get("open_work_orders_without_mappings", 0),
            "unmapped_open_work_orders": parts_assessment.get("unmapped_open_work_orders", []),
            "parts": parts_assessment.get("parts", []),
            "spare_sku_count": len(spare_rows),
            "spare_inventory_snapshot": spare_rows,
            "stock_reservations_created": False,
            "scheduling_clearance": False,
            "operational_clearance": False,
            "reason": parts_assessment.get("note", "Mapped demand is incomplete or unauthenticated."),
        }

        scheduled_items = []
        unscheduled_items = []
        solver_status_name = "NO_PENDING_WORK_ORDERS"
        makespan_hours = 0

        if prepared:
            model = cp_model.CpModel()
            starts = []
            ends = []
            presences = []
            intervals = []
            priority_weights = []

            for index, item in enumerate(prepared):
                duration = item["duration_hours"]
                start = model.new_int_var(0, horizon_hours, f"start_{index}")
                end = model.new_int_var(0, horizon_hours, f"end_{index}")
                presence = model.new_bool_var(f"scheduled_{index}")
                interval = model.new_optional_interval_var(
                    start, duration, end, presence, f"work_{index}"
                )
                model.add(end <= horizon_hours).only_enforce_if(presence)
                # Prevent an absent task from affecting the timing objective.
                model.add(start == 0).only_enforce_if(presence.Not())
                starts.append(start)
                ends.append(end)
                presences.append(presence)
                intervals.append(interval)
                priority_weights.append(item["priority_weight"])

            # Each scheduled job consumes one crew. Crews can work in parallel.
            model.add_cumulative(
                intervals,
                [1] * len(intervals),
                int(technician_capacity),
            )

            scheduled_value = sum(
                presences[i] * priority_weights[i]
                for i in range(len(prepared))
            )
            weighted_start_cost = sum(
                starts[i] * priority_weights[i]
                for i in range(len(prepared))
            )
            # First maximize total priority-weighted work scheduled; then favor
            # earlier starts for higher-priority work.
            priority_scale = horizon_hours * sum(priority_weights) + 1
            model.maximize(scheduled_value * priority_scale - weighted_start_cost)

            solver = cp_model.CpSolver()
            solver.parameters.max_time_in_seconds = 10.0
            solver.parameters.num_search_workers = 8
            status = solver.solve(model)
            if status == cp_model.OPTIMAL:
                solver_status_name = "OPTIMAL"
            elif status == cp_model.FEASIBLE:
                solver_status_name = "FEASIBLE"
            else:
                solver_status_name = solver.status_name(status)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                for index, item in enumerate(prepared):
                    if solver.value(presences[index]):
                        start_offset = int(solver.value(starts[index]))
                        end_offset = int(solver.value(ends[index]))
                        start_dt = created_dt + timedelta(hours=start_offset)
                        end_dt = created_dt + timedelta(hours=end_offset)
                        scheduled_items.append({
                            **item,
                            "schedule_status": "PROPOSED_PENDING_PARTS_VERIFICATION",
                            "start_offset_hours": start_offset,
                            "end_offset_hours": end_offset,
                            "planned_start_utc": start_dt.isoformat(),
                            "planned_end_utc": end_dt.isoformat(),
                            "schedule_note": (
                                "CP-SAT resource-feasible placement under prototype assumptions; "
                                "verify parts, staffing, shifts and approved maintenance data."
                            ),
                        })
                    else:
                        unscheduled_items.append({
                            **item,
                            "schedule_status": "UNSCHEDULED",
                            "start_offset_hours": None,
                            "end_offset_hours": None,
                            "planned_start_utc": None,
                            "planned_end_utc": None,
                            "schedule_note": "Not selected within the planning horizon/capacity by the solver.",
                        })
                if scheduled_items:
                    makespan_hours = max(item["end_offset_hours"] for item in scheduled_items)
            else:
                for item in prepared:
                    unscheduled_items.append({
                        **item,
                        "schedule_status": "UNSCHEDULED",
                        "start_offset_hours": None,
                        "end_offset_hours": None,
                        "planned_start_utc": None,
                        "planned_end_utc": None,
                        "schedule_note": f"Solver did not produce a feasible solution ({solver_status_name}).",
                    })

        all_items = sorted(
            scheduled_items + unscheduled_items,
            key=lambda item: (
                item["start_offset_hours"] is None,
                item["start_offset_hours"] if item["start_offset_hours"] is not None else horizon_hours + 1,
                item["work_orders"] if "work_orders" in item else item["id"],
            ),
        )
        # A schedule is always conditional until job-to-part/BOM links and
        # roster/shift constraints are explicitly defined and checked.
        connection.execute(
            """
            INSERT INTO maintenance_schedule_runs (
                run_id, created_at, solver_status, horizon_days,
                technician_capacity, task_count, scheduled_count,
                unscheduled_count, makespan_hours, assumptions_json,
                spare_assessment_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id, created_at, solver_status_name, int(horizon_days),
                int(technician_capacity), len(prepared), len(scheduled_items),
                len(unscheduled_items), makespan_hours,
                json.dumps(assumptions), json.dumps(spare_assessment),
            ),
        )

        item_insert_rows = []
        for item in all_items:
            item_insert_rows.append((
                run_id,
                str(item["id"]),
                str(item.get("asset_id") or ""),
                str(item["task"]),
                str(item["priority"]),
                str(item["status"]),
                item["schedule_status"],
                int(item["duration_hours"]),
                item["start_offset_hours"],
                item["end_offset_hours"],
                item["planned_start_utc"],
                item["planned_end_utc"],
                item["schedule_note"],
                item["duration_basis"],
            ))

        connection.executemany(
            """
            INSERT INTO maintenance_schedule_items (
                run_id, work_order_id, asset_id, task, priority,
                original_status, schedule_status, duration_hours,
                start_offset_hours, end_offset_hours, planned_start_utc,
                planned_end_utc, schedule_note, duration_basis
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            item_insert_rows,
        )
        connection.commit()

    parts_decision_audit = record_parts_decision_audit(
        context="SCHEDULER_PLAN",
        run_id=run_id,
        assessment=parts_assessment,
        gate=parts_clearance_gate,
    )

    return {
        "run_id": run_id,
        "created_at": created_at,
        "solver": "OR-Tools CP-SAT",
        "solver_status": solver_status_name,
        "horizon_days": int(horizon_days),
        "technician_capacity": int(technician_capacity),
        "task_count": len(prepared),
        "scheduled_count": len(scheduled_items),
        "unscheduled_count": len(unscheduled_items),
        "makespan_hours": makespan_hours,
        "schedule_status": "PROPOSED_PENDING_PARTS_VERIFICATION",
        "operational_clearance": False,
        "parts_clearance_gate": parts_clearance_gate,
        "parts_decision_audit_id": parts_decision_audit["audit_id"],
        "assumptions": assumptions,
        "spare_part_assessment": spare_assessment,
        "items": [
            {
                "work_order_id": item["id"],
                "asset_id": item.get("asset_id"),
                "task": item["task"],
                "priority": item["priority"],
                "original_status": item["status"],
                "schedule_status": item["schedule_status"],
                "duration_hours": item["duration_hours"],
                "start_offset_hours": item["start_offset_hours"],
                "end_offset_hours": item["end_offset_hours"],
                "planned_start_utc": item["planned_start_utc"],
                "planned_end_utc": item["planned_end_utc"],
                "duration_basis": item["duration_basis"],
                "schedule_note": item["schedule_note"],
            }
            for item in all_items
        ],
        "note": (
            "Prototype schedule only. It does not model approved maintenance "
            "instructions, shifts, skill/certification constraints, task dependencies, "
            "or verified spare requirements. Human maintenance planning review is required."
        ),
    }


def get_latest_saved_schedule():
    with closing(get_connection()) as connection:
        _ensure_schedule_tables(connection)
        row = connection.execute(
            "SELECT * FROM maintenance_schedule_runs ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        run = dict(row)
        items = connection.execute(
            "SELECT * FROM maintenance_schedule_items WHERE run_id = ? "
            "ORDER BY (start_offset_hours IS NULL), start_offset_hours, work_order_id",
            (run["run_id"],),
        ).fetchall()
        result = _decode_run(run, items)
        result["schedule_status"] = "PROPOSED_PENDING_PARTS_VERIFICATION"
        result["note"] = (
            "Persisted prototype schedule; verify parts, roster, shifts and approved instructions before use."
        )
        return result


def get_schedule_run_history(limit=20):
    limit = max(1, min(int(limit), 200))
    with closing(get_connection()) as connection:
        _ensure_schedule_tables(connection)
        rows = connection.execute(
            "SELECT run_id, created_at, solver_status, horizon_days, "
            "technician_capacity, task_count, scheduled_count, "
            "unscheduled_count, makespan_hours FROM maintenance_schedule_runs "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
