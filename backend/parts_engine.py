"""Work-order-to-spare-part mapping and inventory assessment for AERO-GUARD NEXUS.

Prototype only. Mapping provenance is user-supplied and not authenticated.
Inventory is assessed from current stock but not reserved; this module does not
approve maintenance or guarantee schedule feasibility.
"""
from contextlib import closing
from datetime import datetime, timezone
import sqlite3
import json

from backend.database import get_connection

ALLOWED_MAPPING_TYPES = {"DEMO_ONLY", "DOCUMENTED"}


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ensure_tables(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS work_order_part_requirements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_order_id TEXT NOT NULL,
            spare_code TEXT NOT NULL,
            quantity_required INTEGER NOT NULL CHECK (quantity_required > 0),
            mapping_type TEXT NOT NULL CHECK (mapping_type IN ('DEMO_ONLY', 'DOCUMENTED')),
            mapping_note TEXT NOT NULL,
            mapping_reference TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (work_order_id, spare_code),
            FOREIGN KEY (work_order_id) REFERENCES work_orders(id),
            FOREIGN KEY (spare_code) REFERENCES spare_parts(code)
        );
        CREATE INDEX IF NOT EXISTS idx_wo_part_requirements_order
            ON work_order_part_requirements(work_order_id);
        CREATE INDEX IF NOT EXISTS idx_wo_part_requirements_spare
            ON work_order_part_requirements(spare_code);
        CREATE TABLE IF NOT EXISTS work_order_part_mapping_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_order_id TEXT NOT NULL,
            spare_code TEXT NOT NULL,
            action TEXT NOT NULL,
            mapping_type TEXT NOT NULL,
            quantity_required INTEGER NOT NULL,
            mapping_note TEXT NOT NULL,
            mapping_reference TEXT,
            created_at TEXT NOT NULL
        );
        """
    )


def save_work_order_part_mapping(
    work_order_id,
    spare_code,
    quantity_required,
    mapping_type="DEMO_ONLY",
    mapping_note="",
    mapping_reference=None,
):
    """Create/update an explicit work order -> spare mapping.

    DOCUMENTED means the operator supplied a reference; the reference itself is
    not authenticated by this prototype.
    """
    order_id = str(work_order_id or "").strip()
    code = str(spare_code or "").strip()
    note = str(mapping_note or "").strip()
    reference = str(mapping_reference or "").strip() or None
    kind = str(mapping_type or "DEMO_ONLY").strip().upper()
    try:
        qty = int(quantity_required)
    except (TypeError, ValueError):
        raise ValueError("quantity_required must be a positive integer.") from None
    if not order_id:
        raise ValueError("work_order_id is required.")
    if not code:
        raise ValueError("spare_code is required.")
    if qty < 1 or qty != quantity_required:
        raise ValueError("quantity_required must be a positive integer.")
    if kind not in ALLOWED_MAPPING_TYPES:
        raise ValueError("mapping_type must be DEMO_ONLY or DOCUMENTED.")
    if len(note) < 5:
        raise ValueError("mapping_note must contain at least 5 characters.")
    if kind == "DOCUMENTED" and not reference:
        raise ValueError("DOCUMENTED mapping requires mapping_reference.")

    now = _now()
    with closing(get_connection()) as connection:
        _ensure_tables(connection)
        order = connection.execute("SELECT id, task, status FROM work_orders WHERE id = ?", (order_id,)).fetchone()
        if order is None:
            raise LookupError("Work order not found.")
        spare = connection.execute("SELECT code, name, stock, required FROM spare_parts WHERE code = ?", (code,)).fetchone()
        if spare is None:
            raise LookupError("Spare part code not found.")
        old = connection.execute(
            "SELECT id FROM work_order_part_requirements WHERE work_order_id = ? AND spare_code = ?",
            (order_id, code),
        ).fetchone()
        action = "UPDATED" if old else "CREATED"
        connection.execute(
            """INSERT INTO work_order_part_requirements
               (work_order_id, spare_code, quantity_required, mapping_type, mapping_note,
                mapping_reference, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(work_order_id, spare_code) DO UPDATE SET
                 quantity_required=excluded.quantity_required,
                 mapping_type=excluded.mapping_type,
                 mapping_note=excluded.mapping_note,
                 mapping_reference=excluded.mapping_reference,
                 updated_at=excluded.updated_at""",
            (order_id, code, qty, kind, note, reference, now, now),
        )
        connection.execute(
            """INSERT INTO work_order_part_mapping_audit
               (work_order_id, spare_code, action, mapping_type, quantity_required,
                mapping_note, mapping_reference, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (order_id, code, action, kind, qty, note, reference, now),
        )
        connection.commit()
    return {
        "saved": True,
        "action": action,
        "mapping": {
            "work_order_id": order_id,
            "task": order["task"],
            "work_order_status": order["status"],
            "spare_code": code,
            "spare_name": spare["name"],
            "quantity_required": qty,
            "current_stock": int(spare["stock"]),
            "mapping_type": kind,
            "mapping_note": note,
            "mapping_reference": reference,
            "mapping_status": "DEMO_ONLY_NOT_OPERATIONAL" if kind == "DEMO_ONLY" else "USER_DECLARED_DOCUMENTED_NOT_AUTHENTICATED",
            "created_or_updated_at": now,
        },
        "note": "This mapping is user-supplied. Inventory is not reserved and mapping references are not independently authenticated.",
    }


def list_work_order_part_mappings(limit=100):
    limit = max(1, min(int(limit), 500))
    with closing(get_connection()) as connection:
        _ensure_tables(connection)
        rows = connection.execute(
            """SELECT r.id, r.work_order_id, w.task, w.priority, w.status AS work_order_status,
                      r.spare_code, s.name AS spare_name, s.stock AS current_stock,
                      s.required AS global_reorder_reference, r.quantity_required,
                      r.mapping_type, r.mapping_note, r.mapping_reference,
                      r.created_at, r.updated_at
               FROM work_order_part_requirements r
               JOIN work_orders w ON w.id = r.work_order_id
               JOIN spare_parts s ON s.code = r.spare_code
               ORDER BY w.priority, w.id, s.code LIMIT ?""",
            (limit,),
        ).fetchall()
    items = [dict(row) for row in rows]
    for item in items:
        item["mapping_status"] = (
            "DEMO_ONLY_NOT_OPERATIONAL" if item["mapping_type"] == "DEMO_ONLY"
            else "USER_DECLARED_DOCUMENTED_NOT_AUTHENTICATED"
        )
    return {"count": len(items), "items": items,
            "note": "spare_parts.required is a global inventory/reorder reference, not a work-order BOM quantity."}


def assess_mapped_inventory():
    """Compare available stock with the sum of explicit mapped demand for open WOs."""
    with closing(get_connection()) as connection:
        _ensure_tables(connection)
        open_orders = connection.execute(
            """SELECT id, task, priority, status FROM work_orders
               WHERE LOWER(TRIM(status)) NOT IN ('completed', 'closed') ORDER BY id"""
        ).fetchall()
        mapped_orders = {row["work_order_id"] for row in connection.execute(
            """SELECT DISTINCT r.work_order_id FROM work_order_part_requirements r
               JOIN work_orders w ON w.id = r.work_order_id
               WHERE LOWER(TRIM(w.status)) NOT IN ('completed', 'closed')"""
        ).fetchall()}
        aggregate_rows = connection.execute(
            """SELECT s.code AS spare_code, s.name AS spare_name, s.stock AS current_stock,
                      SUM(r.quantity_required) AS mapped_open_demand,
                      COUNT(DISTINCT r.work_order_id) AS mapped_open_work_orders,
                      GROUP_CONCAT(DISTINCT r.mapping_type) AS mapping_types
               FROM work_order_part_requirements r
               JOIN work_orders w ON w.id = r.work_order_id
               JOIN spare_parts s ON s.code = r.spare_code
               WHERE LOWER(TRIM(w.status)) NOT IN ('completed', 'closed')
               GROUP BY s.code, s.name, s.stock ORDER BY s.code"""
        ).fetchall()
        unmapped = [dict(row) for row in open_orders if row["id"] not in mapped_orders]
    parts = []
    for row in aggregate_rows:
        item = dict(row)
        item["mapped_open_demand"] = int(item["mapped_open_demand"] or 0)
        item["shortage_quantity"] = max(0, item["mapped_open_demand"] - int(item["current_stock"]))
        item["stock_status"] = "AGGREGATE_SHORTAGE" if item["shortage_quantity"] > 0 else "STOCK_SUFFICIENT_NOT_RESERVED"
        item["mapping_provenance_status"] = (
            "CONTAINS_DEMO_ONLY_MAPPING" if "DEMO_ONLY" in (item["mapping_types"] or "")
            else "USER_DECLARED_DOCUMENTED_NOT_AUTHENTICATED"
        )
        parts.append(item)
    if not parts:
        status = "NOT_VERIFIED_NO_PART_REQUIREMENTS_MAPPED"
    elif unmapped:
        status = "PARTIAL_MAPPING_REVIEW_REQUIRED"
    elif any(p["shortage_quantity"] > 0 for p in parts):
        status = "MAPPED_PARTS_WITH_AGGREGATE_SHORTAGE"
    else:
        status = "MAPPED_STOCK_SUFFICIENT_NOT_RESERVED"
    return {
        "assessment_status": status,
        "open_work_order_count": len(open_orders),
        "open_work_orders_with_mappings": len(mapped_orders),
        "open_work_orders_without_mappings": len(unmapped),
        "unmapped_open_work_orders": unmapped,
        "parts": parts,
        "scope": "Aggregated explicit mappings only; unmapped demand is unknown.",
        "stock_reservations_created": False,
        "scheduling_clearance": False,
        "note": "Current stock is compared with sum of mapped demand across all open work orders. Missing mappings mean incomplete demand; no stock is reserved. DEMO_ONLY mappings are for prototype testing only.",
    }


def get_work_order_part_mapping_audit(limit=50):
    limit = max(1, min(int(limit), 500))
    with closing(get_connection()) as connection:
        _ensure_tables(connection)
        rows = connection.execute(
            """SELECT audit_id, work_order_id, spare_code, action, mapping_type,
                      quantity_required, mapping_note, mapping_reference, created_at
               FROM work_order_part_mapping_audit ORDER BY audit_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return {"count": len(rows), "items": [dict(row) for row in rows]}


# STEP61_PARTS_CLEARANCE_GATE

def build_parts_clearance_gate(assessment):
    """Fail-closed planning gate; this prototype never grants operational clearance."""
    assessment = assessment if isinstance(assessment, dict) else {}
    parts = assessment.get("parts") or []
    blockers = []

    if not parts:
        blockers.append("NO_EXPLICIT_PART_REQUIREMENTS_MAPPED")
    if int(assessment.get("open_work_orders_without_mappings", 0) or 0) > 0:
        blockers.append("UNMAPPED_OPEN_WORK_ORDERS_DEMAND_UNKNOWN")
    if any(int(item.get("shortage_quantity", 0) or 0) > 0 for item in parts):
        blockers.append("AGGREGATE_STOCK_SHORTAGE")
    if any(item.get("mapping_provenance_status") == "CONTAINS_DEMO_ONLY_MAPPING" for item in parts):
        blockers.append("DEMO_ONLY_MAPPING_PRESENT")
    if any(item.get("mapping_provenance_status") == "USER_DECLARED_DOCUMENTED_NOT_AUTHENTICATED" for item in parts):
        blockers.append("MAPPING_REFERENCE_NOT_AUTHENTICATED")
    if assessment.get("stock_reservations_created") is not True:
        blockers.append("STOCK_NOT_RESERVED")
    if assessment.get("scheduling_clearance") is not True:
        blockers.append("PARTS_ASSESSMENT_DOES_NOT_GRANT_SCHEDULING_CLEARANCE")

    # These are permanent prototype boundaries, not conditions a demo mapping can override.
    blockers.append("OPERATIONAL_CLEARANCE_POLICY_NOT_IMPLEMENTED")
    blockers = list(dict.fromkeys(blockers))
    return {
        "gate_status": "BLOCKED_REVIEW_REQUIRED",
        "source_assessment_status": assessment.get("assessment_status", "UNKNOWN"),
        "operational_clearance": False,
        "scheduling_clearance": False,
        "stock_reservations_created": False,
        "blockers": blockers,
        "note": (
            "Fail-closed prototype gate. Mappings are user-supplied, stock is not reserved, "
            "and this endpoint cannot authorize maintenance or operational scheduling."
        ),
    }


def _ensure_parts_decision_audit_table(connection):
    connection.execute(
        """CREATE TABLE IF NOT EXISTS parts_decision_audit (
            audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
            context TEXT NOT NULL,
            run_id TEXT,
            assessment_status TEXT NOT NULL,
            gate_status TEXT NOT NULL,
            blockers_json TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )"""
    )


def record_parts_decision_audit(context, run_id, assessment, gate):
    """Persist the exact parts assessment and fail-closed gate used by a planning run."""
    safe_context = str(context or "UNKNOWN_CONTEXT")[:80]
    safe_run_id = str(run_id or "").strip() or None
    assessment = assessment if isinstance(assessment, dict) else {}
    gate = gate if isinstance(gate, dict) else build_parts_clearance_gate(assessment)
    created_at = _now()
    snapshot = {"assessment": assessment, "gate": gate}
    with closing(get_connection()) as connection:
        _ensure_parts_decision_audit_table(connection)
        cursor = connection.execute(
            """INSERT INTO parts_decision_audit
               (context, run_id, assessment_status, gate_status, blockers_json, snapshot_json, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                safe_context,
                safe_run_id,
                str(assessment.get("assessment_status", "UNKNOWN")),
                str(gate.get("gate_status", "BLOCKED_REVIEW_REQUIRED")),
                json.dumps(gate.get("blockers", []), sort_keys=True),
                json.dumps(snapshot, sort_keys=True, default=str),
                created_at,
            ),
        )
        connection.commit()
        audit_id = int(cursor.lastrowid)
    return {"audit_id": audit_id, "context": safe_context, "run_id": safe_run_id,
            "gate_status": gate.get("gate_status"), "created_at": created_at}


def get_parts_decision_audit(limit=50):
    """Return recent Scheduler/Digital Twin parts-gate decision snapshots."""
    limit = max(1, min(int(limit), 500))
    with closing(get_connection()) as connection:
        _ensure_parts_decision_audit_table(connection)
        rows = connection.execute(
            """SELECT audit_id, context, run_id, assessment_status, gate_status,
                      blockers_json, snapshot_json, created_at
               FROM parts_decision_audit ORDER BY audit_id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        try:
            item["blockers"] = json.loads(item.pop("blockers_json", "[]"))
        except (TypeError, json.JSONDecodeError):
            item["blockers"] = []
        try:
            item["snapshot"] = json.loads(item.pop("snapshot_json", "{}"))
        except (TypeError, json.JSONDecodeError):
            item["snapshot"] = {}
        items.append(item)
    return {"count": len(items), "items": items,
            "note": "Audit records preserve the assessment snapshot and a fail-closed gate; they do not represent operational approval."}
