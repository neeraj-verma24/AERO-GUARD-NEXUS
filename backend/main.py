import json
import uuid
from pathlib import Path
from contextlib import closing
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from backend.evidence_engine import analyze_latest_sensor_screens

from backend.ai_engine import (
    ENSEMBLE_PATH,
    HIGH_DISAGREEMENT_REVIEW_CYCLES,
    LOW_RUL_REVIEW_CYCLES,
    MODEL_PATH,
    get_latest_rul_predictions,
)
from backend.database import (
    get_connection,
    init_db,
    record_evidence_audit_events,
)


# Ensure the local database and all tables exist.
init_db()

from backend.scheduler import (
    generate_and_save_schedule,
    get_latest_saved_schedule,
    get_schedule_run_history,
)
from backend.parts_engine import (
    assess_mapped_inventory,
    get_work_order_part_mapping_audit,
    get_parts_decision_audit,
    list_work_order_part_mappings,
    save_work_order_part_mapping,
)

from backend.digital_twin import (
    compare_scenarios,
    get_latest_scenario_comparison,
    get_scenario_comparison_history,
)
from backend.rag_engine import (
    get_rag_status,
    index_local_documents,
    query_local_documents,
    get_rag_audit_log,
    get_rag_metadata_audit,
    update_document_metadata,
)


from backend.physics_engine import (
    analyze_latest_physics_checks,
    get_physics_rule_status,
)
from fastapi.responses import JSONResponse
import os
from backend.config import get_bool_setting


app = FastAPI(
    title="AERO-GUARD NEXUS",
    description="Predictive Maintenance and Fleet Availability Platform",
    version="0.3.0",
)


@app.on_event("startup")
def _step76_index_packaged_rag_documents():
    """Index only shipped data/manuals files when explicitly enabled by deployment config."""
    try:
        enabled = get_bool_setting("AERO_GUARD_RAG_INDEX_ON_STARTUP", False)
        if not enabled:
            return
        result = index_local_documents()
        print(
            "[AERO-GUARD RAG] startup indexing complete: "
            f"documents_seen={result.get('documents_seen', 0)} "
            f"document_count={result.get('document_count', 0)} "
            f"chunk_count={result.get('chunk_count', 0)} "
            f"errors={len(result.get('errors', []))}"
        )
    except Exception as exc:
        # Keep the API available, but make the RAG indexing failure visible in service logs.
        print(
            "[AERO-GUARD RAG] startup indexing failed: "
            f"{type(exc).__name__}: {str(exc)[:300]}"
        )

# Public demo CORS: replace '*' with the exact frontend origin after Vercel deploy.
_step75_cors_origins = [x.strip() for x in os.getenv('AERO_GUARD_CORS_ORIGINS', '*').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_step75_cors_origins, allow_credentials=False, allow_methods=['GET', 'POST', 'OPTIONS'], allow_headers=['Content-Type', 'Accept'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


SOURCE_DATASET = "NASA C-MAPSS FD001"

class EngineAssetMappingRequest(BaseModel):
    engine_unit: int = Field(ge=1, le=100)
    asset_id: str = Field(min_length=3, max_length=64)
    mapping_type: str = "DEMO_ONLY"
    mapping_note: str = Field(min_length=5, max_length=1000)
    mapping_reference: str | None = None


class WorkOrderPartMappingRequest(BaseModel):
    work_order_id: str = Field(min_length=1, max_length=64)
    spare_code: str = Field(min_length=1, max_length=64)
    quantity_required: int = Field(ge=1, le=100000)
    mapping_type: str = "DEMO_ONLY"
    mapping_note: str = Field(min_length=5, max_length=1000)
    mapping_reference: str | None = Field(default=None, max_length=1000)


class ScenarioComparisonRequest(BaseModel):
    """Optional what-if scenarios; an empty body uses the default four."""
    scenarios: list[dict] | None = None


class RAGQueryRequest(BaseModel):
    """Query the local document index; responses are extractive, not generated advice."""
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class RAGDocumentMetadataRequest(BaseModel):
    """User-provided provenance metadata for an already indexed local document."""
    relative_path: str = Field(..., min_length=1, max_length=255)
    title: str | None = Field(default=None, max_length=250)
    source_organization: str | None = Field(default=None, max_length=200)
    document_type: str | None = Field(default=None, max_length=80)
    revision: str | None = Field(default=None, max_length=80)
    effective_date: str | None = Field(default=None, max_length=10)
    expiry_date: str | None = Field(default=None, max_length=10)
    approval_status: str | None = None
    approval_reference: str | None = Field(default=None, max_length=300)
    reviewed_by: str | None = Field(default=None, max_length=200)
    provenance_notes: str | None = Field(default=None, max_length=1000)



def read_all(query, params=()):
    """Return database rows as JSON-serializable dictionaries."""
    with closing(get_connection()) as connection:
        rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def read_one(query, params=()):
    """Return one database row, or None."""
    with closing(get_connection()) as connection:
        row = connection.execute(query, params).fetchone()
        return dict(row) if row is not None else None


def decode_audit_rows(rows):
    """Decode stored JSON fields into ordinary response objects."""
    decoded = []

    for row in rows:
        item = dict(row)

        try:
            item["review_reasons"] = json.loads(
                item.pop("review_reasons_json", "[]")
            )
        except (TypeError, json.JSONDecodeError):
            item["review_reasons_json"] = item.get(
                "review_reasons_json", "[]"
            )
            item["review_reasons"] = []

        try:
            item["evidence"] = json.loads(
                item.pop("evidence_json", "{}")
            )
        except (TypeError, json.JSONDecodeError):
            item["evidence_json"] = item.get("evidence_json", "{}")
            item["evidence"] = {}

        decoded.append(item)

    return decoded


def build_evidence_decision(prediction, sensor_screen):
    """Combine benchmark RUL, sensor screening and available mapped records."""
    unit = int(prediction["unit"])
    last_cycle = int(prediction["last_observed_cycle"])
    rul = float(prediction["predicted_rul_cycles"])
    disagreement = float(prediction.get("model_disagreement_cycles", 0.0))
    ensemble_size = int(prediction.get("ensemble_size", 0))

    reasons = list(prediction.get("review_reasons") or [])
    if rul <= LOW_RUL_REVIEW_CYCLES:
        reasons.append("LOW_PREDICTED_RUL")
    if disagreement >= HIGH_DISAGREEMENT_REVIEW_CYCLES:
        reasons.append("HIGH_MODEL_DISAGREEMENT")
    if sensor_screen and sensor_screen.get("status") == "SCREEN_POSITIVE":
        reasons.append("SENSOR_OUT_OF_BASELINE_RANGE")
    reasons = sorted(set(reasons))

    mapping = read_one(
        """
        SELECT m.engine_unit, m.asset_id, m.mapping_type, m.mapping_note,
               m.mapping_reference, m.created_at, m.updated_at,
               a.subsystem, a.health, a.status AS asset_status, a.is_demo
        FROM engine_asset_mappings AS m
        JOIN assets AS a ON a.id = m.asset_id
        WHERE m.engine_unit = ?
        """,
        (unit,),
    )

    work_orders = []
    maintenance_status = "NOT_MAPPED"
    if mapping:
        work_orders = read_all(
            """
            SELECT id, task, priority, status, created_at
            FROM work_orders
            WHERE asset_id = ?
            ORDER BY created_at DESC, id
            """,
            (mapping["asset_id"],),
        )
        maintenance_status = "MAPPED_DEMO_ONLY" if mapping["mapping_type"] == "DEMO_ONLY" else "USER_DECLARED_DOCUMENTED_MAPPING"

    spare_summary = read_one(
        """
        SELECT COUNT(*) AS sku_count,
               SUM(CASE WHEN stock < required THEN 1 ELSE 0 END) AS shortage_skus
        FROM spare_parts
        """
    ) or {"sku_count": 0, "shortage_skus": 0}

    # No current workflow has verified an engine-component-to-spare crosswalk.
    # Even a documented asset mapping therefore cannot make the full evidence
    # package complete or imply a maintenance authorization.
    mapping_documented = bool(
        mapping
        and mapping.get("mapping_type") == "DOCUMENTED"
        and mapping.get("mapping_reference")
    )
    evidence_completeness = "PARTIAL" if mapping_documented else "INCOMPLETE"

    decision = "REVIEW_REQUIRED" if reasons else "HOLD_INSUFFICIENT_EVIDENCE"

    sensor_evidence = sensor_screen or {
        "status": "SCREEN_UNAVAILABLE",
        "sensor_count_evaluated": 0,
        "flagged_sensor_count": 0,
        "flagged_sensors": [],
        "limitation": "Sensor screen could not be obtained for this engine.",
    }

    evidence = {
        "rul_prediction": {
            "status": "AVAILABLE",
            "value_cycles": round(rul, 2),
            "review_threshold_cycles": LOW_RUL_REVIEW_CYCLES,
        },
        "model_disagreement": {
            "status": "AVAILABLE",
            "value_cycles": round(disagreement, 2),
            "review_threshold_cycles": HIGH_DISAGREEMENT_REVIEW_CYCLES,
            "interpretation": (
                "Spread among ensemble member predictions; not a calibrated "
                "confidence interval."
            ),
        },
        "sensor_anomaly_check": sensor_evidence,
        "maintenance_history": {
            "status": maintenance_status,
            "asset_mapping": mapping,
            "work_order_count": len(work_orders),
            "work_orders": work_orders,
            "note": (
                "A mapping must be supported by a real documented crosswalk. "
                "DEMO_ONLY links are for UI/testing only."
            ),
        },
        "spare_availability": {
            "status": "NOT_CROSSWALKED",
            "global_inventory_summary": spare_summary,
            "note": (
                "Inventory counts are global. Compatibility between this engine, "
                "its component and individual spare parts is not mapped."
            ),
        },
        "data_scope": {
            "dataset": SOURCE_DATASET,
            "type": "simulated_engine_benchmark",
            "real_aircraft_validation": False,
        },
    }

    return {
        "engine_unit": unit,
        "engine_id": f"FD001-{unit:03d}",
        "last_observed_cycle": last_cycle,
        "predicted_rul_cycles": round(rul, 2),
        "model_disagreement_cycles": round(disagreement, 2),
        "ensemble_size": ensemble_size,
        "sensor_anomaly_screen_status": sensor_evidence.get("status", "UNKNOWN"),
        "anomalous_sensor_count": int(sensor_evidence.get("flagged_sensor_count", 0)),
        "evidence_status": decision,
        "decision": decision,
        "evidence_completeness": evidence_completeness,
        "review_reasons": reasons,
        "evidence": evidence,
    }


def make_audit_event(decision_item, run_id, evaluated_at):
    """Build one database-ready audit snapshot."""
    return {
        "run_id": run_id,
        "evaluated_at": evaluated_at,
        "engine_unit": decision_item["engine_unit"],
        "last_observed_cycle": decision_item["last_observed_cycle"],
        "predicted_rul_cycles": decision_item["predicted_rul_cycles"],
        "model_disagreement_cycles": (
            decision_item["model_disagreement_cycles"]
        ),
        "evidence_status": decision_item["evidence_status"],
        "decision": decision_item["decision"],
        "evidence_completeness": (
            decision_item["evidence_completeness"]
        ),
        "review_reasons_json": json.dumps(
            decision_item["review_reasons"]
        ),
        "evidence_json": json.dumps(
            decision_item["evidence"]
        ),
        "source_dataset": SOURCE_DATASET,
        "ensemble_size": decision_item["ensemble_size"],
        "created_at": evaluated_at,
    }


@app.get("/")
def home():
    return {
        "project": "AERO-GUARD NEXUS",
        "message": "Backend is running",
        "mode": "offline-first",
        "version": "0.3.0",
    }


@app.get("/health")
def health_check():
    database = read_one("SELECT 1 AS connected")

    return {
        "status": "healthy",
        "database": "connected" if database else "unavailable",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/system")
def system_status():
    return {
        "project": "AERO-GUARD NEXUS",
        "backend": "online",
        "database": "connected",
        "ai_engine": (
            "trained_model_available"
            if MODEL_PATH.is_file()
            else "model_missing"
        ),
        "ensemble_available": ENSEMBLE_PATH.is_file(),
        "deployment_mode": "local-first",
        "offline_test_verified": False,
    }


@app.get("/api/v1/assets")
def list_assets():
    items = read_all(
        """
        SELECT id, subsystem, health, status, tone,
               operating_window, is_demo, updated_at
        FROM assets
        ORDER BY id
        """
    )

    return {
        "count": len(items),
        "data_source": "synthetic_demo",
        "items": items,
    }


@app.get("/api/v1/work-orders")
def list_work_orders():
    items = read_all(
        """
        SELECT w.id, w.asset_id, w.task, w.priority,
               w.status, w.created_at
        FROM work_orders AS w
        ORDER BY
            CASE w.priority
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END,
            w.id
        """
    )

    return {
        "count": len(items),
        "data_source": "synthetic_demo",
        "items": items,
    }


@app.get("/api/v1/spares")
def list_spares():
    items = read_all(
        """
        SELECT code, name, stock, required, updated_at
        FROM spare_parts
        ORDER BY code
        """
    )

    for item in items:
        if item["stock"] == 0:
            item["status"] = "Out of stock"
        elif item["stock"] < item["required"] * 2:
            item["status"] = "Limited"
        else:
            item["status"] = "Available"

    return {
        "count": len(items),
        "data_source": "synthetic_demo",
        "items": items,
    }


@app.get("/api/v1/summary")
def dashboard_summary():
    asset_stats = read_one(
        """
        SELECT
            COUNT(*) AS total_assets,
            COALESCE(ROUND(AVG(health), 1), 0) AS average_health,
            SUM(CASE WHEN status = 'Healthy' THEN 1 ELSE 0 END)
                AS healthy_assets,
            SUM(CASE WHEN status != 'Healthy' THEN 1 ELSE 0 END)
                AS assets_needing_review
        FROM assets
        """
    )

    order_stats = read_one(
        """
        SELECT
            COUNT(*) AS total_work_orders,
            SUM(
                CASE
                    WHEN status NOT IN ('Completed', 'Closed')
                    THEN 1 ELSE 0
                END
            ) AS open_work_orders
        FROM work_orders
        """
    )

    spare_stats = read_one(
        """
        SELECT
            COUNT(*) AS spare_skus,
            SUM(CASE WHEN stock < required THEN 1 ELSE 0 END)
                AS shortage_skus
        FROM spare_parts
        """
    )

    return {
        "data_source": "synthetic_demo",
        "assets": asset_stats,
        "work_orders": order_stats,
        "spares": spare_stats,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/ai/rul-predictions")
def ai_rul_predictions():
    """Return ensemble RUL predictions for NASA C-MAPSS FD001."""

    try:
        predictions = get_latest_rul_predictions()

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model or NASA dataset file is unavailable.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RUL inference failed: {type(exc).__name__}",
        ) from exc

    ensemble_size = (
        predictions[0].get("ensemble_size", 0)
        if predictions
        else 0
    )

    review_count = sum(
        item.get("evidence_status") == "REVIEW"
        for item in predictions
    )

    provisional_count = sum(
        item.get("evidence_status") == "PROVISIONAL"
        for item in predictions
    )

    return {
        "model": "Bootstrap HistGradientBoostingRegressor ensemble",
        "ensemble_size": ensemble_size,
        "dataset": SOURCE_DATASET,
        "data_type": "simulated_engine_benchmark",
        "unit_of_measure": "cycles",
        "count": len(predictions),
        "review_count": review_count,
        "provisional_count": provisional_count,
        "uncertainty_method": (
            "Standard deviation across ensemble member predictions; "
            "not a calibrated confidence interval."
        ),
        "predictions": predictions,
        "note": (
            "Experimental benchmark predictions only. "
            "Prototype review thresholds are not validated aircraft "
            "maintenance or operational-clearance decisions."
        ),
    }


@app.post("/api/v1/ai/evidence-gate/run")
def run_evidence_gate():
    """
    Evaluate the latest benchmark predictions and append an audit run.

    GET endpoints do not write records. Use this POST endpoint when
    you intentionally want to create a new auditable evaluation run.
    """
    try:
        predictions = get_latest_rul_predictions()

        run_id = str(uuid.uuid4())
        evaluated_at = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")

        sensor_screens = analyze_latest_sensor_screens()
        decisions = [
            build_evidence_decision(
                item,
                sensor_screens.get(int(item["unit"])),
            )
            for item in predictions
        ]

        events = [
            make_audit_event(item, run_id, evaluated_at)
            for item in decisions
        ]

        record_evidence_audit_events(events)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model or NASA dataset file is unavailable.",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Evidence-gate run failed: {type(exc).__name__}",
        ) from exc

    review_required_count = sum(
        item["decision"] == "REVIEW_REQUIRED"
        for item in decisions
    )

    evidence_incomplete_count = sum(
        item["evidence_completeness"] != "COMPLETE"
        for item in decisions
    )
    sensor_screen_positive_count = sum(
        item.get("sensor_anomaly_screen_status") == "SCREEN_POSITIVE"
        for item in decisions
    )

    return {
        "run_id": run_id,
        "evaluated_at": evaluated_at,
        "dataset": SOURCE_DATASET,
        "data_type": "simulated_engine_benchmark",
        "count": len(decisions),
        "review_required_count": review_required_count,
        "evidence_incomplete_count": evidence_incomplete_count,
        "sensor_screen_positive_count": sensor_screen_positive_count,
        "audit_records_saved": len(events),
        "decisions": decisions,
        "note": (
            "Evidence is incomplete because independent sensor-anomaly "
            "evaluation and maintenance/spare mappings are not integrated. "
            "This output is not an aircraft safety approval."
        ),
    }


@app.get("/api/v1/ai/evidence-gate/latest")
def latest_evidence_gate_run():
    """Return the most recently saved evidence-gate run."""
    run = read_one(
        """
        SELECT
            run_id,
            evaluated_at,
            COUNT(*) AS count,
            SUM(
                CASE WHEN decision = 'REVIEW_REQUIRED'
                THEN 1 ELSE 0 END
            ) AS review_required_count,
            SUM(
                CASE WHEN evidence_completeness != 'COMPLETE'
                THEN 1 ELSE 0 END
            ) AS evidence_incomplete_count,
            MAX(id) AS last_event_id
        FROM evidence_audit_events
        GROUP BY run_id, evaluated_at
        ORDER BY last_event_id DESC
        LIMIT 1
        """
    )

    if run is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No evidence-gate runs found. "
                "POST to /api/v1/ai/evidence-gate/run first."
            ),
        )

    decisions = read_all(
        """
        SELECT *
        FROM evidence_audit_events
        WHERE run_id = ?
        ORDER BY engine_unit
        """,
        (run["run_id"],),
    )

    last_event_id = run.pop("last_event_id", None)
    decoded_decisions = decode_audit_rows(decisions)
    sensor_screen_positive_count = sum(
        (item.get("evidence") or {}).get("sensor_anomaly_check", {}).get("status")
        == "SCREEN_POSITIVE"
        for item in decoded_decisions
    )

    return {
        **run,
        "last_event_id": last_event_id,
        "sensor_screen_positive_count": sensor_screen_positive_count,
        "decisions": decoded_decisions,
        "note": (
            "This is an audit snapshot for the simulated benchmark. "
            "It does not validate aircraft maintenance decisions."
        ),
    }


@app.get("/api/v1/ai/audit-log")
def ai_audit_log(
    limit: int = Query(default=50, ge=1, le=500),
):
    """Return recent evidence-gate audit events."""
    rows = read_all(
        """
        SELECT *
        FROM evidence_audit_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    return {
        "count": len(rows),
        "limit": limit,
        "items": decode_audit_rows(rows),
    }


@app.get("/api/v1/ai/sensor-screen")
def ai_sensor_screen():
    """Return experimental sensor-distribution screening for all test engines."""
    try:
        results_by_unit = analyze_latest_sensor_screens()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="NASA training or test dataset is unavailable.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Sensor screening failed: {type(exc).__name__}",
        ) from exc

    results = list(results_by_unit.values())
    positives = sum(item["status"] == "SCREEN_POSITIVE" for item in results)
    return {
        "dataset": SOURCE_DATASET,
        "method": "Training-distribution quantile screen",
        "lower_quantile": 0.005,
        "upper_quantile": 0.995,
        "count": len(results),
        "screen_positive_count": positives,
        "results": results,
        "note": (
            "A statistical screen only. This is not a calibrated fault model "
            "or proof that an engine/sensor is safe or faulty."
        ),
    }


@app.get("/api/v1/ai/engine-mappings")
def list_engine_asset_mappings():
    """List explicit NASA test-engine to local demo-asset mappings."""
    items = read_all(
        """
        SELECT m.engine_unit, m.asset_id, m.mapping_type, m.mapping_note,
               m.mapping_reference, m.created_at, m.updated_at,
               a.subsystem, a.health, a.status AS asset_status, a.is_demo
        FROM engine_asset_mappings AS m
        JOIN assets AS a ON a.id = m.asset_id
        ORDER BY m.engine_unit
        """
    )
    return {
        "count": len(items),
        "items": items,
        "warning": (
            "NASA units and AG-* demo assets are unrelated unless a documented "
            "crosswalk exists. DEMO_ONLY mappings never count as verified evidence."
        ),
    }


@app.post("/api/v1/ai/engine-mappings")
def save_engine_asset_mapping(payload: EngineAssetMappingRequest):
    """Save a declared mapping; provenance is user-supplied and not verified."""
    mapping_type = payload.mapping_type.strip().upper()
    if mapping_type not in {"DEMO_ONLY", "DOCUMENTED"}:
        raise HTTPException(
            status_code=422,
            detail="mapping_type must be DEMO_ONLY or DOCUMENTED.",
        )

    note = payload.mapping_note.strip()
    reference = (payload.mapping_reference or "").strip() or None
    if len(note) < 5:
        raise HTTPException(
            status_code=422,
            detail="mapping_note must explain the mapping basis (at least 5 characters).",
        )
    if mapping_type == "DOCUMENTED" and not reference:
        raise HTTPException(
            status_code=422,
            detail="DOCUMENTED mappings require a reference or evidence location.",
        )

    asset = read_one("SELECT id FROM assets WHERE id = ?", (payload.asset_id.strip(),))
    if not asset:
        raise HTTPException(status_code=404, detail="Selected asset does not exist.")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with closing(get_connection()) as connection:
        connection.execute(
            """
            INSERT INTO engine_asset_mappings (
                engine_unit, asset_id, mapping_type, mapping_note,
                mapping_reference, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(engine_unit) DO UPDATE SET
                asset_id = excluded.asset_id,
                mapping_type = excluded.mapping_type,
                mapping_note = excluded.mapping_note,
                mapping_reference = excluded.mapping_reference,
                updated_at = excluded.updated_at
            """,
            (
                payload.engine_unit,
                payload.asset_id.strip(),
                mapping_type,
                note,
                reference,
                now,
                now,
            ),
        )
        connection.commit()

    saved = read_one(
        """
        SELECT m.engine_unit, m.asset_id, m.mapping_type, m.mapping_note,
               m.mapping_reference, m.created_at, m.updated_at,
               a.subsystem, a.health, a.status AS asset_status, a.is_demo
        FROM engine_asset_mappings AS m
        JOIN assets AS a ON a.id = m.asset_id
        WHERE m.engine_unit = ?
        """,
        (payload.engine_unit,),
    )

    return {
        "saved": True,
        "mapping": saved,
        "warning": (
            "The supplied mapping reference is not automatically verified. "
            "Use DEMO_ONLY for unrelated demo records. Even DOCUMENTED mappings "
            "do not establish aircraft safety or spare-part compatibility."
        ),
    }


@app.post("/api/v1/scheduling/plan")
def create_maintenance_schedule(
    technician_capacity: int = Query(default=2, ge=1, le=20),
    horizon_days: int = Query(default=7, ge=1, le=30),
):
    """Create and persist a proposed OR-Tools maintenance schedule."""
    try:
        return generate_and_save_schedule(
            technician_capacity=technician_capacity,
            horizon_days=horizon_days,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Scheduling failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/v1/scheduling/latest")
def latest_maintenance_schedule():
    """Return the latest persisted maintenance schedule."""
    try:
        result = get_latest_saved_schedule()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read latest schedule: {type(exc).__name__}: {exc}",
        ) from exc

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="No saved schedule exists. POST to /api/v1/scheduling/plan first.",
        )
    return result


@app.get("/api/v1/scheduling/runs")
def maintenance_schedule_runs(
    limit: int = Query(default=20, ge=1, le=200),
):
    """Return recent persisted schedule runs."""
    return {
        "count": len(get_schedule_run_history(limit=limit)),
        "items": get_schedule_run_history(limit=limit),
    }


@app.post("/api/v1/digital-twin/compare")
def run_digital_twin_comparison(payload: ScenarioComparisonRequest):
    """Run an auditable what-if comparison without changing work orders or schedule history."""
    try:
        return compare_scenarios(payload.scenarios)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Scenario comparison failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/v1/digital-twin/latest")
def latest_digital_twin_comparison():
    """Return the latest saved scenario comparison, if one exists."""
    try:
        result = get_latest_scenario_comparison()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read latest scenario comparison: {type(exc).__name__}: {exc}",
        ) from exc

    if result is None:
        return {"message": "No scenario comparison has been run yet."}
    return result


@app.get("/api/v1/digital-twin/history")
def digital_twin_comparison_history(
    limit: int = Query(default=20, ge=1, le=200),
):
    """Return recent scenario comparison runs."""
    try:
        items = get_scenario_comparison_history(limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read scenario comparison history: {type(exc).__name__}: {exc}",
        ) from exc
    return {"count": len(items), "items": items}


@app.get("/api/v1/rag/status")
def rag_status_api():
    """Report local document-index status."""
    try:
        return get_rag_status()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read RAG status: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/v1/rag/index")
def rag_index_api():
    """Index local .txt/.md files from data/manuals."""
    try:
        return index_local_documents()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Local document indexing failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/v1/rag/documents/metadata")
def rag_document_metadata_api(request: RAGDocumentMetadataRequest):
    """Update provenance fields on an indexed document. Metadata is not independently authenticated."""
    try:
        payload = request.model_dump(exclude={"relative_path"}, exclude_none=True, exclude_unset=True)
        return update_document_metadata(request.relative_path, payload)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not update document provenance: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/v1/rag/query")
def rag_query_api(request: RAGQueryRequest):
    """Retrieve matching source excerpts and abstain when evidence is insufficient."""
    try:
        return query_local_documents(request.query, request.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Local document query failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/v1/rag/audit-log")
def rag_audit_log_api(
    limit: int = Query(default=20, ge=1, le=200),
):
    """Return recent RAG query decisions and source references."""
    try:
        return get_rag_audit_log(limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read RAG audit log: {type(exc).__name__}: {exc}",
        ) from exc




@app.get("/api/v1/rag/metadata-audit")
def rag_metadata_audit_api(
    relative_path: str | None = Query(default=None, max_length=255),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Return document provenance metadata history; no reviewer identity is authenticated."""
    try:
        return get_rag_metadata_audit(relative_path=relative_path, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read RAG metadata audit: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/v1/ai/physics-check")
def ai_physics_check():
    """Run data-integrity sanity checks and explicitly configured local rules."""
    try:
        return analyze_latest_physics_checks()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="NASA C-MAPSS training or test data is unavailable.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Physics sanity check failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.get("/api/v1/ai/physics-check/rules-status")
def ai_physics_rules_status():
    """Report available sensor columns and whether documented limits are configured."""
    try:
        return get_physics_rule_status()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="NASA C-MAPSS training or test data is unavailable.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Physics rule status failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/v1/parts/work-order-mappings")
def create_work_order_part_mapping(payload: WorkOrderPartMappingRequest):
    """Save an explicit work-order/spare requirement; never imply authenticated compatibility."""
    try:
        return save_work_order_part_mapping(
            work_order_id=payload.work_order_id,
            spare_code=payload.spare_code,
            quantity_required=payload.quantity_required,
            mapping_type=payload.mapping_type,
            mapping_note=payload.mapping_note,
            mapping_reference=payload.mapping_reference,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/v1/parts/work-order-mappings")
def list_work_order_part_mappings_api(limit: int = Query(default=100, ge=1, le=500)):
    return list_work_order_part_mappings(limit)


@app.get("/api/v1/parts/assessment")
def assess_mapped_inventory_api():
    return assess_mapped_inventory()


@app.get("/api/v1/parts/mapping-audit")
def work_order_part_mapping_audit_api(limit: int = Query(default=50, ge=1, le=500)):
    return get_work_order_part_mapping_audit(limit)


@app.get("/api/v1/parts/decision-audit")
def parts_decision_audit_api(limit: int = Query(default=50, ge=1, le=500)):
    """List fail-closed parts-gate snapshots emitted by Scheduler and Digital Twin."""
    return get_parts_decision_audit(limit=limit)


# STEP61_PARTS_CLEARANCE_GATE
@app.get("/api/v1/ai/model-metrics")
def ai_model_metrics():
    """Return saved validation, test and baseline comparison metrics."""

    project_root = Path(__file__).resolve().parent.parent

    training_report_path = (
        project_root / "models" / "rul_metrics.json"
    )
    comparison_report_path = (
        project_root / "models" / "rul_model_comparison.json"
    )

    if not training_report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Training metrics not found. Train the RUL model first.",
        )

    if not comparison_report_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Comparison report not found. Run evaluate_rul.py first.",
        )

    try:
        training_metrics = json.loads(
            training_report_path.read_text(encoding="utf-8")
        )
        comparison_metrics = json.loads(
            comparison_report_path.read_text(encoding="utf-8")
        )

    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Saved model metrics could not be read.",
        ) from exc

    return {
        "model": "HistGradientBoostingRegressor baseline evaluation",
        "dataset": SOURCE_DATASET,
        "data_type": "simulated_engine_benchmark",
        "training_evaluation": training_metrics,
        "baseline_comparison": comparison_metrics,
        "note": (
            "These reports evaluate the saved baseline model; they do "
            "not represent the ensemble's independent validation or "
            "validation for real aircraft maintenance."
        ),
    }
# Optional public-demo guard. This is not authentication.
@app.middleware('http')
async def _step75_public_demo_write_guard(request, call_next):
    read_only = os.getenv('AERO_GUARD_DEMO_READ_ONLY', 'false').strip().lower() in {'1', 'true', 'yes', 'on'}
    method = request.method.upper()
    allow_rag_query = method == 'POST' and request.url.path == '/api/v1/rag/query'
    if read_only and method in {'POST', 'PUT', 'PATCH', 'DELETE'} and not allow_rag_query:
        return JSONResponse(status_code=403, content={'detail': 'Public demo is read-only; only RAG query submissions are enabled.'})
    return await call_next(request)
