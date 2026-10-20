# AERO-GUARD NEXUS — Step 73 Demo Readiness

- Generated (UTC): `2026-10-20T11:43:28+00:00`
- Base URL: `http://127.0.0.1:8000`
- Overall: **DEMO_READY_WITH_LIMITATIONS**
- Production readiness: **NO — prototype limitations remain**
- Mode: read-only GET checks; no RAG POST queries were submitted.

## Summary

- PASS: 37
- WARN: 2
- FAIL: 0

## Checks

| Status | Check | Detail |
|---|---|---|
| PASS | GET /health | HTTP 200 |
| PASS | GET /api/v1/system | HTTP 200 |
| PASS | GET /api/v1/summary | HTTP 200 |
| PASS | GET /api/v1/assets | HTTP 200 |
| PASS | GET /api/v1/work-orders | HTTP 200 |
| PASS | GET /api/v1/spares | HTTP 200 |
| PASS | GET /api/v1/ai/model-metrics | HTTP 200 |
| PASS | GET /api/v1/ai/physics-check/rules-status | HTTP 200 |
| PASS | GET /api/v1/ai/evidence-gate/latest | HTTP 200 |
| PASS | GET /api/v1/rag/status | HTTP 200 |
| PASS | GET /api/v1/parts/assessment | HTTP 200 |
| PASS | GET /api/v1/scheduling/latest | HTTP 200 |
| PASS | GET /api/v1/digital-twin/latest | HTTP 200 |
| PASS | GET /api/v1/parts/decision-audit?limit=50 | HTTP 200 |
| PASS | GET /api/v1/rag/audit-log?limit=20 | HTTP 200 |
| PASS | GET /api/v1/ai/rul-predictions | HTTP 200 |
| PASS | File frontend/src/App.jsx | present |
| PASS | File frontend/src/App.css | present |
| PASS | File frontend/dist/index.html | present |
| PASS | Mechanic Co-Pilot references local RAG query endpoint | reference found |
| PASS | System Assurance page is present | reference found |
| PASS | RUL predictions non-empty | count=100 |
| PASS | RUL predicted_rul_cycles within [0,125] | valid=100; malformed=0; min=7.34; max=124.78; out_of_range=0 |
| PASS | RUL baseline_rul_cycles within [0,125] | valid=100; malformed=0; min=5.61; max=125.00; out_of_range=0 |
| PASS | RUL member_prediction_p10_cycles within [0,125] | valid=100; malformed=0; min=2.66; max=124.34; out_of_range=0 |
| PASS | RUL member_prediction_p90_cycles within [0,125] | valid=100; malformed=0; min=7.77; max=125.00; out_of_range=0 |
| PASS | RUL uncertainty limitation shown | Standard deviation across ensemble member predictions; not a calibrated confidence interval. |
| WARN | Engineering rule registry | zero approved rules configured; expected blocker, not a failed API check |
| PASS | Parts assessment does not grant scheduling clearance | observed=False |
| PASS | Parts assessment does not claim stock reservation | observed=False |
| WARN | Unmapped work orders are visible | count=1 |
| PASS | Latest schedule remains proposal-only | PROPOSED_PENDING_PARTS_VERIFICATION |
| PASS | Schedule operational clearance is false | observed=False |
| PASS | Schedule parts gate is blocked | gate_status=BLOCKED_REVIEW_REQUIRED |
| PASS | Latest Digital Twin snapshot | run_id=ed937f82-d558-40fb-ac21-9a7c3b6562a9 |
| PASS | Digital Twin gate remains blocked | gate_status=BLOCKED_REVIEW_REQUIRED |
| PASS | Digital Twin scenario gate fields | scenarios=4; all_blocked=True |
| PASS | RAG audit log is readable | count=18; items=18 |
| PASS | Parts decision audit is readable | count=4; items=4 |

## Known limitations

- NASA C-MAPSS FD001 is simulated benchmark data, not real aircraft telemetry.
- RUL ensemble spread is not a calibrated confidence interval.
- No approved engineering physical rules are configured.
- Parts demo mappings do not establish approved BOMs or stock reservation.
- Scheduler and Digital Twin remain proposal/scenario tools with clearance blocked.
- RAG approval/provenance metadata is user-supplied and not independently authenticated.
