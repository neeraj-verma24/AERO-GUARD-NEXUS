#!/usr/bin/env python3
"""AERO-GUARD NEXUS Step 72 end-to-end prototype acceptance checks.

Default behavior is GET-only. Pass --run-rag-query-tests to POST two known
RAG test queries; each POST intentionally creates a local RAG audit record.
Reports are written under reports/acceptance/. No inventory/work orders are
modified by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8000"


def request_json(base_url: str, path: str, method: str = "GET", body: dict[str, Any] | None = None,
                 timeout: int = 20) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = Request(base_url.rstrip("/") + path, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {"_raw": raw[:1000]}
            return response.status, payload
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"_raw": raw[:1000]}
        return exc.code, payload
    except (URLError, TimeoutError, OSError) as exc:
        return 0, {"_error": str(exc)}


def get_dict(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def items_of(payload: Any) -> list[Any]:
    obj = get_dict(payload)
    items = obj.get("items")
    return items if isinstance(items, list) else []


def record(results: list[dict[str, str]], label: str, ok: bool | None, detail: str) -> None:
    status = "PASS" if ok is True else "FAIL" if ok is False else "WARN"
    results.append({"status": status, "check": label, "detail": str(detail)})
    print(f"[{status}] {label} — {detail}")


def check_http(results: list[dict[str, str]], base: str, path: str, name: str) -> Any:
    code, payload = request_json(base, path)
    record(results, f"GET {path}", code == 200, f"HTTP {code}" + (f"; {payload.get('_error')}" if isinstance(payload, dict) and payload.get("_error") else ""))
    return payload if code == 200 else None


def write_report(results: list[dict[str, str]], rag_tests_requested: bool, base: str) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "reports" / "acceptance"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"step72_acceptance_{stamp}.json"
    md_path = out_dir / f"step72_acceptance_{stamp}.md"
    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("PASS", "WARN", "FAIL")}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "step": 72,
        "base_url": base,
        "rag_query_tests_requested": rag_tests_requested,
        "summary": counts,
        "overall": "PASS_WITH_WARNINGS" if counts["FAIL"] == 0 and counts["WARN"] else "PASS" if counts["FAIL"] == 0 else "FAIL",
        "production_ready": False,
        "note": "Prototype integration evidence only; not aircraft maintenance approval or safety certification.",
        "checks": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# AERO-GUARD NEXUS — Step 72 Acceptance Report", "",
        f"- Generated (UTC): `{payload['generated_at']}`",
        f"- Base URL: `{base}`",
        f"- RAG POST tests requested: `{rag_tests_requested}`",
        f"- Overall: **{payload['overall']}**",
        f"- Production ready: **No**", "",
        "## Summary", "", f"- PASS: {counts['PASS']}", f"- WARN: {counts['WARN']}", f"- FAIL: {counts['FAIL']}", "",
        "## Checks", "", "| Status | Check | Detail |", "|---|---|---|",
    ]
    for item in results:
        detail = item['detail'].replace('|', '\\|').replace('\n', ' ')
        check = item['check'].replace('|', '\\|')
        lines.append(f"| {item['status']} | {check} | {detail} |")
    lines += ["", "Prototype evidence only. RUL uncertainty is not calibrated; C-MAPSS is simulated benchmark data; physics rules are not approved; parts mappings may be demo-only; no operational clearance is granted.", ""]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE, help=f"Local API base URL (default: {DEFAULT_BASE})")
    parser.add_argument("--run-rag-query-tests", action="store_true", help="POST two known RAG queries; each one creates a RAG audit record")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    results: list[dict[str, str]] = []

    print("AERO-GUARD NEXUS — STEP 72 END-TO-END ACCEPTANCE")
    print(f"Base URL: {base}")
    print("Mode: GET-only unless --run-rag-query-tests is supplied.")
    if args.run_rag_query_tests:
        print("NOTICE: Two POST queries will be submitted and recorded in the local RAG audit log.")
    print()

    paths = [
        ("/health", "health"),
        ("/api/v1/system", "system"),
        ("/api/v1/summary", "summary"),
        ("/api/v1/assets", "assets"),
        ("/api/v1/work-orders", "work orders"),
        ("/api/v1/spares", "spares"),
        ("/api/v1/ai/model-metrics", "model metrics"),
        ("/api/v1/ai/physics-check/rules-status", "physics rule status"),
        ("/api/v1/ai/evidence-gate/latest", "latest evidence gate"),
        ("/api/v1/rag/status", "RAG status"),
        ("/api/v1/parts/assessment", "parts assessment"),
        ("/api/v1/scheduling/latest", "latest schedule"),
        ("/api/v1/digital-twin/latest", "latest Digital Twin"),
        ("/api/v1/parts/decision-audit?limit=50", "parts decision audit"),
        ("/api/v1/rag/audit-log?limit=20", "RAG audit log"),
        ("/api/v1/ai/rul-predictions", "RUL predictions"),
    ]
    responses: dict[str, Any] = {}
    for path, name in paths:
        responses[name] = check_http(results, base, path, name)

    # Frontend artifact and integration checks.
    app_file = ROOT / "frontend" / "src" / "App.jsx"
    css_file = ROOT / "frontend" / "src" / "App.css"
    dist_index = ROOT / "frontend" / "dist" / "index.html"
    record(results, "Frontend React source exists", app_file.is_file(), str(app_file.relative_to(ROOT)))
    record(results, "Frontend CSS exists", css_file.is_file(), str(css_file.relative_to(ROOT)))
    record(results, "Production build entry exists", dist_index.is_file(), str(dist_index.relative_to(ROOT)))
    if app_file.is_file():
        app_text = app_file.read_text(encoding="utf-8", errors="replace")
        record(results, "Mechanic Co-Pilot is wired to local RAG query", "/api/v1/rag/query" in app_text, "RAG query endpoint reference found" if "/api/v1/rag/query" in app_text else "Endpoint reference not found in App.jsx")
        record(results, "System Assurance page is present", "System Assurance" in app_text, "Navigation/page label found" if "System Assurance" in app_text else "Expected page label not found")

    # RUL bounds and uncertainty disclosure.
    rul = get_dict(responses.get("RUL predictions"))
    preds = rul.get("predictions") if isinstance(rul.get("predictions"), list) else []
    record(results, "RUL prediction list is non-empty", len(preds) > 0, f"{len(preds)} predictions returned")
    if preds:
        fields = ["predicted_rul_cycles", "baseline_rul_cycles", "member_prediction_p10_cycles", "member_prediction_p90_cycles"]
        for field in fields:
            vals: list[float] = []
            invalid: list[str] = []
            for ix, item in enumerate(preds):
                try:
                    value = float(item[field])
                    vals.append(value)
                    if not (0 <= value <= 125):
                        invalid.append(f"row {ix + 1}={value}")
                except (KeyError, TypeError, ValueError):
                    invalid.append(f"row {ix + 1}=missing/invalid")
            ok = len(vals) == len(preds) and not invalid
            detail = f"min={min(vals):.2f}, max={max(vals):.2f}, out-of-range={len(invalid)}" if vals else "no numeric values"
            record(results, f"RUL field {field} stays within [0,125]", ok, detail)
        uncertainty = str(rul.get("uncertainty_method", ""))
        record(results, "RUL uncertainty disclaimer remains explicit", "not a calibrated confidence interval" in uncertainty.lower(), uncertainty or "uncertainty_method missing")
    else:
        record(results, "RUL output cap range checks", None, "Skipped because RUL API response is unavailable or empty")

    # Physics rules: no real values are inferred from quantiles.
    physics = get_dict(responses.get("physics rule status"))
    rules_info = get_dict(physics.get("engineering_rules"))
    status = str(rules_info.get("status", "UNKNOWN"))
    count = rules_info.get("rule_count")
    record(results, "Physics rule status is explicit", status != "UNKNOWN", f"status={status}; rule_count={count}")
    record(results, "Physics rule source uses simulated benchmark context", "simulated" in str(physics.get("data_type", "")).lower(), str(physics.get("data_type", "missing data_type")))
    if status == "NO_APPROVED_PHYSICAL_RULES_CONFIGURED":
        record(results, "No unapproved engineering rules implicitly configured", count == 0, f"configured rule_count={count}")
    else:
        record(results, "Engineering-rule registry needs qualified review", None, "The endpoint reports configured rules; this script does not validate aircraft applicability")

    # Parts assessment should not grant clearance or claim stock reservations.
    parts = get_dict(responses.get("parts assessment"))
    if parts:
        record(results, "Parts assessment does not grant scheduling clearance", parts.get("scheduling_clearance") is False, f"scheduling_clearance={parts.get('scheduling_clearance')}")
        record(results, "Parts assessment does not claim stock reservation", parts.get("stock_reservations_created") is False, f"stock_reservations_created={parts.get('stock_reservations_created')}")
        part_rows = parts.get("parts") if isinstance(parts.get("parts"), list) else []
        by_code = {str(item.get("spare_code")): item for item in part_rows if isinstance(item, dict)}
        if "SP-118" in by_code:
            p = by_code["SP-118"]
            record(results, "SP-118 sufficient-stock arithmetic", p.get("shortage_quantity") == 0 and p.get("stock_status") == "STOCK_SUFFICIENT_NOT_RESERVED", f"stock={p.get('current_stock')}; demand={p.get('mapped_open_demand')}; shortage={p.get('shortage_quantity')}; status={p.get('stock_status')}")
        else:
            record(results, "SP-118 sufficient-stock arithmetic", None, "SP-118 mapping not present in current assessment")
        if "SP-306" in by_code:
            p = by_code["SP-306"]
            record(results, "SP-306 aggregate shortage is visible", p.get("shortage_quantity", 0) >= 1 and p.get("stock_status") == "AGGREGATE_SHORTAGE", f"stock={p.get('current_stock')}; demand={p.get('mapped_open_demand')}; shortage={p.get('shortage_quantity')}; status={p.get('stock_status')}")
        else:
            record(results, "SP-306 aggregate shortage is visible", None, "SP-306 mapping not present in current assessment")
    else:
        record(results, "Parts clearance remains fail-closed", None, "Parts assessment response unavailable")

    # Scheduler.
    schedule = get_dict(responses.get("latest schedule"))
    if schedule:
        record(results, "Latest schedule remains proposal-only", schedule.get("schedule_status") == "PROPOSED_PENDING_PARTS_VERIFICATION", str(schedule.get("schedule_status")))
        record(results, "Latest schedule operational clearance is false", schedule.get("operational_clearance") is False, f"operational_clearance={schedule.get('operational_clearance')}")
        gate = get_dict(schedule.get("parts_clearance_gate"))
        record(results, "Scheduler parts gate is blocked", gate.get("gate_status") == "BLOCKED_REVIEW_REQUIRED", f"gate_status={gate.get('gate_status', 'missing')}")
    else:
        record(results, "Scheduler fail-closed status", None, "Latest schedule unavailable")

    # Digital Twin structure can differ by version; verify no clearance and gates if exposed.
    twin = get_dict(responses.get("latest Digital Twin"))
    if twin:
        record(results, "Latest Digital Twin snapshot is non-empty", bool(twin), f"run_id={twin.get('run_id', twin.get('comparison_id', 'present'))}")
        scenarios = twin.get("scenarios") if isinstance(twin.get("scenarios"), list) else []
        if scenarios:
            scenario_gates = [get_dict(s).get("parts_clearance_gate") for s in scenarios if isinstance(s, dict)]
            all_gates = len(scenario_gates) == len(scenarios) and all(get_dict(g).get("gate_status") == "BLOCKED_REVIEW_REQUIRED" for g in scenario_gates)
            record(results, "Digital Twin scenarios carry blocked parts gates", all_gates, f"scenarios={len(scenarios)}")
        else:
            record(results, "Digital Twin scenario gate structure", None, "Scenario list is not exposed directly by latest endpoint; check comparison response/audit record")
    else:
        record(results, "Digital Twin snapshot is available", None, "Latest comparison endpoint unavailable")

    # Audit endpoints.
    parts_audit = get_dict(responses.get("parts decision audit"))
    pa_items = items_of(parts_audit)
    record(results, "Parts decision audit is readable", isinstance(parts_audit.get("count"), int) and parts_audit.get("count") >= len(pa_items), f"count={parts_audit.get('count')}; items={len(pa_items)}")
    rag_audit = get_dict(responses.get("RAG audit log"))
    ra_items = items_of(rag_audit)
    record(results, "RAG audit log is readable", isinstance(rag_audit.get("count"), int) and rag_audit.get("count") >= len(ra_items), f"count={rag_audit.get('count')}; items={len(ra_items)}")

    # Optional POST-backed RAG tests: actual query results and subsequent audit presence.
    if args.run_rag_query_tests:
        q_abstain = "Find the sample maintenance notes and show the most relevant passages with their source citations and approval status."
        q_match = "What does the hydraulic inspection checklist require checking, and what should be recorded for review?"
        code_a, abstain = request_json(base, "/api/v1/rag/query", "POST", {"query": q_abstain, "top_k": 5})
        decision_a = get_dict(abstain).get("decision")
        record(results, "RAG query abstains when no supported passage meets threshold", code_a == 200 and decision_a == "INSUFFICIENT_EVIDENCE", f"HTTP {code_a}; decision={decision_a}; audit_id={get_dict(abstain).get('audit_id', 'n/a')}")
        code_b, matched = request_json(base, "/api/v1/rag/query", "POST", {"query": q_match, "top_k": 5})
        matched_obj = get_dict(matched)
        decision_b = matched_obj.get("decision")
        sources = matched_obj.get("sources") if isinstance(matched_obj.get("sources"), list) else []
        record(results, "RAG retrieves matching unverified sample passages", code_b == 200 and decision_b == "UNVERIFIED_SOURCE_MATCH" and int(matched_obj.get("result_count", 0) or 0) > 0, f"HTTP {code_b}; decision={decision_b}; result_count={matched_obj.get('result_count')}; audit_id={matched_obj.get('audit_id', 'n/a')}")
        if sources:
            # The live RAG API nests approval_status inside each source's provenance object.
            provenance_items = [get_dict(get_dict(source).get("provenance")) for source in sources]
            approval_statuses = sorted({str(item.get("approval_status", "MISSING")) for item in provenance_items})
            explicit_unverified = all(item.get("approval_status") == "UNVERIFIED" for item in provenance_items)
            not_authoritative = all(get_dict(source).get("authoritative_eligible") is False for source in sources)
            passed = explicit_unverified and not_authoritative
            record(
                results,
                "Unverified test passages expose UNVERIFIED provenance and are not authoritative-eligible",
                passed,
                f"sources={len(sources)}; approval_statuses={approval_statuses}; all_authoritative_eligible_false={not_authoritative}",
            )
        else:
            record(results, "RAG source authority metadata", None, "Matched response did not expose a sources list; inspect response schema")
        # Verify the two newly-created audit IDs are visible after POST.
        code_log, post_log = request_json(base, "/api/v1/rag/audit-log?limit=20")
        logged_ids = {str(get_dict(item).get("audit_id")) for item in items_of(post_log)}
        expected_ids = {str(get_dict(abstain).get("audit_id")), str(matched_obj.get("audit_id"))}
        expected_ids.discard("None")
        record(results, "Both RAG acceptance queries appear in audit log", code_log == 200 and bool(expected_ids) and expected_ids.issubset(logged_ids), f"HTTP {code_log}; expected_ids={sorted(expected_ids)}; recent_records={len(logged_ids)}")
    else:
        record(results, "RAG active-query acceptance tests", None, "Skipped by default; run with --run-rag-query-tests to POST two queries and create two RAG audit records")

    json_path, md_path = write_report(results, args.run_rag_query_tests, base)
    counts = {s: sum(1 for r in results if r["status"] == s) for s in ("PASS", "WARN", "FAIL")}
    print("\nSUMMARY:")
    print(f"  PASS: {counts['PASS']}")
    print(f"  WARN: {counts['WARN']}")
    print(f"  FAIL: {counts['FAIL']}")
    print(f"  Overall: {'FAIL' if counts['FAIL'] else 'PASS_WITH_WARNINGS' if counts['WARN'] else 'PASS'}")
    print(f"  JSON report: {json_path.relative_to(ROOT)}")
    print(f"  Markdown report: {md_path.relative_to(ROOT)}")
    print("  Production readiness: NO — prototype limitations remain.")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
