"""Repeatable query-level evaluation for the AERO-GUARD NEXUS RAG prototype.

Run from project root: python scripts\\evaluate_rag.py
Requires the current indexed sample_maintenance_notes.md document.
This script queries the active project DB and appends normal RAG audit events.
"""
from __future__ import annotations
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.rag_engine import get_rag_status, query_local_documents, MIN_SCORE

CASES_PATH = ROOT / "tests" / "rag_eval_cases.json"
REPORT_DIR = ROOT / "reports" / "rag_eval"
EXPECTED_SOURCE = "sample_maintenance_notes.md"


def rate(num, den):
    return round(num / den, 4) if den else None


def main():
    if not CASES_PATH.exists():
        print(f"ERROR: Missing evaluation cases: {CASES_PATH}")
        return 2
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    status = get_rag_status()
    docs = {x["relative_path"]: x for x in status.get("documents", [])}
    sample = docs.get(EXPECTED_SOURCE)
    if sample is None:
        print(f"ERROR: {EXPECTED_SOURCE} is not indexed. Run POST /api/v1/rag/index first.")
        return 2

    results = []
    tp = fp = fn = tn = 0
    guard_checks = []
    for case in cases:
        response = query_local_documents(case["query"], top_k=int(case.get("top_k", 5)))
        sources = response.get("sources", [])
        source_paths = sorted({x.get("source", "") for x in sources})
        expected_match = bool(case["expected_match"])
        expected_sources = case.get("expected_sources", [])
        expected_source_found = any(x.get("source") in expected_sources for x in sources)
        actual_match = expected_source_found if expected_match else bool(sources)
        if expected_match and actual_match: tp += 1
        elif expected_match and not actual_match: fn += 1
        elif not expected_match and actual_match: fp += 1
        else: tn += 1
        if expected_match:
            passed = expected_source_found
        else:
            passed = response.get("decision") == "INSUFFICIENT_EVIDENCE" and not sources

        for result in sources:
            if result.get("source") == EXPECTED_SOURCE:
                provenance = result.get("provenance") or {}
                guard_checks.append({
                    "query_id": case["id"],
                    "approval_status": provenance.get("approval_status"),
                    "authoritative_eligible": result.get("authoritative_eligible"),
                    "passed": provenance.get("approval_status") == "UNVERIFIED"
                              and result.get("authoritative_eligible") is False,
                })
        results.append({
            "id": case["id"], "query": case["query"],
            "expected_match": expected_match, "expected_sources": expected_sources,
            "expected_decision": "SOURCE_MATCH" if expected_match else "INSUFFICIENT_EVIDENCE",
            "actual_decision": response.get("decision"),
            "result_count": response.get("result_count", 0),
            "returned_sources": source_paths, "expected_source_found": expected_source_found,
            "case_pass": bool(passed), "audit_id": response.get("audit_id"),
            "top_scores": [x.get("score") for x in sources[:3]],
        })

    precision = rate(tp, tp + fp)
    recall = rate(tp, tp + fn)
    f1 = rate(2 * precision * recall, precision + recall) if precision is not None and recall is not None and precision + recall else 0.0
    accuracy = rate(tp + tn, tp + tn + fp + fn)
    negatives = sum(not c["expected_match"] for c in cases)
    correct_abstentions = sum(
        not c["expected_match"] and r["actual_decision"] == "INSUFFICIENT_EVIDENCE" and r["result_count"] == 0
        for c, r in zip(cases, results)
    )
    guard_passed = bool(guard_checks) and all(x["passed"] for x in guard_checks)
    passed_cases = sum(x["case_pass"] for x in results)
    report = {
        "report_generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_type": "query-level lexical retrieval regression test",
        "dataset_scope": "fictional sample document only; not an aviation-manual benchmark",
        "document_index": {
            "document_count": status.get("document_count"), "chunk_count": status.get("chunk_count"),
            "expected_sample_source": EXPECTED_SOURCE,
            "sample_approval_status": sample.get("approval_status"),
            "sample_authoritative_eligible": sample.get("authoritative_eligible"),
            "min_score_threshold": MIN_SCORE,
        },
        "metrics": {
            "case_count": len(cases), "passed_cases": passed_cases,
            "pass_rate": rate(passed_cases, len(results)),
            "true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "negative_case_count": negatives, "correct_abstentions": correct_abstentions,
            "abstention_accuracy_on_negative_cases": rate(correct_abstentions, negatives),
        },
        "approval_guard": {
            "checks": guard_checks, "passed": guard_passed,
            "note": "Checks that matching fictional sample passages remain UNVERIFIED and non-authoritative; does not authenticate the source.",
        },
        "cases": results,
        "limitations": [
            "Small hand-labelled synthetic query set; query-level rather than corpus-wide passage metrics.",
            "Token-overlap ranking is lexical; semantically similar terms may not match.",
            "Results do not measure real maintenance-manual quality or establish safety.",
            "Source metadata is user-supplied and is not independently authenticated.",
            "Running evaluation appends queries to rag_query_audit.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = REPORT_DIR / "latest.json", REPORT_DIR / "latest.csv"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["id", "query", "expected_match", "expected_sources", "expected_decision", "actual_decision",
              "result_count", "returned_sources", "expected_source_found", "case_pass", "audit_id", "top_scores"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in results:
            row = dict(item)
            for key in ("expected_sources", "returned_sources", "top_scores"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            writer.writerow(row)

    print("=" * 62)
    print("AERO-GUARD NEXUS — RAG EVALUATION")
    print("=" * 62)
    print(f"Cases passed: {passed_cases}/{len(cases)} | pass rate: {report['metrics']['pass_rate']}")
    print(f"Precision: {precision} | Recall: {recall} | F1: {f1} | Accuracy: {accuracy}")
    print(f"Correct abstentions: {correct_abstentions}/{negatives}")
    print(f"Approval guard: {'PASS' if guard_passed else 'FAIL / NOT EXERCISED'}")
    for row in results:
        print(f"[{'PASS' if row['case_pass'] else 'FAIL'}] {row['id']}: {row['actual_decision']} (results={row['result_count']})")
    print(f"JSON report: {json_path}")
    print(f"CSV report : {csv_path}")
    print("Evaluation queries were written to the normal RAG audit table.")
    if passed_cases != len(cases) or not guard_passed:
        print("RESULT: Regression checks failed or were not fully exercised; inspect latest.json.")
        return 1
    print("RESULT: All configured prototype regression checks passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
