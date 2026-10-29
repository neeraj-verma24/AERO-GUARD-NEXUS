"""Offline, extractive document retrieval for AERO-GUARD NEXUS.

Indexes local .txt/.md documents, retrieves matching passages, and abstains
when evidence is weak. It does not generate maintenance instructions.
"""
from __future__ import annotations
from datetime import date, datetime, timezone
from pathlib import Path
from io import BytesIO
import hashlib
import json
import re
import uuid
from contextlib import contextmanager

from backend.database import DB_PATH, get_connection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "manuals"
# DB_PATH is imported from backend.database.
SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}
from backend.config import get_float_setting

MIN_SCORE = get_float_setting(
    "AERO_GUARD_RAG_MIN_SCORE", 0.12, minimum=0.0, maximum=1.0
)
MAX_CHUNK_CHARS = 1400
OVERLAP_CHARS = 180
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "what", "when", "where", "which", "with", "you", "your",
    "do", "does", "can", "could", "should", "would", "about", "please",
}

def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

@contextmanager
def _connect():
    """Use the shared connection factory and always close the connection."""
    conn = get_connection()
    try:
        # Preserve sqlite3 context-manager commit/rollback semantics.
        with conn:
            yield conn
    finally:
        conn.close()

def _ensure_tables(conn):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS rag_documents (
        document_id TEXT PRIMARY KEY, relative_path TEXT NOT NULL UNIQUE,
        sha256 TEXT NOT NULL, byte_size INTEGER NOT NULL,
        indexed_at TEXT NOT NULL, chunk_count INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS rag_chunks (
        chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
        relative_path TEXT NOT NULL, section TEXT, chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL, token_json TEXT NOT NULL, page_number INTEGER,
        FOREIGN KEY(document_id) REFERENCES rag_documents(document_id) ON DELETE CASCADE
    );
    CREATE INDEX IF NOT EXISTS idx_rag_chunks_document
        ON rag_chunks(document_id, chunk_index);
    CREATE TABLE IF NOT EXISTS rag_query_audit (
        audit_id TEXT PRIMARY KEY, queried_at TEXT NOT NULL, query_text TEXT NOT NULL,
        decision TEXT NOT NULL, result_count INTEGER NOT NULL, min_score REAL NOT NULL,
        sources_json TEXT NOT NULL, reason TEXT NOT NULL
    );
    """)
    # Additive migration: never drop or rebuild current documents/chunks/audit rows.
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(rag_documents)")}
    additions = {
        "title": "TEXT",
        "source_organization": "TEXT",
        "document_type": "TEXT",
        "revision": "TEXT",
        "effective_date": "TEXT",
        "expiry_date": "TEXT",
        "approval_status": "TEXT NOT NULL DEFAULT 'UNVERIFIED'",
        "approval_reference": "TEXT",
        "reviewed_by": "TEXT",
        "provenance_notes": "TEXT",
        "metadata_updated_at": "TEXT",
    }
    for column, sql_type in additions.items():
        if column not in columns:
            conn.execute(f"ALTER TABLE rag_documents ADD COLUMN {column} {sql_type}")
    chunk_columns = {row["name"] for row in conn.execute("PRAGMA table_info(rag_chunks)")}
    if "page_number" not in chunk_columns:
        conn.execute("ALTER TABLE rag_chunks ADD COLUMN page_number INTEGER")
    conn.commit()

def _tokens(text):
    return [t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{1,}", text.lower())
            if t not in STOP_WORDS]

def _split_sections(text):
    sections, heading, buffer = [], "Document start", []
    for line in text.splitlines():
        match = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if match:
            content = "\n".join(buffer).strip()
            if content:
                sections.append((heading, content))
            heading, buffer = match.group(2).strip(), []
        else:
            buffer.append(line)
    content = "\n".join(buffer).strip()
    if content:
        sections.append((heading, content))
    return sections

def _chunks(text):
    index = 0
    for section, section_text in _split_sections(text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        for paragraph in paragraphs:
            start = 0
            while start < len(paragraph):
                end = min(len(paragraph), start + MAX_CHUNK_CHARS)
                if end < len(paragraph):
                    candidate = max(paragraph.rfind(". ", start, end), paragraph.rfind("\n", start, end))
                    if candidate > start + MAX_CHUNK_CHARS // 2:
                        end = candidate + 1
                piece = paragraph[start:end].strip()
                if piece and _tokens(piece):
                    yield index, section, piece
                    index += 1
                if end >= len(paragraph):
                    break
                start = max(start + 1, end - OVERLAP_CHARS)

def _safe_relative(path):
    try:
        return path.resolve().relative_to(DOCUMENTS_DIR.resolve()).as_posix()
    except ValueError:
        raise ValueError("Document path must be inside data/manuals.") from None


def _extract_chunks(path, raw):
    """Extract text and return (chunk_index, section, content, page_number).

    PDF page numbers are one-based PDF page indexes, not printed page labels.
    Image-only/scanned PDFs without an OCR text layer are not indexed.
    """
    suffix = path.suffix.lower()
    generated = []
    next_index = 0
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF support requires pypdf. Install it in the project venv with: python -m pip install pypdf"
            ) from exc
        reader = PdfReader(BytesIO(raw), strict=False)
        if reader.is_encrypted:
            raise ValueError("PDF is encrypted/password-protected and was not indexed.")
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            for _, section, content in _chunks(page_text):
                # Page is part of the citation identity. Keep chunk indexes unique per document.
                section_label = section if section != "Document start" else f"PDF page {page_number}"
                generated.append((next_index, section_label, content, page_number))
                next_index += 1
        if not generated:
            raise ValueError(
                "PDF contains no extractable text. It may be scanned/image-only; OCR is not enabled."
            )
        return generated

    text = raw.decode("utf-8-sig", errors="replace")
    for idx, section, content in _chunks(text):
        generated.append((idx, section, content, None))
    return generated

APPROVAL_STATUSES = {"UNVERIFIED", "PENDING_REVIEW", "APPROVED", "REJECTED", "EXPIRED"}


def _current_approval_state(row):
    """Evaluate stored approval metadata against required fields and current dates."""
    status = (row["approval_status"] or "UNVERIFIED").upper()
    if status != "APPROVED":
        return status
    required = ("source_organization", "document_type", "revision", "effective_date",
                "approval_reference", "reviewed_by")
    if any(not (row[field] or "").strip() for field in required):
        return "APPROVAL_METADATA_INCOMPLETE"
    try:
        effective = date.fromisoformat(row["effective_date"])
        expiry = date.fromisoformat(row["expiry_date"]) if row["expiry_date"] else None
    except (ValueError, TypeError):
        return "APPROVAL_DATE_INVALID"
    today = date.today()
    if effective > today:
        return "NOT_YET_EFFECTIVE"
    if expiry and expiry < today:
        return "EXPIRED"
    return "APPROVED"


def _provenance(row):
    state = _current_approval_state(row)
    return {
        "title": row["title"] or Path(row["relative_path"]).name,
        "source_organization": row["source_organization"],
        "document_type": row["document_type"],
        "revision": row["revision"],
        "effective_date": row["effective_date"],
        "expiry_date": row["expiry_date"],
        "approval_status": state,
        "approval_reference": row["approval_reference"],
        "reviewed_by": row["reviewed_by"],
        "provenance_notes": row["provenance_notes"],
        "sha256": row["sha256"],
        "indexed_at": row["indexed_at"],
        "metadata_updated_at": row["metadata_updated_at"],
    }


def update_document_metadata(relative_path, metadata):
    """Update provenance fields. User-supplied metadata is not independently authenticated."""
    relative_path = (relative_path or "").replace("\\", "/").strip()
    if (not relative_path or relative_path.startswith("/")
            or ".." in Path(relative_path).parts):
        raise ValueError("relative_path must identify a file inside data/manuals.")
    allowed = {"title", "source_organization", "document_type", "revision",
               "effective_date", "expiry_date", "approval_status", "approval_reference",
               "reviewed_by", "provenance_notes"}
    unknown = set(metadata) - allowed
    if unknown:
        raise ValueError("Unsupported metadata fields: " + ", ".join(sorted(unknown)))

    with _connect() as conn:
        _ensure_tables(conn)
        row = conn.execute("SELECT * FROM rag_documents WHERE relative_path = ?",
                           (relative_path,)).fetchone()
        if row is None:
            raise LookupError("Document not found in index. Run the local index endpoint first.")
        candidate = {key: row[key] for key in allowed}
        candidate.update({key: value for key, value in metadata.items() if value is not None})
        status = str(candidate.get("approval_status") or "UNVERIFIED").strip().upper()
        if status not in APPROVAL_STATUSES:
            raise ValueError("approval_status must be one of: " + ", ".join(sorted(APPROVAL_STATUSES)))
        candidate["approval_status"] = status
        for field in ("effective_date", "expiry_date"):
            value = candidate.get(field)
            if value:
                try:
                    date.fromisoformat(value)
                except (ValueError, TypeError):
                    raise ValueError(f"{field} must use YYYY-MM-DD.") from None
        if status == "APPROVED":
            required = ("source_organization", "document_type", "revision", "effective_date",
                        "approval_reference", "reviewed_by")
            missing = [field for field in required if not str(candidate.get(field) or "").strip()]
            if missing:
                raise ValueError("APPROVED requires: " + ", ".join(missing))
            if candidate.get("expiry_date") and date.fromisoformat(candidate["expiry_date"]) < date.today():
                raise ValueError("Cannot mark an already expired document as APPROVED.")
        # Cap metadata field lengths to keep the audit surface controlled.
        limits = {"title": 250, "source_organization": 200, "document_type": 80,
                  "revision": 80, "approval_reference": 300, "reviewed_by": 200,
                  "provenance_notes": 1000}
        for field, limit in limits.items():
            value = candidate.get(field)
            if value is not None and len(str(value)) > limit:
                raise ValueError(f"{field} must be {limit} characters or fewer.")
        to_update = {key: candidate[key] for key in allowed}
        to_update["metadata_updated_at"] = _now()
        assignments = ", ".join(f"{key} = ?" for key in to_update)
        conn.execute(f"UPDATE rag_documents SET {assignments} WHERE relative_path = ?",
                     [*to_update.values(), relative_path])
        conn.commit()
        updated = conn.execute("SELECT * FROM rag_documents WHERE relative_path = ?",
                               (relative_path,)).fetchone()
    return {"updated": True, "relative_path": relative_path,
            "provenance": _provenance(updated),
            "note": "Metadata is user-supplied and not independently authenticated. APPROVED is an administrative flag, not a safety certification."}


def index_local_documents():
    """Index local .txt/.md files under data/manuals; never accesses the network."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in DOCUMENTS_DIR.rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
                   and not any(part.startswith(".") for part in p.relative_to(DOCUMENTS_DIR).parts))
    processed, errors = [], []
    with _connect() as conn:
        _ensure_tables(conn)
        present = {_safe_relative(p) for p in files}
        existing = conn.execute("SELECT document_id, relative_path FROM rag_documents").fetchall()
        for row in existing:
            if row["relative_path"] not in present:
                conn.execute("DELETE FROM rag_documents WHERE document_id=?", (row["document_id"],))
        for path in files:
            relative = _safe_relative(path)
            try:
                raw = path.read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                generated = _extract_chunks(path, raw)
                old = conn.execute("SELECT document_id, sha256 FROM rag_documents WHERE relative_path=?",
                                   (relative,)).fetchone()
                if not generated:
                    # Do not leave stale previously indexed passages retrievable
                    # when a file is emptied or becomes non-indexable.
                    if old:
                        conn.execute("DELETE FROM rag_documents WHERE document_id=?", (old["document_id"],))
                    errors.append({"path": relative, "error": "No indexable text found; document is not available for retrieval."})
                    continue
                if old and old["sha256"] == digest:
                    count = conn.execute("SELECT COUNT(*) FROM rag_chunks WHERE document_id=?",
                                         (old["document_id"],)).fetchone()[0]
                    processed.append({"path": relative, "status": "UNCHANGED", "chunk_count": count})
                    continue
                doc_id = old["document_id"] if old else str(uuid.uuid4())
                if old:
                    conn.execute("DELETE FROM rag_chunks WHERE document_id=?", (doc_id,))
                    # A changed file hash invalidates prior approval for the old content.
                    conn.execute("""UPDATE rag_documents SET approval_status='UNVERIFIED',
                        approval_reference=NULL, reviewed_by=NULL, metadata_updated_at=NULL
                        WHERE document_id=?""", (doc_id,))
                conn.execute("""INSERT INTO rag_documents
                    (document_id, relative_path, sha256, byte_size, indexed_at, chunk_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(relative_path) DO UPDATE SET sha256=excluded.sha256,
                    byte_size=excluded.byte_size, indexed_at=excluded.indexed_at,
                    chunk_count=excluded.chunk_count""",
                    (doc_id, relative, digest, len(raw), _now(), len(generated)))
                conn.executemany("""INSERT INTO rag_chunks
                    (chunk_id, document_id, relative_path, section, chunk_index, content, token_json, page_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    [(str(uuid.uuid4()), doc_id, relative, section, idx, content,
                      json.dumps(sorted(set(_tokens(content)))), page_number)
                     for idx, section, content, page_number in generated])
                processed.append({"path": relative, "status": "INDEXED_UNVERIFIED", "chunk_count": len(generated)})
            except Exception as exc:
                errors.append({"path": relative, "error": str(exc)[:500],
                               "error_type": type(exc).__name__})
        conn.commit()
        document_count = conn.execute("SELECT COUNT(*) FROM rag_documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
    return {"indexed_at": _now(), "documents_directory": str(DOCUMENTS_DIR),
            "supported_extensions": sorted(SUPPORTED_SUFFIXES), "documents_seen": len(files),
            "document_count": document_count, "chunk_count": chunk_count,
            "processed": processed, "errors": errors,
            "note": "New or changed files default to UNVERIFIED. NASA C-MAPSS data is not a maintenance manual."}

def _pdf_support_available():
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False


def get_rag_status():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        _ensure_tables(conn)
        docs = conn.execute("""SELECT relative_path, sha256, byte_size, indexed_at, chunk_count,
            title, source_organization, document_type, revision, effective_date, expiry_date,
            approval_status, approval_reference, reviewed_by, provenance_notes, metadata_updated_at
            FROM rag_documents ORDER BY relative_path""").fetchall()
        chunks = conn.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
    items = []
    for row in docs:
        item = dict(row)
        item["approval_status"] = _current_approval_state(row)
        item["authoritative_eligible"] = item["approval_status"] == "APPROVED"
        items.append(item)
    return {"documents_directory": str(DOCUMENTS_DIR), "document_count": len(items),
            "chunk_count": chunks, "documents": items,
            "retrieval_method": "offline lexical token-overlap ranking",
            "answer_generation": "EXTRACTIVE_ONLY; no LLM-generated maintenance instructions",
            "approval_policy": "Only APPROVED documents with complete provenance and currently valid dates are eligible as authoritative evidence.",
            "manual_evidence_available": any(item["authoritative_eligible"] for item in items),
            "pdf_support_available": _pdf_support_available(),
            "pdf_note": "Text-layer PDFs are supported when pypdf is installed; scanned/image-only PDFs need OCR and will be skipped."}

def query_local_documents(query, top_k=5):
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")
    if len(query) > 1000:
        raise ValueError("query must be 1000 characters or fewer")
    if not 1 <= int(top_k) <= 10:
        raise ValueError("top_k must be between 1 and 10")
    query_tokens = set(_tokens(query))
    audit_id, matches = str(uuid.uuid4()), []
    with _connect() as conn:
        _ensure_tables(conn)
        rows = conn.execute("""SELECT c.chunk_id, c.relative_path, c.section, c.chunk_index,
            c.content, c.token_json, c.page_number, d.title, d.source_organization, d.document_type,
            d.revision, d.effective_date, d.expiry_date, d.approval_status,
            d.approval_reference, d.reviewed_by, d.provenance_notes, d.sha256,
            d.indexed_at, d.metadata_updated_at
            FROM rag_chunks c JOIN rag_documents d ON d.document_id=c.document_id""").fetchall()
        for row in rows:
            document_tokens = set(json.loads(row["token_json"]))
            overlap = query_tokens & document_tokens
            if not query_tokens or not document_tokens or not overlap:
                continue
            score = 0.65 * (len(overlap) / max(1, len(query_tokens | document_tokens)))
            score += 0.35 * (len(overlap) / max(1, len(query_tokens)))
            provenance = _provenance(row)
            eligible = provenance["approval_status"] == "APPROVED"
            matches.append({"chunk_id": row["chunk_id"], "source": row["relative_path"],
                "section": row["section"], "chunk_index": row["chunk_index"],
                "page_number": row["page_number"],
                "citation": (f"{row['relative_path']} — PDF page {row['page_number']}" if row["page_number"] else f"{row['relative_path']} — section {row['section']}"),
                "score": round(score, 4), "query_token_overlap": sorted(overlap),
                "excerpt": row["content"], "provenance": provenance,
                "authoritative_eligible": eligible})
        matches.sort(key=lambda item: (-item["score"], item["source"], item["chunk_index"]))
        selected = [item for item in matches if item["score"] >= MIN_SCORE][:int(top_k)]
        approved = [item for item in selected if item["authoritative_eligible"]]
        if approved:
            decision = "APPROVED_SOURCE_MATCH_FOUND"
            reason = "A matching source is marked APPROVED and has complete, currently valid provenance metadata; independently confirm applicability."
            answer = "Matching excerpts from documents marked APPROVED are provided. Confirm applicability and current official instructions before any maintenance use."
        elif selected:
            decision = "UNVERIFIED_SOURCE_MATCH"
            reason = "Matching passages exist, but none is from a currently APPROVED document."
            answer = "Unverified excerpts are shown for retrieval testing only. They are not authoritative maintenance guidance."
        else:
            decision = "INSUFFICIENT_EVIDENCE"
            reason = "No indexed passage met the retrieval threshold; no maintenance instruction was generated."
            answer = "INSUFFICIENT_EVIDENCE: no relevant indexed source passage was found. Add an authorized document, set complete provenance metadata, and complete review."
        audit_sources = [{"source": item["source"], "section": item["section"],
                          "page_number": item["page_number"], "citation": item["citation"],
                          "chunk_id": item["chunk_id"], "score": item["score"],
                          "approval_status": item["provenance"]["approval_status"],
                          "authoritative_eligible": item["authoritative_eligible"]}
                         for item in selected]
        conn.execute("""INSERT INTO rag_query_audit
            (audit_id, queried_at, query_text, decision, result_count, min_score, sources_json, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (audit_id, _now(), query, decision, len(selected), MIN_SCORE,
             json.dumps(audit_sources), reason))
        conn.commit()
    return {"audit_id": audit_id, "queried_at": _now(), "query": query,
            "decision": decision, "answer": answer, "min_score_threshold": MIN_SCORE,
            "result_count": len(selected), "sources": selected, "reason": reason,
            "safety_note": "Metadata is user-supplied and not independently authenticated. Lexical relevance does not prove correctness, authority, currency, or aircraft applicability."}

def get_rag_audit_log(limit=20):
    limit = max(1, min(int(limit), 200))
    with _connect() as conn:
        _ensure_tables(conn)
        rows = conn.execute("""SELECT audit_id, queried_at, query_text, decision, result_count,
            min_score, sources_json, reason FROM rag_query_audit
            ORDER BY queried_at DESC, rowid DESC LIMIT ?""", (limit,)).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["sources"] = json.loads(item.pop("sources_json"))
        items.append(item)
    return {"count": len(items), "items": items}
