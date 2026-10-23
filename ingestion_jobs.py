"""
Upload-and-ingest support for the FastAPI layer.

This module does NOT re-implement any pipeline logic. Extraction, chunking,
embedding and the database insert all run inside ingest_pdfs.process_pdf().
What lives here is only what an HTTP upload needs around it:

  * safe staging of the uploaded bytes (size cap, PDF checks)
  * a safe paper ID derived from the filename (same convention as
    ingest_pdfs: filename stem == paper_id)
  * duplicate detection that never overwrites anything
  * a single-job lock so two ingestions can never run at once
  * a background thread plus an in-memory job registry the UI can poll
  * cleanup of everything a failed job created, and friendly error messages
"""

import logging
import re
import tempfile
import threading
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

from ingest_pdfs import (
    CHUNKS_DIR,
    EMBEDDINGS_DIR,
    EXTRACTED_DIR,
    FAILURE_LOG,
    PDF_DIR,
    connect_database,
    process_pdf,
    pymupdf,
)

log = logging.getLogger("ingestion_jobs")

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB per PDF
READ_CHUNK_BYTES = 1024 * 1024
STAGING_DIR = PDF_DIR.parent / "uploads_tmp"
MAX_JOBS_REMEMBERED = 100

_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{name}{number}" for name in ("COM", "LPT") for number in range(1, 10)
}


class IngestError(Exception):
    """An expected, user-facing failure with an HTTP status and a stable code."""

    def __init__(self, status_code, code, message, **extra):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.extra = extra


@dataclass
class Job:
    job_id: str
    filename: str
    paper_id: str
    status: str = "processing"  # processing | completed | failed
    message: str | None = None
    error_code: str | None = None
    pages: int | None = None
    chunks: int | None = None
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    finished_at: str | None = None


_ingest_lock = threading.Lock()  # held for the whole life of one job
_registry_lock = threading.Lock()
_jobs: "OrderedDict[str, Job]" = OrderedDict()
_active_job_id: str | None = None


# ---------------------------------------------------------------------------
# Public status helpers
# ---------------------------------------------------------------------------
def status():
    return {
        "busy": _ingest_lock.locked(),
        "active_job_id": _active_job_id,
        "max_upload_bytes": MAX_UPLOAD_BYTES,
    }


def get_job(job_id):
    with _registry_lock:
        job = _jobs.get(job_id)
        return asdict(job) if job else None


# ---------------------------------------------------------------------------
# Filename -> paper_id
# ---------------------------------------------------------------------------
def sanitize_paper_id(filename):
    """
    Turn an uploaded filename into a safe paper ID (and PDF file stem).
    Strips any directory part, keeps letters, digits, dot, dash, underscore.
    arXiv-style names such as 1805.01956v1.pdf pass through unchanged.
    """
    base = re.split(r"[\\/]", filename or "")[-1].strip()
    if not base.lower().endswith(".pdf"):
        raise IngestError(422, "not_pdf", "Only PDF files (.pdf) can be uploaded.")

    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", base[:-4]).strip("._-")
    stem = stem[:100].strip("._-")
    if not stem:
        raise IngestError(
            422,
            "bad_name",
            "The file name has no usable letters or digits. Rename the file and try again.",
        )
    if stem.split(".")[0].upper() in _WINDOWS_RESERVED:
        stem = f"paper_{stem}"
    return stem


# ---------------------------------------------------------------------------
# Staging + validation
# ---------------------------------------------------------------------------
def _stage_upload(fileobj):
    """Stream the upload to a temp file, enforcing the size cap while reading."""
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=STAGING_DIR, suffix=".part", delete=False
    )
    path = Path(handle.name)
    size = 0
    try:
        with handle:
            while True:
                block = fileobj.read(READ_CHUNK_BYTES)
                if not block:
                    break
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise IngestError(
                        413,
                        "too_large",
                        f"This file is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
                    )
                handle.write(block)
        if size == 0:
            raise IngestError(422, "empty", "This file is empty.")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _validate_pdf(path):
    with path.open("rb") as handle:
        head = handle.read(1024)
    if b"%PDF-" not in head:
        raise IngestError(422, "not_pdf", "This file isn't a PDF.")

    try:
        with pymupdf.open(str(path)) as document:
            if document.needs_pass or document.is_encrypted:
                raise IngestError(
                    422,
                    "encrypted",
                    "This PDF is password-protected. Remove the password and try again.",
                )
            if len(document) == 0:
                raise IngestError(422, "invalid_pdf", "This PDF has no pages.")
    except IngestError:
        raise
    except Exception as error:
        log.warning("PDF validation failed: %s: %s", type(error).__name__, error)
        raise IngestError(
            422,
            "invalid_pdf",
            "This PDF is damaged or can't be read.",
        ) from error


def _ensure_folders():
    """ingest_pdfs creates these once at import; recreate them if someone
    deleted a folder while the API was running."""
    for folder in (PDF_DIR, EXTRACTED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _artifact_paths(paper_id):
    return [
        EXTRACTED_DIR / f"{paper_id}.txt",
        CHUNKS_DIR / f"{paper_id}.json",
        EMBEDDINGS_DIR / f"{paper_id}.json",
    ]


def _check_not_duplicate(paper_id):
    """Raise unless this paper ID is completely unused. Never overwrites."""
    # 1) Database: case-insensitive, so ids that differ only by case collide
    #    the same way they would on a case-insensitive Windows disk.
    try:
        conn = connect_database()
    except Exception as error:
        raise IngestError(
            503,
            "database",
            "Can't reach the database. Check that PostgreSQL is running, then try again.",
        ) from error
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT paper_id FROM papers WHERE lower(paper_id) = lower(%s);",
                (paper_id,),
            )
            row = cursor.fetchone()
    except Exception as error:
        raise IngestError(
            503, "database", "The database check failed. Try again in a moment."
        ) from error
    finally:
        conn.close()

    if row:
        raise IngestError(
            409,
            "duplicate",
            f"{row[0]} is already in the library, so it wasn't uploaded again. "
            "If this is a different paper, rename the file and upload it again.",
            paper_id=row[0],
        )

    # 2) A same-named PDF already sitting in the papers folder (any case).
    wanted = f"{paper_id}.pdf".lower()
    if any(item.name.lower() == wanted for item in PDF_DIR.iterdir()):
        raise IngestError(
            409,
            "duplicate_file",
            f"A file named {paper_id}.pdf is already in the papers folder but "
            "isn't in the library yet. Run ingest_pdfs.py to add it, or rename "
            "this file and upload it again.",
            paper_id=paper_id,
        )

    # 3) Leftover intermediate files from an earlier run for this ID.
    leftovers = [path.name for path in _artifact_paths(paper_id) if path.exists()]
    if leftovers:
        raise IngestError(
            409,
            "stale_files",
            f"Intermediate files for {paper_id} already exist ({', '.join(leftovers)}). "
            "Remove them or rename this file so nothing is overwritten.",
            paper_id=paper_id,
        )


# ---------------------------------------------------------------------------
# Accepting an upload
# ---------------------------------------------------------------------------
def accept_upload(filename, fileobj, embedding_model):
    """
    Validate the upload, claim the single ingestion slot, place the PDF in
    dataset/papers (exclusive create: never overwrites) and start a
    background job that runs ingest_pdfs.process_pdf().
    Returns the new job as a dict. Raises IngestError for expected failures.
    """
    global _active_job_id

    paper_id = sanitize_paper_id(filename)
    _ensure_folders()
    staged = _stage_upload(fileobj)

    try:
        _validate_pdf(staged)

        if not _ingest_lock.acquire(blocking=False):
            raise IngestError(
                409,
                "busy",
                "Another paper is being processed. It will be your turn when it finishes.",
            )
        lock_owned_here = True

        try:
            _check_not_duplicate(paper_id)

            final_path = PDF_DIR / f"{paper_id}.pdf"
            try:
                # "xb" = create exclusively; raises if the file exists.
                with staged.open("rb") as source, final_path.open("xb") as target:
                    while True:
                        block = source.read(READ_CHUNK_BYTES)
                        if not block:
                            break
                        target.write(block)
            except FileExistsError as error:
                raise IngestError(
                    409,
                    "duplicate_file",
                    f"A file named {paper_id}.pdf already exists in the papers folder.",
                    paper_id=paper_id,
                ) from error
            except OSError as error:
                final_path.unlink(missing_ok=True)
                raise IngestError(
                    500, "disk", "The PDF couldn't be saved. Check free disk space."
                ) from error

            job = Job(
                job_id=uuid.uuid4().hex[:12],
                filename=Path(re.split(r"[\\/]", filename)[-1]).name,
                paper_id=paper_id,
            )
            with _registry_lock:
                _jobs[job.job_id] = job
                while len(_jobs) > MAX_JOBS_REMEMBERED:
                    _jobs.popitem(last=False)
            _active_job_id = job.job_id

            try:
                threading.Thread(
                    target=_run_job,
                    args=(job, final_path, embedding_model),
                    name=f"ingest-{job.job_id}",
                    daemon=True,
                ).start()
            except Exception:
                final_path.unlink(missing_ok=True)
                _active_job_id = None
                raise

            lock_owned_here = False  # the job thread now owns (and releases) it
            return asdict(job)
        finally:
            if lock_owned_here:
                _ingest_lock.release()
    finally:
        staged.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The background job
# ---------------------------------------------------------------------------
def describe_failure(error):
    """Map an exception to (error_code, user-facing message)."""
    text = str(error)
    lowered = text.lower()

    if isinstance(error, IngestError):
        return error.code, error.message
    if isinstance(error, psycopg2.Error):
        return (
            "database",
            "The database couldn't save this paper, so nothing was added. "
            "Check that PostgreSQL is running, then try again.",
        )
    if isinstance(error, ValueError):
        if "no selectable text" in lowered or "no non-empty chunks" in lowered:
            return (
                "no_text",
                "This PDF has no selectable text. It looks like a scanned document, "
                "and OCR isn't supported yet.",
            )
        if "encrypted" in lowered or "password" in lowered:
            return (
                "encrypted",
                "This PDF is password-protected. Remove the password and try again.",
            )
    if type(error).__name__ in {"FileDataError", "EmptyFileError"} or any(
        phrase in lowered for phrase in ("cannot open", "broken document", "no objects found")
    ):
        return "invalid_pdf", "This PDF is damaged or can't be read."
    return (
        "failed",
        "Ingestion failed unexpectedly, so nothing was added. "
        "Details are in ingestion_failures.log in the project folder.",
    )


def _log_failure(job, error):
    try:
        with FAILURE_LOG.open("a", encoding="utf-8") as handle:
            handle.write(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Upload via API\n"
                f"PDF: {job.filename}\n"
                f"Error: {type(error).__name__}: {error}\n"
                f"{'-' * 60}\n"
            )
    except OSError:
        pass


def _run_job(job, pdf_path, embedding_model):
    """Runs ingest_pdfs.process_pdf() and records the outcome on the job."""
    global _active_job_id
    conn = None
    delete_artifacts = True
    final_status = "failed"
    try:
        conn = connect_database()
        result = process_pdf(pdf_path, conn, embedding_model)

        if result["status"] != "SUCCESS":
            # Someone else (e.g. the command-line script) ingested this ID
            # between our check and now. Leave their data completely alone.
            delete_artifacts = False
            raise IngestError(
                409,
                "duplicate",
                f"{job.paper_id} was added by another process, so this upload was skipped.",
            )

        job.pages = result["pages"]
        job.chunks = result["inserted"]
        job.message = (
            f"Added {job.paper_id}: {result['pages']} pages, "
            f"{result['inserted']} searchable chunks."
        )
        final_status = "completed"
        log.info("Upload ingested: %s", job.paper_id)

    except Exception as error:
        job.error_code, job.message = describe_failure(error)
        if not isinstance(error, IngestError):
            log.exception("Upload ingestion failed for %s", job.filename)
            # ingest_pdfs sends logging to ingestion.log, so also show it here.
            print(f"Ingestion error for {job.filename}: {type(error).__name__}: {error}")
        _log_failure(job, error)

        # Undo everything this job created. The database insert is already
        # transactional (process_pdf rolls back), so no rows remain.
        pdf_path.unlink(missing_ok=True)
        if delete_artifacts:
            for path in _artifact_paths(job.paper_id):
                path.unlink(missing_ok=True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        job.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _active_job_id = None
        _ingest_lock.release()
        # Publish the final status only after the slot is free, so a client
        # that sees "completed" can immediately start the next upload.
        job.status = final_status
