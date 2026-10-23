import psycopg2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Upload support. This wraps the existing ingest_pdfs.process_pdf(); it does
# not re-implement extraction, chunking, embedding or database inserts.
import ingestion_jobs

# Reuse the existing, tested RAG functions.
from rag_assistant import (
    retrieve_chunks,
    generate_answer,
    validate_citations,
    assess_original_question,
    verify_answer_claims,
    # Reuse the embedding model rag_assistant already loaded, so uploads don't
    # load a second copy into memory.
    model as embedding_model,
    # Reuse the existing database settings instead of duplicating them.
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
)

app = FastAPI(
    title="AI Research Paper Intelligence Assistant",
    description="API for research paper semantic search and evidence-based answers.",
    version="1.2.0",
)

@app.middleware("http")
async def reject_oversized_uploads(request, call_next):
    """Refuse an oversized upload from its Content-Length header, before the
    body is read. ingestion_jobs enforces the same cap again while streaming."""
    if request.method == "POST" and request.url.path == "/ingest/upload":
        declared = request.headers.get("content-length", "")
        # Allow a little overhead for the multipart envelope.
        if declared.isdigit() and int(declared) > (
            ingestion_jobs.MAX_UPLOAD_BYTES + 1024 * 1024
        ):
            limit_mb = ingestion_jobs.MAX_UPLOAD_BYTES // (1024 * 1024)
            return JSONResponse(
                status_code=413,
                content={
                    "detail": {
                        "code": "too_large",
                        "message": f"This file is larger than the {limit_mb} MB limit.",
                    }
                },
            )
    return await call_next(request)


# Allow the React development server to call this API.
# (Added after the middleware above so CORS wraps it and 413 responses still
# carry CORS headers the browser can read.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)


def get_connection():
    """Open a short-lived read connection using the existing DB settings."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


@app.get("/")
def home():
    return {
        "message": "AI Research Paper Intelligence Assistant API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "api": "running"}


@app.post("/chat")
def chat(request: ChatRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Please enter a non-empty question.",
        )

    try:
        # Step 1: Retrieve relevant chunks from the existing database.
        results = retrieve_chunks(question, top_k=request.top_k)

        if not results:
            return {
                "question": question,
                "answer": "No relevant research-paper chunks were found.",
                "sources": [],
                "citation_validation": {
                    "status": "not_checked",
                    "found": [],
                    "invalid": [],
                },
                "question_assessment": None,
                "claim_verification": None,
                "warnings": [],
            }

        # Step 2: Generate the answer using the existing RAG pipeline.
        answer = generate_answer(question, results)

        # Step 3: Validate citations against retrieved paper/page pairs.
        found_citations, invalid_citations = validate_citations(
            answer, results
        )

        citation_validation = {
            "status": (
                "passed"
                if found_citations and not invalid_citations
                else "warning"
            ),
            "found": [
                {"paper_id": paper_id, "page": page}
                for paper_id, page in found_citations
            ],
            "invalid": invalid_citations,
        }

        # Step 4: Assess the original question independently.
        warnings = []
        try:
            question_assessment = assess_original_question(
                question, results
            )
        except Exception as error:
            question_assessment = None
            warnings.append(
                "Original-question assessment failed: "
                + type(error).__name__
            )

        # Step 5: Verify factual claims using the existing verifier.
        try:
            claim_verification = verify_answer_claims(
                question, answer, results
            )
        except Exception as error:
            claim_verification = None
            warnings.append(
                "Answer-claim verification failed: "
                + type(error).__name__
            )

        # Step 6: Return source details to the frontend.
        sources = []
        for rank, row in enumerate(results, start=1):
            chunk_id, paper_id, page_number, chunk_text, distance = row

            sources.append({
                "rank": rank,
                "chunk_id": str(chunk_id),
                "paper_id": str(paper_id),
                "page_number": int(page_number),
                "distance": float(distance),
                "text": str(chunk_text),
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "citation_validation": citation_validation,
            "question_assessment": question_assessment,
            "claim_verification": claim_verification,
            "warnings": warnings,
        }

    except Exception as error:
        # Keep implementation details out of the public API response.
        print(f"Chat API error: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=500,
            detail=(
                "The research assistant could not process this question. "
                "Check the backend terminal for the error."
            ),
        ) from error


# =========================================================
# READ-ONLY ENDPOINTS ADDED FOR THE FRONTEND
# None of these write to the database or call Gemini.
# =========================================================

@app.post("/search")
def search(request: SearchRequest):
    """Semantic search only: calls the existing retrieve_chunks()."""
    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=422,
            detail="Please enter a non-empty search query.",
        )

    try:
        results = retrieve_chunks(query, top_k=request.top_k)

        return {
            "query": query,
            "top_k": request.top_k,
            "results": [
                {
                    "rank": rank,
                    "chunk_id": str(chunk_id),
                    "paper_id": str(paper_id),
                    "page_number": int(page_number),
                    "distance": float(distance),
                    "text": str(chunk_text),
                }
                for rank, (
                    chunk_id, paper_id, page_number, chunk_text, distance
                ) in enumerate(results, start=1)
            ],
        }

    except Exception as error:
        print(f"Search API error: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Search failed. Check the backend terminal for the error."
            ),
        ) from error


@app.get("/stats")
def stats():
    """Real counts from the existing tables (same queries as verify_database.py)."""
    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM papers;")
                total_papers = int(cursor.fetchone()[0])

                cursor.execute("SELECT COUNT(*) FROM paper_chunks;")
                total_chunks = int(cursor.fetchone()[0])

                cursor.execute(
                    "SELECT COUNT(*) FROM paper_chunks "
                    "WHERE embedding IS NULL;"
                )
                chunks_without_embeddings = int(cursor.fetchone()[0])

                cursor.execute(
                    "SELECT COUNT(DISTINCT paper_id) FROM paper_chunks;"
                )
                papers_with_chunks = int(cursor.fetchone()[0])
        finally:
            conn.close()

        return {
            "total_papers": total_papers,
            "total_chunks": total_chunks,
            "papers_with_chunks": papers_with_chunks,
            "chunks_without_embeddings": chunks_without_embeddings,
        }

    except Exception as error:
        print(f"Stats API error: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=500,
            detail="Could not read database statistics.",
        ) from error


@app.get("/papers")
def list_papers(
    q: str = Query(default="", max_length=100),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    """
    Paginated paper list. The papers table stores no titles, so papers are
    identified by paper_id. Optional q filters paper_id (case-insensitive).
    """
    term = q.strip()
    # Escape LIKE wildcards so user input is matched literally.
    escaped = (
        term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )
    pattern = f"%{escaped}%"
    offset = (page - 1) * page_size

    try:
        conn = get_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM papers WHERE paper_id ILIKE %s;",
                    (pattern,),
                )
                total = int(cursor.fetchone()[0])

                cursor.execute(
                    """
                    SELECT
                        p.paper_id,
                        p.source,
                        p.total_pages,
                        COUNT(c.chunk_id) AS chunk_count
                    FROM papers p
                    LEFT JOIN paper_chunks c ON c.paper_id = p.paper_id
                    WHERE p.paper_id ILIKE %s
                    GROUP BY p.paper_id, p.source, p.total_pages
                    ORDER BY p.paper_id
                    LIMIT %s OFFSET %s;
                    """,
                    (pattern, page_size, offset),
                )
                rows = cursor.fetchall()
        finally:
            conn.close()

        return {
            "page": page,
            "page_size": page_size,
            "total": total,
            "items": [
                {
                    "paper_id": str(paper_id),
                    "source": source,
                    "total_pages": (
                        int(total_pages) if total_pages is not None else None
                    ),
                    "chunk_count": int(chunk_count),
                    "indexed": int(chunk_count) > 0,
                }
                for paper_id, source, total_pages, chunk_count in rows
            ],
        }

    except Exception as error:
        print(f"Papers API error: {type(error).__name__}: {error}")
        raise HTTPException(
            status_code=500,
            detail="Could not read the paper list.",
        ) from error


# =========================================================
# PDF UPLOAD AND INGESTION
# =========================================================

def _ingest_http_error(error: ingestion_jobs.IngestError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": error.message, **error.extra},
    )


@app.get("/ingest/status")
def ingest_status():
    """Whether an ingestion is running, and the upload size limit."""
    return ingestion_jobs.status()


@app.get("/ingest/jobs/{job_id}")
def ingest_job(job_id: str):
    job = ingestion_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "job_not_found",
                "message": (
                    "This job is no longer tracked (the API may have restarted). "
                    "Check the Library to see whether the paper was added."
                ),
            },
        )
    return job


try:

    @app.post("/ingest/upload", status_code=202)
    def upload_paper(file: UploadFile = File(...)):
        """
        Validate and store one PDF, then ingest it in the background with the
        existing pipeline. Returns a job to poll at /ingest/jobs/{job_id}.
        """
        try:
            return ingestion_jobs.accept_upload(
                file.filename or "", file.file, embedding_model
            )
        except ingestion_jobs.IngestError as error:
            raise _ingest_http_error(error) from error
        except Exception as error:
            print(f"Upload API error: {type(error).__name__}: {error}")
            raise HTTPException(
                status_code=500,
                detail={
                    "code": "failed",
                    "message": (
                        "The upload could not be processed. "
                        "Check the backend terminal for the error."
                    ),
                },
            ) from error

except RuntimeError:
    # FastAPI refuses to define a file-upload route without python-multipart.
    # Keep the rest of the API running and explain the fix instead.
    @app.post("/ingest/upload", status_code=503)
    def upload_unavailable():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "missing_dependency",
                "message": (
                    "Uploads need the python-multipart package. "
                    "Run: pip install python-multipart, then restart the API."
                ),
            },
        )
