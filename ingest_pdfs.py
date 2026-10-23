import re
import json
import math
import logging
from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


# =========================================================
# CONFIGURATION: PRESERVES YOUR EXISTING DATABASE/TABLES
# =========================================================
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "research_paper_assistant"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBEDDING_BATCH_SIZE = 32

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "dataset" / "papers"
EXTRACTED_DIR = BASE_DIR / "dataset" / "extracted_text"
CHUNKS_DIR = BASE_DIR / "dataset" / "chunks"
EMBEDDINGS_DIR = BASE_DIR / "dataset" / "embeddings"
RUN_LOG = BASE_DIR / "ingestion.log"
FAILURE_LOG = BASE_DIR / "ingestion_failures.log"

for folder in (PDF_DIR, EXTRACTED_DIR, CHUNKS_DIR, EMBEDDINGS_DIR):
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=RUN_LOG,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8"
)


# =========================================================
# TEXT CLEANING / CHUNKING
# =========================================================
def clean_text(text):
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(
        char for char in text
        if char in "\n\t" or ord(char) >= 32
    ).strip()


def chunk_page(text, paper_id, page_number):
    """Split one page into 1500-character chunks with 200 overlap."""
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    chunk_number = 1
    step = CHUNK_SIZE - CHUNK_OVERLAP

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "chunk_id": f"{paper_id}_page{page_number}_chunk{chunk_number}",
                "paper_id": paper_id,
                "page_number": int(page_number),
                "chunk_number": int(chunk_number),
                "chunk_text": chunk_text
            })
            chunk_number += 1

        if end >= len(text):
            break
        start += step

    return chunks


# =========================================================
# PDF EXTRACTION
# =========================================================
def extract_pdf(pdf_path):
    page_texts = []

    with pymupdf.open(str(pdf_path)) as document:
        if document.is_encrypted:
            raise ValueError("PDF is encrypted or password-protected.")

        page_count = len(document)
        for page_index, page in enumerate(document, start=1):
            text = clean_text(page.get_text("text", sort=True))
            page_texts.append({
                "page_number": page_index,
                "text": text
            })

    if not page_texts:
        raise ValueError("PDF has no pages.")

    if not any(item["text"] for item in page_texts):
        raise ValueError(
            "No selectable text found. This may be a scanned PDF; OCR is not "
            "enabled in this script."
        )

    return page_texts, page_count


def save_extracted_text(paper_id, page_texts):
    path = EXTRACTED_DIR / f"{paper_id}.txt"
    with path.open("w", encoding="utf-8") as file:
        for page in page_texts:
            file.write(f"\n--- PAGE {page['page_number']} ---\n")
            file.write(page["text"])
            file.write("\n")
    return path


def save_json(path, data):
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    temp_path.replace(path)


# =========================================================
# EMBEDDINGS
# =========================================================
def add_embeddings(chunks, model):
    if not chunks:
        return chunks

    texts = [chunk["chunk_text"] for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    for chunk, vector in zip(chunks, vectors):
        values = [float(value) for value in vector.tolist()]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid embedding generated for {chunk['chunk_id']}")
        chunk["embedding"] = values

    return chunks


# =========================================================
# DATABASE HELPERS
# =========================================================
def connect_database():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def paper_id_from_pdf(pdf_path):
    # Existing project convention: PDF filename stem is paper_id.
    # Example: 1805.01956v1.pdf -> 1805.01956v1
    paper_id = pdf_path.stem.strip()
    if not paper_id or "\x00" in paper_id:
        raise ValueError("Invalid paper ID derived from PDF filename.")
    return paper_id


def source_from_paper_id(paper_id):
    if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", paper_id):
        return "arXiv"
    return "PDF"


def insert_paper_and_chunks(conn, paper_id, pdf_path, page_count, chunks):
    new_paper = False
    inserted = 0
    duplicates = 0

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO papers (paper_id, source, pdf_path, total_pages)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (paper_id) DO NOTHING;
                """,
                (
                    paper_id,
                    source_from_paper_id(paper_id),
                    str(Path("dataset") / "papers" / pdf_path.name),
                    page_count
                )
            )
            new_paper = cursor.rowcount == 1

            for chunk in chunks:
                vector_text = "[" + ",".join(
                    format(value, ".12g") for value in chunk["embedding"]
                ) + "]"

                cursor.execute(
                    """
                    INSERT INTO paper_chunks
                        (chunk_id, paper_id, page_number, chunk_number,
                         chunk_text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (chunk_id) DO NOTHING;
                    """,
                    (
                        chunk["chunk_id"],
                        paper_id,
                        int(chunk["page_number"]),
                        int(chunk["chunk_number"]),
                        clean_text(chunk["chunk_text"]),
                        vector_text
                    )
                )

                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    duplicates += 1

        conn.commit()
        return new_paper, inserted, duplicates

    except Exception:
        conn.rollback()
        raise


def already_ingested(conn, paper_id):
    with conn.cursor() as cursor:
        cursor.execute("SELECT 1 FROM papers WHERE paper_id = %s;", (paper_id,))
        paper_exists = cursor.fetchone() is not None

        cursor.execute(
            "SELECT COUNT(*) FROM paper_chunks WHERE paper_id = %s;",
            (paper_id,)
        )
        chunk_count = int(cursor.fetchone()[0])

    return paper_exists and chunk_count > 0, chunk_count


# =========================================================
# PROCESS ONE PDF END TO END
# =========================================================
def process_pdf(pdf_path, conn, model):
    paper_id = paper_id_from_pdf(pdf_path)
    done, existing_count = already_ingested(conn, paper_id)

    # Do not waste time re-embedding papers already in the database.
    if done:
        return {
            "status": "SKIPPED_ALREADY_INGESTED",
            "paper_id": paper_id,
            "chunks": existing_count,
            "inserted": 0,
            "duplicates": 0,
            "new_paper": False,
            "pages": None
        }

    page_texts, page_count = extract_pdf(pdf_path)
    save_extracted_text(paper_id, page_texts)

    chunks = []
    for page in page_texts:
        chunks.extend(
            chunk_page(page["text"], paper_id, page["page_number"])
        )

    if not chunks:
        raise ValueError("No non-empty chunks were created from this PDF.")

    save_json(CHUNKS_DIR / f"{paper_id}.json", chunks)

    chunks = add_embeddings(chunks, model)
    save_json(EMBEDDINGS_DIR / f"{paper_id}.json", chunks)

    new_paper, inserted, duplicates = insert_paper_and_chunks(
        conn, paper_id, pdf_path, page_count, chunks
    )

    return {
        "status": "SUCCESS",
        "paper_id": paper_id,
        "chunks": len(chunks),
        "inserted": inserted,
        "duplicates": duplicates,
        "new_paper": new_paper,
        "pages": page_count
    }


# =========================================================
# MAIN: PROCESS ALL PDFs IN dataset/papers
# =========================================================
def main():
    print("\n==============================================")
    print("AI Research Paper Assistant - PDF Ingestion")
    print("==============================================")
    print("PDF folder:", PDF_DIR)
    print("Chunk size:", CHUNK_SIZE, "| Overlap:", CHUNK_OVERLAP)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in dataset/papers.")
        print("Place PDF files in that folder and run again.")
        return

    print("PDF files discovered:", len(pdf_files))
    print("\nLoading embedding model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    print("Embedding model loaded.")

    conn = None
    processed = 0
    skipped = 0
    failed = 0
    new_papers = 0
    new_chunks = 0
    duplicate_chunks = 0

    FAILURE_LOG.write_text(
        "PDF ingestion failures for this run\n\n",
        encoding="utf-8"
    )

    try:
        print("\nConnecting to PostgreSQL...")
        conn = connect_database()
        print("Database connected.")

        with FAILURE_LOG.open("a", encoding="utf-8") as failure_file:
            for index, pdf_path in enumerate(pdf_files, start=1):
                print("\n" + "-" * 60)
                print(f"[{index}/{len(pdf_files)}] {pdf_path.name}")

                try:
                    result = process_pdf(pdf_path, conn, model)

                    if result["status"] == "SKIPPED_ALREADY_INGESTED":
                        skipped += 1
                        print("SKIPPED: already in database")
                        print("Existing chunks:", result["chunks"])
                        continue

                    processed += 1
                    new_papers += int(result["new_paper"])
                    new_chunks += result["inserted"]
                    duplicate_chunks += result["duplicates"]

                    print("Status:", result["status"])
                    print("Paper ID:", result["paper_id"])
                    print("Pages:", result["pages"])
                    print("Chunks generated:", result["chunks"])
                    print("New chunks inserted:", result["inserted"])
                    print("Duplicate chunk IDs skipped:", result["duplicates"])

                except Exception as error:
                    failed += 1
                    conn.rollback()
                    logging.exception("Failed processing %s", pdf_path.name)
                    failure_file.write(
                        f"PDF: {pdf_path.name}\n"
                        f"Error: {type(error).__name__}: {error}\n"
                        f"{'-' * 60}\n"
                    )
                    failure_file.flush()
                    print("FAILED:", type(error).__name__, str(error))
                    print("Details saved in ingestion_failures.log")

    except Exception as error:
        print("Database connection error:", type(error).__name__, str(error))

    finally:
        if conn is not None:
            conn.close()
            print("\nDatabase connection closed.")

    print("\n==============================================")
    print("INGESTION SUMMARY")
    print("==============================================")
    print("PDFs discovered:", len(pdf_files))
    print("New PDFs processed:", processed)
    print("Already-ingested PDFs skipped:", skipped)
    print("PDFs failed:", failed)
    print("New paper rows inserted:", new_papers)
    print("New chunks inserted:", new_chunks)
    print("Duplicate chunks skipped:", duplicate_chunks)
    print("Extracted text:", EXTRACTED_DIR)
    print("Chunk JSON files:", CHUNKS_DIR)
    print("Embedding JSON files:", EMBEDDINGS_DIR)
    print("Run log:", RUN_LOG)
    print("Failure log:", FAILURE_LOG)
    print("==============================================")
    print(
        "Note: image-only/scanned PDFs may need OCR. "
        "Review ingestion_failures.log for any failures."
    )


if __name__ == "__main__":
    main()
