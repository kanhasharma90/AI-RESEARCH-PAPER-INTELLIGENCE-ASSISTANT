import json
import re
import math
import psycopg2
from pathlib import Path
from datetime import datetime


# ==========================================
# 1. DATABASE CONFIGURATION
# ==========================================

DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "research_paper_assistant"
DB_USER = "postgres"
DB_PASSWORD = "postgres"


# ==========================================
# 2. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "dataset" / "embeddings"
PDF_DIR = BASE_DIR / "dataset" / "papers"
ERROR_LOG = BASE_DIR / "failed_imports.log"


# ==========================================
# 3. LOAD AND VALIDATE ONE JSON FILE
# ==========================================

def load_and_validate_json(json_path):
    with json_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("JSON root must be a list of chunks.")

    if not chunks:
        raise ValueError("JSON file contains no chunks.")

    required_fields = {
        "chunk_id",
        "paper_id",
        "page_number",
        "chunk_text",
        "embedding"
    }

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk #{index} is not a JSON object.")

        missing = required_fields - set(chunk.keys())
        if missing:
            raise ValueError(
                f"Chunk #{index} is missing fields: "
                + ", ".join(sorted(missing))
            )

    paper_id = str(chunks[0]["paper_id"]).strip()
    if not paper_id:
        raise ValueError("paper_id is empty.")

    if any(str(chunk["paper_id"]).strip() != paper_id for chunk in chunks):
        raise ValueError("This JSON contains more than one paper_id.")

    prepared_chunks = []
    page_numbers = []
    embedding_dimension = None
    removed_null_characters = 0

    for index, chunk in enumerate(chunks, start=1):
        chunk_id = str(chunk["chunk_id"]).strip()
        if not chunk_id:
            raise ValueError(f"Chunk #{index} has an empty chunk_id.")

        try:
            page_number = int(chunk["page_number"])
        except (TypeError, ValueError):
            raise ValueError(f"Chunk #{index} has an invalid page_number.")

        if page_number < 1:
            raise ValueError(f"Chunk #{index} has page_number less than 1.")

        chunk_text = str(chunk["chunk_text"])
        null_count = chunk_text.count("\x00")
        if null_count:
            chunk_text = chunk_text.replace("\x00", "")
            removed_null_characters += null_count

        if not chunk_text.strip():
            raise ValueError(f"Chunk #{index} has empty chunk_text after cleaning.")

        raw_embedding = chunk["embedding"]
        if not isinstance(raw_embedding, list) or not raw_embedding:
            raise ValueError(
                f"Chunk #{index} must contain a non-empty embedding list."
            )

        try:
            embedding = [float(value) for value in raw_embedding]
        except (TypeError, ValueError):
            raise ValueError(
                f"Chunk #{index} contains a non-numeric embedding value."
            )

        if not all(math.isfinite(value) for value in embedding):
            raise ValueError(
                f"Chunk #{index} contains a non-finite embedding value."
            )

        if embedding_dimension is None:
            embedding_dimension = len(embedding)
        elif len(embedding) != embedding_dimension:
            raise ValueError(
                f"Embedding dimensions are inconsistent in {json_path.name}."
            )

        if chunk.get("chunk_number") is not None:
            try:
                chunk_number = int(chunk["chunk_number"])
            except (TypeError, ValueError):
                raise ValueError(f"Chunk #{index} has invalid chunk_number.")
        else:
            match = re.search(r"_chunk(\d+)$", chunk_id)
            chunk_number = int(match.group(1)) if match else index

        prepared_chunks.append({
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "page_number": page_number,
            "chunk_number": chunk_number,
            "chunk_text": chunk_text,
            "embedding": embedding
        })
        page_numbers.append(page_number)

    return paper_id, max(page_numbers), prepared_chunks, removed_null_characters


# ==========================================
# 4. IMPORT ONE PAPER TRANSACTIONALLY
# ==========================================

def insert_one_paper(cursor, conn, json_path):
    paper_id, total_pages, chunks, removed_null_characters = (
        load_and_validate_json(json_path)
    )

    pdf_path = f"dataset/papers/{paper_id}.pdf"
    expected_pdf = PDF_DIR / f"{paper_id}.pdf"

    if not expected_pdf.exists():
        print(
            f"Note: matching PDF not found at {expected_pdf}. "
            "Loading the JSON embeddings anyway."
        )

    try:
        cursor.execute(
            """
            INSERT INTO papers
                (paper_id, source, pdf_path, total_pages)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (paper_id) DO NOTHING;
            """,
            (paper_id, "arXiv", pdf_path, total_pages)
        )
        paper_inserted = cursor.rowcount == 1

        inserted_chunks = 0
        duplicate_chunks = 0

        for chunk in chunks:
            embedding_string = "[" + ",".join(
                format(value, ".12g") for value in chunk["embedding"]
            ) + "]"

            cursor.execute(
                """
                INSERT INTO paper_chunks
                    (
                        chunk_id,
                        paper_id,
                        page_number,
                        chunk_number,
                        chunk_text,
                        embedding
                    )
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id) DO NOTHING;
                """,
                (
                    chunk["chunk_id"],
                    chunk["paper_id"],
                    chunk["page_number"],
                    chunk["chunk_number"],
                    chunk["chunk_text"],
                    embedding_string
                )
            )

            if cursor.rowcount == 1:
                inserted_chunks += 1
            else:
                duplicate_chunks += 1

        conn.commit()

        return {
            "paper_id": paper_id,
            "total_chunks": len(chunks),
            "paper_inserted": paper_inserted,
            "inserted_chunks": inserted_chunks,
            "duplicate_chunks": duplicate_chunks,
            "removed_null_characters": removed_null_characters
        }

    except Exception:
        conn.rollback()
        raise


# ==========================================
# 5. LOG FAILURES
# ==========================================

def log_failure(filename, error):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with ERROR_LOG.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"[{timestamp}] File: {filename}\n"
            f"Error: {type(error).__name__}: {error}\n"
            f"{'-' * 70}\n"
        )


# ==========================================
# 6. MAIN: LOAD ALL EMBEDDING JSON FILES
# ==========================================

def main():
    print("\n==========================================")
    print(" Multi-Paper Embedding Loader")
    print("==========================================")
    print("Embeddings directory:", JSON_DIR)
    print("NULL characters in chunk text will be removed before database insert.")

    if not JSON_DIR.exists():
        print("\nERROR: Embeddings directory does not exist.")
        print("Expected:", JSON_DIR)
        return

    json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        print("\nNo JSON files found in the embeddings directory.")
        return

    print("JSON files found:", len(json_files))
    print("Connecting to PostgreSQL...")

    conn = None
    cursor = None

    successful_files = 0
    failed_files = 0
    total_chunks_seen = 0
    papers_inserted = 0
    chunks_inserted = 0
    duplicate_chunks = 0
    null_characters_removed = 0

    # Start a fresh log for this run so it contains only current failures.
    ERROR_LOG.write_text(
        f"Import failure log - {datetime.now().isoformat(timespec='seconds')}\n\n",
        encoding="utf-8"
    )

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        print("Connected successfully!")

        for index, json_path in enumerate(json_files, start=1):
            print("\n" + "-" * 60)
            print(f"[{index}/{len(json_files)}] Processing: {json_path.name}")

            try:
                summary = insert_one_paper(cursor, conn, json_path)

                successful_files += 1
                total_chunks_seen += summary["total_chunks"]
                papers_inserted += int(summary["paper_inserted"])
                chunks_inserted += summary["inserted_chunks"]
                duplicate_chunks += summary["duplicate_chunks"]
                null_characters_removed += summary["removed_null_characters"]

                print("Paper ID:", summary["paper_id"])
                print("Chunks in JSON:", summary["total_chunks"])
                print("New paper rows inserted:", int(summary["paper_inserted"]))
                print("New chunks inserted:", summary["inserted_chunks"])
                print("Duplicate chunks skipped:", summary["duplicate_chunks"])

                if summary["removed_null_characters"]:
                    print(
                        "NULL characters removed:",
                        summary["removed_null_characters"]
                    )

                print("Status: SUCCESS")

            except Exception as error:
                failed_files += 1
                conn.rollback()
                log_failure(json_path.name, error)
                print("Status: FAILED")
                print(f"Reason: {type(error).__name__}: {error}")
                print("Failure recorded in:", ERROR_LOG.name)

    except Exception as error:
        print("\nDatabase connection or processing error:")
        print(f"{type(error).__name__}: {error}")

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
            print("\nDatabase connection closed.")

    print("\n==========================================")
    print("FINAL IMPORT SUMMARY")
    print("==========================================")
    print("JSON files discovered:", len(json_files))
    print("Files imported successfully:", successful_files)
    print("Files failed:", failed_files)
    print("Chunks across successful files:", total_chunks_seen)
    print("New paper rows inserted:", papers_inserted)
    print("New chunks inserted:", chunks_inserted)
    print("Duplicate chunks skipped:", duplicate_chunks)
    print("NULL characters removed:", null_characters_removed)
    print("Failure log:", ERROR_LOG)
    print("==========================================")


if __name__ == "__main__":
    main()
