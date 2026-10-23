import json
import re
import math
import psycopg2
from pathlib import Path


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

# This script expects the following project structure:
# project_folder/
#   load_one_paper.py
#   dataset/
#     embeddings/
#       paper_id_1.json
#       paper_id_2.json
#     papers/
#       paper_id_1.pdf
#       paper_id_2.pdf

BASE_DIR = Path(__file__).resolve().parent
JSON_DIR = BASE_DIR / "dataset" / "embeddings"
PDF_DIR = BASE_DIR / "dataset" / "papers"


# ==========================================
# 3. VALIDATE AND PREPARE ONE JSON FILE
# ==========================================

def load_and_validate_json(json_path):
    """
    Load one paper's embedding JSON and validate it before inserting
    anything into the database.
    """
    with json_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not isinstance(chunks, list):
        raise ValueError("JSON root must be a list of chunks.")

    if not chunks:
        raise ValueError("JSON file contains no chunks.")

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk #{index} is not a JSON object.")

        required_fields = {
            "chunk_id",
            "paper_id",
            "page_number",
            "chunk_text",
            "embedding"
        }
        missing_fields = required_fields - set(chunk.keys())

        if missing_fields:
            raise ValueError(
                f"Chunk #{index} is missing fields: "
                + ", ".join(sorted(missing_fields))
            )

    paper_id = str(chunks[0]["paper_id"]).strip()
    if not paper_id:
        raise ValueError("paper_id is empty.")

    # Do not accidentally insert chunks from different papers
    # while registering only the first paper ID.
    mismatched_ids = [
        index for index, chunk in enumerate(chunks, start=1)
        if str(chunk["paper_id"]).strip() != paper_id
    ]
    if mismatched_ids:
        raise ValueError(
            f"JSON contains chunks belonging to multiple paper IDs. "
            f"First paper ID is {paper_id}."
        )

    prepared_chunks = []
    embedding_dimension = None
    page_numbers = []

    for index, chunk in enumerate(chunks, start=1):
        try:
            page_number = int(chunk["page_number"])
        except (TypeError, ValueError):
            raise ValueError(
                f"Chunk #{index} has an invalid page_number."
            )

        if page_number < 1:
            raise ValueError(
                f"Chunk #{index} has page_number less than 1."
            )

        chunk_id = str(chunk["chunk_id"]).strip()
        if not chunk_id:
            raise ValueError(f"Chunk #{index} has an empty chunk_id.")

        chunk_text = str(chunk["chunk_text"])
        if not chunk_text.strip():
            raise ValueError(f"Chunk #{index} has empty chunk_text.")

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

        if not all(
            math.isfinite(value)
            for value in embedding
        ):
            raise ValueError(
                f"Chunk #{index} contains a non-finite embedding value."
            )

        if embedding_dimension is None:
            embedding_dimension = len(embedding)
        elif len(embedding) != embedding_dimension:
            raise ValueError(
                f"Embedding dimensions are inconsistent in {json_path.name}."
            )

        # Prefer a provided chunk_number; otherwise extract the suffix
        # from IDs such as 1805.01956v1_page2_chunk3.
        if chunk.get("chunk_number") is not None:
            try:
                chunk_number = int(chunk["chunk_number"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Chunk #{index} has an invalid chunk_number."
                )
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

    return paper_id, max(page_numbers), prepared_chunks


# ==========================================
# 4. INSERT ONE PAPER IN A TRANSACTION
# ==========================================

def insert_one_paper(cursor, conn, json_path):
    """
    Import one JSON file. Commit the entire paper only if all its chunks
    are processed successfully. Existing paper IDs/chunk IDs are skipped.
    """
    paper_id, total_pages, chunks = load_and_validate_json(json_path)

    pdf_path = f"dataset/papers/{paper_id}.pdf"

    paper_pdf_path = PDF_DIR / f"{paper_id}.pdf"
    if not paper_pdf_path.exists():
        print(
            f"Note: PDF file not found at {paper_pdf_path}. "
            "The JSON embeddings will still be loaded."
        )

    try:
        cursor.execute(
            """
            INSERT INTO papers
                (paper_id, source, pdf_path, total_pages)
            VALUES
                (%s, %s, %s, %s)
            ON CONFLICT (paper_id)
            DO NOTHING;
            """,
            (paper_id, "arXiv", pdf_path, total_pages)
        )
        paper_inserted = cursor.rowcount == 1

        inserted_chunks = 0
        duplicate_chunks = 0

        for chunk in chunks:
            embedding_string = (
                "[" + ",".join(
                    format(value, ".12g")
                    for value in chunk["embedding"]
                ) + "]"
            )

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
                VALUES
                    (%s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id)
                DO NOTHING;
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

            # rowcount is 1 for an inserted row and 0 when ON CONFLICT
            # DO NOTHING skipped a duplicate.
            if cursor.rowcount == 1:
                inserted_chunks += 1
            else:
                duplicate_chunks += 1

        # This paper is committed only after all its chunks succeed.
        conn.commit()

        return {
            "paper_id": paper_id,
            "total_chunks": len(chunks),
            "paper_inserted": paper_inserted,
            "inserted_chunks": inserted_chunks,
            "duplicate_chunks": duplicate_chunks
        }

    except Exception:
        # Roll back this paper's insertions so a failed JSON does not leave
        # a partially inserted paper in the database.
        conn.rollback()
        raise


# ==========================================
# 5. MAIN: LOAD ALL PAPER JSON FILES
# ==========================================

def main():
    print("\n==========================================")
    print(" Multi-Paper Embedding Loader")
    print("==========================================")
    print("JSON directory:", JSON_DIR)

    if not JSON_DIR.exists():
        print("\nERROR: Embeddings directory does not exist.")
        print("Create this folder and put paper JSON files in it:")
        print(JSON_DIR)
        return

    json_files = sorted(JSON_DIR.glob("*.json"))

    if not json_files:
        print("\nNo JSON files found in the embeddings directory.")
        print("Expected files like: dataset/embeddings/1805.01956v1.json")
        return

    print(f"\nJSON files found: {len(json_files)}")
    print("Connecting to PostgreSQL...")

    conn = None
    cursor = None

    successful_files = 0
    failed_files = 0
    total_chunks_seen = 0
    papers_inserted = 0
    chunks_inserted = 0
    duplicate_chunks = 0

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

        for json_path in json_files:
            print("\n" + "-" * 60)
            print("Processing:", json_path.name)

            try:
                summary = insert_one_paper(cursor, conn, json_path)

                successful_files += 1
                total_chunks_seen += summary["total_chunks"]
                papers_inserted += int(summary["paper_inserted"])
                chunks_inserted += summary["inserted_chunks"]
                duplicate_chunks += summary["duplicate_chunks"]

                print("Paper ID:", summary["paper_id"])
                print("Chunks in JSON:", summary["total_chunks"])
                print("New paper rows inserted:", int(summary["paper_inserted"]))
                print("New chunks inserted:", summary["inserted_chunks"])
                print("Duplicate chunks skipped:", summary["duplicate_chunks"])
                print("Status: SUCCESS")

            except Exception as error:
                failed_files += 1
                # The insert function rolls back database insert errors.
                # Also roll back here for errors raised during validation.
                conn.rollback()
                print("Status: FAILED")
                print("Reason:", error)

    except Exception as error:
        print("\nDatabase connection or processing error:")
        print(error)

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
    print("==========================================")


if __name__ == "__main__":
    main()
