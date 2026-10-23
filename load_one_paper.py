import json
import psycopg2
import os

# -----------------------------
# DATABASE CONFIGURATION
# -----------------------------
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "research_paper_assistant"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# -----------------------------
# JSON FILE
# -----------------------------
JSON_FILE = "dataset/embeddings/1805.01956v1.json"

# -----------------------------
# CONNECT TO DATABASE
# -----------------------------
print("Connecting to PostgreSQL...")

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

cursor = conn.cursor()

print("Connected successfully!")

# -----------------------------
# LOAD JSON
# -----------------------------
with open(JSON_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

print("Chunks found:", len(chunks))

# -----------------------------
# PAPER INFORMATION
# -----------------------------
paper_id = chunks[0]["paper_id"]
total_pages = max(chunk["page_number"] for chunk in chunks)

# Insert paper information
cursor.execute("""
    INSERT INTO papers
        (paper_id, source, pdf_path, total_pages)
    VALUES
        (%s, %s, %s, %s)
    ON CONFLICT (paper_id)
    DO NOTHING;
""", (
    paper_id,
    "arXiv",
    f"dataset/papers/{paper_id}.pdf",
    total_pages
))

# -----------------------------
# INSERT CHUNKS + EMBEDDINGS
# -----------------------------
inserted = 0

for chunk in chunks:

    embedding = chunk["embedding"]

    # Convert Python list to pgvector format
    embedding_string = "[" + ",".join(map(str, embedding)) + "]"

    cursor.execute("""
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
    """, (
        chunk["chunk_id"],
        chunk["paper_id"],
        chunk["page_number"],
        chunk["chunk_id"].split("_chunk")[-1],
        chunk["chunk_text"],
        embedding_string
    ))

    inserted += 1

# -----------------------------
# SAVE CHANGES
# -----------------------------
conn.commit()

print("--------------------------------")
print("Paper ID:", paper_id)
print("Total chunks:", len(chunks))
print("Chunks processed:", inserted)
print("Data inserted successfully!")
print("--------------------------------")

cursor.close()
conn.close()

print("Database connection closed.")