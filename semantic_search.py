from sentence_transformers import SentenceTransformer
import psycopg2

# -----------------------------
# DATABASE CONFIGURATION
# -----------------------------
DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "research_paper_assistant"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Model loaded!")

# -----------------------------
# USER QUESTION
# -----------------------------
question = input("\nEnter your question: ")

# Create question embedding
query_embedding = model.encode(
    question,
    normalize_embeddings=True
)

# Convert embedding to pgvector format
embedding_string = "[" + ",".join(map(str, query_embedding)) + "]"

# -----------------------------
# CONNECT TO DATABASE
# -----------------------------
conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD
)

cursor = conn.cursor()

# -----------------------------
# SEMANTIC SEARCH
# -----------------------------
cursor.execute("""
    SELECT
        chunk_id,
        paper_id,
        page_number,
        chunk_text,
        embedding <=> %s::vector AS distance
    FROM paper_chunks
    ORDER BY embedding <=> %s::vector
    LIMIT 5;
""", (
    embedding_string,
    embedding_string
))

results = cursor.fetchall()

# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\n========== SEARCH RESULTS ==========\n")

for rank, row in enumerate(results, start=1):

    chunk_id = row[0]
    paper_id = row[1]
    page_number = row[2]
    chunk_text = row[3]
    distance = row[4]

    print(f"Result #{rank}")
    print("Chunk ID:", chunk_id)
    print("Paper ID:", paper_id)
    print("Page:", page_number)
    print("Distance:", round(distance, 4))
    print("Text:", chunk_text[:500])
    print("-" * 80)

cursor.close()
conn.close()