
import psycopg2

DB_HOST = "localhost"
DB_PORT = 5433
DB_NAME = "research_paper_assistant"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

conn = None

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    with conn.cursor() as cursor:

        # Count unique papers
        cursor.execute("SELECT COUNT(*) FROM papers;")
        total_papers = cursor.fetchone()[0]

        # Count chunks
        cursor.execute("SELECT COUNT(*) FROM paper_chunks;")
        total_chunks = cursor.fetchone()[0]

        # Count chunks without embeddings
        cursor.execute("""
            SELECT COUNT(*)
            FROM paper_chunks
            WHERE embedding IS NULL;
        """)
        missing_embeddings = cursor.fetchone()[0]

        # Count distinct papers represented in chunks
        cursor.execute("""
            SELECT COUNT(DISTINCT paper_id)
            FROM paper_chunks;
        """)
        papers_with_chunks = cursor.fetchone()[0]

        print("\n========== DATABASE VERIFICATION ==========")
        print("Total papers:", total_papers)
        print("Total chunks:", total_chunks)
        print("Papers represented in chunks:", papers_with_chunks)
        print("Chunks without embeddings:", missing_embeddings)
        print("===========================================")

except Exception as error:
    print("Database verification failed:", error)

finally:
    if conn is not None:
        conn.close()
        print("Database connection closed.")
