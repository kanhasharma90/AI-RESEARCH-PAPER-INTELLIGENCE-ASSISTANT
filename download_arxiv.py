import arxiv
import requests
import os
import time

# -----------------------------
# SETTINGS
# -----------------------------

DOWNLOAD_LIMIT = 500

DOWNLOAD_FOLDER = "dataset/papers"

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

# Search AI + ML papers
search = arxiv.Search(
    query="cat:cs.AI OR cat:cs.LG",
    max_results=DOWNLOAD_LIMIT,
    sort_by=arxiv.SortCriterion.Relevance
)

client = arxiv.Client(
    page_size=10,
    delay_seconds=3,
    num_retries=3
)

# -----------------------------
# DOWNLOAD
# -----------------------------

count = 0

for result in client.results(search):

    count += 1

    paper_id = result.get_short_id()

    print(f"\n[{count}/{DOWNLOAD_LIMIT}]")
    print("Title:", result.title)
    print("arXiv ID:", paper_id)

    # Remove version number if present
    safe_id = paper_id.replace("/", "_")

    pdf_path = os.path.join(
        DOWNLOAD_FOLDER,
        safe_id + ".pdf"
    )

    # Skip if already downloaded
    if os.path.exists(pdf_path):
        print("Already downloaded.")
        continue

    try:

        response = requests.get(
            result.pdf_url,
            timeout=60
        )

        response.raise_for_status()

        with open(pdf_path, "wb") as f:
            f.write(response.content)

        print("Downloaded:", pdf_path)

    except Exception as e:

        print("Download failed:", e)

    time.sleep(3)

print("\nDONE!")
print("Total papers processed:", count)