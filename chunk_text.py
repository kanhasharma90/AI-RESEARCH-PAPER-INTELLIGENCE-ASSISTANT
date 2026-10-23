import os
import re
import json


# ==========================================
# SETTINGS
# ==========================================

INPUT_FOLDER = "dataset/extracted_text"
OUTPUT_FOLDER = "dataset/chunks"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# EXTRACT PAGE-WISE TEXT
# ==========================================

def extract_pages(text):

    pages = []

    # Find PAGE markers
    pattern = r"={20,}\s*PAGE\s+(\d+)\s*={20,}"

    matches = list(re.finditer(pattern, text))

    for i, match in enumerate(matches):

        page_number = int(match.group(1))

        start = match.end()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        page_text = text[start:end].strip()

        if page_text:
            pages.append({
                "page_number": page_number,
                "text": page_text
            })

    return pages


# ==========================================
# CREATE CHUNKS
# ==========================================

def create_chunks(page_text, page_number, paper_id):

    chunks = []

    start = 0
    chunk_number = 1

    text_length = len(page_text)

    while start < text_length:

        end = start + CHUNK_SIZE

        chunk_text = page_text[start:end].strip()

        if chunk_text:

            chunk_id = (
                f"{paper_id}_"
                f"page{page_number}_"
                f"chunk{chunk_number}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "page_number": page_number,
                "chunk_text": chunk_text
            })

        # Move forward while keeping overlap
        start = end - CHUNK_OVERLAP

        chunk_number += 1

    return chunks


# ==========================================
# PROCESS ONE FILE
# ==========================================

def process_file(filename):

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    paper_id = os.path.splitext(filename)[0]

    with open(
        input_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    pages = extract_pages(text)

    all_chunks = []

    for page in pages:

        page_chunks = create_chunks(
            page["text"],
            page["page_number"],
            paper_id
        )

        all_chunks.extend(page_chunks)

    return all_chunks


# ==========================================
# PROCESS ALL TEXT FILES
# ==========================================

text_files = [
    file
    for file in os.listdir(INPUT_FOLDER)
    if file.lower().endswith(".txt")
]

print("Total text files:", len(text_files))

total_chunks = 0


for index, filename in enumerate(
    text_files,
    start=1
):

    print(
        f"\n[{index}/{len(text_files)}] "
        f"Processing: {filename}"
    )

    try:

        chunks = process_file(filename)

        paper_id = os.path.splitext(filename)[0]

        output_file = os.path.join(
            OUTPUT_FOLDER,
            paper_id + ".json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                chunks,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            "Chunks created:",
            len(chunks)
        )

        total_chunks += len(chunks)

    except Exception as e:

        print(
            "ERROR:",
            e
        )


# ==========================================
# FINAL RESULT
# ==========================================

print("\n===================================")
print("CHUNKING COMPLETED")
print("===================================")

print(
    "Total chunks:",
    total_chunks
)