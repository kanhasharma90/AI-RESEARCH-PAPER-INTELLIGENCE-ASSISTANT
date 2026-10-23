import os
import json
from sentence_transformers import SentenceTransformer


# ==========================================
# SETTINGS
# ==========================================

INPUT_FOLDER = "dataset/chunks"
OUTPUT_FOLDER = "dataset/embeddings"

MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully!")


# ==========================================
# FIND CHUNK FILES
# ==========================================

json_files = [
    file
    for file in os.listdir(INPUT_FOLDER)
    if file.lower().endswith(".json")
]

print("Total papers:", len(json_files))


# ==========================================
# PROCESS EACH PAPER
# ==========================================

total_chunks = 0

for index, filename in enumerate(json_files, start=1):

    print(f"\n[{index}/{len(json_files)}] Processing {filename}")

    input_path = os.path.join(
        INPUT_FOLDER,
        filename
    )

    output_path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )

    try:

        # Read chunks
        with open(
            input_path,
            "r",
            encoding="utf-8"
        ) as file:

            chunks = json.load(file)


        # Extract text from every chunk
        texts = [
            chunk["chunk_text"]
            for chunk in chunks
        ]


        # Generate embeddings
        embeddings = model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True
        )


        # Add embedding to each chunk
        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            chunk["embedding"] = embedding.tolist()


        # Save result
        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                chunks,
                file,
                indent=4,
                ensure_ascii=False
            )


        total_chunks += len(chunks)

        print(
            "✓ Embeddings generated:",
            len(chunks)
        )


    except Exception as e:

        print("✗ ERROR:", e)


# ==========================================
# FINAL RESULT
# ==========================================

print("\n======================================")
print("EMBEDDING GENERATION COMPLETED")
print("======================================")

print("Total chunks embedded:", total_chunks)