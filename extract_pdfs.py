import fitz
import os

# -----------------------------
# FOLDERS
# -----------------------------

PDF_FOLDER = "dataset/papers"
OUTPUT_FOLDER = "dataset/extracted_text"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# -----------------------------
# PROCESS ALL PDFs
# -----------------------------

pdf_files = [
    file for file in os.listdir(PDF_FOLDER)
    if file.lower().endswith(".pdf")
]

print("Total PDFs found:", len(pdf_files))


for index, filename in enumerate(pdf_files, start=1):

    pdf_path = os.path.join(PDF_FOLDER, filename)

    paper_id = os.path.splitext(filename)[0]

    output_path = os.path.join(
        OUTPUT_FOLDER,
        paper_id + ".txt"
    )

    print(f"\n[{index}/{len(pdf_files)}] Processing: {filename}")

    try:

        doc = fitz.open(pdf_path)

        with open(output_path, "w", encoding="utf-8") as output:

            output.write(f"PAPER ID: {paper_id}\n")
            output.write(f"NUMBER OF PAGES: {len(doc)}\n\n")

            for page_number, page in enumerate(doc, start=1):

                text = page.get_text("text")

                output.write("\n")
                output.write("=" * 80 + "\n")
                output.write(f"PAGE {page_number}\n")
                output.write("=" * 80 + "\n\n")

                output.write(text)

        doc.close()

        print("✓ Extracted successfully")

    except Exception as e:

        print("✗ Error:", e)


print("\n================================")
print("TEXT EXTRACTION COMPLETED")
print("================================")