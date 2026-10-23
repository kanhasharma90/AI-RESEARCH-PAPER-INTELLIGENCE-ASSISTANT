import fitz

pdf_path = "dataset/papers/1805.00779v1.pdf"

doc = fitz.open(pdf_path)

print("Number of pages:", len(doc))

for page_number, page in enumerate(doc, start=1):

    text = page.get_text("text")

    print("\n")
    print("=" * 80)
    print(f"PAGE {page_number}")
    print("=" * 80)

    print(text[:2000])

doc.close()