from pathlib import Path

from pypdf import PdfReader


# This is the PDF we will read. Add a PDF named "sample.pdf" to the Data folder.
pdf_path = Path("Data/sample.pdf")

if not pdf_path.exists():
    print(f"PDF not found: {pdf_path}")
    print("Add a PDF named 'sample.pdf' to the Data folder, then run this file again.")
    raise SystemExit(1)

reader = PdfReader(pdf_path)

print(f"Number of pages: {len(reader.pages)}\n")

# We combine text from every page. This combined text will be used by RAG later.
full_text = ""

for page_number, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""

    print(f"--- Page {page_number} ---")
    print(page_text[:500])
    print()

    full_text += page_text + "\n"

print(f"Total characters extracted: {len(full_text)}")
