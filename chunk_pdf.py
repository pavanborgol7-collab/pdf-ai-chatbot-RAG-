from pathlib import Path
import sys

from pypdf import PdfReader


# PDFs may contain symbols such as bullets; UTF-8 lets the Windows terminal print them.
sys.stdout.reconfigure(encoding="utf-8")


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping pieces without breaking words when possible."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer a natural break so a word is not split in half.
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        # Move forward, while keeping a small shared section with the next chunk.
        start = end - overlap

    return chunks


pdf_path = Path("Data/sample.pdf")

if not pdf_path.exists():
    print(f"PDF not found: {pdf_path}")
    raise SystemExit(1)

reader = PdfReader(pdf_path)
full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

chunks = split_text(full_text)

print(f"Original text length: {len(full_text)} characters")
print(f"Number of chunks: {len(chunks)}")

for number, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {number} ({len(chunk)} characters) ---")
    print(chunk)
