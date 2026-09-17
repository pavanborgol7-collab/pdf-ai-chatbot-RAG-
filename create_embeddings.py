import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from pypdf import PdfReader


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Split document text into overlapping chunks for semantic search."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def create_embeddings(chunks: list[str]) -> list[list[float]]:
    """Ask the free, local Ollama server for one vector per text chunk."""
    request_body = json.dumps({"model": EMBEDDING_MODEL, "input": chunks}).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except URLError:
        print("Could not connect to Ollama.")
        print("Open the Ollama app, then verify with: ollama list")
        raise SystemExit(1)

    return response_data["embeddings"]


pdf_path = Path("Data/sample.pdf")
output_path = Path("Data/embeddings.json")

if not pdf_path.exists():
    print(f"PDF not found: {pdf_path}")
    raise SystemExit(1)

reader = PdfReader(pdf_path)
document_text = "\n".join(page.extract_text() or "" for page in reader.pages)
chunks = split_text(document_text)

embeddings = create_embeddings(chunks)

records = [
    {"chunk_number": number, "text": chunk, "embedding": embedding}
    for number, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1)
]

output_path.write_text(json.dumps(records), encoding="utf-8")

print(f"Created local embeddings for {len(records)} chunks.")
print(f"Saved vectors to: {output_path}")
print(f"Each vector has {len(records[0]['embedding'])} numbers.")
