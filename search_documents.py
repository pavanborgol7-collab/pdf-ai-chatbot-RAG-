import json
import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


OLLAMA_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDINGS_PATH = Path("Data/embeddings.json")


def create_embedding(text: str) -> list[float]:
    """Convert one question into a vector using the same model as the PDF chunks."""
    request_body = json.dumps({"model": EMBEDDING_MODEL, "input": text}).encode("utf-8")
    request = Request(
        OLLAMA_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))["embeddings"][0]
    except URLError:
        print("Could not connect to Ollama. Open the Ollama app and try again.")
        raise SystemExit(1)


def cosine_similarity(first: list[float], second: list[float]) -> float:
    """Measure how similar two vectors point in the same direction."""
    dot_product = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(a * a for a in first))
    second_length = math.sqrt(sum(b * b for b in second))
    return dot_product / (first_length * second_length)


if not EMBEDDINGS_PATH.exists():
    print("No embeddings found. Run 'python create_embeddings.py' first.")
    raise SystemExit(1)

question = input("Ask a question about the PDF: ").strip()
if not question:
    print("Please enter a question.")
    raise SystemExit(1)

records = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
question_embedding = create_embedding(question)

for record in records:
    record["similarity"] = cosine_similarity(question_embedding, record["embedding"])

best_matches = sorted(records, key=lambda record: record["similarity"], reverse=True)[:3]

print("\nMost relevant PDF chunks:")
for match in best_matches:
    print(f"\n--- Chunk {match['chunk_number']} | score: {match['similarity']:.3f} ---")
    print(match["text"])
