import json
import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


EMBEDDING_URL = "http://localhost:11434/api/embed"
GENERATE_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
EMBEDDINGS_PATH = Path("Data/embeddings.json")
NUMBER_OF_SOURCES = 3


def ollama_post(url: str, body: dict) -> dict:
    """Send one request to the local Ollama server."""
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError:
        print("Could not connect to Ollama. Open the Ollama app and try again.")
        raise SystemExit(1)


def create_embedding(text: str) -> list[float]:
    response = ollama_post(EMBEDDING_URL, {"model": EMBEDDING_MODEL, "input": text})
    return response["embeddings"][0]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(a * a for a in first))
    second_length = math.sqrt(sum(b * b for b in second))
    return dot_product / (first_length * second_length)


def retrieve_sources(question: str) -> list[dict]:
    records = json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))
    question_embedding = create_embedding(question)

    for record in records:
        record["similarity"] = cosine_similarity(question_embedding, record["embedding"])

    return sorted(records, key=lambda record: record["similarity"], reverse=True)[:NUMBER_OF_SOURCES]


if not EMBEDDINGS_PATH.exists():
    print("No embeddings found. Run 'python create_embeddings.py' first.")
    raise SystemExit(1)

question = input("Ask a question about the PDF: ").strip()
if not question:
    print("Please enter a question.")
    raise SystemExit(1)

sources = retrieve_sources(question)
context = "\n\n".join(
    f"[Source chunk {source['chunk_number']}]\n{source['text']}" for source in sources
)

prompt = f"""You are a helpful study assistant. Answer the question using only the source text below.
If the answer is not in the sources, say that the PDF does not provide enough information.
Use simple, clear language. Cite supporting source chunks in square brackets, for example [Chunk 16].

Source text:
{context}

Question: {question}

Answer:"""

response = ollama_post(
    GENERATE_URL,
    {"model": LLM_MODEL, "prompt": prompt, "stream": False},
)

print("\nAnswer:\n")
print(response["response"].strip())
print("\nRetrieved sources:")
for source in sources:
    print(f"- Chunk {source['chunk_number']} (similarity: {source['similarity']:.3f})")
