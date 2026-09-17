import json
import math
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import streamlit as st


EMBEDDING_URL = "http://localhost:11434/api/embed"
GENERATE_URL = "http://localhost:11434/api/generate"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
EMBEDDINGS_PATH = Path("Data/embeddings.json")
NUMBER_OF_SOURCES = 3


def ollama_post(url: str, body: dict) -> dict:
    """Send one request to the Ollama server running on this computer."""
    request = Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def create_embedding(text: str) -> list[float]:
    response = ollama_post(EMBEDDING_URL, {"model": EMBEDDING_MODEL, "input": text})
    return response["embeddings"][0]


def cosine_similarity(first: list[float], second: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(first, second))
    first_length = math.sqrt(sum(a * a for a in first))
    second_length = math.sqrt(sum(b * b for b in second))
    return dot_product / (first_length * second_length)


@st.cache_data(show_spinner=False)
def load_records() -> list[dict]:
    return json.loads(EMBEDDINGS_PATH.read_text(encoding="utf-8"))


def retrieve_sources(question: str) -> list[dict]:
    records = load_records()
    question_embedding = create_embedding(question)

    for record in records:
        record["similarity"] = cosine_similarity(question_embedding, record["embedding"])

    return sorted(records, key=lambda record: record["similarity"], reverse=True)[:NUMBER_OF_SOURCES]


def answer_question(question: str, sources: list[dict]) -> str:
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
    response = ollama_post(GENERATE_URL, {"model": LLM_MODEL, "prompt": prompt, "stream": False})
    return response["response"].strip()


st.set_page_config(page_title="PDF RAG Study Dashboard", page_icon="📚", layout="wide")

st.title("📚 PDF RAG Study Dashboard")
st.caption("Ask questions about your Operating System notes. Everything runs locally with Ollama.")

if not EMBEDDINGS_PATH.exists():
    st.error("No document index was found. Run `python create_embeddings.py` first.")
    st.stop()

try:
    indexed_chunks = len(load_records())
except (OSError, json.JSONDecodeError):
    st.error("The document index could not be read. Run `python create_embeddings.py` again.")
    st.stop()

metric_one, metric_two, metric_three = st.columns(3)
with metric_one:
    st.metric("Document", "OS Semester 1", help="Your indexed study PDF")
with metric_two:
    st.metric("Knowledge chunks", indexed_chunks, help="Text pieces available for semantic search")
with metric_three:
    st.metric("Privacy mode", "Local", help="No paid API calls are used")

with st.sidebar:
    st.title("📚 Study RAG")
    st.success("System ready")
    st.caption("Your study material is indexed and searchable.")
    st.divider()
    st.caption("EMBEDDING MODEL")
    st.code(EMBEDDING_MODEL)
    st.caption("ANSWER MODEL")
    st.code(LLM_MODEL)
    st.divider()
    with st.expander("How this works"):
        st.markdown(
            "1. Your question becomes a vector.\n"
            "2. The closest PDF chunks are retrieved.\n"
            "3. Ollama answers using those sources."
        )
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            with st.expander("View retrieved PDF sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"**Chunk {source['chunk_number']}** · similarity `{source['similarity']:.3f}`"
                    )
                    st.write(source["text"])
                    st.divider()

suggested_question = None
if not st.session_state.messages:
    st.markdown("#### Try a question")
    suggestion_one, suggestion_two, suggestion_three = st.columns(3)
    with suggestion_one:
        if st.button("What is CPU scheduling?", use_container_width=True):
            suggested_question = "What is CPU scheduling and why is it needed?"
    with suggestion_two:
        if st.button("Process vs thread", use_container_width=True):
            suggested_question = "What is the difference between a process and a thread?"
    with suggestion_three:
        if st.button("Explain IPC", use_container_width=True):
            suggested_question = "What is interprocess communication and why is it needed?"

question = st.chat_input("Ask anything about your Operating System notes") or suggested_question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant notes and writing an answer..."):
            try:
                sources = retrieve_sources(question)
                answer = answer_question(question, sources)
            except URLError:
                st.error("Could not connect to Ollama. Open the Ollama app and try again.")
                st.stop()
            except KeyError:
                st.error("A required Ollama model is missing. Run `ollama list` to verify your models.")
                st.stop()

        st.markdown(answer)
        with st.expander("View retrieved PDF sources"):
            for source in sources:
                st.markdown(
                    f"**Chunk {source['chunk_number']}** · similarity `{source['similarity']:.3f}`"
                )
                st.write(source["text"])
                st.divider()

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
