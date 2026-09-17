# 🤖 PDF AI Chatbot — RAG

An AI-powered PDF chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from PDF documents.

## 🚀 Features

- 📄 Extracts text from PDF documents
- ✂️ Splits PDF text into smaller chunks
- 🧠 Creates document embeddings
- 🔎 Searches for relevant document information
- 🦙 Uses Ollama for local AI inference
- 💬 Answers questions based on PDF content

## 🧠 Models Used

### Chat / Generation Model
`llama3.2:3b`

### Embedding Model
`nomic-embed-text`

Both models are run locally using Ollama.

## 🔄 RAG Pipeline

```text
PDF Document
     ↓
Text Extraction
     ↓
Text Chunking
     ↓
Document Embeddings
     ↓
Similarity / Document Search
     ↓
Relevant Context
     ↓
Ollama LLM
     ↓
AI Answer

git clone https://github.com/pavanborgol7-collab/pdf-ai-chatbot-RAG.git
cd pdf-ai-chatbot-RAG

pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull nomic-embed-text

streamlit run app.py








