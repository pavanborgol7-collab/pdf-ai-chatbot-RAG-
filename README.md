# 🤖 PDF AI Chatbot — RAG

An AI-powered PDF chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from PDF documents.

## 🚀 Features

- 📄 Extracts text from PDF documents
- ✂️ Splits documents into smaller chunks
- 🧠 Creates embeddings for document chunks
- 🔎 Searches for relevant information
- 🤖 Uses Ollama for local LLM inference
- 💬 Answers questions based on the uploaded PDF
- 🔒 Runs the LLM locally using Ollama

## 🧠 How It Works

PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embeddings
 ↓
Document Search
 ↓
Relevant Context
 ↓
Ollama LLM
 ↓
AI Answer

## 🛠️ Technologies Used

- Python
- RAG (Retrieval-Augmented Generation)
- Ollama
- PyPDF
- Embeddings
- Vector/Document Search
- LangChain (if used in the project)

## 📂 Project Structure

```text
pdf-ai-chatbot-RAG/
│
├── app.py
├── ask_pdf.py
├── chunk_pdf.py
├── create_embeddings.py
├── extract_pdf.py
├── search_documents.py
├── requirements.txt
├── .gitignore
└── README.md
