# AI-Driven Retrieval-Augmented Generation (RAG) Pipeline

A production-grade RAG pipeline built with FastAPI, LangChain, FAISS, and HuggingFace embeddings (all-MiniLM-L6-v2) designed for fast semantic document querying and automated context synthesis.

## Key Features
* **Dense Vector Search:** Uses HuggingFace sentence-transformers (all-MiniLM-L6-v2) to convert unstructured documents into 384-dimensional dense embeddings.
* **Ultra-Fast Similarity Retrieval:** Leverages FAISS (Facebook AI Similarity Search) for low-latency L2 vector indexing and context extraction.
* **High-Performance REST API:** Built on FastAPI with async execution and strict Pydantic payload validation.

## Tech Stack
* **Framework:** Python, FastAPI, Uvicorn
* **Orchestration:** LangChain
* **Vector Store:** FAISS
* **Embeddings:** HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

## Getting Started
1. Clone Repository: git clone https://github.com/Altamashazam7/rag-document-qa.git
2. Setup Environment: python -m venv venv; .\venv\Scripts\Activate
3. Install Dependencies: pip install -r requirements.txt
4. Start API Server: uvicorn main:app --reload
5. Access Docs: http://localhost:8000/docs
