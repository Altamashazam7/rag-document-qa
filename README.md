# Document Q&A Engine with RAG Pipeline

A high-performance Retrieval-Augmented Generation (RAG) backend service built with Python, FastAPI, LangChain, and FAISS. It enables semantic vector similarity search over unstructured PDF documents with source page citations.

## Key Features
- **PDF Ingestion & Processing:** Automated text extraction using `pypdf`.
- **Sliding-Window Chunking:** Text chunking strategy (500 tokens with 50-token overlap) to preserve semantic context and reduce prompt token bloat.
- **In-Memory Vector Search:** High-speed vector indexing and similarity search using **FAISS** and `all-MiniLM-L6-v2` embeddings.
- **RESTful API:** Clean FastAPI endpoints for uploading documents and performing grounded Q&A.

## Architecture
```text
[PDF Upload] -> [Text Extraction] -> [500-Token Chunking] -> [HuggingFace Embeddings] -> [FAISS Vector Store]
                                                                                               |
[User Question] ----------------------------> [Similarity Search] <----------------------------+
                                                    |
                                    [Top-K Context + Page Citations]