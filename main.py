import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize FastAPI App
app = FastAPI(
    title="Automated HR Candidate Screening RAG Engine",
    description="Local vector-based PDF resume screening using FAISS and HuggingFace Embeddings.",
    version="1.0.0"
)

# Directory Configuration
UPLOAD_DIR = "uploaded_resumes"
INDEX_DIR = "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize Local Embedding Model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Global Vector Store Instance
vector_store = None

# Reload persistent vector database from disk if available on startup
if os.path.exists(INDEX_DIR):
    try:
        vector_store = FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        print("Existing FAISS vector index loaded successfully from disk.")
    except Exception as e:
        print(f"Failed to load existing index: {e}")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "HR Candidate Screening RAG API",
        "docs": "Visit http://127.0.0.1:8000/docs to test endpoints"
    }


@app.post("/upload-batch")
async def upload_batch_resumes(
    files: List[UploadFile] = File(..., description="Select multiple resume PDFs")
):
    """
    Upload multiple resume PDFs simultaneously into the local FAISS vector store.
    Attaches candidate file metadata to each chunk for precise source attribution.
    """
    global vector_store
    processed_files = []
    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse PDF and assign source candidate metadata
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_candidate"] = file.filename

            chunks = text_splitter.split_documents(docs)
            all_chunks.extend(chunks)
            processed_files.append(file.filename)
        except Exception as e:
            print(f"Error processing file {file.filename}: {e}")

    if not all_chunks:
        raise HTTPException(
            status_code=400, 
            detail="No valid text could be extracted from the uploaded PDF resumes."
        )

    # Initialize or extend the FAISS index
    if vector_store is None:
        vector_store = FAISS.from_documents(all_chunks, embeddings)
    else:
        vector_store.add_documents(all_chunks)

    # Persist FAISS index snapshot to disk
    vector_store.save_local(INDEX_DIR)

    return {
        "status": "success",
        "processed_resumes": processed_files,
        "total_chunks_indexed": len(all_chunks),
        "message": "Resumes indexed and vector database persisted to disk."
    }


@app.post("/query")
async def query_candidates(request: QueryRequest):
    """
    Perform semantic search across all uploaded candidate resumes.
    Returns matching text passages tagged with source candidate filenames.
    """
    global vector_store
    if vector_store is None:
        raise HTTPException(
            status_code=400, 
            detail="No resumes indexed yet. Please upload resumes first."
        )

    # Similarity search in FAISS vector space
    results = vector_store.similarity_search(request.question, k=request.top_k)

    retrieved_context = []
    for doc in results:
        retrieved_context.append({
            "candidate_file": doc.metadata.get("source_candidate", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,
            "content": doc.page_content
        })

    return {
        "query": request.question,
        "matches_found": len(retrieved_context),
        "results": retrieved_context
    }