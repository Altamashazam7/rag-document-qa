import os
import shutil
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI(
    title="Automated HR Candidate Screening RAG Engine",
    version="1.0.0"
)

UPLOAD_DIR = "uploaded_resumes"
INDEX_DIR = "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variables for lazy loading
embeddings = None
vector_store = None

def get_embeddings():
    """Lazy load embedding model to save startup memory."""
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    return embeddings

def get_vector_store():
    """Lazy load vector store on demand."""
    global vector_store
    if vector_store is None and os.path.exists(INDEX_DIR):
        try:
            vector_store = FAISS.load_local(
                INDEX_DIR, 
                get_embeddings(), 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Failed to load existing index: {e}")
    return vector_store

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

@app.get("/")
async def root():
    return {"status": "online", "service": "HR Candidate Screening RAG API"}

@app.post("/upload-batch")
async def upload_batch_resumes(
    files: List[UploadFile] = File(..., description="Select multiple resume PDFs")
):
    global vector_store
    processed_files = []
    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

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
        raise HTTPException(status_code=400, detail="No valid text extracted.")

    embeds = get_embeddings()
    v_store = get_vector_store()

    if v_store is None:
        vector_store = FAISS.from_documents(all_chunks, embeds)
    else:
        v_store.add_documents(all_chunks)
        vector_store = v_store

    vector_store.save_local(INDEX_DIR)

    return {
        "status": "success",
        "processed_resumes": processed_files,
        "total_chunks_indexed": len(all_chunks)
    }

@app.post("/query")
async def query_candidates(request: QueryRequest):
    v_store = get_vector_store()
    if v_store is None:
        raise HTTPException(status_code=400, detail="No resumes indexed yet.")

    results = v_store.similarity_search(request.question, k=request.top_k)

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