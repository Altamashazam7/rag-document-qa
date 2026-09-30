import os
import shutil
import gc
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

app = FastAPI(title="Automated HR Candidate Screening RAG Engine")

UPLOAD_DIR = "uploaded_resumes"
INDEX_DIR = "faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)

embeddings = None
vector_store = None

def get_embeddings():
    global embeddings
    if embeddings is None:
        # FastEmbed runs ONNX model on CPU with minimal RAM (~120MB)
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return embeddings

def get_vector_store():
    global vector_store
    if vector_store is None and os.path.exists(INDEX_DIR):
        try:
            vector_store = FAISS.load_local(
                INDEX_DIR, 
                get_embeddings(), 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Error loading existing vector index: {e}")
    return vector_store

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

@app.get("/")
async def root():
    return {"status": "online", "service": "HR Candidate Screening RAG API"}

@app.post("/upload-batch")
async def upload_batch_resumes(files: List[UploadFile] = File(...)):
    global vector_store
    processed_files = []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    embeds = get_embeddings()

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
            
            if not chunks:
                continue

            v_store = get_vector_store()
            if v_store is None and vector_store is None:
                vector_store = FAISS.from_documents(chunks, embeds)
            else:
                target_store = vector_store if vector_store else v_store
                target_store.add_documents(chunks)
                vector_store = target_store

            processed_files.append(file.filename)
            gc.collect()

        except Exception as e:
            print(f"Failed to process candidate resume {file.filename}: {e}")

    if not processed_files:
        raise HTTPException(
            status_code=400, 
            detail="No valid text could be extracted from the uploaded PDF resumes."
        )

    if vector_store:
        vector_store.save_local(INDEX_DIR)

    return {
        "status": "success",
        "processed_resumes": processed_files
    }

@app.post("/query")
async def query_candidates(request: QueryRequest):
    v_store = get_vector_store()
    if v_store is None:
        raise HTTPException(
            status_code=400, 
            detail="No candidate resumes have been indexed yet. Please upload resumes first."
        )

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
