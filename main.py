import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from rag_service import rag_engine

app = FastAPI(title="Document Q&A RAG Engine")

os.makedirs("uploads", exist_ok=True)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "RAG Document Q&A API is live!"}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join("uploads", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        num_chunks = rag_engine.process_pdf(file_path)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": num_chunks,
            "message": "PDF successfully processed and indexed into FAISS vector store."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: QueryRequest):
    try:
        result = rag_engine.query(request.question)
        return {
            "status": "success",
            "question": result["question"],
            "retrieved_context": result["retrieved_context"],
            "sources": result["sources"]
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))