import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGEngine:
    def __init__(self):
        print("Loading Embedding Model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None

    def process_pdf(self, file_path: str) -> int:
        """Loads PDF, chunks text (500 tokens / 50 overlap), and stores in FAISS."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Sliding-window chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)

        # Index chunks into in-memory FAISS vector store
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        return len(chunks)

    def query(self, question: str, top_k: int = 3) -> dict:
        """Retrieves top_k chunks using similarity search and returns context."""
        if not self.vector_store:
            raise ValueError("No document indexed yet. Please upload a PDF first.")

        relevant_docs = self.vector_store.similarity_search(question, k=top_k)
        
        sources = []
        context_text = ""
        
        for doc in relevant_docs:
            page_num = doc.metadata.get("page", 0) + 1
            content = doc.page_content.strip()
            sources.append({"page": page_num, "content": content})
            context_text += f"\n[Page {page_num}]: {content}\n"

        return {
            "question": question,
            "retrieved_context": context_text,
            "sources": sources
        }

# Global instance
rag_engine = RAGEngine()