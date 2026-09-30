import streamlit as st
import requests

# Backend API configuration
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="HR Candidate Screening RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Automated HR Candidate Screening System")
st.markdown("Upload candidate resume PDFs and perform local semantic search.")

# Sidebar - System Status
st.sidebar.header("System Status")
try:
    health_response = requests.get(f"{API_URL}/")
    if health_response.status_code == 200:
        st.sidebar.success("Backend API: Online")
    else:
        st.sidebar.error("Backend API: Error")
except Exception:
    st.sidebar.error("Backend API: Offline (Start uvicorn)")

# Tab Layout
tab1, tab2 = st.tabs(["📤 Upload Resumes", "🔍 Screen Candidates"])

# --- TAB 1: BATCH PDF UPLOAD ---
with tab1:
    st.header("Upload Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF resumes to index",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Index Resumes", type="primary"):
        if not uploaded_files:
            st.warning("Please select at least one PDF file.")
        else:
            files_payload = [
                ("files", (file.name, file.getvalue(), "application/pdf"))
                for file in uploaded_files
            ]
            with st.spinner("Processing & indexing resumes into FAISS vector store..."):
                try:
                    response = requests.post(f"{API_URL}/upload-batch", files=files_payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Successfully processed {len(data['processed_resumes'])} resumes!")
                        st.json(data)
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend server: {e}")

# --- TAB 2: CANDIDATE SEARCH & SCREENING ---
with tab2:
    st.header("Candidate Screening & Querying")
    
    query_text = st.text_input(
        "Enter job requirements or technical criteria:",
        placeholder="e.g., Which candidates have experience with Python, Docker, and FAISS?"
    )
    top_k = st.slider("Number of top matching passages to retrieve", min_value=1, max_value=10, value=4)

    if st.button("Search Candidates", type="primary"):
        if not query_text.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Searching FAISS vector database..."):
                try:
                    payload = {"question": query_text, "top_k": top_k}
                    response = requests.post(f"{API_URL}/query", json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.subheader(f"Matches Found: {data['matches_found']}")
                        
                        for idx, match in enumerate(data['results'], 1):
                            with st.expander(f"Match #{idx} — Candidate: {match['candidate_file']} (Page {match['page']})", expanded=True):
                                st.write(match['content'])
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend server: {e}")