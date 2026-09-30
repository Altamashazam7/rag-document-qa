import streamlit as st
import requests

# Backend API configuration - pointing to live Render instance
API_URL = "https://rag-document-qa-qn4l.onrender.com"

st.set_page_config(
    page_title="HR Candidate Screening RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Automated HR Candidate Screening System")
st.markdown("Upload candidate resume PDFs and perform local semantic search.")

# --- SIDEBAR: SYSTEM STATUS ---
st.sidebar.header("System Status")

try:
    # 15s timeout to give Render cold starts a chance to finish responding
    health_response = requests.get(f"{API_URL}/", timeout=15)
    if health_response.status_code == 200:
        st.sidebar.success("Backend API: Online")
    else:
        st.sidebar.error(f"Backend API Error: HTTP {health_response.status_code}")
except Exception:
    st.sidebar.error("Backend API: Offline / Waking up...")
    st.sidebar.info("Render free tier takes 30–50s to wake up on first load.")
    if st.sidebar.button("Retry Connection"):
        st.rerun()

# --- MAIN TAB LAYOUT ---
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
                    # Extended 180s timeout for multi-file PDF processing & embedding generation
                    response = requests.post(f"{API_URL}/upload-batch", files=files_payload, timeout=180)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Successfully processed {len(data['processed_resumes'])} resume(s)!")
                        st.json(data)
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Try uploading fewer PDFs at once.")
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
                    # 60s timeout for similarity query search
                    response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                    if response.status_code == 200:
                        data = response.json()
                        st.subheader(f"Matches Found: {data['matches_found']}")
                        
                        if data['matches_found'] == 0:
                            st.info("No matching content found for this query.")
                        else:
                            for idx, match in enumerate(data['results'], 1):
                                candidate_name = match.get('candidate_file', 'Unknown File')
                                page_num = match.get('page', 'N/A')
                                with st.expander(f"Match #{idx} — Candidate: {candidate_name} (Page {page_num})", expanded=True):
                                    st.write(match['content'])
                    else:
                        st.error(f"Error {response.status_code}: {response.text}")
                except requests.exceptions.Timeout:
                    st.error("Query timed out. Please try again.")
                except Exception as e:
                    st.error(f"Failed to connect to backend server: {e}")
