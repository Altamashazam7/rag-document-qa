import streamlit as st
import requests

API_URL = "https://rag-document-qa-qn4l.onrender.com"

st.set_page_config(
    page_title="HR Candidate Screening RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Automated HR Candidate Screening System")
st.markdown("Upload candidate resume PDFs and perform local semantic search.")

# --- HELPER FUNCTION: SAFE JSON EXTRACTION ---
def safe_parse_json(response):
    """Safely parse JSON or fallback to raw text if server returned HTML/text."""
    try:
        return response.json()
    except Exception:
        return None

# --- SIDEBAR SYSTEM HEALTH MONITOR ---
st.sidebar.header("System Status")

backend_online = False
try:
    # 20s timeout allows sleeping Render containers to wake up on first ping
    health_response = requests.get(f"{API_URL}/", timeout=20)
    if health_response.status_code == 200:
        backend_online = True
        st.sidebar.success("Backend API: Online")
    else:
        st.sidebar.error(f"Backend API Error: HTTP {health_response.status_code}")
except Exception:
    st.sidebar.warning("Backend API: Offline / Waking up...")
    st.sidebar.caption("Free cloud servers take 30–50s to wake up on first visit.")
    if st.sidebar.button("Wake Up Backend"):
        st.rerun()

# --- TAB NAVIGATION ---
tab1, tab2 = st.tabs(["📤 Upload Resumes", "🔍 Screen Candidates"])

# --- TAB 1: RESUME UPLOAD & INDEXING ---
with tab1:
    st.header("Upload Candidate Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF resumes to index",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Index Resumes", type="primary"):
        if not backend_online:
            st.error("Backend server is currently offline or waking up. Please click 'Wake Up Backend' in the sidebar.")
        elif not uploaded_files:
            st.warning("Please select at least one PDF resume.")
        else:
            files_payload = [
                ("files", (file.name, file.getvalue(), "application/pdf"))
                for file in uploaded_files
            ]
            with st.spinner("Processing PDF text & generating FAISS vector embeddings..."):
                try:
                    # 180s timeout accommodates batch vector embedding creation
                    response = requests.post(f"{API_URL}/upload-batch", files=files_payload, timeout=180)
                    data = safe_parse_json(response)

                    if response.status_code == 200:
                        if data:
                            st.success(f"Successfully indexed {len(data.get('processed_resumes', []))} resume(s)!")
                            st.json(data)
                        else:
                            st.error("Response returned 200 OK but was not valid JSON.")
                    else:
                        error_detail = data.get('detail', response.text) if data else response.text
                        st.error(f"Error {response.status_code}: {error_detail}")
                        
                except requests.exceptions.Timeout:
                    st.error("Request timed out. Please try uploading in smaller batches of 1–2 files.")
                except Exception as e:
                    st.error(f"Failed to communicate with backend server: {e}")

# --- TAB 2: CANDIDATE SEARCH & SCREENING ---
with tab2:
    st.header("Candidate Screening & Querying")
    
    query_text = st.text_input(
        "Enter job requirements or technical criteria:",
        placeholder="e.g., Which candidates have experience with Python, Docker, and FAISS?"
    )
    top_k = st.slider("Number of top matching passages to retrieve", min_value=1, max_value=10, value=4)

    if st.button("Search Candidates", type="primary"):
        if not backend_online:
            st.error("Backend server is currently offline or waking up. Please click 'Wake Up Backend' in the sidebar.")
        elif not query_text.strip():
            st.warning("Please enter a query or job description.")
        else:
            with st.spinner("Searching FAISS vector database..."):
                try:
                    payload = {"question": query_text, "top_k": top_k}
                    response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                    data = safe_parse_json(response)

                    if response.status_code == 200:
                        if data:
                            st.subheader(f"Matches Found: {data.get('matches_found', 0)}")
                            
                            if data.get('matches_found', 0) == 0:
                                st.info("No relevant matches found for your query.")
                            else:
                                for idx, match in enumerate(data.get('results', []), 1):
                                    candidate_name = match.get('candidate_file', 'Unknown Candidate')
                                    page_num = match.get('page', 'N/A')
                                    with st.expander(f"Match #{idx} — Candidate: {candidate_name} (Page {page_num})", expanded=True):
                                        st.write(match['content'])
                        else:
                            st.error("Response returned 200 OK but was not valid JSON.")
                    else:
                        error_detail = data.get('detail', response.text) if data else response.text
                        st.error(f"Error {response.status_code}: {error_detail}")

                except requests.exceptions.Timeout:
                    st.error("Query timed out. Please try again.")
                except Exception as e:
                    st.error(f"Failed to communicate with backend server: {e}")
