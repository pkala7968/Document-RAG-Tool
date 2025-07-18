import streamlit as st
import requests

UPLOAD_URL = "http://localhost:8000/upload"
QUERY_URL = "http://localhost:8000/query"

# Session ID to track uploaded docs
if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.title("Document RAG Chatbot")
st.sidebar.title("Upload Documents")

# Upload files only once
docs = st.sidebar.file_uploader("Upload files", type=["pdf", "docx", "txt", "jpg", "jpeg"], accept_multiple_files=True)

question = st.text_input("Ask a question based on the uploaded documents")

if st.button("Submit") and question:
    with st.spinner("Processing..."):
        try:
            if docs and st.session_state.session_id is None:
                # First time: upload files and ask question
                files = [("files", (doc.name, doc, doc.type)) for doc in docs]
                res = requests.post(UPLOAD_URL, files=files, data={"question": question})
                if res.status_code == 200:
                    result = res.json()
                    st.session_state.session_id = result["doc_id"]
                else:
                    st.error(f"Upload failed: {res.status_code} - {res.text}")
                    st.stop()

            elif st.session_state.session_id:
                # Follow-up question
                res = requests.post(QUERY_URL, data={
                    "question": question,
                    "session_id": st.session_state.session_id
                })
                if res.status_code == 200:
                    result = res.json()
                else:
                    st.error(f"Query failed: {res.status_code} - {res.text}")
                    st.stop()
            else:
                st.warning("Please upload documents before asking a question.")
                st.stop()

            # Show answer and sources
            st.subheader("Answer:")
            st.write(result["answer"])

            st.subheader("Sources:")
            sources = result.get("sources", [])
            if sources:
                st.table(sources)
            else:
                st.write("No sources found.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")