import streamlit as st
import requests

UPLOAD_URL = "https://backend-doc-rag.fly.dev/upload"
QUERY_URL = "https://backend-doc-rag.fly.dev/query"

# Session ID to track uploaded docs
if "session_id" not in st.session_state:
    st.session_state.session_id = None

st.title("Document RAG Chatbot")
st.sidebar.title("Upload Documents")

# Upload files only once
docs = st.sidebar.file_uploader("Upload files", type=["pdf", "docx", "txt", "jpg", "jpeg"], accept_multiple_files=True)

question = st.text_input("Ask a question based on the uploaded documents")

if st.sidebar.button("Upload Documents"):
    if docs:
        with st.spinner("Uploading documents..."):
            files = [("files", (doc.name, doc, doc.type)) for doc in docs]
            res = requests.post(UPLOAD_URL, files=files)
            if res.status_code == 200:
                st.success("Documents uploaded and indexed!")
                st.session_state.session_id = "shared"
            else:
                st.error(f"Upload failed: {res.status_code} - {res.text}")
    else:
        st.warning("Please upload at least one document.")

if st.button("Submit"):
    if question and st.session_state.get("session_id"):
        with st.spinner("Processing..."):
            res = requests.post(QUERY_URL, data={
                "question": question
            })
            if res.status_code == 200:
                result = res.json()
                st.session_state.answer = result["answer"]
                st.session_state.sources = result["sources"]
            else:
                st.error(f"Query failed: {res.status_code} - {res.text}")
    else:
        st.warning("Please upload documents and enter a question.")
    
# After question is submitted, show answer + evaluation
if "answer" in st.session_state and "sources" in st.session_state:
    st.subheader("Answer:")
    st.write(st.session_state.answer)

    st.subheader("Sources:")
    if st.session_state.sources:
        st.table(st.session_state.sources)
    else:
        st.write("No sources found.")

    if st.button("Evaluate Answer"):
        with st.spinner("Evaluating answer accuracy..."):
            eval_payload = {
                "answer": st.session_state.answer,
                "sources": st.session_state.sources
            }
            eval_res = requests.post("https://backend-doc-rag.fly.dev/evaluate", json=eval_payload)
            if eval_res.status_code == 200:
                eval_data = eval_res.json()
                st.subheader("Evaluation Scores:")
                st.write(f"**Factual Accuracy:** {eval_data['factual_accuracy']}")
                st.write(f"**Completeness:** {eval_data['completeness']}")
                st.write(f"**Hallucination:** {eval_data['hallucination']}")
                st.write(f"**Comment:** {eval_data['comment']}")
            else:
                st.error("Evaluation failed.")