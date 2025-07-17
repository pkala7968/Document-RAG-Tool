import os
import streamlit as st
from dotenv import load_dotenv
import requests

FASTAPI_URL = "http://localhost:8000/query"  # Or your deployed URL


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("Document RAG Tool")
st.sidebar.title("Upload & Ask")

docs = st.sidebar.file_uploader("Upload files", type=["pdf", "docx", "txt", "jpg", "jpeg"], accept_multiple_files=True)
question = st.text_input("Ask a question based on the documents")

if st.button("Submit") and docs and question:
    with st.spinner("Sending to backend..."):
        files = [("files", (doc.name, doc, doc.type)) for doc in docs]
        json_data = {"question": question}

        try:
            res = requests.post(FASTAPI_URL, files=files, data={"question": question})
            if res.status_code == 200:
                result = res.json()

                st.subheader("Answer:")
                st.write(result["answer"])

                st.subheader("Sources:")
                source_data = result.get("sources", [])
                if source_data:
                    st.table(source_data)
                else:
                    st.write("No sources found.")
            else:
                st.error(f"Backend error: {res.status_code} - {res.text}")

        except Exception as e:
            st.error(f"Request failed: {e}")