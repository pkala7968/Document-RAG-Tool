import os
import streamlit as st
from ocr import process_document
from vectorstore import vectorstore_from_docs
from llm import get_conversational_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("Dicument RAG Tool")
st.sidebar.title("Add you Documents")
st.sidebar.subheader("Upload your PDFs to the app to start querying them.")

docs= st.sidebar.file_uploader("Upload your Documents", type=["pdf","docx","txt","jpg/jpeg"], accept_multiple_files=True, key="pdf_uploader")

process_docs= st.sidebar.button("Load docs")

if process_docs and docs:
    all_docs = []

    for doc in docs:
        with st.spinner(f"Processing {doc.name}..."):
            langchain_docs = process_document(doc)

            # Split while preserving metadata
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = splitter.split_documents(langchain_docs)

            all_docs.extend(splits)

    vectorstore = vectorstore_from_docs(all_docs)
    retriever = vectorstore.as_retriever()
    chain = get_conversational_chain(retriever)

    st.session_state.chain = chain
    st.success("Documents loaded and ready!")

user_input = st.text_input("Ask a question about your documents:")

if user_input:
    with st.spinner("Generating answer..."):
        response = st.session_state.chain({"question": user_input})
        st.write(response["answer"])

        # Show citations
        st.markdown("**Sources:**")
        for doc in response["source_documents"]:
            st.markdown(f"- `{doc.metadata.get('source')}`, Page {doc.metadata.get('page')}")