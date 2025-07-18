from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import os
from app.models.schema import QueryResponse, Source
from app.services.ocr import process_document
from app.services.llm import get_conversational_chain
from app.services.vectorstore import vectorstore_from_docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.services.embeddings import embeddings

router = APIRouter()

FAISS_INDEX_DIR = "faiss_index"  # single shared index

# Upload endpoint - handles document uploads and indexing
@router.post("/upload")
async def upload_docs(files: List[UploadFile] = File(...)):
    all_docs = []

    for file in files:
        content = await file.read()
        file.file.seek(0)
        file_obj = file.file

        raw_docs = process_document(file_obj, filename=file.filename)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(raw_docs)
        all_docs.extend(split_docs)

    vectorstore = vectorstore_from_docs(all_docs)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_DIR)

    return {"message": "Documents uploaded and indexed successfully."}

# Query endpoint - handles question answering
@router.post("/query", response_model=QueryResponse)
async def query_docs(question: str = Form(...)):
    if not os.path.exists(FAISS_INDEX_DIR):
        return QueryResponse(
            doc_id="shared",
            question=question,
            answer="No documents uploaded yet.",
            sources=[]
        )

    vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    chain = get_conversational_chain(retriever)

    response = chain.invoke({"question": question})

    sources = []
    if response.get("answer", "").strip().lower() != "answer is not available in the context":
        for doc in response.get("source_documents", []):
            sources.append(Source(
                file=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page", 1)
            ))

    return QueryResponse(
        doc_id="shared",
        question=question,
        answer=response.get("answer", "No answer returned."),
        sources=sources
    )

# Health check
@router.get("/status")
def status():
    return {"status": "OK"}