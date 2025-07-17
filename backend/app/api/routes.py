from fastapi import APIRouter, UploadFile, File
from typing import List
from fastapi import Form
from app.models.schema import QueryResponse, Source
from app.services.ocr import process_document
from app.services.llm import get_conversational_chain
from app.services.vectorstore import vectorstore_from_docs
from langchain.text_splitter import RecursiveCharacterTextSplitter

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_docs(
    files: List[UploadFile] = File(...),
    question: str = Form(...)
):
    all_docs = []
    for file in files:
        content = await file.read()
        file.file.seek(0)
        file_obj = file.file

        # Pass one file at a time
        raw_docs = process_document(file_obj, filename=file.filename)

        # Chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(raw_docs)
        all_docs.extend(split_docs)

    # Vectorstore and retrieval
    vectorstore = vectorstore_from_docs(all_docs)
    retriever = vectorstore.as_retriever()
    chain = get_conversational_chain(retriever)

    response = chain.invoke({"question": question})
    
    # Build structured response
    source_list = []

    if response["answer"].strip().lower() == "answer is not available in the context":
        return QueryResponse(
        doc_id="unknown",
        question=question,
        answer=response["answer"],
        sources=source_list
    )
    
    for doc in response["source_documents"]:
        source_list.append(Source(file=doc.metadata["source"], page=doc.metadata.get("page", 1)))

    return QueryResponse(
        doc_id="some-id",
        question=question,
        answer=response["answer"],
        sources=source_list
    )

@router.get("/status")
def status():
    return {"status": "OK"}