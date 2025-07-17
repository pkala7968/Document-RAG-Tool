from fastapi import APIRouter, UploadFile, File
from typing import List
from app.models.schema import QueryRequest, QueryResponse, Source
from app.services.ocr import process_document
from app.services.llm import get_conversational_chain

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_docs(files: List[UploadFile] = File(...), query: QueryRequest = None):
    docs = process_document(files)
    response = get_conversational_chain(docs, query.question)

    sources = [
        Source(file=doc.metadata["source"], page=doc.metadata.get("page", 1))
        for doc in response["source_documents"]
    ]

    return QueryResponse(
        doc_id="some-id",
        question=query.question,
        answer=response["answer"],
        sources=sources
    )

@router.get("/status")
def status():
    return {"status": "OK"}
