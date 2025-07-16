from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

from app.services.vectorstore import query_similar_chunks
from app.services.llm import generate_answer_with_citations, generate_themes

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

class DocumentAnswer(BaseModel):
    doc_id: str
    page: int
    answer: str

class QueryResponse(BaseModel):
    document_answers: List[DocumentAnswer]
    theme_summary: str

@router.post("/", response_model=QueryResponse)
async def query_documents(query: QueryRequest):
    try:
        # Step 1: Semantic search
        chunks = query_similar_chunks(query.question)

        # Step 2: Generate answers for each document chunk
        doc_answers = generate_answer_with_citations(query.question, chunks)

        # Step 3: Generate theme summary
        themes = generate_themes(doc_answers)

        return {
            "document_answers": doc_answers,
            "theme_summary": themes
        }
    except Exception as e:
        return {
            "document_answers": [],
            "theme_summary": f"Error: {str(e)}"
        }