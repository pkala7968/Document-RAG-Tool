from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    file: str
    page: int

class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    sources: List[Source]