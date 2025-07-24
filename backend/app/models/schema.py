from pydantic import BaseModel
from typing import List

class Source(BaseModel):
    file: str
    page: int

class QueryResponse(BaseModel):
    doc_id: str
    question: str
    answer: str
    sources: List[Source]

class EvaluateRequest(BaseModel):
    answer: str
    sources: List[Source]