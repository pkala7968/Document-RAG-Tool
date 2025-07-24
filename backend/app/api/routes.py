from fastapi import APIRouter, UploadFile, File, Form
from typing import List
import os
from app.models.schema import QueryResponse, Source, EvaluateRequest
from app.services.ocr import process_document
from app.services.llm import get_conversational_chain, evaluate_with_llm
from app.services.vectorstore import vectorstore_from_docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from app.services.embeddings import embeddings

router = APIRouter()

FAISS_INDEX_DIR = "data/faiss_index"  # single shared index
UPLOAD_DIR = "data/uploaded_files"

# Upload endpoint - handles document uploads and indexing
@router.post("/upload")
async def upload_docs(files: List[UploadFile] = File(...)):
    all_docs = []
    uploaded_filenames = []

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    for file in files:
        try:
            # Save to disk
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())

            # Re-open using binary mode for processing
            with open(file_path, "rb") as f:
                raw_docs = process_document(f, filename=file.filename)

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = splitter.split_documents(raw_docs)
            all_docs.extend(split_docs)

            uploaded_filenames.append(file.filename)

        except Exception as e:
            print(f"[❌ Error Uploading] {file.filename}: {e}")
            continue

    if not all_docs:
        return {"error": "No documents were successfully processed."}

    # Create vectorstore index
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    vectorstore = vectorstore_from_docs(all_docs)
    vectorstore.save_local(FAISS_INDEX_DIR)

    return {
        "message": "Documents uploaded and indexed successfully.",
        "files": uploaded_filenames
    }

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

@router.post("/evaluate")
async def evaluate_answer(payload: EvaluateRequest):
    context_chunks = []
    failed_sources = []

    for src in payload.sources:
        file_path = os.path.join(UPLOAD_DIR, src.file)

        if not os.path.isfile(file_path):
            print(f"[⚠️ Missing File] {file_path} not found.")
            failed_sources.append(src.file)
            continue

        try:
            with open(file_path, "rb") as f:
                docs = process_document(f, filename=src.file)

            # If page is specified and docs are paginated (e.g., PDF)
            if src.page and len(docs) > 1:
                page_index = int(src.page) - 1
                if 0 <= page_index < len(docs):
                    context_chunks.append(docs[page_index].page_content)
                else:
                    print(f"[⚠️ Page Index Out of Range] Page {src.page} in {src.file}")
                    context_chunks.append(" ".join(doc.page_content for doc in docs))  # fallback to full
            else:
                # For non-paginated docs or page not specified
                context_chunks.append(" ".join(doc.page_content for doc in docs))

        except Exception as e:
            print(f"[❌ Error Processing] {src.file} (Page {src.page}): {e}")
            failed_sources.append(src.file)
            continue

    if not context_chunks:
        return {
            "factual_accuracy": "0%",
            "completeness": "0%",
            "hallucination": "100%",
            "comment": f"Failed to extract any content from: {failed_sources or [src.file for src in payload.sources]}"
        }

    combined_context = "\n\n".join(context_chunks)

    try:
        evaluation = evaluate_with_llm(combined_context, payload.answer)
        return evaluation
    except Exception as e:
        return {
            "factual_accuracy": "0%",
            "completeness": "0%",
            "hallucination": "100%",
            "comment": f"Evaluation failed: {str(e)}"
        }

# Health check
@router.get("/status")
def status():
    return {"status": "OK"}