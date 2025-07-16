from fastapi import APIRouter, UploadFile, File
from app.services.ocr import process_document
import os
from uuid import uuid4

router = APIRouter()

@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid4())
    file_ext = os.path.splitext(file.filename)[1]
    save_path = f"backend/data/uploads/{file_id}{file_ext}"

    with open(save_path, "wb") as f:
        content = await file.read()
        f.write(content)

    text_chunks = process_document(save_path, file_id)
    return {"file_id": file_id, "chunks": len(text_chunks)}