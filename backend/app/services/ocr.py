import pytesseract
from PIL import Image
import fitz 
import os
from app.config import settings

def ocr_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=settings.OCR_LANG)
    return text

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            chunks.append({
                "text": text.strip(),
                "page": page_num + 1
            })
    return chunks

def process_document(file_path, file_id):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        text = ocr_image(file_path)
        return [{"text": text, "page": 1}]
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return [{"text": text, "page": 1}]
    else:
        raise ValueError("Unsupported file type")