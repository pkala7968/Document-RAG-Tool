from langchain.schema import Document
import pytesseract
from PIL import Image
import fitz
import docx
import os

def process_document(file_obj, filename="uploaded_file.pdf"):
    name = filename
    ext = os.path.splitext(name)[1].lower()

    text_chunks = []

    if ext in [".jpg", ".jpeg", ".png"]:
        image = Image.open(file_obj)
        text = pytesseract.image_to_string(image, lang="eng")
        text_chunks = [Document(page_content=text, metadata={"source": name, "page": 1})]

    elif ext == ".pdf":
        pdf = fitz.open(stream=file_obj.read(), filetype="pdf")
        for i, page in enumerate(pdf):
            text = page.get_text().strip()
            if text:
                text_chunks.append(Document(page_content=text, metadata={"source": name, "page": i + 1}))

    elif ext == ".docx":
        doc = docx.Document(file_obj)
        full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        text_chunks = [Document(page_content=full_text, metadata={"source": name, "page": 1})]

    elif ext == ".txt":
        text = file_obj.read().decode("utf-8")
        text_chunks = [Document(page_content=text, metadata={"source": name, "page": 1})]

    else:
        raise ValueError("Unsupported file type")

    return text_chunks