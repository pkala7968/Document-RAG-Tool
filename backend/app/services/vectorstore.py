from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import GOOGLE_API_KEY
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "faiss_index"

def vectorstore_from_docs(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(str(DATA_PATH))
    return vectorstore 
