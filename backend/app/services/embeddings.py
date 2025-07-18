from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import GOOGLE_API_KEY

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)