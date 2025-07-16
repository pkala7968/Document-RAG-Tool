import os
from chromadb import Client
from chromadb.config import Settings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings

# Setup ChromaDB client with persistent storage
client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="backend/data/chroma"))
collection = client.get_or_create_collection(name="documents")

# Use OpenAI for embeddings
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=settings.GOOGLE_API_KEY)

def embed_and_store_chunks(chunks, doc_id):
    for idx, chunk in enumerate(chunks):
        text = chunk['text']
        metadata = {
            "doc_id": doc_id,
            "page": chunk.get("page", 1)
        }
        try:
            embedding = embedding_model.embed_query(text)
            collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
                ids=[f"{doc_id}-{idx}"]
            )
        except Exception as e:
            print(f"Failed to embed chunk {idx} from {doc_id}: {e}")

def query_similar_chunks(query, top_k=5):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    docs = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        docs.append({
            "text": doc,
            "doc_id": meta['doc_id'],
            "page": meta['page']
        })
    return docs
