from fastapi import FastAPI
from app.api import upload_routes, query_routes

app = FastAPI(title="Document QA + Theme Bot")

# Register API routes
app.include_router(upload_routes.router, prefix="/upload", tags=["Upload"])
app.include_router(query_routes.router, prefix="/query", tags=["Query"])

@app.get("/")
def root():
    return {"message": "Welcome to Wasserstoff AI DocBot"}