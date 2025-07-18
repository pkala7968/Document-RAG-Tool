from fastapi import FastAPI
from app.api.routes import router
import uvicorn

app = FastAPI()
app.include_router(router)

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)