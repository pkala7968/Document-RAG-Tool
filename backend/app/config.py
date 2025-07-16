import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DATA_DIR = "backend/data"
    OCR_LANG = "eng"

settings = Settings()