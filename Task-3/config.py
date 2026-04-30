#config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GEMINI_MODEL = "google/gemini-2.0-flash-001"
MAX_TOKENS   = 10000
TEMPERATURE  = 0.2

MODEL_FOLDER = GEMINI_MODEL.replace("/", "_")