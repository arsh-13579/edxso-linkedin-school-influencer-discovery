import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip().strip('"').strip("'")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    DB_PATH = "data/school_influencers.db"

    TARGET_PERSONAS = [
        "School & Education Consultant / Advisor",
        "Former Principal, School Director, Owner & Trustee",
        "Education Entrepreneur & School Network Professional",
        "Experienced Multi-School Ecosystem Professional",
        "School Decision-Maker Relationship Builder"
    ]
