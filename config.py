# config.py - Simplified to use RapidAPI as the single source
import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class TennisAPIEndpoints(BaseModel):
    """Simplified: Only includes RapidAPI as the main data source."""
    rapidapi_base: str = Field(default="https://tennisapi1.p.rapidapi.com/api/tennis")

class TennisAPICredentials(BaseModel):
    """Simplified: Only includes RapidAPI credentials."""
    rapidapi_key: Optional[str] = Field(default=None)
    rapidapi_host: str = Field(default="tennisapi1.p.rapidapi.com")

class TennisIntelligenceConfig(BaseModel):
    enable_betting_intelligence: bool = Field(default=True)

class DatabaseConfig(BaseModel):
    database_url: str = Field(default="sqlite:///tennis_intelligence.db")
    enable_caching: bool = Field(default=True)

class TennisAPIConfig(BaseModel):
    """Master tennis API configuration, simplified."""
    endpoints: TennisAPIEndpoints = Field(default_factory=TennisAPIEndpoints)
    credentials: TennisAPICredentials = Field(default_factory=TennisAPICredentials)
    intelligence: TennisIntelligenceConfig = Field(default_factory=TennisIntelligenceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    request_timeout_seconds: int = Field(default=30)

def load_tennis_config() -> TennisAPIConfig:
    """Load tennis configuration from environment variables for RapidAPI."""
    endpoints = TennisAPIEndpoints(
        rapidapi_base=os.getenv("TENNIS_RAPIDAPI_BASE", "https://tennisapi1.p.rapidapi.com/api/tennis"),
    )
    credentials = TennisAPICredentials(
        rapidapi_key=os.getenv("TENNIS_RAPIDAPI_KEY"),
        rapidapi_host=os.getenv("TENNIS_RAPIDAPI_HOST", "tennisapi1.p.rapidapi.com"),
    )
    return TennisAPIConfig(endpoints=endpoints, credentials=credentials)

tennis_config = load_tennis_config()

def validate_tennis_config() -> Dict[str, bool]:
    return {"has_api_credentials": bool(tennis_config.credentials.rapidapi_key)}

# --- Settings for RAG/LLM System (for backward compatibility) ---
class Settings:
    def __init__(self):
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_gemini_api_key_here")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.LOCAL_API_SERVER_HOST = os.getenv("LOCAL_API_SERVER_HOST", "127.0.0.1")
        self.LOCAL_API_SERVER_PORT = int(os.getenv("LOCAL_API_SERVER_PORT", "8000"))
        self.KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./kb_docs"))
        self.VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "./vector_store"))
        self.VECTOR_STORE_PATH = self.VECTOR_STORE_DIR / "tennis_faiss.index"
        self.VECTOR_STORE_METADATA_PATH = self.VECTOR_STORE_DIR / "tennis_faiss_metadata.pkl"
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()