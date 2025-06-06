# config.py - COMPLETE WORKING VERSION
import os
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TennisAPIEndpoints(BaseModel):
    """Tennis API endpoint configuration"""
    primary_live_api: str = Field(default="https://api.edgeai.pro/api/tennis")
    rapidapi_base: str = Field(default="https://tennisapi1.p.rapidapi.com/api/tennis")
    sportsdata_base: str = Field(default="https://api.sportsdata.io/v3/tennis")
    backup_endpoints: List[str] = Field(default_factory=list)


class TennisAPICredentials(BaseModel):
    """Tennis API credentials configuration"""
    rapidapi_key: Optional[str] = Field(default=None)
    rapidapi_host: str = Field(default="tennisapi1.p.rapidapi.com")
    sportsdata_key: Optional[str] = Field(default=None)
    edgeai_key: Optional[str] = Field(default=None)


class TennisIntelligenceConfig(BaseModel):
    """Tennis intelligence analysis configuration"""
    default_analysis_depth: str = Field(default="comprehensive")
    enable_live_odds: bool = Field(default=True)
    enable_h2h_analysis: bool = Field(default=True)
    enable_form_analysis: bool = Field(default=True)
    enable_surface_analysis: bool = Field(default=True)
    enable_betting_intelligence: bool = Field(default=True)

    # Analysis timeframes
    recent_form_matches: int = Field(default=10)
    head_to_head_limit: int = Field(default=20)
    ranking_history_days: int = Field(default=365)

    # Betting analysis settings
    value_threshold: float = Field(default=0.05)  # 5% edge minimum
    risk_tolerance: str = Field(default="medium")
    betting_confidence_levels: List[str] = Field(default=["low", "medium", "high", "very_high"])


class DatabaseConfig(BaseModel):
    """Database configuration for tennis data"""
    database_url: str = Field(default="sqlite:///tennis_intelligence.db")
    enable_caching: bool = Field(default=True)
    cache_duration_minutes: int = Field(default=30)
    auto_update_rankings: bool = Field(default=True)
    auto_update_interval_hours: int = Field(default=24)


class TennisAPIConfig(BaseModel):
    """Master tennis API configuration"""
    endpoints: TennisAPIEndpoints = Field(default_factory=TennisAPIEndpoints)
    credentials: TennisAPICredentials = Field(default_factory=TennisAPICredentials)
    intelligence: TennisIntelligenceConfig = Field(default_factory=TennisIntelligenceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # Global settings
    request_timeout_seconds: int = Field(default=30)
    max_retries: int = Field(default=3)
    rate_limit_calls_per_minute: int = Field(default=60)
    enable_fallback_data: bool = Field(default=True)

    class Config:
        env_prefix = "TENNIS_"


# Create global configuration instance
def load_tennis_config() -> TennisAPIConfig:
    """Load tennis configuration from environment variables"""

    # API Endpoints from environment
    endpoints = TennisAPIEndpoints(
        primary_live_api=os.getenv("TENNIS_PRIMARY_API", "https://api.edgeai.pro/api/tennis"),
        rapidapi_base=os.getenv("TENNIS_RAPIDAPI_BASE", "https://tennisapi1.p.rapidapi.com/api/tennis"),
        sportsdata_base=os.getenv("TENNIS_SPORTSDATA_BASE", "https://api.sportsdata.io/v3/tennis"),
        backup_endpoints=os.getenv("TENNIS_BACKUP_ENDPOINTS", "").split(",") if os.getenv(
            "TENNIS_BACKUP_ENDPOINTS") else []
    )

    # API Credentials from environment
    credentials = TennisAPICredentials(
        rapidapi_key=os.getenv("TENNIS_RAPIDAPI_KEY"),
        rapidapi_host=os.getenv("TENNIS_RAPIDAPI_HOST", "tennisapi1.p.rapidapi.com"),
        sportsdata_key=os.getenv("TENNIS_SPORTSDATA_KEY"),
        edgeai_key=os.getenv("TENNIS_EDGEAI_KEY")
    )

    # Intelligence configuration from environment
    intelligence = TennisIntelligenceConfig(
        default_analysis_depth=os.getenv("TENNIS_ANALYSIS_DEPTH", "comprehensive"),
        enable_live_odds=os.getenv("TENNIS_ENABLE_LIVE_ODDS", "true").lower() == "true",
        enable_h2h_analysis=os.getenv("TENNIS_ENABLE_H2H", "true").lower() == "true",
        enable_form_analysis=os.getenv("TENNIS_ENABLE_FORM", "true").lower() == "true",
        enable_surface_analysis=os.getenv("TENNIS_ENABLE_SURFACE", "true").lower() == "true",
        enable_betting_intelligence=os.getenv("TENNIS_ENABLE_BETTING", "true").lower() == "true",
        recent_form_matches=int(os.getenv("TENNIS_FORM_MATCHES", "10")),
        head_to_head_limit=int(os.getenv("TENNIS_H2H_LIMIT", "20")),
        ranking_history_days=int(os.getenv("TENNIS_RANKING_DAYS", "365")),
        value_threshold=float(os.getenv("TENNIS_VALUE_THRESHOLD", "0.05")),
        risk_tolerance=os.getenv("TENNIS_RISK_TOLERANCE", "medium")
    )

    # Database configuration from environment
    database = DatabaseConfig(
        database_url=os.getenv("TENNIS_DATABASE_URL", "sqlite:///tennis_intelligence.db"),
        enable_caching=os.getenv("TENNIS_ENABLE_CACHING", "true").lower() == "true",
        cache_duration_minutes=int(os.getenv("TENNIS_CACHE_MINUTES", "30")),
        auto_update_rankings=os.getenv("TENNIS_AUTO_UPDATE", "true").lower() == "true",
        auto_update_interval_hours=int(os.getenv("TENNIS_UPDATE_HOURS", "24"))
    )

    return TennisAPIConfig(
        endpoints=endpoints,
        credentials=credentials,
        intelligence=intelligence,
        database=database,
        request_timeout_seconds=int(os.getenv("TENNIS_TIMEOUT", "30")),
        max_retries=int(os.getenv("TENNIS_MAX_RETRIES", "3")),
        rate_limit_calls_per_minute=int(os.getenv("TENNIS_RATE_LIMIT", "60")),
        enable_fallback_data=os.getenv("TENNIS_ENABLE_FALLBACK", "true").lower() == "true"
    )


# Global configuration instance
tennis_config = load_tennis_config()


# Configuration validation
def validate_tennis_config() -> Dict[str, bool]:
    """Validate tennis configuration and return status"""
    validation_results = {
        "has_primary_api": bool(tennis_config.endpoints.primary_live_api),
        "has_api_credentials": bool(tennis_config.credentials.rapidapi_key or tennis_config.credentials.sportsdata_key),
        "intelligence_enabled": tennis_config.intelligence.enable_betting_intelligence,
        "database_configured": bool(tennis_config.database.database_url),
        "fallback_available": tennis_config.enable_fallback_data
    }

    return validation_results


# CRITICAL: Export settings for backward compatibility with api_server.py
settings = tennis_config


# Also create old-style attributes for backward compatibility
class Settings:
    def __init__(self):
        # LLM Configuration
        self.LLM_PROVIDER = "gemini"
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "your_gemini_api_key_here")
        self.GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))

        # Ollama Configuration
        self.OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
        self.OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

        # API Server Configuration
        self.LOCAL_API_SERVER_HOST = os.getenv("LOCAL_API_SERVER_HOST", "127.0.0.1")
        self.LOCAL_API_SERVER_PORT = int(os.getenv("LOCAL_API_SERVER_PORT", "8000"))
        self.REMOTE_BACKEND_URL = os.getenv("REMOTE_BACKEND_URL")

        # RAG Configuration
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))

        # Directory Configuration
        self.KNOWLEDGE_BASE_DIR = Path(os.getenv("KNOWLEDGE_BASE_DIR", "./kb_docs"))
        self.VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", "./vector_store"))
        self.VECTOR_STORE_PATH = self.VECTOR_STORE_DIR / "tennis_faiss.index"
        self.VECTOR_STORE_METADATA_PATH = self.VECTOR_STORE_DIR / "tennis_faiss_metadata.pkl"

        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


# Create settings instance for backward compatibility
settings = Settings()

# Also keep tennis_config available
tennis_config = load_tennis_config()

if __name__ == "__main__":
    print("🎾 Tennis API Configuration System")
    print("=" * 50)

    # Load and display configuration
    config = load_tennis_config()
    print(f"✅ Configuration loaded successfully")

    # Validate configuration
    validation = validate_tennis_config()
    print(f"\n📋 Configuration Validation:")
    for check, status in validation.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check}: {status}")

    # Show configuration summary
    print(f"\n⚙️  Configuration Summary:")
    print(f"   • Primary API: {config.endpoints.primary_live_api}")
    print(f"   • Intelligence Level: {config.intelligence.default_analysis_depth}")
    print(f"   • Betting Intelligence: {config.intelligence.enable_betting_intelligence}")
    print(f"   • Database: {config.database.database_url}")
    print(f"   • Caching: {config.database.enable_caching}")
    print(f"   • Fallback Data: {config.enable_fallback_data}")

    settings = tennis_config