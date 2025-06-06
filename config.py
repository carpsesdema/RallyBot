# config/tennis_api_config.py - PROFESSIONAL CONFIGURATION SYSTEM
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


# Environment template for clients
def generate_env_template() -> str:
    """Generate environment variable template for clients"""
    template = """
# Tennis API Configuration - Professional Setup
# Copy this to your .env file and configure with your API keys

# === PRIMARY API ENDPOINTS ===
TENNIS_PRIMARY_API=https://api.edgeai.pro/api/tennis
TENNIS_RAPIDAPI_BASE=https://tennisapi1.p.rapidapi.com/api/tennis
TENNIS_SPORTSDATA_BASE=https://api.sportsdata.io/v3/tennis
TENNIS_BACKUP_ENDPOINTS=https://api.backup1.com,https://api.backup2.com

# === API CREDENTIALS ===
TENNIS_RAPIDAPI_KEY=your_rapidapi_key_here
TENNIS_RAPIDAPI_HOST=tennisapi1.p.rapidapi.com
TENNIS_SPORTSDATA_KEY=your_sportsdata_key_here
TENNIS_EDGEAI_KEY=your_edgeai_key_here

# === INTELLIGENCE SETTINGS ===
TENNIS_ANALYSIS_DEPTH=comprehensive  # basic, standard, comprehensive, elite
TENNIS_ENABLE_LIVE_ODDS=true
TENNIS_ENABLE_H2H=true
TENNIS_ENABLE_FORM=true
TENNIS_ENABLE_SURFACE=true
TENNIS_ENABLE_BETTING=true

# === ANALYSIS PARAMETERS ===
TENNIS_FORM_MATCHES=10           # Number of recent matches for form analysis
TENNIS_H2H_LIMIT=20              # Maximum H2H matches to analyze
TENNIS_RANKING_DAYS=365          # Days of ranking history
TENNIS_VALUE_THRESHOLD=0.05      # Minimum edge for bet recommendations (5%)
TENNIS_RISK_TOLERANCE=medium     # low, medium, high, very_high

# === DATABASE CONFIGURATION ===
TENNIS_DATABASE_URL=sqlite:///tennis_intelligence.db  # or postgresql://user:pass@host/db
TENNIS_ENABLE_CACHING=true
TENNIS_CACHE_MINUTES=30
TENNIS_AUTO_UPDATE=true
TENNIS_UPDATE_HOURS=24

# === API PERFORMANCE ===
TENNIS_TIMEOUT=30                # Request timeout in seconds
TENNIS_MAX_RETRIES=3             # Max retry attempts
TENNIS_RATE_LIMIT=60             # Calls per minute
TENNIS_ENABLE_FALLBACK=true      # Enable fallback data when APIs unavailable

# === PROFESSIONAL FEATURES ===
TENNIS_ENABLE_ANALYTICS=true     # Enable usage analytics
TENNIS_LOG_LEVEL=INFO            # DEBUG, INFO, WARNING, ERROR
TENNIS_METRICS_ENABLED=true      # Enable performance metrics
"""
    return template


if __name__ == "__main__":
    # Configuration testing
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

    # Generate environment template
    print(f"\n📄 Environment template generated")
    template_path = Path("tennis_api.env.template")
    with open(template_path, "w") as f:
        f.write(generate_env_template())
    print(f"   Template saved to: {template_path}")

    # Export settings for backward compatibility
    settings = tennis_config