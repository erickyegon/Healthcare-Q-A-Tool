"""
Configuration settings for Healthcare Q&A Tool.
Manages environment variables and application settings.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Euri AI API Configuration
    euri_api_key: str = Field(..., description="Euri AI API key")
    euri_base_url: str = Field("https://api.euri.ai/v1", description="Euri AI base URL")
    
    # Chroma DB Configuration
    chroma_persist_directory: str = Field("./data/chroma_db", description="Chroma DB persistence directory")
    chroma_collection_name: str = Field("healthcare_articles", description="Chroma collection name")
    
    # PubMed API Configuration
    pubmed_email: Optional[str] = Field(None, description="Email for PubMed API")
    pubmed_tool_name: str = Field("healthcare_qa_tool", description="Tool name for PubMed API")
    pubmed_max_retries: int = Field(3, description="Maximum retries for PubMed API")
    pubmed_rate_limit_delay: float = Field(1.0, description="Rate limit delay in seconds")
    
    # Application Configuration
    log_level: str = Field("INFO", description="Logging level")
    max_articles_per_search: int = Field(100, description="Maximum articles per search")
    chunk_size: int = Field(1000, description="Text chunk size for processing")
    chunk_overlap: int = Field(200, description="Text chunk overlap")
    
    # Vector Store Configuration
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", 
        description="Embedding model name"
    )
    vector_dimension: int = Field(384, description="Vector dimension")
    
    # Q&A System Configuration
    llm_model: str = Field("gpt-4.1-nano", description="LLM model name")
    max_context_length: int = Field(4000, description="Maximum context length")
    temperature: float = Field(0.1, description="LLM temperature")
    max_tokens: int = Field(500, description="Maximum tokens for response")

    # Authentication & Security Configuration
    enable_authentication: bool = Field(True, description="Enable user authentication")
    jwt_secret_key: str = Field("your_jwt_secret_key_here_change_in_production", description="JWT secret key")
    session_timeout_hours: int = Field(8, description="Session timeout in hours")
    require_email_verification: bool = Field(False, description="Require email verification")
    
    @property
    def data_directory(self) -> Path:
        """Get the data directory path."""
        return Path("./data")
    
    @property
    def chroma_path(self) -> Path:
        """Get the Chroma DB path."""
        return Path(self.chroma_persist_directory)
    
    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.data_directory.mkdir(exist_ok=True)
        self.chroma_path.parent.mkdir(parents=True, exist_ok=True)
    
    def get_api_key(self) -> str:
        """Get the Euri API key."""
        return self.euri_api_key
    
    def get_llm_config(self) -> dict:
        """Get LLM configuration for Euri client."""
        return {
            "model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.euri_api_key,
            "base_url": self.euri_base_url
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings
