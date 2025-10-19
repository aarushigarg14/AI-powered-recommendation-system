"""
Configuration settings for the AI Furniture Recommendation Platform
Handles environment variables and application settings
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Application settings from environment variables"""
    
    def __init__(self):
        # Server settings
        self.host: str = os.getenv("HOST", "0.0.0.0")
        self.port: int = int(os.getenv("PORT", "8000"))
        self.debug: bool = os.getenv("DEBUG", "true").lower() == "true"
        self.environment: str = os.getenv("ENVIRONMENT", "development")
        
        # CORS settings
        self.frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
        self.allowed_origins: List[str] = [
            self.frontend_url,
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "https://localhost:3000"
        ]
        
        # Data paths
        self.data_path: str = os.getenv("DATA_PATH", "../data/intern_data_ikarus.csv")
        self.cleaned_data_path: str = os.getenv("CLEANED_DATA_PATH", "../data/cleaned_furniture_data.csv")
        
        # Pinecone settings
        self.pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
        self.pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "furniture-recommendations")
        
        # OpenAI settings (optional)
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        
        # Hugging Face settings
        self.huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
        
        # Model settings
        self.embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.genai_model: str = os.getenv("GENAI_MODEL", "google/flan-t5-small")
        self.max_results: int = int(os.getenv("MAX_RESULTS", "20"))
        self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        
        # Cache settings
        self.enable_cache: bool = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
        
        # Validate required settings
        self._validate_settings()
    
    def _validate_settings(self):
        """Validate critical settings"""
        if not self.data_path:
            raise ValueError("DATA_PATH environment variable is required")
        
        # Warning for missing optional settings
        if not self.pinecone_api_key:
            print("⚠️  Warning: PINECONE_API_KEY not set. Vector database features will be limited.")
        
        if not self.openai_api_key:
            print("ℹ️  Info: OPENAI_API_KEY not set. Using local models for text generation.")
        
        if not self.huggingface_token:
            print("ℹ️  Info: HUGGINGFACE_TOKEN not set. Rate limits may apply for model downloads.")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"
    
    def __repr__(self):
        """String representation of settings (hiding sensitive data)"""
        return f"""Settings(
    host={self.host},
    port={self.port},
    debug={self.debug},
    environment={self.environment},
    pinecone_configured={'✅' if self.pinecone_api_key else '❌'},
    openai_configured={'✅' if self.openai_api_key else '❌'},
    embedding_model={self.embedding_model},
    max_results={self.max_results}
)"""