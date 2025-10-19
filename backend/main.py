"""
AI-Powered Furniture Recommendation & Analytics Platform
FastAPI Backend Server

This server provides:
- Semantic search endpoint for furniture recommendations
- Analytics endpoint for business insights
- AI-powered product description generation
- Vector database integration with Pinecone
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
import os
from dotenv import load_dotenv
import logging
from typing import Optional, List, Dict, Any
import asyncio

# Import route modules
from routes.search import router as search_router
from routes.analytics import router as analytics_router
from routes.health import router as health_router

# Import models and utilities
from models.data_manager import DataManager
from models.ai_models import AIModelManager
from utils.config import Settings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize settings
settings = Settings()

# Global managers
data_manager: Optional[DataManager] = None
ai_manager: Optional[AIModelManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global data_manager, ai_manager
    
    logger.info("🚀 Starting AI Furniture Recommendation Platform...")
    
    try:
        # Initialize data manager
        logger.info("📊 Loading furniture dataset...")
        data_manager = DataManager(settings.data_path)
        await data_manager.load_data()
        
        # Initialize AI models
        logger.info("🤖 Initializing AI models...")
        ai_manager = AIModelManager(settings)
        await ai_manager.initialize_models()
        
        # Initialize vector database
        logger.info("🔍 Setting up vector database...")
        await ai_manager.setup_vector_database(data_manager.get_clean_data())
        
        logger.info("✅ Server initialization complete!")
        
        # Store managers in app state
        app.state.data_manager = data_manager
        app.state.ai_manager = ai_manager
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize server: {str(e)}")
        raise
    
    finally:
        logger.info("🛑 Shutting down server...")
        if ai_manager:
            await ai_manager.cleanup()
        logger.info("👋 Server shutdown complete!")

# Create FastAPI app
app = FastAPI(
    title="AI Furniture Recommendation Platform",
    description="Intelligent furniture discovery system powered by AI, NLP, CV, and GenAI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(search_router, prefix="/api", tags=["Search"])
app.include_router(analytics_router, prefix="/api", tags=["Analytics"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI-Powered Furniture Recommendation Platform",
        "version": "1.0.0",
        "features": [
            "Semantic Search",
            "AI-Generated Descriptions",
            "Analytics Dashboard",
            "Multi-Modal Embeddings"
        ],
        "endpoints": {
            "search": "/api/search",
            "analytics": "/api/analytics", 
            "health": "/api/health",
            "docs": "/docs"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses"""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "type": "internal_error"
        }
    )

# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "message": "Request failed",
            "type": "http_error",
            "status_code": exc.status_code
        }
    )

# Dependency to get data manager
async def get_data_manager():
    """Dependency to get data manager instance"""
    if not hasattr(app.state, 'data_manager') or app.state.data_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Data manager not available. Server may be starting up."
        )
    return app.state.data_manager

# Dependency to get AI manager
async def get_ai_manager():
    """Dependency to get AI manager instance"""
    if not hasattr(app.state, 'ai_manager') or app.state.ai_manager is None:
        raise HTTPException(
            status_code=503,
            detail="AI manager not available. Server may be starting up."
        )
    return app.state.ai_manager

# Additional middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests for debugging"""
    start_time = asyncio.get_event_loop().time()
    
    response = await call_next(request)
    
    process_time = asyncio.get_event_loop().time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response

# Health check with dependency
@app.get("/api/status")
async def detailed_status(
    data_manager: DataManager = Depends(get_data_manager),
    ai_manager: AIModelManager = Depends(get_ai_manager)
):
    """Detailed server status with component health"""
    try:
        # Check data manager status
        data_status = await data_manager.health_check()
        
        # Check AI manager status  
        ai_status = await ai_manager.health_check()
        
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "components": {
                "data_manager": data_status,
                "ai_manager": ai_status,
                "vector_database": ai_manager.is_vector_db_ready(),
                "embedding_model": ai_manager.is_embedding_model_ready(),
                "genai_model": ai_manager.is_genai_model_ready()
            },
            "dataset": {
                "total_products": len(data_manager.get_clean_data()),
                "categories": data_manager.get_category_count(),
                "valid_prices": data_manager.get_valid_price_count()
            }
        }
    
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "Server components not ready"
            }
        )

# Run server
if __name__ == "__main__":
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )