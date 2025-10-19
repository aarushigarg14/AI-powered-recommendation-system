"""
Simplified AI-Powered Furniture Recommendation Platform
FastAPI Backend Server - Quick Start Version

This is a lightweight version that starts quickly without heavy AI initialization.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Furniture Recommendation Platform - Quick Start",
    description="Lightweight furniture discovery system for development",
    version="1.0.0-dev",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None
    max_results: Optional[int] = Field(8, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]

# Mock furniture data
MOCK_FURNITURE = [
    {
        "id": "chair-001",
        "title": "Modern Ergonomic Office Chair",
        "price": 299.99,
        "category": "Office Furniture",
        "material": "Fabric",
        "color": "Black",
        "brand": "ComfortCorp",
        "description": "Experience ultimate comfort with this modern ergonomic office chair featuring lumbar support and breathable fabric.",
        "images": ["https://via.placeholder.com/300x200?text=Office+Chair"],
        "rating": 4.5,
        "similarity_score": 0.95
    },
    {
        "id": "table-001",
        "title": "Scandinavian Oak Dining Table",
        "price": 599.99,
        "category": "Dining Room",
        "material": "Oak Wood",
        "color": "Natural",
        "brand": "Nordic Design",
        "description": "Beautifully crafted Scandinavian dining table made from solid oak wood, perfect for family gatherings.",
        "images": ["https://via.placeholder.com/300x200?text=Dining+Table"],
        "rating": 4.8,
        "similarity_score": 0.87
    },
    {
        "id": "sofa-001",
        "title": "Luxury Velvet Sectional Sofa",
        "price": 1299.99,
        "category": "Living Room",
        "material": "Velvet",
        "color": "Navy Blue",
        "brand": "Elegance Home",
        "description": "Indulge in luxury with this premium velvet sectional sofa featuring deep seating and elegant design.",
        "images": ["https://via.placeholder.com/300x200?text=Velvet+Sofa"],
        "rating": 4.7,
        "similarity_score": 0.82
    },
    {
        "id": "shelf-001",
        "title": "Minimalist Bookshelf Storage Unit",
        "price": 199.99,
        "category": "Storage",
        "material": "MDF",
        "color": "White",
        "brand": "Simple Living",
        "description": "Clean and modern bookshelf perfect for organizing books and decorative items in any room.",
        "images": ["https://via.placeholder.com/300x200?text=Bookshelf"],
        "rating": 4.3,
        "similarity_score": 0.75
    },
    {
        "id": "bed-001",
        "title": "Platform Bed Frame King Size",
        "price": 449.99,
        "category": "Bedroom",
        "material": "Pine Wood",
        "color": "Walnut",
        "brand": "Sleep Haven",
        "description": "Sturdy platform bed frame with modern design, perfect for contemporary bedrooms.",
        "images": ["https://via.placeholder.com/300x200?text=Bed+Frame"],
        "rating": 4.6,
        "similarity_score": 0.78
    }
]

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI-Powered Furniture Recommendation Platform - Quick Start",
        "version": "1.0.0-dev",
        "status": "running",
        "features": [
            "Mock Furniture Search",
            "Analytics Dashboard",
            "Health Monitoring"
        ],
        "endpoints": {
            "search": "/api/search",
            "analytics": "/api/analytics",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0-dev",
        components={
            "api": "operational",
            "database": "mock",
            "ai_models": "mock"
        }
    )

# Search endpoint
@app.post("/api/search")
async def search_furniture(request: SearchRequest):
    """Search for furniture products"""
    start_time = time.time()
    
    try:
        logger.info(f"Search query: '{request.query}'")
        
        # Simple keyword-based filtering
        query_lower = request.query.lower()
        results = []
        
        for furniture in MOCK_FURNITURE:
            # Check if query matches title, category, material, or color
            matches = any([
                query_lower in furniture['title'].lower(),
                query_lower in furniture['category'].lower(),
                query_lower in furniture['material'].lower(),
                query_lower in furniture['color'].lower(),
                any(word in furniture['title'].lower() for word in query_lower.split())
            ])
            
            if matches:
                results.append(furniture)
        
        # If no matches, return all products
        if not results:
            results = MOCK_FURNITURE
        
        # Limit results
        results = results[:request.max_results]
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "message": f"Found {len(results)} products matching '{request.query}'",
            "query": request.query,
            "session_id": request.session_id,
            "results_count": len(results),
            "results": results,
            "processing_time": round(processing_time, 3),
            "data": {
                "results": results
            }
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {
            "success": False,
            "message": "Search encountered an error. Please try again.",
            "query": request.query,
            "session_id": request.session_id,
            "results_count": 0,
            "results": [],
            "processing_time": round(time.time() - start_time, 3)
        }

# Analytics endpoint
@app.get("/api/analytics")
async def get_analytics():
    """Get analytics data"""
    
    # Mock analytics data
    analytics_data = {
        "success": True,
        "data": {
            "summary": {
                "total_products": len(MOCK_FURNITURE),
                "total_categories": len(set(f['category'] for f in MOCK_FURNITURE)),
                "average_price": round(sum(f['price'] for f in MOCK_FURNITURE) / len(MOCK_FURNITURE), 2),
                "price_range": {
                    "min": min(f['price'] for f in MOCK_FURNITURE),
                    "max": max(f['price'] for f in MOCK_FURNITURE)
                }
            },
            "categories": [
                {"name": "Office Furniture", "count": 1, "percentage": 20},
                {"name": "Dining Room", "count": 1, "percentage": 20},
                {"name": "Living Room", "count": 1, "percentage": 20},
                {"name": "Storage", "count": 1, "percentage": 20},
                {"name": "Bedroom", "count": 1, "percentage": 20}
            ],
            "price_distribution": [
                {"range": "Under $300", "count": 2, "percentage": 40},
                {"range": "$300-$600", "count": 1, "percentage": 20},
                {"range": "$600-$1000", "count": 1, "percentage": 20},
                {"range": "Over $1000", "count": 1, "percentage": 20}
            ],
            "top_brands": [
                {"name": "ComfortCorp", "count": 1, "revenue": 299.99},
                {"name": "Nordic Design", "count": 1, "revenue": 599.99},
                {"name": "Elegance Home", "count": 1, "revenue": 1299.99},
                {"name": "Simple Living", "count": 1, "revenue": 199.99},
                {"name": "Sleep Haven", "count": 1, "revenue": 449.99}
            ],
            "materials": [
                {"name": "Fabric", "count": 1, "percentage": 20},
                {"name": "Oak Wood", "count": 1, "percentage": 20},
                {"name": "Velvet", "count": 1, "percentage": 20},
                {"name": "MDF", "count": 1, "percentage": 20},
                {"name": "Pine Wood", "count": 1, "percentage": 20}
            ],
            "monthly_trends": [
                {"month": "Jan", "products_added": 15, "searches": 1250, "revenue": 45000},
                {"month": "Feb", "products_added": 12, "searches": 1100, "revenue": 38000},
                {"month": "Mar", "products_added": 18, "searches": 1350, "revenue": 52000},
                {"month": "Apr", "products_added": 22, "searches": 1450, "revenue": 61000},
                {"month": "May", "products_added": 25, "searches": 1600, "revenue": 68000},
                {"month": "Jun", "products_added": 20, "searches": 1500, "revenue": 58000}
            ]
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return analytics_data

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later."
        }
    )

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting simple furniture recommendation API...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")