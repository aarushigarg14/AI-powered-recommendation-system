"""
Working AI-Powered Furniture Recommendation Platform
FastAPI Backend Server - Using CSV Dataset

This server uses the intern_data_ikarus.csv dataset for search functionality.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import time
import csv
import re
import ast
import os
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Furniture Recommendation Platform - Working Version",
    description="Furniture discovery system using real CSV dataset",
    version="1.0.0-working",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8001", "http://127.0.0.1:8001"],
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

# Global variable to store loaded furniture data
_furniture_dataset: Optional[List[Dict[str, Any]]] = None

def load_furniture_dataset() -> List[Dict[str, Any]]:
    """Load furniture data from CSV file"""
    global _furniture_dataset
    
    if _furniture_dataset is not None:
        return _furniture_dataset
    
    # Construct path to CSV file
    current_dir = Path(__file__).parent  # backend directory
    project_dir = current_dir.parent  # aarushi project final directory
    csv_path = project_dir / "data" / "intern_data_ikarus.csv"
    
    logger.info(f"Loading furniture dataset from: {csv_path}")
    
    if not csv_path.exists():
        logger.error(f"CSV file not found at: {csv_path}")
        return []
    
    try:
        furniture_data = []
        with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                try:
                    # Parse price
                    price = None
                    if row.get('price') and row['price'].strip():
                        price_str = re.sub(r'[^0-9.]', '', row['price'])
                        if price_str:
                            price = float(price_str)
                    
                    # Parse categories (convert string representation of list to actual list)
                    categories = []
                    if row.get('categories'):
                        try:
                            categories = ast.literal_eval(row['categories'])
                            if not isinstance(categories, list):
                                categories = [str(categories)]
                        except (ValueError, SyntaxError):
                            categories = [row['categories']]
                    
                    # Parse images (convert string representation of list to actual list)
                    images = []
                    if row.get('images'):
                        try:
                            images = ast.literal_eval(row['images'])
                            if not isinstance(images, list):
                                images = [str(images)]
                            # Clean up image URLs (remove extra spaces)
                            images = [img.strip() for img in images if img and img.strip()]
                        except (ValueError, SyntaxError):
                            images = [row['images']] if row['images'] else []
                    
                    # Extract primary category from categories list
                    primary_category = None
                    if categories:
                        # Use the most specific category (usually the last one)
                        primary_category = categories[-1] if isinstance(categories, list) else str(categories)
                    
                    # Clean and prepare the product data
                    product = {
                        "id": row.get('uniq_id', f"product-{len(furniture_data)}"),
                        "title": row.get('title', '').strip(),
                        "price": price,
                        "category": primary_category,
                        "material": row.get('material', '').strip() or None,
                        "color": row.get('color', '').strip() or None,
                        "brand": row.get('brand', '').strip() or None,
                        "description": row.get('description', '').strip() or row.get('title', '').strip(),
                        "original_description": row.get('description', '').strip(),
                        "images": images,
                        "primary_image": images[0] if images else None,
                        "categories": categories,
                        "manufacturer": row.get('manufacturer', '').strip() or None,
                        "country_of_origin": row.get('country_of_origin', '').strip() or None,
                        "package_dimensions": row.get('package_dimensions', '').strip() or None,
                        "similarity_score": 1.0  # Will be calculated during search
                    }
                    
                    # Only add products with valid titles
                    if product['title']:
                        furniture_data.append(product)
                        
                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    continue
                    
        _furniture_dataset = furniture_data
        logger.info(f"Successfully loaded {len(furniture_data)} furniture products from CSV")
        return furniture_data
        
    except Exception as e:
        logger.error(f"Error loading furniture dataset: {e}")
        return []

def search_furniture_dataset(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Search furniture dataset based on query"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        logger.warning("No furniture dataset available")
        return []
    
    query_lower = query.lower()
    query_words = query_lower.split()
    
    scored_products = []
    
    for product in dataset:
        score = 0.0
        
        # Score based on title matching
        title_lower = product['title'].lower()
        for word in query_words:
            if word in title_lower:
                # Exact word match gets higher score
                if word == title_lower or f' {word} ' in f' {title_lower} ':
                    score += 3.0
                else:
                    score += 1.0
        
        # Score based on category matching
        if product['category']:
            category_lower = product['category'].lower()
            for word in query_words:
                if word in category_lower:
                    score += 2.0
        
        # Score based on categories list matching
        if product['categories']:
            for category in product['categories']:
                category_lower = str(category).lower()
                for word in query_words:
                    if word in category_lower:
                        score += 1.5
        
        # Score based on description matching
        description_lower = product['description'].lower()
        for word in query_words:
            if word in description_lower:
                score += 1.0
        
        # Score based on material matching
        if product['material']:
            material_lower = product['material'].lower()
            for word in query_words:
                if word in material_lower:
                    score += 2.0
        
        # Score based on color matching
        if product['color']:
            color_lower = product['color'].lower()
            for word in query_words:
                if word in color_lower:
                    score += 2.0
        
        # Score based on brand matching
        if product['brand']:
            brand_lower = product['brand'].lower()
            for word in query_words:
                if word in brand_lower:
                    score += 1.5
        
        # Only include products with some relevance
        if score > 0:
            product_copy = product.copy()
            product_copy['similarity_score'] = round(score, 2)
            scored_products.append(product_copy)
    
    # Sort by score (descending) and return top results
    scored_products.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # If no scored results, return some random products
    if not scored_products:
        logger.info(f"No direct matches for '{query}', returning random products")
        random_products = random.sample(dataset, min(max_results, len(dataset)))
        for product in random_products:
            product['similarity_score'] = 0.1
        return random_products
    
    return scored_products[:max_results]

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    dataset_size = len(load_furniture_dataset())
    
    return {
        "message": "AI-Powered Furniture Recommendation Platform - Working Version",
        "version": "1.0.0-working",
        "status": "running",
        "dataset_info": {
            "source": "intern_data_ikarus.csv",
            "products_loaded": dataset_size
        },
        "features": [
            "Real CSV Dataset Search",
            "Advanced Scoring Algorithm",
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
    dataset = load_furniture_dataset()
    
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0-working",
        components={
            "api": "operational",
            "dataset": f"loaded ({len(dataset)} products)",
            "search": "operational"
        }
    )

# Search endpoint
@app.post("/api/search")
async def search_furniture(request: SearchRequest):
    """Search for furniture products using the CSV dataset"""
    start_time = time.time()
    
    try:
        logger.info(f"Search query: '{request.query}'")
        
        # Search the real dataset
        results = search_furniture_dataset(request.query, request.max_results)
        
        processing_time = time.time() - start_time
        
        # Generate a helpful response message
        if results:
            message = f"Found {len(results)} products matching '{request.query}'"
            if results[0].get('similarity_score', 0) > 2.0:
                message += " with high relevance"
            elif results[0].get('similarity_score', 0) < 0.5:
                message += " - showing similar products"
        else:
            message = f"No products found matching '{request.query}'"
        
        return {
            "success": True,
            "message": message,
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
    """Get analytics data from the loaded dataset"""
    
    dataset = load_furniture_dataset()
    
    if not dataset:
        return {"success": False, "message": "No dataset loaded"}
    
    # Calculate real analytics from the dataset
    total_products = len(dataset)
    
    # Category analysis
    category_counts = {}
    for product in dataset:
        if product.get('category'):
            category = product['category']
            category_counts[category] = category_counts.get(category, 0) + 1
    
    categories = [
        {
            "name": cat, 
            "count": count, 
            "percentage": round((count / total_products) * 100, 1)
        }
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    # Price analysis
    prices = [p['price'] for p in dataset if p.get('price') is not None]
    avg_price = sum(prices) / len(prices) if prices else 0
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    
    # Material analysis
    material_counts = {}
    for product in dataset:
        if product.get('material'):
            material = product['material']
            material_counts[material] = material_counts.get(material, 0) + 1
    
    materials = [
        {
            "name": mat, 
            "count": count, 
            "percentage": round((count / total_products) * 100, 1)
        }
        for mat, count in sorted(material_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    # Brand analysis
    brand_counts = {}
    for product in dataset:
        if product.get('brand'):
            brand = product['brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    top_brands = [
        {
            "name": brand, 
            "count": count,
            "percentage": round((count / total_products) * 100, 1)
        }
        for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    ]
    
    analytics_data = {
        "success": True,
        "data": {
            "summary": {
                "total_products": total_products,
                "total_categories": len(category_counts),
                "total_brands": len(brand_counts),
                "total_materials": len(material_counts),
                "average_price": round(avg_price, 2) if avg_price else 0,
                "price_range": {
                    "min": min_price,
                    "max": max_price
                },
                "products_with_prices": len(prices),
                "products_with_images": len([p for p in dataset if p.get('images')])
            },
            "categories": categories,
            "materials": materials,
            "top_brands": top_brands,
            "price_distribution": [
                {"range": "Under $50", "count": len([p for p in prices if p < 50]), "percentage": round((len([p for p in prices if p < 50]) / len(prices)) * 100, 1) if prices else 0},
                {"range": "$50-$200", "count": len([p for p in prices if 50 <= p < 200]), "percentage": round((len([p for p in prices if 50 <= p < 200]) / len(prices)) * 100, 1) if prices else 0},
                {"range": "$200-$500", "count": len([p for p in prices if 200 <= p < 500]), "percentage": round((len([p for p in prices if 200 <= p < 500]) / len(prices)) * 100, 1) if prices else 0},
                {"range": "$500-$1000", "count": len([p for p in prices if 500 <= p < 1000]), "percentage": round((len([p for p in prices if 500 <= p < 1000]) / len(prices)) * 100, 1) if prices else 0},
                {"range": "Over $1000", "count": len([p for p in prices if p >= 1000]), "percentage": round((len([p for p in prices if p >= 1000]) / len(prices)) * 100, 1) if prices else 0}
            ]
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    return analytics_data

# Debug endpoint to view sample data
@app.get("/api/debug/sample")
async def get_sample_data():
    """Get a sample of the loaded dataset for debugging"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        return {"error": "No dataset loaded"}
    
    # Return first 3 products as sample
    sample = dataset[:3]
    
    return {
        "total_products": len(dataset),
        "sample_products": sample,
        "sample_size": len(sample)
    }

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
    logger.info("Starting working furniture recommendation API with CSV dataset...")
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
