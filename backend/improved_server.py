"""
Improved AI-Powered Furniture Recommendation Platform
FastAPI Backend Server - Production Ready

This server uses the intern_data_ikarus.csv dataset with better configuration.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Furniture Recommendation Platform",
    description="Intelligent furniture discovery system using real CSV dataset with advanced search",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware with comprehensive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    max_results: Optional[int] = Field(8, ge=1, le=20, description="Maximum results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    dataset_info: Dict[str, Any]

# Global variable to store loaded furniture data
_furniture_dataset: Optional[List[Dict[str, Any]]] = None

def load_furniture_dataset() -> List[Dict[str, Any]]:
    """Load furniture data from CSV file with improved error handling"""
    global _furniture_dataset
    
    if _furniture_dataset is not None:
        logger.info(f"Using cached dataset with {len(_furniture_dataset)} products")
        return _furniture_dataset
    
    # Construct path to CSV file
    current_dir = Path(__file__).parent  # backend directory
    project_dir = current_dir.parent  # aarushi project final directory
    csv_path = project_dir / "data" / "intern_data_ikarus.csv"
    
    logger.info(f"Loading furniture dataset from: {csv_path}")
    
    if not csv_path.exists():
        logger.error(f"CSV file not found at: {csv_path}")
        logger.info(f"Current directory: {current_dir}")
        logger.info(f"Project directory: {project_dir}")
        # List files in data directory if it exists
        data_dir = project_dir / "data"
        if data_dir.exists():
            logger.info(f"Files in data directory: {list(data_dir.glob('*.csv'))}")
        return []
    
    try:
        furniture_data = []
        with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            logger.info(f"CSV headers: {reader.fieldnames}")
            
            for row_num, row in enumerate(reader, 1):
                try:
                    # Parse price with better error handling
                    price = None
                    if row.get('price') and row['price'].strip():
                        price_str = re.sub(r'[^0-9.]', '', row['price'])
                        if price_str:
                            try:
                                price = float(price_str)
                            except ValueError:
                                logger.debug(f"Could not parse price '{row['price']}' in row {row_num}")
                    
                    # Parse categories safely
                    categories = []
                    if row.get('categories'):
                        try:
                            categories = ast.literal_eval(row['categories'])
                            if not isinstance(categories, list):
                                categories = [str(categories)]
                        except (ValueError, SyntaxError):
                            categories = [row['categories']]
                    
                    # Parse images safely
                    images = []
                    if row.get('images'):
                        try:
                            images = ast.literal_eval(row['images'])
                            if not isinstance(images, list):
                                images = [str(images)]
                            # Clean up image URLs
                            images = [img.strip() for img in images if img and img.strip()]
                        except (ValueError, SyntaxError):
                            if row['images'].strip():
                                images = [row['images'].strip()]
                    
                    # Extract primary category
                    primary_category = None
                    if categories:
                        primary_category = categories[-1] if isinstance(categories, list) else str(categories)
                    
                    # Clean and prepare the product data
                    product = {
                        "id": row.get('uniq_id', f"product-{row_num}"),
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
                    if product['title'] and len(product['title']) > 2:
                        furniture_data.append(product)
                        
                except Exception as e:
                    logger.warning(f"Error processing row {row_num}: {e}")
                    continue
                    
        _furniture_dataset = furniture_data
        logger.info(f"Successfully loaded {len(furniture_data)} furniture products from CSV")
        
        # Log some statistics
        categories = set()
        brands = set() 
        materials = set()
        prices = []
        
        for product in furniture_data:
            if product.get('category'):
                categories.add(product['category'])
            if product.get('brand'):
                brands.add(product['brand'])
            if product.get('material'):
                materials.add(product['material'])
            if product.get('price') is not None:
                prices.append(product['price'])
        
        logger.info(f"Dataset statistics: {len(categories)} categories, {len(brands)} brands, {len(materials)} materials")
        if prices:
            logger.info(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}, Average: ${sum(prices)/len(prices):.2f}")
        
        return furniture_data
        
    except Exception as e:
        logger.error(f"Error loading furniture dataset: {e}")
        return []

def search_furniture_dataset(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Enhanced search function with better scoring"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        logger.warning("No furniture dataset available")
        return []
    
    query_lower = query.lower()
    query_words = [word for word in query_lower.split() if len(word) > 1]
    
    if not query_words:
        return random.sample(dataset, min(max_results, len(dataset)))
    
    scored_products = []
    
    for product in dataset:
        score = 0.0
        
        # Title matching (highest weight)
        title_lower = product['title'].lower()
        for word in query_words:
            if word in title_lower:
                # Exact word boundary match
                if f' {word} ' in f' {title_lower} ':
                    score += 5.0
                # Word at start or end
                elif title_lower.startswith(word) or title_lower.endswith(word):
                    score += 4.0
                # Partial match
                else:
                    score += 2.0
        
        # Category matching
        if product.get('category'):
            category_lower = product['category'].lower()
            for word in query_words:
                if word in category_lower:
                    score += 3.0
        
        # Categories list matching
        if product.get('categories'):
            for category in product['categories']:
                category_lower = str(category).lower()
                for word in query_words:
                    if word in category_lower:
                        score += 2.0
        
        # Description matching
        if product.get('description'):
            description_lower = product['description'].lower()
            for word in query_words:
                if word in description_lower:
                    score += 1.5
        
        # Material matching
        if product.get('material'):
            material_lower = product['material'].lower()
            for word in query_words:
                if word in material_lower:
                    score += 3.0
        
        # Color matching
        if product.get('color'):
            color_lower = product['color'].lower()
            for word in query_words:
                if word in color_lower:
                    score += 3.0
        
        # Brand matching
        if product.get('brand'):
            brand_lower = product['brand'].lower()
            for word in query_words:
                if word in brand_lower:
                    score += 2.0
        
        # Only include products with some relevance
        if score > 0:
            product_copy = product.copy()
            product_copy['similarity_score'] = round(score, 2)
            scored_products.append(product_copy)
    
    # Sort by score (descending)
    scored_products.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # If no scored results, return some random products
    if not scored_products:
        logger.info(f"No matches for '{query}', returning random products")
        random_products = random.sample(dataset, min(max_results, len(dataset)))
        for product in random_products:
            product['similarity_score'] = 0.1
        return random_products
    
    return scored_products[:max_results]

# Startup event to load dataset
@app.on_event("startup")
async def startup_event():
    """Load dataset on startup"""
    logger.info("Starting AI Furniture Recommendation Platform...")
    dataset = load_furniture_dataset()
    logger.info(f"Startup complete with {len(dataset)} products loaded")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with comprehensive system information"""
    dataset = load_furniture_dataset()
    
    return {
        "message": "AI-Powered Furniture Recommendation Platform",
        "version": "1.0.0",
        "status": "operational",
        "dataset_info": {
            "source": "intern_data_ikarus.csv",
            "products_loaded": len(dataset),
            "status": "ready" if dataset else "error"
        },
        "features": [
            "Real CSV Dataset Search",
            "Intelligent Scoring Algorithm", 
            "Multi-field Search (title, category, material, color, brand)",
            "Analytics Dashboard",
            "Health Monitoring"
        ],
        "endpoints": {
            "search": "/api/search",
            "analytics": "/api/analytics", 
            "health": "/api/health",
            "sample": "/api/debug/sample",
            "docs": "/docs"
        },
        "server_info": {
            "host": "localhost", 
            "port": 8001,
            "cors": "enabled",
            "environment": "development"
        }
    }

# Health check endpoint
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with dataset verification"""
    dataset = load_furniture_dataset()
    
    return HealthResponse(
        status="healthy" if dataset else "unhealthy",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        version="1.0.0",
        dataset_info={
            "products_loaded": len(dataset),
            "source": "intern_data_ikarus.csv",
            "status": "loaded" if dataset else "failed"
        }
    )

# Enhanced search endpoint
@app.post("/api/search")
async def search_furniture(request: SearchRequest):
    """Search for furniture products using the CSV dataset"""
    start_time = time.time()
    
    try:
        logger.info(f"Search query: '{request.query}' (max_results: {request.max_results})")
        
        # Search the dataset
        results = search_furniture_dataset(request.query, request.max_results)
        
        processing_time = time.time() - start_time
        
        # Generate response message
        if results:
            top_score = results[0].get('similarity_score', 0)
            if top_score > 4.0:
                relevance = "with high relevance"
            elif top_score > 1.0:
                relevance = "with good relevance"
            else:
                relevance = "- showing similar products"
            
            message = f"Found {len(results)} products matching '{request.query}' {relevance}"
        else:
            message = f"No products found matching '{request.query}'"
        
        response = {
            "success": True,
            "message": message,
            "query": request.query,
            "session_id": request.session_id,
            "results_count": len(results),
            "results": results,
            "processing_time": round(processing_time, 4),
            "search_info": {
                "query_words": len(request.query.split()),
                "max_score": results[0]['similarity_score'] if results else 0,
                "min_score": results[-1]['similarity_score'] if results else 0
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return {
            "success": False,
            "message": f"Search encountered an error: {str(e)}",
            "query": request.query,
            "session_id": request.session_id,
            "results_count": 0,
            "results": [],
            "processing_time": round(time.time() - start_time, 4)
        }

# Analytics endpoint  
@app.get("/api/analytics")
async def get_analytics():
    """Get comprehensive analytics data from the loaded dataset"""
    
    dataset = load_furniture_dataset()
    
    if not dataset:
        return {"success": False, "message": "No dataset loaded", "data": {}}
    
    # Calculate comprehensive analytics
    total_products = len(dataset)
    
    # Category analysis
    category_counts = {}
    for product in dataset:
        if product.get('category'):
            category = product['category']
            category_counts[category] = category_counts.get(category, 0) + 1
    
    # Material analysis
    material_counts = {}
    for product in dataset:
        if product.get('material'):
            material = product['material']
            material_counts[material] = material_counts.get(material, 0) + 1
    
    # Brand analysis
    brand_counts = {}
    for product in dataset:
        if product.get('brand'):
            brand = product['brand']
            brand_counts[brand] = brand_counts.get(brand, 0) + 1
    
    # Price analysis
    prices = [p['price'] for p in dataset if p.get('price') is not None]
    avg_price = sum(prices) / len(prices) if prices else 0
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0
    
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
                "products_with_images": len([p for p in dataset if p.get('images')]),
                "products_with_descriptions": len([p for p in dataset if p.get('description')])
            },
            "categories": [
                {
                    "name": cat, 
                    "count": count, 
                    "percentage": round((count / total_products) * 100, 1)
                }
                for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            ],
            "materials": [
                {
                    "name": mat, 
                    "count": count, 
                    "percentage": round((count / total_products) * 100, 1)
                }
                for mat, count in sorted(material_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            ],
            "top_brands": [
                {
                    "name": brand, 
                    "count": count,
                    "percentage": round((count / total_products) * 100, 1)
                }
                for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            ],
            "price_distribution": [
                {"range": "Under $50", "count": len([p for p in prices if p < 50]), "percentage": round((len([p for p in prices if p < 50]) / len(prices)) * 100, 1) if prices else 0},
                {"range": "$50-$200", "count": len([p for p in prices if 50 <= p < 200]), "percentage": round((len([p for p in prices if 50 <= p < 200]) / len(prices)) * 100, 1) if prices else 0},
                {"range": "$200-$500", "count": len([p for p in prices if 200 <= p < 500]), "percentage": round((len([p for p in prices if 200 <= p < 500]) / len(prices)) * 100, 1) if prices else 0},
                {"range": "$500+", "count": len([p for p in prices if p >= 500]), "percentage": round((len([p for p in prices if p >= 500]) / len(prices)) * 100, 1) if prices else 0}
            ]
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generated_in": time.time()
    }
    
    return analytics_data

# Debug endpoint
@app.get("/api/debug/sample")
async def get_sample_data():
    """Get sample products for debugging"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        return {"error": "No dataset loaded", "dataset_path": "data/intern_data_ikarus.csv"}
    
    return {
        "total_products": len(dataset),
        "sample_products": dataset[:5],
        "dataset_info": {
            "first_product_keys": list(dataset[0].keys()) if dataset else [],
            "source": "intern_data_ikarus.csv"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Enhanced global exception handler"""
    logger.error(f"Unhandled exception on {request.url}: {str(exc)}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_url": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Load dataset at startup to verify it works
    logger.info("Pre-loading dataset to verify configuration...")
    dataset = load_furniture_dataset()
    
    if not dataset:
        logger.error("Failed to load dataset! Check that intern_data_ikarus.csv exists in the data/ directory")
        exit(1)
    
    logger.info(f"Dataset verified: {len(dataset)} products loaded")
    logger.info("Starting improved furniture recommendation API...")
    
    # Use localhost instead of 0.0.0.0 for better Windows compatibility
    uvicorn.run(
        app, 
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0
        port=8001, 
        log_level="info",
        access_log=True
    )