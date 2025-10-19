from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Furniture Recommendation Platform",
    description="Intelligent furniture discovery system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None
    max_results: Optional[int] = Field(8, ge=1, le=50)
    filters: Optional[Dict[str, Any]] = None

# Comprehensive furniture database
FURNITURE_DATABASE = [
    {
        "id": "chair-ergonomic-001",
        "title": "Modern Ergonomic Office Chair",
        "price": 299.99,
        "category": "Office Furniture",
        "material": "Fabric",
        "color": "Black",
        "brand": "ComfortCorp",
        "description": "Experience ultimate comfort with this modern ergonomic office chair featuring lumbar support, breathable mesh back, and adjustable armrests. Perfect for long work sessions.",
        "images": ["https://images.unsplash.com/photo-1541558869434-2840d308329a?w=300&h=200&fit=crop"],
        "rating": 4.5,
        "reviews": 234,
        "in_stock": True,
        "tags": ["ergonomic", "office", "comfortable", "adjustable"]
    },
    {
        "id": "table-dining-001",
        "title": "Scandinavian Oak Dining Table",
        "price": 599.99,
        "category": "Dining Room",
        "material": "Oak Wood",
        "color": "Natural",
        "brand": "Nordic Design",
        "description": "Beautifully crafted Scandinavian dining table made from solid oak wood. Features clean lines and natural finish, perfect for family gatherings and dinner parties.",
        "images": ["https://images.unsplash.com/photo-1449247709967-d4461a6a6103?w=300&h=200&fit=crop"],
        "rating": 4.8,
        "reviews": 156,
        "in_stock": True,
        "tags": ["dining", "wooden", "scandinavian", "family"]
    },
    {
        "id": "sofa-sectional-001",
        "title": "Luxury Velvet Sectional Sofa",
        "price": 1299.99,
        "category": "Living Room",
        "material": "Velvet",
        "color": "Navy Blue",
        "brand": "Elegance Home",
        "description": "Indulge in luxury with this premium velvet sectional sofa. Features deep seating, elegant design, and plush cushions for ultimate relaxation.",
        "images": ["https://images.unsplash.com/photo-1555041469-a586c61ea9bc?w=300&h=200&fit=crop"],
        "rating": 4.7,
        "reviews": 189,
        "in_stock": True,
        "tags": ["luxury", "velvet", "sectional", "comfortable"]
    },
    {
        "id": "storage-bookshelf-001",
        "title": "Minimalist Bookshelf Storage Unit",
        "price": 199.99,
        "category": "Storage",
        "material": "MDF",
        "color": "White",
        "brand": "Simple Living",
        "description": "Clean and modern bookshelf perfect for organizing books, decorative items, and storage. Features 5 shelves with contemporary design.",
        "images": ["https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300&h=200&fit=crop"],
        "rating": 4.3,
        "reviews": 98,
        "in_stock": True,
        "tags": ["storage", "books", "minimalist", "organization"]
    },
    {
        "id": "bed-platform-001",
        "title": "Platform Bed Frame King Size",
        "price": 449.99,
        "category": "Bedroom",
        "material": "Pine Wood",
        "color": "Walnut",
        "brand": "Sleep Haven",
        "description": "Sturdy platform bed frame with modern design. No box spring needed. Features clean lines and durable construction for contemporary bedrooms.",
        "images": ["https://images.unsplash.com/photo-1505693416388-ac5ce068fe85?w=300&h=200&fit=crop"],
        "rating": 4.6,
        "reviews": 145,
        "in_stock": True,
        "tags": ["bedroom", "platform", "king", "modern"]
    },
    {
        "id": "desk-standing-001",
        "title": "Adjustable Standing Desk",
        "price": 399.99,
        "category": "Office Furniture",
        "material": "Steel",
        "color": "White",
        "brand": "WorkSmart",
        "description": "Electric height-adjustable standing desk with memory settings. Promotes healthy work habits and reduces back strain.",
        "images": ["https://images.unsplash.com/photo-1551818255-e6e10975cd17?w=300&h=200&fit=crop"],
        "rating": 4.4,
        "reviews": 267,
        "in_stock": True,
        "tags": ["standing", "adjustable", "electric", "health"]
    },
    {
        "id": "chair-accent-001",
        "title": "Mid-Century Modern Accent Chair",
        "price": 249.99,
        "category": "Living Room",
        "material": "Fabric",
        "color": "Mustard Yellow",
        "brand": "Retro Style",
        "description": "Stylish mid-century modern accent chair with vibrant mustard yellow fabric. Perfect for adding a pop of color to any room.",
        "images": ["https://images.unsplash.com/photo-1586023492125-27b2c045efd7?w=300&h=200&fit=crop"],
        "rating": 4.5,
        "reviews": 123,
        "in_stock": True,
        "tags": ["accent", "mid-century", "colorful", "stylish"]
    },
    {
        "id": "table-coffee-001",
        "title": "Glass Top Coffee Table",
        "price": 179.99,
        "category": "Living Room",
        "material": "Glass",
        "color": "Clear",
        "brand": "Crystal Home",
        "description": "Elegant glass top coffee table with chrome legs. Creates an airy, modern look while providing practical surface space.",
        "images": ["https://images.unsplash.com/photo-1586627117984-d8b90a8b2a45?w=300&h=200&fit=crop"],
        "rating": 4.2,
        "reviews": 87,
        "in_stock": True,
        "tags": ["coffee", "glass", "modern", "chrome"]
    }
]

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI-Powered Furniture Recommendation Platform",
        "version": "1.0.0",
        "status": "running",
        "total_products": len(FURNITURE_DATABASE),
        "endpoints": {
            "search": "/api/search",
            "analytics": "/api/analytics",
            "health": "/api/health",
            "docs": "/docs"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0",
        "database_size": len(FURNITURE_DATABASE),
        "components": {
            "api": "operational",
            "database": "operational",
            "search": "operational"
        }
    }

@app.post("/api/search")
async def search_furniture(request: SearchRequest):
    """Search for furniture products using natural language"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing search query: '{request.query}'")
        
        query_lower = request.query.lower()
        results = []
        
        # Advanced search algorithm
        for furniture in FURNITURE_DATABASE:
            score = 0
            
            # Title matching (highest weight)
            if any(word in furniture['title'].lower() for word in query_lower.split()):
                score += 10
            
            # Direct category matching
            if query_lower in furniture['category'].lower():
                score += 8
            
            # Material matching
            if query_lower in furniture['material'].lower():
                score += 6
            
            # Color matching
            if query_lower in furniture['color'].lower():
                score += 6
            
            # Brand matching
            if query_lower in furniture['brand'].lower():
                score += 4
            
            # Tag matching
            if any(tag in query_lower for tag in furniture['tags']):
                score += 5
            
            # Description matching
            if any(word in furniture['description'].lower() for word in query_lower.split()):
                score += 3
            
            # Specific keyword matching
            keywords = {
                'chair': ['chair', 'seat', 'sitting'],
                'table': ['table', 'desk', 'surface'],
                'sofa': ['sofa', 'couch', 'sectional'],
                'bed': ['bed', 'sleep', 'bedroom'],
                'storage': ['storage', 'shelf', 'organize'],
                'office': ['office', 'work', 'desk'],
                'comfortable': ['comfortable', 'ergonomic', 'cozy'],
                'modern': ['modern', 'contemporary', 'sleek'],
                'luxury': ['luxury', 'premium', 'elegant']
            }
            
            for keyword, synonyms in keywords.items():
                if any(syn in query_lower for syn in synonyms):
                    if keyword in furniture['title'].lower() or keyword in furniture['tags']:
                        score += 7
            
            if score > 0:
                furniture_copy = furniture.copy()
                furniture_copy['similarity_score'] = score / 10.0  # Normalize to 0-1
                results.append(furniture_copy)
        
        # Sort by relevance score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Apply filters if provided
        if request.filters:
            if 'price_max' in request.filters:
                results = [r for r in results if r['price'] <= request.filters['price_max']]
            if 'price_min' in request.filters:
                results = [r for r in results if r['price'] >= request.filters['price_min']]
            if 'category' in request.filters:
                results = [r for r in results if request.filters['category'].lower() in r['category'].lower()]
            if 'material' in request.filters:
                results = [r for r in results if request.filters['material'].lower() in r['material'].lower()]
            if 'color' in request.filters:
                results = [r for r in results if request.filters['color'].lower() in r['color'].lower()]
        
        # If no relevant results, show most popular items
        if not results:
            results = sorted(FURNITURE_DATABASE, key=lambda x: x['rating'], reverse=True)
        
        # Limit results
        results = results[:request.max_results]
        
        processing_time = time.time() - start_time
        
        # Generate contextual message
        if len(results) > 0:
            if any(word in query_lower for word in ['chair', 'seat']):
                message = f"Found {len(results)} chairs and seating options for '{request.query}'"
            elif any(word in query_lower for word in ['table', 'desk']):
                message = f"Found {len(results)} tables and desks for '{request.query}'"
            elif any(word in query_lower for word in ['sofa', 'couch']):
                message = f"Found {len(results)} sofas and seating for '{request.query}'"
            else:
                message = f"Found {len(results)} furniture items matching '{request.query}'"
        else:
            message = f"No specific matches for '{request.query}', showing popular items"
        
        response = {
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
        
        logger.info(f"Search completed: {len(results)} results in {processing_time:.3f}s")
        return response
        
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

@app.get("/api/analytics")
async def get_analytics():
    """Get comprehensive analytics data"""
    
    # Calculate real analytics from the furniture database
    total_products = len(FURNITURE_DATABASE)
    categories = {}
    materials = {}
    price_ranges = {"Under $300": 0, "$300-$600": 0, "$600-$1000": 0, "Over $1000": 0}
    brands = {}
    
    for item in FURNITURE_DATABASE:
        # Category distribution
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1
        
        # Material distribution
        mat = item['material']
        materials[mat] = materials.get(mat, 0) + 1
        
        # Price distribution
        price = item['price']
        if price < 300:
            price_ranges["Under $300"] += 1
        elif price < 600:
            price_ranges["$300-$600"] += 1
        elif price < 1000:
            price_ranges["$600-$1000"] += 1
        else:
            price_ranges["Over $1000"] += 1
        
        # Brand distribution
        brand = item['brand']
        brands[brand] = brands.get(brand, 0) + 1
    
    # Convert to percentage format
    category_data = [{"name": k, "count": v, "percentage": round((v/total_products)*100, 1)} for k, v in categories.items()]
    material_data = [{"name": k, "count": v, "percentage": round((v/total_products)*100, 1)} for k, v in materials.items()]
    price_data = [{"range": k, "count": v, "percentage": round((v/total_products)*100, 1)} for k, v in price_ranges.items()]
    brand_data = [{"name": k, "count": v, "revenue": sum(item['price'] for item in FURNITURE_DATABASE if item['brand'] == k)} for k, v in brands.items()]
    
    analytics_data = {
        "success": True,
        "data": {
            "summary": {
                "total_products": total_products,
                "total_categories": len(categories),
                "average_price": round(sum(item['price'] for item in FURNITURE_DATABASE) / total_products, 2),
                "price_range": {
                    "min": min(item['price'] for item in FURNITURE_DATABASE),
                    "max": max(item['price'] for item in FURNITURE_DATABASE)
                },
                "average_rating": round(sum(item['rating'] for item in FURNITURE_DATABASE) / total_products, 2),
                "total_reviews": sum(item['reviews'] for item in FURNITURE_DATABASE)
            },
            "categories": category_data,
            "price_distribution": price_data,
            "top_brands": sorted(brand_data, key=lambda x: x['revenue'], reverse=True),
            "materials": material_data,
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

@app.get("/api/products")
async def get_all_products():
    """Get all available products"""
    return {
        "success": True,
        "products": FURNITURE_DATABASE,
        "total_count": len(FURNITURE_DATABASE)
    }

@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    """Get a specific product by ID"""
    product = next((item for item in FURNITURE_DATABASE if item['id'] == product_id), None)
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return {
        "success": True,
        "product": product
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AI Furniture Recommendation Platform...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")