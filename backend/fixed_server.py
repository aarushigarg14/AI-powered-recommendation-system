"""
Fixed AI-Powered Furniture Recommendation Platform
FastAPI Backend Server - Fixed Version

This server correctly loads the intern_data_ikarus.csv dataset and fixes the host binding issue.
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Furniture Recommendation Platform - Fixed",
    description="Furniture discovery system using your CSV dataset - Fixed version",
    version="1.0.1-fixed"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None
    max_results: Optional[int] = Field(8, ge=1, le=20)
    filters: Optional[Dict[str, Any]] = None

# Global dataset storage
_furniture_dataset: Optional[List[Dict[str, Any]]] = None

def load_furniture_dataset() -> List[Dict[str, Any]]:
    """Load furniture data from CSV file with detailed logging"""
    global _furniture_dataset
    
    if _furniture_dataset is not None:
        logger.info(f"Using cached dataset: {len(_furniture_dataset)} products")
        return _furniture_dataset
    
    # Find CSV file
    current_dir = Path(__file__).parent
    project_dir = current_dir.parent
    csv_path = project_dir / "data" / "intern_data_ikarus.csv"
    
    logger.info(f"Looking for CSV at: {csv_path}")
    logger.info(f"CSV exists: {csv_path.exists()}")
    
    if not csv_path.exists():
        # Try alternative paths
        alt_paths = [
            Path("data/intern_data_ikarus.csv"),
            Path("../data/intern_data_ikarus.csv"),
            Path("D:/aarushi project final/data/intern_data_ikarus.csv")
        ]
        
        for alt_path in alt_paths:
            logger.info(f"Trying alternative path: {alt_path}")
            if alt_path.exists():
                csv_path = alt_path
                logger.info(f"Found CSV at alternative path: {csv_path}")
                break
        else:
            logger.error(f"CSV file not found. Checked paths: {[csv_path] + alt_paths}")
            return []
    
    try:
        furniture_data = []
        logger.info(f"Opening CSV file: {csv_path}")
        
        with open(csv_path, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = reader.fieldnames
            logger.info(f"CSV headers: {headers}")
            
            for row_num, row in enumerate(reader, 1):
                if row_num <= 3:  # Log first few rows for debugging
                    logger.info(f"Row {row_num} sample: title='{row.get('title', '')[:50]}...', price='{row.get('price', '')}'")
                
                try:
                    # Parse price
                    price = None
                    if row.get('price') and row['price'].strip():
                        price_str = re.sub(r'[^\d.]', '', row['price'])
                        if price_str:
                            try:
                                price = float(price_str)
                            except ValueError:
                                pass
                    
                    # Parse categories
                    categories = []
                    if row.get('categories'):
                        try:
                            categories = ast.literal_eval(row['categories'])
                            if not isinstance(categories, list):
                                categories = [str(categories)]
                        except:
                            categories = [row['categories']]
                    
                    # Parse images
                    images = []
                    if row.get('images'):
                        try:
                            images = ast.literal_eval(row['images'])
                            if not isinstance(images, list):
                                images = [str(images)]
                            images = [img.strip() for img in images if img and img.strip()]
                        except:
                            if row['images'].strip():
                                images = [row['images'].strip()]
                    
                    # Get primary category
                    primary_category = None
                    if categories:
                        primary_category = categories[-1] if isinstance(categories, list) else str(categories)
                    
                    # Create product
                    product = {
                        "id": row.get('uniq_id', f"prod-{row_num}"),
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
                        "similarity_score": 1.0
                    }
                    
                    # Add if has valid title
                    if product['title'] and len(product['title']) > 2:
                        furniture_data.append(product)
                        
                except Exception as e:
                    if row_num <= 5:  # Only log first few errors
                        logger.warning(f"Error processing row {row_num}: {e}")
                    continue
        
        _furniture_dataset = furniture_data
        logger.info(f"✅ Successfully loaded {len(furniture_data)} products from CSV")
        
        if furniture_data:
            # Log sample data
            sample = furniture_data[0]
            logger.info(f"Sample product: {sample['title'][:50]}... | Price: ${sample['price']} | Category: {sample['category']}")
            
            # Statistics
            with_prices = len([p for p in furniture_data if p.get('price')])
            with_images = len([p for p in furniture_data if p.get('images')])
            categories = set(p['category'] for p in furniture_data if p.get('category'))
            
            logger.info(f"Statistics: {with_prices} with prices, {with_images} with images, {len(categories)} categories")
        
        return furniture_data
        
    except Exception as e:
        logger.error(f"❌ Failed to load CSV: {e}")
        return []

def parse_query_requirements(query: str) -> dict:
    """Parse query for special requirements like relevance levels"""
    query_lower = query.lower()
    requirements = {
        'relevance_filter': None,
        'clean_query': query,
        'sort_order': 'desc'  # default: highest relevance first
    }
    
    # Check for relevance requirements - handle various patterns
    if ('low relevance' in query_lower or 'low quality' in query_lower or 
        'with low relevance' in query_lower or 'under low relevance' in query_lower):
        requirements['relevance_filter'] = 'low'
        requirements['sort_order'] = 'asc'  # lowest relevance first
        # Remove relevance terms from query
        clean_query = query_lower
        for phrase in ['with low relevance', 'under low relevance', 'low relevance', 'low quality']:
            clean_query = clean_query.replace(phrase, '')
        clean_query = ' '.join(clean_query.split())  # clean extra spaces
        requirements['clean_query'] = clean_query
    elif ('high relevance' in query_lower or 'high quality' in query_lower or 
          'with high relevance' in query_lower or 'under high relevance' in query_lower):
        requirements['relevance_filter'] = 'high'
        # Remove relevance terms from query
        clean_query = query_lower
        for phrase in ['with high relevance', 'under high relevance', 'high relevance', 'high quality']:
            clean_query = clean_query.replace(phrase, '')
        clean_query = ' '.join(clean_query.split())
        requirements['clean_query'] = clean_query
    elif ('medium relevance' in query_lower or 'moderate relevance' in query_lower or
          'with medium relevance' in query_lower):
        requirements['relevance_filter'] = 'medium'
        clean_query = query_lower
        for phrase in ['with medium relevance', 'medium relevance', 'moderate relevance']:
            clean_query = clean_query.replace(phrase, '')
        clean_query = ' '.join(clean_query.split())
        requirements['clean_query'] = clean_query
    
    return requirements

def search_furniture_dataset(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Search furniture dataset with enhanced scoring and query parsing"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        logger.warning("No dataset available for search")
        return []
    
    # Parse query for special requirements
    requirements = parse_query_requirements(query)
    clean_query = requirements['clean_query']
    relevance_filter = requirements['relevance_filter']
    sort_order = requirements['sort_order']
    
    query_lower = clean_query.lower().strip()
    query_words = [word for word in query_lower.split() if len(word) > 1]
    
    logger.info(f"Searching for: '{clean_query}' (words: {query_words}, filter: {relevance_filter})")
    logger.info(f"Original query: '{query}' -> Clean query: '{clean_query}', Filter: {relevance_filter}")
    
    if not query_words:
        return random.sample(dataset, min(max_results, len(dataset)))
    
    scored_products = []
    
    for product in dataset:
        score = 0.0
        
        # Title matching (highest priority)
        if product.get('title'):
            title_lower = product['title'].lower()
            for word in query_words:
                if word in title_lower:
                    # Exact word match
                    if f' {word} ' in f' {title_lower} ' or title_lower.startswith(word) or title_lower.endswith(word):
                        score += 5.0
                    else:
                        score += 2.0
        
        # Category matching
        if product.get('category'):
            category_lower = product['category'].lower()
            for word in query_words:
                if word in category_lower:
                    score += 4.0
        
        # Categories list
        if product.get('categories'):
            for cat in product['categories']:
                cat_lower = str(cat).lower()
                for word in query_words:
                    if word in cat_lower:
                        score += 3.0
        
        # Description matching
        if product.get('description'):
            desc_lower = product['description'].lower()
            for word in query_words:
                if word in desc_lower:
                    score += 1.5
        
        # Material, color, brand matching
        for field in ['material', 'color', 'brand']:
            if product.get(field):
                field_lower = product[field].lower()
                for word in query_words:
                    if word in field_lower:
                        score += 3.0
        
        # Only include products with meaningful relevance (score > 0.5 for partial matches)
        if score > 0.5:
            product_copy = product.copy()
            product_copy['similarity_score'] = round(score, 2)
            scored_products.append(product_copy)
    
    # Filter by relevance if specified
    if relevance_filter and scored_products:
        if relevance_filter == 'low':
            # For low relevance, we want products that match the search term but with lower scores
            # Keep products with scores between 1.0 and 5.0 (still relevant to search but lower priority)
            scored_products = [p for p in scored_products if 1.0 <= p['similarity_score'] <= 5.0]
        elif relevance_filter == 'medium':
            # Keep products with medium relevance scores (5.0 <= score <= 10.0)
            scored_products = [p for p in scored_products if 5.0 <= p['similarity_score'] <= 10.0]
        elif relevance_filter == 'high':
            # Keep only products with high relevance scores (> 10.0)
            scored_products = [p for p in scored_products if p['similarity_score'] > 10.0]
    
    # Sort by score (ascending for low relevance, descending for others)
    reverse_sort = sort_order == 'desc'
    scored_products.sort(key=lambda x: x['similarity_score'], reverse=reverse_sort)
    
    logger.info(f"Found {len(scored_products)} matching products (filter: {relevance_filter})")
    if scored_products:
        logger.info(f"Top result: {scored_products[0]['title'][:50]}... (score: {scored_products[0]['similarity_score']})")
    
    # Return results or limited random suggestions if no matches
    if scored_products:
        return scored_products[:max_results]
    else:
        logger.info(f"No matches found for '{clean_query}' (filter: {relevance_filter}), returning limited random suggestions")
        # Only return max 3 random products when no matches found
        num_random = min(3, max_results, len(dataset))
        random_products = random.sample(dataset, num_random)
        for p in random_products:
            p['similarity_score'] = 0.1
        return random_products

# Routes
@app.get("/")
async def root():
    """Root endpoint"""
    dataset = load_furniture_dataset()
    
    return {
        "message": "AI Furniture Recommendation Platform - Fixed",
        "version": "1.0.1-fixed",
        "status": "running",
        "dataset_info": {
            "source": "intern_data_ikarus.csv",
            "products_loaded": len(dataset),
            "status": "loaded" if dataset else "failed"
        },
        "endpoints": {
            "search": "/api/search",
            "analytics": "/api/analytics",
            "health": "/api/health",
            "sample": "/api/debug/sample"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check"""
    dataset = load_furniture_dataset()
    
    return {
        "status": "healthy" if dataset else "unhealthy", 
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "products": len(dataset),
            "status": "loaded" if dataset else "failed"
        }
    }

@app.post("/api/search")
async def search_furniture(request: SearchRequest):
    """Enhanced search endpoint"""
    start_time = time.time()
    
    try:
        logger.info(f"🔍 Search request: '{request.query}' (max: {request.max_results})")
        
        # Parse query to check for relevance requirements
        requirements = parse_query_requirements(request.query)
        relevance_filter = requirements.get('relevance_filter')
        clean_query = requirements.get('clean_query', request.query)
        
        results = search_furniture_dataset(request.query, request.max_results)
        processing_time = time.time() - start_time
        
        if results:
            top_score = results[0].get('similarity_score', 0)
            
            # Custom message based on relevance filter
            if relevance_filter == 'low':
                message = f"Found {len(results)} products matching '{clean_query}' with low relevance (as requested)"
            elif relevance_filter == 'medium':
                message = f"Found {len(results)} products matching '{clean_query}' with medium relevance"
            elif relevance_filter == 'high':
                message = f"Found {len(results)} products matching '{clean_query}' with high relevance"
            else:
                # Standard relevance classification
                if top_score > 4.0:
                    relevance = "with high relevance"
                    message = f"Found {len(results)} products matching '{request.query}' {relevance}"
                elif top_score > 1.0:
                    relevance = "with good relevance"
                    message = f"Found {len(results)} products matching '{request.query}' {relevance}"
                elif top_score > 0.5:
                    message = f"Found {len(results)} products related to '{request.query}'"
                else:
                    message = f"No exact matches for '{request.query}', showing {len(results)} random suggestions"
        else:
            if relevance_filter:
                message = f"No products found matching '{clean_query}' with {relevance_filter} relevance"
            else:
                message = f"No products found for '{request.query}'"
        
        return {
            "success": True,
            "message": message,
            "query": request.query,
            "session_id": request.session_id,
            "results_count": len(results),
            "results": results,
            "processing_time": round(processing_time, 3)
        }
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return {
            "success": False,
            "message": f"Search failed: {str(e)}",
            "query": request.query,
            "session_id": request.session_id,
            "results_count": 0,
            "results": [],
            "processing_time": round(time.time() - start_time, 3)
        }

@app.get("/api/analytics")
async def get_analytics():
    """Analytics endpoint with real data"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        return {"success": False, "message": "No dataset loaded"}
    
    # Calculate analytics
    total = len(dataset)
    
    categories = {}
    brands = {}
    materials = {}
    prices = []
    
    for product in dataset:
        if product.get('category'):
            cat = product['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        if product.get('brand'):
            brand = product['brand']
            brands[brand] = brands.get(brand, 0) + 1
            
        if product.get('material'):
            mat = product['material']
            materials[mat] = materials.get(mat, 0) + 1
            
        if product.get('price'):
            prices.append(product['price'])
    
    return {
        "success": True,
        "data": {
            "summary": {
                "total_products": total,
                "total_categories": len(categories),
                "total_brands": len(brands),
                "total_materials": len(materials),
                "products_with_prices": len(prices),
                "average_price": round(sum(prices) / len(prices), 2) if prices else 0,
                "price_range": {
                    "min": min(prices) if prices else 0,
                    "max": max(prices) if prices else 0
                }
            },
            "top_categories": [
                {"name": cat, "count": count, "percentage": round(count/total*100, 1)}
                for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "top_brands": [
                {"name": brand, "count": count, "percentage": round(count/total*100, 1)}
                for brand, count in sorted(brands.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "top_materials": [
                {"name": mat, "count": count, "percentage": round(count/total*100, 1)}
                for mat, count in sorted(materials.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/api/debug/sample")
async def get_sample_data():
    """Debug endpoint to see sample data"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        return {"error": "No dataset loaded"}
    
    return {
        "total_products": len(dataset),
        "first_5_products": dataset[:5],
        "random_product": random.choice(dataset) if dataset else None
    }

if __name__ == "__main__":
    import uvicorn
    
    # Pre-load and verify dataset
    print("🔄 Loading dataset...")
    dataset = load_furniture_dataset()
    
    if dataset:
        print(f"✅ Dataset loaded successfully: {len(dataset)} products")
        print(f"📝 Sample: {dataset[0]['title'][:50]}...")
    else:
        print("❌ Failed to load dataset!")
        print("🔍 Check that 'intern_data_ikarus.csv' exists in the 'data' folder")
        exit(1)
    
    print("🚀 Starting server on http://localhost:8001")
    
    uvicorn.run(
        app,
        host="127.0.0.1",  # Use localhost instead of 0.0.0.0
        port=8001,
        log_level="info"
    )