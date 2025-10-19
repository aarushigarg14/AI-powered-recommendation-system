"""
Enhanced AI-Powered Furniture Recommendation Platform
FastAPI Backend Server - Complete with Health Endpoint and Enhanced Search
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
from datetime import datetime
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Furniture Recommendation Platform - Enhanced",
    description="Furniture discovery system with comprehensive search across all fields",
    version="2.0.0-enhanced",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000", 
        "http://localhost:8001", 
        "http://127.0.0.1:8001"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Pydantic models
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None
    max_results: Optional[int] = Field(20, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    dataset_loaded: bool
    dataset_size: int

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
                    
                    # Parse categories
                    categories = []
                    if row.get('categories'):
                        try:
                            categories = ast.literal_eval(row['categories'])
                            if not isinstance(categories, list):
                                categories = [str(categories)]
                        except (ValueError, SyntaxError):
                            categories = [row['categories']]
                    
                    # Parse images
                    images = []
                    if row.get('images'):
                        try:
                            images = ast.literal_eval(row['images'])
                            if not isinstance(images, list):
                                images = [str(images)]
                            images = [img.strip() for img in images if img and img.strip()]
                        except (ValueError, SyntaxError):
                            images = [row['images']] if row['images'] else []
                    
                    # Extract primary category
                    primary_category = None
                    if categories:
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
                        "similarity_score": 1.0
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

def search_furniture_dataset(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """Search furniture dataset based on query across all fields"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        logger.warning("No furniture dataset available")
        return []
    
    # Parse price and relevance requirements from query
    relevance_requirement = None
    max_price = None
    min_price = None
    clean_query = query.lower()
    
    # Check for price specifications in the query
    price_patterns = [
        (r'\bunder\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
        (r'\bbelow\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
        (r'\bless\s+than\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
        (r'\bup\s+to\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
        (r'\bover\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
        (r'\babove\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
        (r'\bmore\s+than\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
        (r'\bbetween\s*\$?(\d+(?:\.\d{2})?)\s*and\s*\$?(\d+(?:\.\d{2})?)\b', 'range')
    ]
    
    for pattern, price_type in price_patterns:
        match = re.search(pattern, clean_query)
        if match:
            if price_type == 'max':
                max_price = float(match.group(1))
                # Remove price specification from query
                clean_query = re.sub(pattern, '', clean_query).strip()
            elif price_type == 'min':
                min_price = float(match.group(1))
                clean_query = re.sub(pattern, '', clean_query).strip()
            elif price_type == 'range':
                min_price = float(match.group(1))
                max_price = float(match.group(2))
                clean_query = re.sub(pattern, '', clean_query).strip()
            break
    
    # Check for relevance specifications in the query
    relevance_patterns = [
        (r'\b(?:with|under|having)\s+(?:low|poor|bad)\s+relevance\b', 'low'),
        (r'\b(?:with|under|having)\s+(?:high|good|strong)\s+relevance\b', 'high'),
        (r'\b(?:low|poor|bad)\s+relevance\b', 'low'),
        (r'\b(?:high|good|strong)\s+relevance\b', 'high')
    ]
    
    for pattern, req_type in relevance_patterns:
        if re.search(pattern, clean_query):
            relevance_requirement = req_type
            # Remove relevance specification from query
            clean_query = re.sub(pattern, '', clean_query).strip()
            break
    
    query_words = clean_query.split()
    
    scored_products = []
    
    for product in dataset:
        score = 0.0
        
        # Score based on title matching (highest priority)
        if product.get('title'):
            title_lower = product['title'].lower()
            for word in query_words:
                if word in title_lower:
                    # Exact word match gets higher score
                    if word == title_lower or f' {word} ' in f' {title_lower} ':
                        score += 4.0
                    else:
                        score += 2.0
        
        # Score based on brand matching
        if product.get('brand'):
            brand_lower = product['brand'].lower()
            for word in query_words:
                if word in brand_lower:
                    score += 3.0
        
        # Score based on description matching
        if product.get('description'):
            description_lower = product['description'].lower()
            for word in query_words:
                if word in description_lower:
                    score += 1.5
        
        # Score based on price matching
        if product.get('price'):
            price_str = str(product['price'])
            for word in query_words:
                if word in price_str or (word.replace('$', '') in price_str):
                    score += 2.0
        
        # Score based on categories list matching
        if product.get('categories'):
            for category in product['categories']:
                if category:
                    category_lower = str(category).lower()
                    for word in query_words:
                        if word in category_lower:
                            score += 2.5
        
        # Score based on primary category matching
        if product.get('category'):
            category_lower = product['category'].lower()
            for word in query_words:
                if word in category_lower:
                    score += 2.0
        
        # Score based on images matching
        if product.get('images'):
            for image in product['images']:
                if image:
                    image_lower = str(image).lower()
                    for word in query_words:
                        if word in image_lower:
                            score += 1.0
        
        # Score based on manufacturer matching
        if product.get('manufacturer'):
            manufacturer_lower = product['manufacturer'].lower()
            for word in query_words:
                if word in manufacturer_lower:
                    score += 2.5
        
        # Score based on package_dimensions matching
        if product.get('package_dimensions'):
            dimensions_lower = product['package_dimensions'].lower()
            for word in query_words:
                if word in dimensions_lower:
                    score += 1.0
        
        # Score based on country_of_origin matching
        if product.get('country_of_origin'):
            country_lower = product['country_of_origin'].lower()
            for word in query_words:
                if word in country_lower:
                    score += 1.5
        
        # Score based on material matching
        if product.get('material'):
            material_lower = product['material'].lower()
            for word in query_words:
                if word in material_lower:
                    score += 2.0
        
        # Score based on color matching
        if product.get('color'):
            color_lower = product['color'].lower()
            for word in query_words:
                if word in color_lower:
                    score += 2.0
        
        # Score based on uniq_id matching
        if product.get('id'):
            id_lower = str(product['id']).lower()
            for word in query_words:
                if word in id_lower:
                    score += 1.0
        
        # Only include products with some relevance
        if score > 0:
            product_copy = product.copy()
            product_copy['similarity_score'] = round(score, 2)
            scored_products.append(product_copy)
    
    # Sort by score (descending)
    scored_products.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Apply price filtering if specified
    if max_price is not None or min_price is not None:
        price_filtered_products = []
        for product in scored_products:
            product_price = product.get('price')
            if product_price is not None:
                # Check price constraints
                if max_price is not None and product_price > max_price:
                    continue  # Skip products over max price
                if min_price is not None and product_price < min_price:
                    continue  # Skip products under min price
                price_filtered_products.append(product)
            else:
                # If no price information, include only if no strict price filter
                if max_price is None and min_price is None:
                    price_filtered_products.append(product)
        scored_products = price_filtered_products
    
    # Apply relevance filtering if specified
    if relevance_requirement:
        if relevance_requirement == 'low':
            # Filter for low relevance (scores between 0.1 and 5.0)
            scored_products = [p for p in scored_products if 0.1 <= p['similarity_score'] <= 5.0]
        elif relevance_requirement == 'high':
            # Filter for high relevance (scores above 8.0)
            scored_products = [p for p in scored_products if p['similarity_score'] > 8.0]
    
    # If no scored results, return some random products
    if not scored_products:
        logger.info(f"No direct matches for '{query}', returning random products")
        random_products = random.sample(dataset, min(max_results, len(dataset)))
        for i, product in enumerate(random_products):
            product_copy = product.copy()
            product_copy['similarity_score'] = round(1.0 + (i * 0.1), 2)
            scored_products.append(product_copy)
    
    return scored_products[:max_results]

def generate_response_message(query: str, results_count: int, products: List[Dict[str, Any]] = None, max_price: float = None, min_price: float = None) -> str:
    """Generate a natural language response message"""
    
    if results_count == 0:
        price_info = ""
        if max_price is not None:
            price_info = f" under ${max_price}"
        elif min_price is not None:
            price_info = f" over ${min_price}"
        elif max_price is not None and min_price is not None:
            price_info = f" between ${min_price} and ${max_price}"
        return f"I couldn't find any furniture matching '{query}'{price_info} across title, brand, description, price, categories, manufacturer, materials, colors, and other product details. Try adjusting your search terms."
    
    # Check if this was a relevance-filtered search
    relevance_info = ""
    if products and len(products) > 0:
        avg_score = sum(p.get('similarity_score', 0) for p in products) / len(products)
        if 'low relevance' in query.lower() or 'with low relevance' in query.lower() or 'under low relevance' in query.lower():
            if avg_score <= 5.0:
                relevance_info = " with low relevance"
            else:
                relevance_info = " with mixed relevance"
        elif 'high relevance' in query.lower():
            if avg_score > 8.0:
                relevance_info = " with high relevance"
            else:
                relevance_info = " with mixed relevance"
        elif avg_score > 10.0:
            relevance_info = " with high relevance"
        elif avg_score <= 3.0:
            relevance_info = " with low relevance"
    
    # Clean query for display
    clean_query = query
    relevance_patterns = [
        r'\b(?:with|under|having)\s+(?:low|poor|bad)\s+relevance\b',
        r'\b(?:with|under|having)\s+(?:high|good|strong)\s+relevance\b',
        r'\b(?:low|poor|bad)\s+relevance\b',
        r'\b(?:high|good|strong)\s+relevance\b'
    ]
    
    for pattern in relevance_patterns:
        clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE).strip()
    
    # Add price filtering information
    price_info = ""
    if max_price is not None and min_price is not None:
        price_info = f" (${min_price} - ${max_price})"
    elif max_price is not None:
        price_info = f" (under ${max_price})"
    elif min_price is not None:
        price_info = f" (over ${min_price})"
    
    # Generate message
    if results_count == 1:
        message = f"Found 1 product matching '{clean_query}'{price_info}{relevance_info}"
    else:
        message = f"Found {results_count} products matching '{clean_query}'{price_info}{relevance_info}"
    
    message += " - searched across title, brand, description, categories, materials, colors, manufacturer, and all product details."
    
    return message

# Analytics Functions
def get_analytics_data() -> Dict[str, Any]:
    """Generate analytics data based on the actual furniture dataset"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        return {"error": "No dataset available"}
    
    # Basic stats
    total_products = len(dataset)
    products_with_prices = [p for p in dataset if p.get('price') is not None]
    
    # Price analysis
    prices = [p['price'] for p in products_with_prices]
    price_stats = {
        "min_price": min(prices) if prices else 0,
        "max_price": max(prices) if prices else 0,
        "avg_price": sum(prices) / len(prices) if prices else 0,
        "products_with_prices": len(products_with_prices),
        "products_without_prices": total_products - len(products_with_prices)
    }
    
    # Price distribution
    price_ranges = {
        "Under $50": len([p for p in prices if p < 50]) if prices else 0,
        "$50-$100": len([p for p in prices if 50 <= p < 100]) if prices else 0,
        "$100-$200": len([p for p in prices if 100 <= p < 200]) if prices else 0,
        "$200-$500": len([p for p in prices if 200 <= p < 500]) if prices else 0,
        "$500+": len([p for p in prices if p >= 500]) if prices else 0
    }
    
    # Category analysis
    categories = [p.get('category') for p in dataset if p.get('category')]
    category_counts = Counter(categories)
    top_categories = dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Brand analysis
    brands = [p.get('brand') for p in dataset if p.get('brand')]
    brand_counts = Counter(brands)
    top_brands = dict(sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Material analysis
    materials = [p.get('material') for p in dataset if p.get('material')]
    material_counts = Counter(materials)
    top_materials = dict(sorted(material_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Color analysis
    colors = [p.get('color') for p in dataset if p.get('color')]
    color_counts = Counter(colors)
    top_colors = dict(sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Data completeness analysis
    completeness = {
        "total_products": total_products,
        "with_title": len([p for p in dataset if p.get('title')]),
        "with_price": len(products_with_prices),
        "with_category": len([p for p in dataset if p.get('category')]),
        "with_brand": len([p for p in dataset if p.get('brand')]),
        "with_material": len([p for p in dataset if p.get('material')]),
        "with_color": len([p for p in dataset if p.get('color')]),
        "with_images": len([p for p in dataset if p.get('images')]),
        "with_description": len([p for p in dataset if p.get('description')])
    }
    
    return {
        "overview": {
            "total_products": total_products,
            "unique_categories": len(category_counts),
            "unique_brands": len(brand_counts),
            "unique_materials": len(material_counts),
            "unique_colors": len(color_counts)
        },
        "price_stats": price_stats,
        "price_distribution": price_ranges,
        "top_categories": top_categories,
        "top_brands": top_brands,
        "top_materials": top_materials,
        "top_colors": top_colors,
        "data_completeness": completeness,
        "generated_at": datetime.now().isoformat()
    }

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    dataset = load_furniture_dataset()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0-enhanced",
        "dataset_loaded": len(dataset) > 0,
        "dataset_size": len(dataset)
    }

@app.post("/api/search")
async def search_furniture(request: SearchRequest):
    """Search for furniture products using Pinecone or fallback to keyword search"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing search query: '{request.query}'")
        
        # Handle greetings and conversational queries
        query_lower = request.query.lower().strip()
        greeting_patterns = [
            r'^(hi|hello|hey|greetings?)!?$',
            r'^(good\s+(morning|afternoon|evening|day))!?$',
            r'^(how\s+(are\s+you|do\s+you\s+do))\??$'
        ]
        
        for pattern in greeting_patterns:
            if re.match(pattern, query_lower):
                greeting_responses = [
                    "Hello! 👋 I'm your AI furniture assistant. I can help you find the perfect furniture using smart semantic search. Try asking me about 'comfortable sofas' or 'wooden dining chairs'!",
                    "Hi there! 🛋️ Welcome to our AI-powered furniture store! I can understand what you're looking for and find similar items. What type of furniture are you interested in today?",
                    "Hey! 😊 I'm here to help you discover amazing furniture using advanced AI search. Just describe what you need - like 'modern coffee table' or 'cozy bedroom chair' - and I'll find the best matches!",
                    "Greetings! 🏠 I'm your smart furniture shopping assistant powered by Pinecone AI. I can understand the meaning behind your searches, not just keywords. What can I help you find today?"
                ]
                import random
                selected_response = random.choice(greeting_responses)
                
                # Get a few featured products to show
                dataset = load_furniture_dataset()
                featured_products = random.sample(dataset, min(6, len(dataset)))
                for product in featured_products:
                    product['similarity_score'] = 95.0  # High score for featured items
                
                return {
                    "success": True,
                    "message": selected_response,
                    "query": request.query,
                    "session_id": request.session_id,
                    "results_count": len(featured_products),
                    "results": featured_products,
                    "search_method": "greeting (conversational AI)",
                    "processing_time": round(time.time() - start_time, 3)
                }
        
        # Handle other conversational queries
        conversational_patterns = [
            (r'(what\s+can\s+you\s+do|what\s+do\s+you\s+sell|help\s*me?)\??', 
             "I can help you find furniture using AI-powered semantic search! 🤖 I understand what you mean, not just keywords. Try searching for things like: 'comfortable living room seating', 'storage solutions', 'modern dining furniture', or 'bedroom essentials'. What are you looking for?"),
            
            (r'(thank\s+you|thanks|thx)!?', 
             "You're welcome! 😊 Happy to help you find the perfect furniture. Feel free to search for anything else you need!"),
            
            (r'(bye|goodbye|see\s+you|exit)!?', 
             "Goodbye! 👋 Thanks for using our AI furniture assistant. Come back anytime you need help finding great furniture!"),
            
            (r'(how\s+does\s+this\s+work|how\s+to\s+search)\??', 
             "Great question! 💡 I use advanced AI to understand what you're looking for. Instead of just matching keywords, I understand meaning and context. Just describe what you want in natural language - like 'cozy reading chair' or 'space-saving storage' - and I'll find the best matches!")
        ]
        
        for pattern, response in conversational_patterns:
            if re.search(pattern, query_lower):
                # Get some random featured products for conversational responses
                dataset = load_furniture_dataset()
                featured_products = random.sample(dataset, min(4, len(dataset)))
                for product in featured_products:
                    product['similarity_score'] = 90.0
                
                return {
                    "success": True,
                    "message": response,
                    "query": request.query,
                    "session_id": request.session_id,
                    "results_count": len(featured_products),
                    "results": featured_products,
                    "search_method": "conversational AI",
                    "processing_time": round(time.time() - start_time, 3)
                }
        
        # Try Pinecone semantic search first
        products = []
        search_method = "keyword"
        
        try:
            from backend.services.pinecone_service import PineconeService
            pinecone_service = PineconeService()
            
            # Extract price filters for Pinecone
            filters = {}
            query_lower = request.query.lower()
            
            # Check for price specifications in the query
            price_patterns = [
                (r'\bunder\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
                (r'\bbelow\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
                (r'\bless\s+than\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
                (r'\bup\s+to\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
                (r'\bover\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
                (r'\babove\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
                (r'\bmore\s+than\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
                (r'\bbetween\s*\$?(\d+(?:\.\d{2})?)\s*and\s*\$?(\d+(?:\.\d{2})?)\b', 'range')
            ]
            
            for pattern, price_type in price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    if price_type == 'max':
                        filters['max_price'] = float(match.group(1))
                    elif price_type == 'min':
                        filters['min_price'] = float(match.group(1))
                    elif price_type == 'range':
                        filters['min_price'] = float(match.group(1))
                        filters['max_price'] = float(match.group(2))
                    break
            
            # Perform Pinecone semantic search
            pinecone_results = pinecone_service.semantic_search(
                request.query, 
                max_results=request.max_results,
                filters=filters if filters else None
            )
            
            if pinecone_results:
                # Convert Pinecone results to expected format
                dataset = load_furniture_dataset()
                product_lookup = {p['id']: p for p in dataset}
                
                products = []
                for result in pinecone_results:
                    # Get full product data from dataset
                    full_product = product_lookup.get(result['id'])
                    if full_product:
                        # Update with Pinecone similarity score
                        full_product = full_product.copy()
                        full_product['similarity_score'] = result['similarity_score']
                        products.append(full_product)
                
                search_method = "semantic (Pinecone)"
                logger.info(f"Using Pinecone semantic search - found {len(products)} results")
            else:
                raise Exception("No Pinecone results")
                
        except Exception as e:
            logger.warning(f"Pinecone search failed: {e}. Falling back to keyword search.")
            # Fallback to original keyword search
            products = search_furniture_dataset(request.query, request.max_results)
            search_method = "keyword (fallback)"
        
        logger.info(f"Search method used: {search_method}")
        
        # Extract price constraints for message generation from filters or re-parse
        max_price = filters.get('max_price') if 'filters' in locals() else None
        min_price = filters.get('min_price') if 'filters' in locals() else None
        
        if max_price is None and min_price is None:
            # Re-parse for keyword search fallback
            query_lower = request.query.lower()
            price_patterns = [
                (r'\bunder\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
                (r'\bbelow\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
                (r'\bless\s+than\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
                (r'\bup\s+to\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
                (r'\bover\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
                (r'\babove\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
                (r'\bmore\s+than\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
                (r'\bbetween\s*\$?(\d+(?:\.\d{2})?)\s*and\s*\$?(\d+(?:\.\d{2})?)\b', 'range')
            ]
            
            for pattern, price_type in price_patterns:
                match = re.search(pattern, query_lower)
                if match:
                    if price_type == 'max':
                        max_price = float(match.group(1))
                    elif price_type == 'min':
                        min_price = float(match.group(1))
                    elif price_type == 'range':
                        min_price = float(match.group(1))
                        max_price = float(match.group(2))
                    break
        
        # Extract price constraints for message generation
        max_price = None
        min_price = None
        query_lower = request.query.lower()
        
        price_patterns = [
            (r'\bunder\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
            (r'\bbelow\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
            (r'\bless\s+than\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
            (r'\bup\s+to\s*\$?(\d+(?:\.\d{2})?)\b', 'max'),
            (r'\bover\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
            (r'\babove\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
            (r'\bmore\s+than\s*\$?(\d+(?:\.\d{2})?)\b', 'min'),
            (r'\bbetween\s*\$?(\d+(?:\.\d{2})?)\s*and\s*\$?(\d+(?:\.\d{2})?)\b', 'range')
        ]
        
        for pattern, price_type in price_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if price_type == 'max':
                    max_price = float(match.group(1))
                elif price_type == 'min':
                    min_price = float(match.group(1))
                elif price_type == 'range':
                    min_price = float(match.group(1))
                    max_price = float(match.group(2))
                break
        
        # Generate response message with search method info
        base_message = generate_response_message(request.query, len(products), products, max_price, min_price)
        
        if search_method == "semantic (Pinecone)":
            message = base_message + " 🤖 Powered by AI semantic search."
        else:
            message = base_message + " 🔍 Using keyword search."
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        response = {
            "success": True,
            "message": message,
            "query": request.query,
            "session_id": request.session_id,
            "results_count": len(products),
            "results": products,
            "search_method": search_method,
            "processing_time": round(processing_time, 3)
        }
        
        logger.info(f"Search completed in {processing_time:.3f}s. Found {len(products)} results.")
        return response
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        processing_time = time.time() - start_time
        
        return {
            "success": False,
            "message": "I encountered an error while searching. Please try again with a different query.",
            "query": request.query,
            "session_id": request.session_id,
            "results_count": 0,
            "results": [],
            "processing_time": round(processing_time, 3)
        }

@app.get("/api/analytics")
async def get_analytics():
    """Get analytics data for the furniture dataset"""
    try:
        analytics = get_analytics_data()
        
        if "error" in analytics:
            raise HTTPException(status_code=503, detail=analytics["error"])
        
        return {
            "success": True,
            "data": analytics
        }
        
    except Exception as e:
        logger.error(f"Analytics generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics data")

@app.get("/api/categories")
async def get_categories():
    """Get all available categories"""
    try:
        dataset = load_furniture_dataset()
        categories = [p.get('category') for p in dataset if p.get('category')]
        category_counts = Counter(categories)
        
        return {
            "success": True,
            "data": {
                "total_categories": len(category_counts),
                "categories": dict(sorted(category_counts.items(), key=lambda x: x[1], reverse=True))
            }
        }
        
    except Exception as e:
        logger.error(f"Categories fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch categories")

@app.get("/api/brands")
async def get_brands():
    """Get all available brands"""
    try:
        dataset = load_furniture_dataset()
        brands = [p.get('brand') for p in dataset if p.get('brand')]
        brand_counts = Counter(brands)
        
        return {
            "success": True,
            "data": {
                "total_brands": len(brand_counts),
                "brands": dict(sorted(brand_counts.items(), key=lambda x: x[1], reverse=True))
            }
        }
        
    except Exception as e:
        logger.error(f"Brands fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch brands")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Furniture Recommendation Platform - Enhanced API",
        "version": "2.0.0-enhanced",
        "docs": "/docs",
        "health": "/health",
        "analytics": "/api/analytics",
        "categories": "/api/categories",
        "brands": "/api/brands"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Load dataset on startup
    logger.info("Loading furniture dataset...")
    dataset = load_furniture_dataset()
    logger.info(f"Dataset loaded with {len(dataset)} products")
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")