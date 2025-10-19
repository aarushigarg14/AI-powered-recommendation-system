"""
Search API routes for furniture recommendations
Handles natural language queries and returns AI-powered results
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import asyncio
import csv
import re
import ast
import os
from pathlib import Path

from models.data_manager import DataManager
from models.ai_models import AIModelManager
from utils.helpers import validate_search_query, extract_keywords

logger = logging.getLogger(__name__)
router = APIRouter()

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural language search query")
    session_id: Optional[str] = Field(None, description="Session ID for conversation context")
    max_results: Optional[int] = Field(8, ge=1, le=50, description="Maximum number of results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional search filters")

class ProductResult(BaseModel):
    id: str
    title: str
    price: Optional[float]
    category: Optional[str]
    material: Optional[str]
    color: Optional[str]
    brand: Optional[str]
    description: str
    original_description: Optional[str]
    images: List[str]
    primary_image: Optional[str]
    categories: List[str]
    similarity_score: Optional[float]

class SearchResponse(BaseModel):
    success: bool
    message: str
    query: str
    session_id: Optional[str]
    results_count: int
    products: List[ProductResult]
    query_analysis: Optional[Dict[str, Any]]
    processing_time: float
    cached: bool = False

class ConversationContext(BaseModel):
    session_id: str
    previous_queries: List[str] = []
    preferences: Dict[str, Any] = {}
    last_updated: datetime = Field(default_factory=datetime.now)

# Session storage (in-memory for now, should be Redis in production)
conversation_sessions: Dict[str, ConversationContext] = {}

# Global variable to store loaded furniture data
_furniture_dataset: Optional[List[Dict[str, Any]]] = None

def load_furniture_dataset() -> List[Dict[str, Any]]:
    """Load furniture data from CSV file"""
    global _furniture_dataset
    
    if _furniture_dataset is not None:
        return _furniture_dataset
    
    # Construct path to CSV file
    current_dir = Path(__file__).parent.parent  # backend directory
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
    """Search furniture dataset based on query across all fields"""
    dataset = load_furniture_dataset()
    
    if not dataset:
        logger.warning("No furniture dataset available, falling back to mock data")
        return get_mock_search_results(query, max_results)
    
    # Parse relevance requirements from query
    relevance_requirement = None
    clean_query = query.lower()
    
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
        
        # Score based on price matching (if query contains price-related terms)
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
        
        # Score based on images matching (search in image URLs/names)
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
        import random
        random_products = random.sample(dataset, min(max_results, len(dataset)))
        for i, product in enumerate(random_products):
            product_copy = product.copy()
            product_copy['similarity_score'] = round(1.0 + (i * 0.1), 2)  # Low scores for random results
            scored_products.append(product_copy)
    
    return scored_products[:max_results]

def get_mock_search_results(query: str, max_results: int = 8) -> List[Dict[str, Any]]:
    """Mock search results for development"""
    # Sample furniture products
    mock_products = [
        {
            "id": "mock-1",
            "title": "Modern Ergonomic Office Chair",
            "price": 299.99,
            "category": "Office Furniture",
            "material": "Fabric",
            "color": "Black",
            "brand": "ComfortCorp",
            "description": "Experience ultimate comfort with this modern ergonomic office chair featuring lumbar support and breathable fabric.",
            "original_description": "Office chair with ergonomic design",
            "images": ["https://via.placeholder.com/300x200?text=Office+Chair"],
            "primary_image": "https://via.placeholder.com/300x200?text=Office+Chair",
            "categories": ["Office", "Seating"],
            "similarity_score": 0.95
        },
        {
            "id": "mock-2", 
            "title": "Scandinavian Oak Dining Table",
            "price": 599.99,
            "category": "Dining Room",
            "material": "Oak Wood",
            "color": "Natural",
            "brand": "Nordic Design",
            "description": "Beautifully crafted Scandinavian dining table made from solid oak wood, perfect for family gatherings.",
            "original_description": "Oak dining table",
            "images": ["https://via.placeholder.com/300x200?text=Dining+Table"],
            "primary_image": "https://via.placeholder.com/300x200?text=Dining+Table",
            "categories": ["Dining", "Tables"],
            "similarity_score": 0.87
        },
        {
            "id": "mock-3",
            "title": "Luxury Velvet Sectional Sofa",
            "price": 1299.99,
            "category": "Living Room",
            "material": "Velvet",
            "color": "Navy Blue",
            "brand": "Elegance Home",
            "description": "Indulge in luxury with this premium velvet sectional sofa featuring deep seating and elegant design.",
            "original_description": "Velvet sectional sofa",
            "images": ["https://via.placeholder.com/300x200?text=Velvet+Sofa"],
            "primary_image": "https://via.placeholder.com/300x200?text=Velvet+Sofa",
            "categories": ["Living Room", "Seating"],
            "similarity_score": 0.82
        },
        {
            "id": "mock-4",
            "title": "Minimalist Bookshelf Storage Unit",
            "price": 199.99,
            "category": "Storage",
            "material": "MDF",
            "color": "White",
            "brand": "Simple Living",
            "description": "Clean and modern bookshelf perfect for organizing books and decorative items in any room.",
            "original_description": "White bookshelf unit",
            "images": ["https://via.placeholder.com/300x200?text=Bookshelf"],
            "primary_image": "https://via.placeholder.com/300x200?text=Bookshelf", 
            "categories": ["Storage", "Shelving"],
            "similarity_score": 0.75
        }
    ]
    
    # Filter based on query keywords (simple matching)
    query_lower = query.lower()
    filtered_products = []
    
    for product in mock_products:
        # Simple relevance scoring based on keyword matching
        title_match = any(word in product['title'].lower() for word in query_lower.split())
        category_match = any(word in product['category'].lower() for word in query_lower.split())
        material_match = any(word in product['material'].lower() for word in query_lower.split())
        
        if title_match or category_match or material_match:
            filtered_products.append(product)
    
    # If no matches, return all products
    if not filtered_products:
        filtered_products = mock_products
    
    return filtered_products[:max_results]

@router.post("/search")
async def search_furniture(
    request: SearchRequest
):
    """
    Search for furniture products using natural language queries
    
    Supports queries like:
    - "Show me modern beds under $500"
    - "I need a comfortable grey sofa"
    - "Find wooden dining tables for 6 people"
    """
    start_time = asyncio.get_event_loop().time()
    
    try:
        logger.info(f"Processing search query: '{request.query}' for session: {request.session_id}")
        
        # Validate and analyze query
        query_validation = validate_search_query(request.query)
        if not query_validation['valid']:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid query",
                    "message": "Please provide a valid search query with at least 2 characters.",
                    "query": request.query
                }
            )
        
        # Update conversation context
        session_context = None
        if request.session_id:
            session_context = await update_conversation_context(
                request.session_id, 
                request.query,
                query_validation
            )
        
        # Extract filters from query if not provided
        search_filters = request.filters or {}
        if query_validation.get('extracted_info'):
            extracted_info = query_validation['extracted_info']
            
            if 'max_price' in extracted_info and 'max_price' not in search_filters:
                search_filters['max_price'] = extracted_info['max_price']
            
            if 'colors' in extracted_info and 'color' not in search_filters:
                search_filters['color'] = extracted_info['colors'][0]  # Use first color
            
            if 'materials' in extracted_info and 'material' not in search_filters:
                search_filters['material'] = extracted_info['materials'][0]  # Use first material
        
        # Use real furniture dataset with enhanced search across all fields
        products = search_furniture_dataset(request.query, request.max_results)
        
        # Convert to response format (using dictionaries for simplicity)
        product_results = []
        for product in products:
            product_result = {
                "id": product['id'],
                "title": product['title'],
                "price": product['price'],
                "category": product['category'],
                "material": product['material'],
                "color": product['color'],
                "brand": product['brand'],
                "description": product['description'],
                "original_description": product.get('original_description'),
                "images": product.get('images', []),
                "primary_image": product.get('primary_image'),
                "categories": product.get('categories', []),
                "similarity_score": product.get('similarity_score')
            }
            product_results.append(product_result)
        
        # Calculate processing time
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Generate response message
        message = await generate_response_message(
            request.query, 
            len(product_results), 
            query_validation,
            search_filters,
            products
        )
        
        response = {
            "success": True,
            "message": message,
            "query": request.query,
            "session_id": request.session_id,
            "results_count": len(product_results),
            "results": product_results,
            "query_analysis": query_validation,
            "processing_time": round(processing_time, 3)
        }
        
        logger.info(f"Search completed in {processing_time:.3f}s. Found {len(product_results)} results.")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        processing_time = asyncio.get_event_loop().time() - start_time
        
        # Return error response
        return {
            "success": False,
            "message": "I apologize, but I encountered an error while searching. Please try again with a different query.",
            "query": request.query,
            "session_id": request.session_id,
            "results_count": 0,
            "results": [],
            "processing_time": round(processing_time, 3)
        }

async def update_conversation_context(
    session_id: str, 
    query: str,
    query_analysis: Dict[str, Any]
) -> ConversationContext:
    """Update conversation context for the session"""
    
    if session_id not in conversation_sessions:
        conversation_sessions[session_id] = ConversationContext(session_id=session_id)
    
    context = conversation_sessions[session_id]
    
    # Add query to history
    context.previous_queries.append(query)
    
    # Keep only last 10 queries
    if len(context.previous_queries) > 10:
        context.previous_queries = context.previous_queries[-10:]
    
    # Update preferences based on query analysis
    if query_analysis.get('extracted_info'):
        extracted_info = query_analysis['extracted_info']
        
        # Update price preferences
        if 'max_price' in extracted_info:
            context.preferences['preferred_max_price'] = extracted_info['max_price']
        
        # Update color preferences
        if 'colors' in extracted_info:
            if 'preferred_colors' not in context.preferences:
                context.preferences['preferred_colors'] = []
            for color in extracted_info['colors']:
                if color not in context.preferences['preferred_colors']:
                    context.preferences['preferred_colors'].append(color)
        
        # Update material preferences
        if 'materials' in extracted_info:
            if 'preferred_materials' not in context.preferences:
                context.preferences['preferred_materials'] = []
            for material in extracted_info['materials']:
                if material not in context.preferences['preferred_materials']:
                    context.preferences['preferred_materials'].append(material)
    
    context.last_updated = datetime.now()
    return context

async def generate_response_message(
    query: str,
    results_count: int,
    query_analysis: Dict[str, Any],
    filters: Dict[str, Any],
    products: List[Dict[str, Any]] = None
) -> str:
    """Generate a natural language response message"""
    
    if results_count == 0:
        return f"I couldn't find any furniture matching '{query}' across title, brand, description, price, categories, manufacturer, materials, colors, and other product details. Try adjusting your search terms."
    
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
    
    # Clean query for display (remove relevance specifications)
    clean_query = query
    relevance_patterns = [
        r'\b(?:with|under|having)\s+(?:low|poor|bad)\s+relevance\b',
        r'\b(?:with|under|having)\s+(?:high|good|strong)\s+relevance\b',
        r'\b(?:low|poor|bad)\s+relevance\b',
        r'\b(?:high|good|strong)\s+relevance\b'
    ]
    
    for pattern in relevance_patterns:
        clean_query = re.sub(pattern, '', clean_query, flags=re.IGNORECASE).strip()
    
    # Base message
    message_parts = []
    
    if results_count == 1:
        message_parts.append(f"Found 1 product matching '{clean_query}'{relevance_info}")
    else:
        message_parts.append(f"Found {results_count} products matching '{clean_query}'{relevance_info}")
    
    # Add filter information
    filter_descriptions = []
    if filters.get('max_price'):
        filter_descriptions.append(f"under ${filters['max_price']}")
    if filters.get('color'):
        filter_descriptions.append(f"in {filters['color']}")
    if filters.get('material'):
        filter_descriptions.append(f"made of {filters['material']}")
    if filters.get('category'):
        filter_descriptions.append(f"in {filters['category']} category")
    
    if filter_descriptions:
        message_parts.append(f" ({', '.join(filter_descriptions)})")
    
    # Add search coverage information
    message_parts.append(" - searched across title, brand, description, categories, materials, colors, manufacturer, and all product details.")
    
    return " ".join(message_parts)

@router.get("/search/suggestions")
async def get_search_suggestions(
    category: Optional[str] = None,
    data_manager: DataManager = Depends()
) -> Dict[str, Any]:
    """Get search suggestions for the chat interface"""
    
    try:
        metadata = data_manager.get_metadata()
        
        # Base suggestions
        suggestions = [
            "Show me modern sofas under $500",
            "I need a comfortable office chair",
            "Find wooden dining tables for 6 people",
            "What storage solutions do you have?",
            "Show me leather furniture",
            "I want a grey sectional sofa",
            "Find bedroom furniture under $300",
            "Show me outdoor patio furniture"
        ]
        
        # Category-specific suggestions
        if category:
            category_lower = category.lower()
            if 'living' in category_lower:
                suggestions = [
                    "Show me modern living room sofas",
                    "I need a comfortable sectional sofa",
                    "Find coffee tables under $200",
                    "Show me TV stands and media centers"
                ]
            elif 'bedroom' in category_lower:
                suggestions = [
                    "Show me platform beds",
                    "I need matching nightstands",
                    "Find bedroom dressers with mirrors",
                    "Show me comfortable mattresses"
                ]
            elif 'dining' in category_lower:
                suggestions = [
                    "Show me dining tables for 6 people",
                    "I need matching dining chairs",
                    "Find bar stools for kitchen island",
                    "Show me buffets and sideboards"
                ]
        
        # Popular categories for quick access
        popular_categories = []
        if metadata.get('categories', {}).get('main_categories'):
            top_cats = list(metadata['categories']['main_categories'].keys())[:6]
            popular_categories = [cat for cat in top_cats if cat != 'Unknown']
        
        return {
            "suggestions": suggestions,
            "popular_categories": popular_categories,
            "quick_filters": {
                "price_ranges": ["Under $100", "Under $300", "Under $500", "Under $1000"],
                "materials": ["Wood", "Metal", "Leather", "Fabric"],
                "colors": ["Black", "White", "Brown", "Grey", "Blue"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get suggestions: {str(e)}")
        return {
            "suggestions": [
                "Show me furniture",
                "I need a chair",
                "Find tables",
                "Show me storage"
            ],
            "popular_categories": [],
            "quick_filters": {}
        }

@router.get("/search/session/{session_id}")
async def get_session_context(session_id: str) -> Dict[str, Any]:
    """Get conversation context for a session"""
    
    if session_id not in conversation_sessions:
        return {
            "session_id": session_id,
            "previous_queries": [],
            "preferences": {},
            "exists": False
        }
    
    context = conversation_sessions[session_id]
    return {
        "session_id": session_id,
        "previous_queries": context.previous_queries[-5:],  # Last 5 queries
        "preferences": context.preferences,
        "last_updated": context.last_updated.isoformat(),
        "exists": True
    }

@router.delete("/search/session/{session_id}")
async def clear_session_context(session_id: str) -> Dict[str, str]:
    """Clear conversation context for a session"""
    
    if session_id in conversation_sessions:
        del conversation_sessions[session_id]
        return {"message": f"Session {session_id} cleared successfully"}
    
    return {"message": f"Session {session_id} not found"}