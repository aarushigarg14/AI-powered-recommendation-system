"""
Utility functions for data processing, validation, and common helpers
"""

import re
import ast
import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Data Processing Utilities
def safe_parse_list(val: Any) -> List[str]:
    """
    Safely parse string representations of lists
    
    Args:
        val: Value to parse (string, list, or other)
        
    Returns:
        List of strings
    """
    if pd.isna(val) or val == '' or val is None:
        return []
    
    try:
        if isinstance(val, list):
            return [str(item).strip() for item in val if item]
        
        if isinstance(val, str):
            val = val.strip()
            
            # Handle list-like strings
            if val.startswith('[') and val.endswith(']'):
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return [str(item).strip().strip('\'"') for item in parsed if item]
            
            # Handle comma-separated values
            if ',' in val:
                return [item.strip().strip('\'"') for item in val.split(',') if item.strip()]
            
            # Single value
            return [val.strip().strip('\'"')]
        
        return [str(val)]
    
    except Exception as e:
        logger.warning(f"Failed to parse list value '{val}': {e}")
        return [str(val)] if val else []

def clean_price(price_str: Any) -> Optional[float]:
    """
    Clean and convert price strings to float
    
    Args:
        price_str: Price string or value
        
    Returns:
        Float price or None if invalid
    """
    if pd.isna(price_str) or price_str == '' or price_str is None:
        return None
    
    try:
        # Handle numeric values
        if isinstance(price_str, (int, float)):
            return float(price_str) if price_str > 0 else None
        
        # Clean string price
        price_cleaned = re.sub(r'[^\d.]', '', str(price_str))
        
        if price_cleaned and '.' in price_cleaned:
            # Ensure only one decimal point
            parts = price_cleaned.split('.')
            if len(parts) == 2:
                price_cleaned = f"{parts[0]}.{parts[1]}"
            else:
                price_cleaned = parts[0]
        
        return float(price_cleaned) if price_cleaned else None
    
    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse price '{price_str}': {e}")
        return None

def validate_image_url(url: str) -> bool:
    """
    Validate if a URL is a valid image URL
    
    Args:
        url: URL string to validate
        
    Returns:
        Boolean indicating if URL is valid
    """
    if not url or not isinstance(url, str):
        return False
    
    try:
        parsed = urlparse(url.strip())
        
        # Check if URL has scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            return False
        
        # Check for common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'}
        path_lower = parsed.path.lower()
        
        # URL ends with image extension or contains image indicators
        has_image_ext = any(path_lower.endswith(ext) for ext in image_extensions)
        has_image_indicator = any(indicator in url.lower() for indicator in ['image', 'img', 'photo', 'pic'])
        
        return has_image_ext or has_image_indicator
    
    except Exception:
        return False

def create_combined_text(row: pd.Series) -> str:
    """
    Create combined text for embeddings from product data
    
    Args:
        row: Pandas Series with product data
        
    Returns:
        Combined text string
    """
    text_parts = []
    
    # Add title
    if pd.notna(row.get('title')) and str(row.get('title')).strip() not in ['nan', 'None', '']:
        text_parts.append(str(row['title']).strip())
    
    # Add description
    if pd.notna(row.get('description')) and str(row.get('description')).strip() not in ['nan', 'None', '']:
        desc = str(row['description']).strip()
        if len(desc) > 500:  # Truncate very long descriptions
            desc = desc[:500] + "..."
        text_parts.append(desc)
    
    # Add categories
    categories = row.get('categories_list', [])
    if isinstance(categories, list) and categories:
        # Take first 3 most specific categories
        relevant_cats = categories[:3]
        text_parts.append(' '.join(relevant_cats))
    
    # Add material
    if pd.notna(row.get('material')) and str(row.get('material')).strip() not in ['nan', 'None', '']:
        text_parts.append(f"material: {row['material']}")
    
    # Add color
    if pd.notna(row.get('color')) and str(row.get('color')).strip() not in ['nan', 'None', '']:
        text_parts.append(f"color: {row['color']}")
    
    # Add brand
    if pd.notna(row.get('brand')) and str(row.get('brand')).strip() not in ['nan', 'None', '']:
        text_parts.append(f"brand: {row['brand']}")
    
    return ' '.join(text_parts)

# Text Processing Utilities
def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract important keywords from text
    
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of keywords
    """
    if not text:
        return []
    
    # Simple keyword extraction (can be enhanced with NLP)
    text = text.lower()
    
    # Remove common stopwords
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'can', 'cannot'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    keywords = [word for word in words if word not in stopwords]
    
    # Count frequency and return top keywords
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:max_keywords]]

# Validation Utilities
def validate_search_query(query: str) -> Dict[str, Any]:
    """
    Validate and analyze search query
    
    Args:
        query: Search query string
        
    Returns:
        Dictionary with validation results and extracted information
    """
    result = {
        'valid': False,
        'query': query.strip() if query else '',
        'word_count': 0,
        'has_price_filter': False,
        'has_color_filter': False,
        'has_material_filter': False,
        'extracted_info': {}
    }
    
    if not query or not isinstance(query, str):
        return result
    
    clean_query = query.strip()
    if len(clean_query) < 2:
        return result
    
    result['valid'] = True
    result['query'] = clean_query
    result['word_count'] = len(clean_query.split())
    
    # Extract price information
    price_patterns = [
        r'under\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'below\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'less\s*than\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)\s*or\s*less',
        r'max\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)',
        r'maximum\s*\$?(\d+(?:,\d{3})*(?:\.\d{2})?)'
    ]
    
    for pattern in price_patterns:
        match = re.search(pattern, clean_query.lower())
        if match:
            try:
                price = float(match.group(1).replace(',', ''))
                result['has_price_filter'] = True
                result['extracted_info']['max_price'] = price
                break
            except ValueError:
                continue
    
    # Extract color information
    colors = [
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown',
        'black', 'white', 'gray', 'grey', 'beige', 'cream', 'ivory', 'gold',
        'silver', 'bronze', 'navy', 'maroon', 'teal', 'turquoise', 'lime',
        'magenta', 'cyan', 'tan', 'khaki', 'olive', 'coral', 'salmon'
    ]
    
    query_lower = clean_query.lower()
    found_colors = [color for color in colors if color in query_lower]
    if found_colors:
        result['has_color_filter'] = True
        result['extracted_info']['colors'] = found_colors
    
    # Extract material information
    materials = [
        'wood', 'wooden', 'metal', 'steel', 'iron', 'aluminum', 'leather',
        'fabric', 'cotton', 'linen', 'velvet', 'plastic', 'glass', 'marble',
        'granite', 'ceramic', 'bamboo', 'oak', 'pine', 'maple', 'mahogany',
        'teak', 'walnut', 'cherry', 'birch'
    ]
    
    found_materials = [material for material in materials if material in query_lower]
    if found_materials:
        result['has_material_filter'] = True
        result['extracted_info']['materials'] = found_materials
    
    return result

# Caching Utilities
class SimpleCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if datetime.now() > entry['expires']:
            del self.cache[key]
            return None
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL"""
        expires = datetime.now() + timedelta(seconds=ttl or self.default_ttl)
        self.cache[key] = {
            'value': value,
            'expires': expires
        }
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)

# Performance Utilities
async def batch_process(items: List[Any], batch_size: int = 10, delay: float = 0.1):
    """
    Process items in batches with delay
    
    Args:
        items: List of items to process
        batch_size: Size of each batch
        delay: Delay between batches in seconds
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        yield batch
        
        if delay > 0 and i + batch_size < len(items):
            await asyncio.sleep(delay)

def format_price(price: Optional[float]) -> str:
    """
    Format price for display
    
    Args:
        price: Price value
        
    Returns:
        Formatted price string
    """
    if price is None or pd.isna(price):
        return "Price not available"
    
    if price == 0:
        return "Free"
    
    return f"${price:.2f}"

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to specified length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].rsplit(' ', 1)[0] + suffix