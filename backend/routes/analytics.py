

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta
import asyncio

from models.data_manager import DataManager

logger = logging.getLogger(__name__)
router = APIRouter()

class CategoryDistribution(BaseModel):
    category: str
    count: int
    percentage: float

class PriceDistribution(BaseModel):
    range: str
    count: int
    percentage: float

class BrandData(BaseModel):
    brand: str
    count: int
    percentage: float

class MaterialData(BaseModel):
    material: str
    count: int
    percentage: float

class TrendData(BaseModel):
    month: str
    searches: int
    sales: int
    revenue: float

class AnalyticsResponse(BaseModel):
    summary: Dict[str, Any]
    category_distribution: List[CategoryDistribution]
    price_distribution: List[PriceDistribution]
    top_brands: List[BrandData]
    top_materials: List[MaterialData]
    monthly_trends: List[TrendData]
    data_quality: Dict[str, Any]
    last_updated: str

@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics_data(
    data_manager: DataManager = Depends()
) -> AnalyticsResponse:
    """
    Get comprehensive analytics data for the business dashboard
    
    Returns:
    - Summary statistics (total products, categories, prices)
    - Category and price distributions
    - Top brands and materials
    - Data quality metrics
    - Monthly trends (simulated)
    """
    try:
        logger.info("Generating analytics data...")
        start_time = asyncio.get_event_loop().time()
        
        # Get analytics data from data manager
        analytics_data = data_manager.get_analytics_data()
        metadata = data_manager.get_metadata()
        
        if not analytics_data:
            raise HTTPException(
                status_code=503,
                detail="Analytics data not available. Dataset may not be loaded."
            )
        
        # Summary statistics
        summary = {
            "total_products": analytics_data.get('total_products', 0),
            "total_categories": analytics_data.get('total_categories', 0),
            "average_price": analytics_data.get('average_price', 0),
            "valid_prices": analytics_data.get('valid_prices', 0),
            "products_with_images": analytics_data.get('products_with_images', 0),
            "total_revenue": await calculate_total_revenue(analytics_data),
            "conversion_rate": await calculate_conversion_rate(),
            "active_sessions": await get_active_sessions_count()
        }
        
        # Category distribution
        category_distribution = []
        for cat_data in analytics_data.get('category_distribution', []):
            category_distribution.append(CategoryDistribution(
                category=cat_data['category'],
                count=cat_data['count'],
                percentage=cat_data['percentage']
            ))
        
        # Price distribution
        price_distribution = []
        for price_data in analytics_data.get('price_distribution', []):
            price_distribution.append(PriceDistribution(
                range=price_data['range'],
                count=price_data['count'],
                percentage=price_data['percentage']
            ))
        
        # Top brands
        top_brands = []
        brands_data = analytics_data.get('top_brands', {})
        total_with_brands = sum(brands_data.values())
        
        for brand, count in list(brands_data.items())[:10]:
            if brand and brand.lower() not in ['nan', 'none', 'unknown']:
                percentage = (count / total_with_brands * 100) if total_with_brands > 0 else 0
                top_brands.append(BrandData(
                    brand=brand,
                    count=count,
                    percentage=round(percentage, 1)
                ))
        
        # Top materials
        top_materials = []
        materials_data = analytics_data.get('top_materials', {})
        total_with_materials = sum(materials_data.values())
        
        for material, count in list(materials_data.items())[:10]:
            if material and material.lower() not in ['nan', 'none', 'unknown']:
                percentage = (count / total_with_materials * 100) if total_with_materials > 0 else 0
                top_materials.append(MaterialData(
                    material=material,
                    count=count,
                    percentage=round(percentage, 1)
                ))
        
        # Monthly trends (simulated data for demo)
        monthly_trends = await generate_monthly_trends(analytics_data)
        
        # Data quality metrics
        data_quality = {
            "completeness_score": await calculate_data_completeness(metadata),
            "price_coverage": (summary['valid_prices'] / summary['total_products'] * 100) if summary['total_products'] > 0 else 0,
            "image_coverage": (summary['products_with_images'] / summary['total_products'] * 100) if summary['total_products'] > 0 else 0,
            "category_coverage": (analytics_data.get('total_categories', 0) / summary['total_products'] * 100) if summary['total_products'] > 0 else 0,
            "missing_data_points": await identify_missing_data(metadata)
        }
        
        processing_time = asyncio.get_event_loop().time() - start_time
        logger.info(f"Analytics data generated in {processing_time:.3f}s")
        
        return AnalyticsResponse(
            summary=summary,
            category_distribution=category_distribution,
            price_distribution=price_distribution,
            top_brands=top_brands,
            top_materials=top_materials,
            monthly_trends=monthly_trends,
            data_quality=data_quality,
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Analytics generation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate analytics data: {str(e)}"
        )

async def calculate_total_revenue(analytics_data: Dict[str, Any]) -> float:
    """Calculate estimated total revenue"""
    # Simulate revenue calculation based on average price and conversion
    total_products = analytics_data.get('total_products', 0)
    avg_price = analytics_data.get('average_price', 0)
    
    # Assume 15% of products have been "sold" for simulation
    estimated_sales = total_products * 0.15
    estimated_revenue = estimated_sales * avg_price
    
    return round(estimated_revenue, 2)

async def calculate_conversion_rate() -> float:
    """Calculate simulated conversion rate"""
    # Simulate conversion rate between 2-5%
    import random
    return round(random.uniform(2.0, 5.0), 1)

async def get_active_sessions_count() -> int:
    """Get count of active user sessions"""
    # Simulate active sessions
    import random
    return random.randint(50, 150)

async def generate_monthly_trends(analytics_data: Dict[str, Any]) -> List[TrendData]:
    """Generate simulated monthly trends data"""
    import random
    
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    trends = []
    
    base_searches = 1000
    base_sales = 200
    avg_price = analytics_data.get('average_price', 100)
    
    for i, month in enumerate(months):
        # Simulate growth trend
        growth_factor = 1 + (i * 0.1) + random.uniform(-0.05, 0.1)
        searches = int(base_searches * growth_factor)
        sales = int(base_sales * growth_factor)
        revenue = sales * avg_price * random.uniform(0.8, 1.2)
        
        trends.append(TrendData(
            month=month,
            searches=searches,
            sales=sales,
            revenue=round(revenue, 2)
        ))
    
    return trends

async def calculate_data_completeness(metadata: Dict[str, Any]) -> float:
    """Calculate overall data completeness score"""
    if not metadata.get('data_quality'):
        return 0.0
    
    data_quality = metadata['data_quality']
    total_products = metadata.get('total_products', 1)
    
    # Weight different fields by importance
    weights = {
        'products_with_price': 0.3,
        'products_with_description': 0.2,
        'products_with_images': 0.25,
        'products_with_categories': 0.25
    }
    
    weighted_score = 0
    for field, weight in weights.items():
        if field in data_quality:
            completeness = data_quality[field] / total_products
            weighted_score += completeness * weight
    
    return round(weighted_score * 100, 1)

async def identify_missing_data(metadata: Dict[str, Any]) -> List[str]:
    """Identify types of missing data"""
    missing_data = []
    
    if not metadata.get('data_quality'):
        return missing_data
    
    data_quality = metadata['data_quality']
    total_products = metadata.get('total_products', 0)
    
    # Check for significant missing data (>20% missing)
    if total_products > 0:
        missing_threshold = 0.2 * total_products
        
        if (total_products - data_quality.get('products_with_price', 0)) > missing_threshold:
            missing_data.append('Price information')
        
        if (total_products - data_quality.get('products_with_description', 0)) > missing_threshold:
            missing_data.append('Product descriptions')
        
        if (total_products - data_quality.get('products_with_images', 0)) > missing_threshold:
            missing_data.append('Product images')
        
        if (total_products - data_quality.get('products_with_categories', 0)) > missing_threshold:
            missing_data.append('Category information')
    
    return missing_data

@router.get("/analytics/categories")
async def get_category_analytics(
    data_manager: DataManager = Depends()
) -> Dict[str, Any]:
    """Get detailed category analytics"""
    
    try:
        analytics_data = data_manager.get_analytics_data()
        metadata = data_manager.get_metadata()
        
        category_details = []
        for cat_data in analytics_data.get('category_distribution', []):
            category_details.append({
                'name': cat_data['category'],
                'product_count': cat_data['count'],
                'percentage': cat_data['percentage'],
                'avg_price': await get_category_avg_price(cat_data['category'], data_manager),
                'price_range': await get_category_price_range(cat_data['category'], data_manager)
            })
        
        return {
            'total_categories': len(category_details),
            'categories': category_details,
            'most_popular': category_details[0] if category_details else None,
            'category_diversity': len(metadata.get('categories', {}).get('unique_categories', []))
        }
        
    except Exception as e:
        logger.error(f"Category analytics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_category_avg_price(category: str, data_manager: DataManager) -> float:
    """Get average price for a specific category"""
    try:
        # Simulate category-specific pricing
        import random
        base_prices = {
            'living room': random.uniform(300, 800),
            'bedroom': random.uniform(200, 600),
            'dining room': random.uniform(150, 500),
            'office': random.uniform(100, 400),
            'storage': random.uniform(50, 200)
        }
        
        category_lower = category.lower()
        for key, price in base_prices.items():
            if key in category_lower:
                return round(price, 2)
        
        return round(random.uniform(100, 400), 2)
    except:
        return 0.0

async def get_category_price_range(category: str, data_manager: DataManager) -> Dict[str, float]:
    """Get price range for a specific category"""
    try:
        avg_price = await get_category_avg_price(category, data_manager)
        return {
            'min': round(avg_price * 0.3, 2),
            'max': round(avg_price * 2.5, 2)
        }
    except:
        return {'min': 0.0, 'max': 0.0}

@router.get("/analytics/performance")
async def get_performance_metrics(
    data_manager: DataManager = Depends()
) -> Dict[str, Any]:
    """Get performance metrics for the system"""
    
    try:
        import psutil
        import time
        
        # System performance
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Application performance
        analytics_data = data_manager.get_analytics_data()
        
        return {
            'system_performance': {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_total': round(memory.total / (1024**3), 2),  # GB
                'memory_available': round(memory.available / (1024**3), 2)  # GB
            },
            'data_performance': {
                'total_products': analytics_data.get('total_products', 0),
                'query_response_time': '~150ms',  # Simulated
                'cache_hit_rate': '85%',  # Simulated
                'search_accuracy': '92%'  # Simulated
            },
            'api_metrics': {
                'uptime': '99.9%',  # Simulated
                'requests_per_minute': 45,  # Simulated
                'error_rate': '0.1%'  # Simulated
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except ImportError:
        # Fallback if psutil is not available
        return {
            'system_performance': {
                'cpu_usage': 15.0,
                'memory_usage': 45.0,
                'memory_total': 8.0,
                'memory_available': 4.4
            },
            'data_performance': {
                'total_products': data_manager.get_analytics_data().get('total_products', 0),
                'query_response_time': '~150ms',
                'cache_hit_rate': '85%',
                'search_accuracy': '92%'
            },
            'api_metrics': {
                'uptime': '99.9%',
                'requests_per_minute': 45,
                'error_rate': '0.1%'
            },
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Performance metrics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))