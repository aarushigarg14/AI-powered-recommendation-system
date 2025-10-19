"""
Data Manager for Furniture Dataset
Handles loading, cleaning, and processing of furniture data
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import ast
import re
from datetime import datetime

from utils.helpers import safe_parse_list, clean_price, create_combined_text, validate_image_url

logger = logging.getLogger(__name__)

class DataManager:
    """Manages furniture dataset loading, cleaning, and processing"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.raw_data: Optional[pd.DataFrame] = None
        self.clean_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False
        
    async def load_data(self) -> None:
        """Load and clean the furniture dataset"""
        try:
            logger.info(f"Loading dataset from {self.data_path}")
            
            # Check if file exists
            if not self.data_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
            
            # Load raw data
            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Loaded raw dataset with {len(self.raw_data)} rows and {len(self.raw_data.columns)} columns")
            
            # Clean and process data
            await self._clean_data()
            await self._generate_metadata()
            
            self.is_loaded = True
            logger.info("Dataset loading and processing completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    async def _clean_data(self) -> None:
        """Clean and preprocess the dataset"""
        logger.info("Starting data cleaning process...")
        
        # Create a copy for cleaning
        df = self.raw_data.copy()
        
        # Clean price column
        df['price_numeric'] = df['price'].apply(clean_price)
        
        # Parse categories and images as lists
        df['categories_list'] = df['categories'].apply(safe_parse_list)
        df['images_list'] = df['images'].apply(safe_parse_list)
        
        # Clean text columns
        text_columns = ['title', 'brand', 'description', 'material', 'color', 'manufacturer']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], np.nan)
        
        # Validate and clean image URLs
        df['valid_images'] = df['images_list'].apply(self._filter_valid_images)
        df['image_count'] = df['valid_images'].apply(len)
        df['primary_image'] = df['valid_images'].apply(lambda x: x[0] if x else None)
        
        # Create combined text for embeddings
        df['combined_text'] = df.apply(create_combined_text, axis=1)
        
        # Add derived columns
        df['has_price'] = df['price_numeric'].notna()
        df['has_description'] = df['description'].notna() & (df['description'] != 'nan')
        df['category_count'] = df['categories_list'].apply(len)
        
        # Extract main category (first category)
        df['main_category'] = df['categories_list'].apply(lambda x: x[0] if x else 'Unknown')
        
        # Clean dimensions if available
        if 'package_dimensions' in df.columns:
            df['dimensions_cleaned'] = df['package_dimensions'].apply(self._parse_dimensions)
        
        # Remove duplicates based on unique_id
        if 'uniq_id' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['uniq_id'], keep='first')
            duplicates_removed = initial_count - len(df)
            if duplicates_removed > 0:
                logger.info(f"Removed {duplicates_removed} duplicate products")
        
        # Store cleaned data
        self.clean_data = df
        logger.info(f"Data cleaning completed. Final dataset: {len(df)} products")
    
    def _filter_valid_images(self, images_list: List[str]) -> List[str]:
        """Filter and validate image URLs"""
        if not isinstance(images_list, list):
            return []
        
        valid_images = []
        for img_url in images_list:
            if img_url and isinstance(img_url, str):
                img_url = img_url.strip()
                if validate_image_url(img_url):
                    valid_images.append(img_url)
        
        return valid_images
    
    def _parse_dimensions(self, dim_str: str) -> Optional[Dict[str, float]]:
        """Parse package dimensions string"""
        if pd.isna(dim_str) or not isinstance(dim_str, str):
            return None
        
        try:
            # Extract numbers from dimension string
            numbers = re.findall(r'(\d+(?:\.\d+)?)', dim_str.replace(',', ''))
            if len(numbers) >= 3:
                return {
                    'length': float(numbers[0]),
                    'width': float(numbers[1]),
                    'height': float(numbers[2])
                }
        except:
            pass
        
        return None
    
    async def _generate_metadata(self) -> None:
        """Generate metadata about the dataset"""
        if self.clean_data is None:
            return
        
        df = self.clean_data
        
        # Basic statistics
        self.metadata = {
            'total_products': len(df),
            'unique_products': df['uniq_id'].nunique() if 'uniq_id' in df.columns else len(df),
            'columns': list(df.columns),
            'data_quality': {
                'products_with_price': df['has_price'].sum(),
                'products_with_description': df['has_description'].sum(),
                'products_with_images': (df['image_count'] > 0).sum(),
                'products_with_categories': (df['category_count'] > 0).sum()
            },
            'price_stats': {
                'min_price': float(df['price_numeric'].min()) if df['price_numeric'].notna().any() else 0,
                'max_price': float(df['price_numeric'].max()) if df['price_numeric'].notna().any() else 0,
                'avg_price': float(df['price_numeric'].mean()) if df['price_numeric'].notna().any() else 0,
                'median_price': float(df['price_numeric'].median()) if df['price_numeric'].notna().any() else 0
            },
            'categories': {
                'unique_categories': self._get_unique_categories(),
                'main_categories': df['main_category'].value_counts().head(10).to_dict()
            },
            'brands': df['brand'].value_counts().head(10).to_dict() if 'brand' in df.columns else {},
            'materials': df['material'].value_counts().head(10).to_dict() if 'material' in df.columns else {},
            'colors': df['color'].value_counts().head(10).to_dict() if 'color' in df.columns else {},
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info("Dataset metadata generated successfully")
    
    def _get_unique_categories(self) -> List[str]:
        """Get all unique categories from the dataset"""
        all_categories = set()
        for cat_list in self.clean_data['categories_list']:
            if isinstance(cat_list, list):
                all_categories.update(cat_list)
        return sorted(list(all_categories))
    
    # Public methods for accessing data
    def get_clean_data(self) -> pd.DataFrame:
        """Get the cleaned dataset"""
        if not self.is_loaded or self.clean_data is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        return self.clean_data
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        return self.metadata
    
    def get_product_by_id(self, product_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific product by ID"""
        if not self.is_loaded:
            return None
        
        product = self.clean_data[self.clean_data['uniq_id'] == product_id]
        if product.empty:
            return None
        
        return product.iloc[0].to_dict()
    
    def search_products(
        self, 
        query: str = None,
        category: str = None,
        min_price: float = None,
        max_price: float = None,
        material: str = None,
        color: str = None,
        limit: int = 20
    ) -> pd.DataFrame:
        """Search products based on various criteria"""
        if not self.is_loaded:
            return pd.DataFrame()
        
        df = self.clean_data.copy()
        
        # Apply filters
        if category:
            df = df[df['categories_list'].apply(
                lambda x: any(category.lower() in cat.lower() for cat in x) if isinstance(x, list) else False
            )]
        
        if min_price is not None:
            df = df[df['price_numeric'] >= min_price]
        
        if max_price is not None:
            df = df[df['price_numeric'] <= max_price]
        
        if material:
            df = df[df['material'].str.contains(material, case=False, na=False)]
        
        if color:
            df = df[df['color'].str.contains(color, case=False, na=False)]
        
        if query:
            # Simple text search in combined_text
            df = df[df['combined_text'].str.contains(query, case=False, na=False)]
        
        return df.head(limit)
    
    def get_category_count(self) -> int:
        """Get total number of unique categories"""
        return len(self.metadata.get('categories', {}).get('unique_categories', []))
    
    def get_valid_price_count(self) -> int:
        """Get number of products with valid prices"""
        return self.metadata.get('data_quality', {}).get('products_with_price', 0)
    
    def get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data for the dashboard"""
        if not self.is_loaded:
            return {}
        
        df = self.clean_data
        
        # Category distribution
        category_dist = []
        for category, count in self.metadata['categories']['main_categories'].items():
            if category != 'Unknown':
                percentage = (count / len(df)) * 100
                category_dist.append({
                    'category': category,
                    'count': count,
                    'percentage': round(percentage, 1)
                })
        
        # Price range distribution
        price_ranges = [
            (0, 50, '$0-50'),
            (50, 100, '$50-100'),
            (100, 200, '$100-200'),
            (200, 500, '$200-500'),
            (500, 1000, '$500-1000'),
            (1000, float('inf'), '$1000+')
        ]
        
        price_dist = []
        valid_prices = df['price_numeric'].dropna()
        total_with_price = len(valid_prices)
        
        for min_price, max_price, label in price_ranges:
            if max_price == float('inf'):
                count = len(valid_prices[valid_prices >= min_price])
            else:
                count = len(valid_prices[(valid_prices >= min_price) & (valid_prices < max_price)])
            
            if total_with_price > 0:
                percentage = (count / total_with_price) * 100
                price_dist.append({
                    'range': label,
                    'count': count,
                    'percentage': round(percentage, 1)
                })
        
        return {
            'total_products': len(df),
            'total_categories': self.get_category_count(),
            'average_price': round(self.metadata['price_stats']['avg_price'], 2),
            'valid_prices': total_with_price,
            'products_with_images': (df['image_count'] > 0).sum(),
            'category_distribution': category_dist,
            'price_distribution': price_dist,
            'top_brands': dict(list(self.metadata['brands'].items())[:5]),
            'top_materials': dict(list(self.metadata['materials'].items())[:5]),
            'top_colors': dict(list(self.metadata['colors'].items())[:5])
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for data manager"""
        return {
            'status': 'healthy' if self.is_loaded else 'not_loaded',
            'data_loaded': self.is_loaded,
            'total_products': len(self.clean_data) if self.clean_data is not None else 0,
            'data_path': str(self.data_path),
            'last_loaded': self.metadata.get('generated_at', 'never')
        }