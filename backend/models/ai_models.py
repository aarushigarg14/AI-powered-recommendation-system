"""
AI Models Manager
Handles NLP embeddings, computer vision, vector database, and generative AI
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import json
from datetime import datetime
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from PIL import Image
import requests
import io
import os

from utils.config import Settings
from utils.helpers import SimpleCache, validate_search_query, extract_keywords

logger = logging.getLogger(__name__)

class AIModelManager:
    """Manages all AI models and vector database operations"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Model instances
        self.embedding_model: Optional[SentenceTransformer] = None
        self.genai_model: Optional[Any] = None
        self.genai_tokenizer: Optional[Any] = None
        
        # Vector database (simplified in-memory for now)
        self.vector_store: Dict[str, Any] = {}
        self.embeddings_cache: Dict[str, np.ndarray] = {}
        
        # Caching
        self.query_cache = SimpleCache(default_ttl=settings.cache_ttl)
        self.description_cache = SimpleCache(default_ttl=settings.cache_ttl * 2)
        
        # Model status
        self.embedding_model_ready = False
        self.genai_model_ready = False
        self.vector_db_ready = False
        
    async def initialize_models(self) -> None:
        """Initialize all AI models"""
        try:
            logger.info("Initializing AI models...")
            
            # Initialize embedding model
            await self._load_embedding_model()
            
            # Initialize generative AI model
            await self._load_genai_model()
            
            logger.info("AI models initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {str(e)}")
            raise
    
    async def _load_embedding_model(self) -> None:
        """Load sentence transformer model for embeddings"""
        try:
            logger.info(f"Loading embedding model: {self.settings.embedding_model}")
            
            # Load in a separate thread to avoid blocking
            def load_model():
                return SentenceTransformer(self.settings.embedding_model)
            
            loop = asyncio.get_event_loop()
            self.embedding_model = await loop.run_in_executor(None, load_model)
            
            self.embedding_model_ready = True
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    async def _load_genai_model(self) -> None:
        """Load generative AI model for product descriptions"""
        try:
            logger.info(f"Loading GenAI model: {self.settings.genai_model}")
            
            def load_genai():
                tokenizer = AutoTokenizer.from_pretrained(self.settings.genai_model)
                model = AutoModelForSeq2SeqLM.from_pretrained(self.settings.genai_model)
                return tokenizer, model
            
            loop = asyncio.get_event_loop()
            self.genai_tokenizer, self.genai_model = await loop.run_in_executor(None, load_genai)
            
            self.genai_model_ready = True
            logger.info("GenAI model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load GenAI model: {str(e)}. Will use fallback descriptions.")
            # Don't raise - we can work without GenAI
    
    async def setup_vector_database(self, data: pd.DataFrame) -> None:
        """Set up vector database with product embeddings"""
        try:
            logger.info("Setting up vector database...")
            
            if not self.embedding_model_ready:
                logger.warning("Embedding model not ready. Skipping vector database setup.")
                return
            
            # Generate embeddings for all products
            await self._generate_product_embeddings(data)
            
            self.vector_db_ready = True
            logger.info(f"Vector database setup completed with {len(self.vector_store)} products")
            
        except Exception as e:
            logger.error(f"Failed to setup vector database: {str(e)}")
            raise
    
    async def _generate_product_embeddings(self, data: pd.DataFrame) -> None:
        """Generate embeddings for all products"""
        logger.info("Generating product embeddings...")
        
        batch_size = 32
        total_products = len(data)
        
        for i in range(0, total_products, batch_size):
            batch = data.iloc[i:i + batch_size]
            texts = batch['combined_text'].tolist()
            product_ids = batch['uniq_id'].tolist()
            
            # Generate embeddings for batch
            def encode_batch():
                return self.embedding_model.encode(texts, convert_to_numpy=True)
            
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, encode_batch)
            
            # Store embeddings and metadata
            for j, (product_id, embedding) in enumerate(zip(product_ids, embeddings)):
                product_data = batch.iloc[j].to_dict()
                
                self.vector_store[product_id] = {
                    'embedding': embedding,
                    'metadata': {
                        'id': product_id,
                        'title': product_data.get('title', ''),
                        'price': product_data.get('price_numeric'),
                        'category': product_data.get('main_category', ''),
                        'material': product_data.get('material'),
                        'color': product_data.get('color'),
                        'brand': product_data.get('brand'),
                        'description': product_data.get('description'),
                        'images': product_data.get('valid_images', []),
                        'primary_image': product_data.get('primary_image'),
                        'categories': product_data.get('categories_list', [])
                    }
                }
            
            # Progress logging
            processed = min(i + batch_size, total_products)
            logger.info(f"Processed {processed}/{total_products} product embeddings")
            
            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)
    
    async def search_similar_products(
        self, 
        query: str, 
        max_results: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar products using semantic search"""
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, max_results, filters)
            cached_result = self.query_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached search results")
                return cached_result
            
            if not self.vector_db_ready or not self.embedding_model_ready:
                logger.warning("Vector database or embedding model not ready. Using fallback search.")
                return await self._fallback_search(query, max_results)
            
            # Generate query embedding
            def encode_query():
                return self.embedding_model.encode([query], convert_to_numpy=True)[0]
            
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(None, encode_query)
            
            # Calculate similarities
            similarities = []
            for product_id, product_data in self.vector_store.items():
                product_embedding = product_data['embedding']
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, product_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding)
                )
                
                similarities.append({
                    'product_id': product_id,
                    'similarity': float(similarity),
                    'metadata': product_data['metadata']
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Apply filters if provided
            if filters:
                similarities = self._apply_filters(similarities, filters)
            
            # Get top results
            top_results = similarities[:max_results]
            
            # Generate AI descriptions for results
            results_with_descriptions = []
            for result in top_results:
                metadata = result['metadata']
                
                # Generate AI description
                ai_description = await self.generate_product_description(
                    metadata['title'], 
                    metadata.get('description', ''),
                    metadata.get('category', ''),
                    metadata.get('material', ''),
                    metadata.get('color', '')
                )
                
                product_result = {
                    'id': metadata['id'],
                    'title': metadata['title'],
                    'price': metadata['price'],
                    'category': metadata['category'],
                    'material': metadata['material'],
                    'color': metadata['color'],
                    'brand': metadata['brand'],
                    'description': ai_description,
                    'original_description': metadata.get('description', ''),
                    'images': metadata.get('images', []),
                    'primary_image': metadata.get('primary_image'),
                    'categories': metadata.get('categories', []),
                    'similarity_score': result['similarity']
                }
                
                results_with_descriptions.append(product_result)
            
            # Cache results
            self.query_cache.set(cache_key, results_with_descriptions)
            
            logger.info(f"Found {len(results_with_descriptions)} similar products for query: {query}")
            return results_with_descriptions
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            # Return fallback search results
            return await self._fallback_search(query, max_results)
    
    async def _fallback_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Fallback search when vector search is not available"""
        # Simple keyword-based search
        results = []
        query_lower = query.lower()
        keywords = extract_keywords(query)
        
        for product_id, product_data in self.vector_store.items():
            metadata = product_data['metadata']
            title = (metadata.get('title', '') or '').lower()
            description = (metadata.get('description', '') or '').lower()
            category = (metadata.get('category', '') or '').lower()
            
            # Simple scoring based on keyword matches
            score = 0
            for keyword in keywords:
                if keyword in title:
                    score += 3
                if keyword in description:
                    score += 2
                if keyword in category:
                    score += 1
            
            if score > 0:
                ai_description = await self.generate_product_description(
                    metadata['title'], 
                    metadata.get('description', ''),
                    metadata.get('category', ''),
                    metadata.get('material', ''),
                    metadata.get('color', '')
                )
                
                results.append({
                    'id': metadata['id'],
                    'title': metadata['title'],
                    'price': metadata['price'],
                    'category': metadata['category'],
                    'material': metadata['material'],
                    'color': metadata['color'],
                    'brand': metadata['brand'],
                    'description': ai_description,
                    'original_description': metadata.get('description', ''),
                    'images': metadata.get('images', []),
                    'primary_image': metadata.get('primary_image'),
                    'categories': metadata.get('categories', []),
                    'similarity_score': score / 10.0  # Normalize
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:max_results]
    
    def _apply_filters(self, similarities: List[Dict], filters: Dict[str, Any]) -> List[Dict]:
        """Apply filters to search results"""
        filtered = []
        
        for result in similarities:
            metadata = result['metadata']
            include = True
            
            # Price filters
            if filters.get('min_price') is not None:
                if not metadata.get('price') or metadata['price'] < filters['min_price']:
                    include = False
            
            if filters.get('max_price') is not None:
                if not metadata.get('price') or metadata['price'] > filters['max_price']:
                    include = False
            
            # Category filter
            if filters.get('category'):
                categories = metadata.get('categories', [])
                if not any(filters['category'].lower() in cat.lower() for cat in categories):
                    include = False
            
            # Material filter
            if filters.get('material'):
                material = metadata.get('material', '') or ''
                if filters['material'].lower() not in material.lower():
                    include = False
            
            # Color filter
            if filters.get('color'):
                color = metadata.get('color', '') or ''
                if filters['color'].lower() not in color.lower():
                    include = False
            
            if include:
                filtered.append(result)
        
        return filtered
    
    async def generate_product_description(
        self, 
        title: str, 
        original_description: str = "",
        category: str = "",
        material: str = "",
        color: str = ""
    ) -> str:
        """Generate AI-powered product description"""
        try:
            # Create cache key
            cache_key = hashlib.md5(
                f"{title}_{original_description}_{category}_{material}_{color}".encode()
            ).hexdigest()
            
            # Check cache
            cached_desc = self.description_cache.get(cache_key)
            if cached_desc:
                return cached_desc
            
            if not self.genai_model_ready:
                # Use enhanced fallback description
                description = self._generate_fallback_description(title, category, material, color)
                self.description_cache.set(cache_key, description)
                return description
            
            # Create prompt for GenAI
            prompt = self._create_description_prompt(title, original_description, category, material, color)
            
            # Generate description
            def generate():
                inputs = self.genai_tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
                
                with torch.no_grad():
                    outputs = self.genai_model.generate(
                        inputs.input_ids,
                        max_length=100,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.genai_tokenizer.eos_token_id
                    )
                
                description = self.genai_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return description.strip()
            
            loop = asyncio.get_event_loop()
            ai_description = await loop.run_in_executor(None, generate)
            
            # Post-process description
            ai_description = self._post_process_description(ai_description, title)
            
            # Cache the result
            self.description_cache.set(cache_key, ai_description)
            
            return ai_description
            
        except Exception as e:
            logger.error(f"Failed to generate AI description: {str(e)}")
            # Return fallback description
            return self._generate_fallback_description(title, category, material, color)
    
    def _create_description_prompt(self, title: str, original_desc: str, category: str, material: str, color: str) -> str:
        """Create prompt for GenAI model"""
        prompt_parts = [
            "Write a creative, engaging 1-2 sentence product description for this furniture item:",
            f"Product: {title}"
        ]
        
        if category:
            prompt_parts.append(f"Category: {category}")
        if material:
            prompt_parts.append(f"Material: {material}")
        if color:
            prompt_parts.append(f"Color: {color}")
        
        prompt_parts.append("Description:")
        
        return " ".join(prompt_parts)
    
    def _post_process_description(self, description: str, title: str) -> str:
        """Post-process generated description"""
        # Remove the original prompt if it appears
        if "Description:" in description:
            description = description.split("Description:")[-1].strip()
        
        # Ensure it doesn't start with the title
        if description.lower().startswith(title.lower()):
            description = description[len(title):].strip()
            if description.startswith(':') or description.startswith('-'):
                description = description[1:].strip()
        
        # Capitalize first letter
        if description:
            description = description[0].upper() + description[1:]
        
        # Ensure it ends with a period
        if description and not description.endswith(('.', '!', '?')):
            description += '.'
        
        return description or self._generate_fallback_description(title, '', '', '')
    
    def _generate_fallback_description(self, title: str, category: str, material: str, color: str) -> str:
        """Generate fallback description when AI is not available"""
        templates = [
            "Discover the perfect blend of style and functionality with this {adjective} {category}.",
            "Transform your space with this {adjective} {category} that combines quality craftsmanship with modern design.",
            "Experience comfort and elegance with this beautifully crafted {category} piece.",
            "Add sophistication to your home with this {adjective} {category} featuring premium quality construction.",
            "Create your ideal living space with this thoughtfully designed {category} that offers both style and durability."
        ]
        
        # Choose appropriate adjective based on available info
        adjectives = []
        if color:
            adjectives.append(color.lower())
        if material:
            if 'wood' in material.lower():
                adjectives.append('wooden')
            elif 'metal' in material.lower():
                adjectives.append('metal')
            elif 'leather' in material.lower():
                adjectives.append('leather')
        
        adjective = ', '.join(adjectives) if adjectives else 'stylish'
        category_name = category.lower() if category else 'furniture piece'
        
        import random
        template = random.choice(templates)
        description = template.format(adjective=adjective, category=category_name)
        
        return description
    
    def _generate_cache_key(self, query: str, max_results: int, filters: Dict[str, Any]) -> str:
        """Generate cache key for search results"""
        key_data = {
            'query': query,
            'max_results': max_results,
            'filters': filters or {}
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    # Status checking methods
    def is_embedding_model_ready(self) -> bool:
        """Check if embedding model is ready"""
        return self.embedding_model_ready
    
    def is_genai_model_ready(self) -> bool:
        """Check if GenAI model is ready"""
        return self.genai_model_ready
    
    def is_vector_db_ready(self) -> bool:
        """Check if vector database is ready"""
        return self.vector_db_ready
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for AI models"""
        return {
            'embedding_model': 'ready' if self.embedding_model_ready else 'not_ready',
            'genai_model': 'ready' if self.genai_model_ready else 'not_ready',
            'vector_database': 'ready' if self.vector_db_ready else 'not_ready',
            'products_indexed': len(self.vector_store),
            'cache_size': {
                'query_cache': self.query_cache.size(),
                'description_cache': self.description_cache.size()
            }
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up AI models...")
        
        # Clear caches
        self.query_cache.clear()
        self.description_cache.clear()
        
        # Clear vector store
        self.vector_store.clear()
        self.embeddings_cache.clear()
        
        # Model cleanup would go here if needed
        
        logger.info("AI models cleanup completed")