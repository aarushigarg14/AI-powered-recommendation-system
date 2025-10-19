"""
Pinecone Vector Database Service
Handles vector embeddings and semantic search for furniture products
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class PineconeService:
    def __init__(self):
        """Initialize Pinecone service with API key and embedding model"""
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "furniture-recommendations")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "384"))
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        
        # Initialize embedding model
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or create Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone index: {e}")
            self.index = None
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for a text string"""
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return []
    
    def create_product_text(self, product: Dict[str, Any]) -> str:
        """Create searchable text from product data"""
        text_parts = []
        
        # Add title (most important)
        if product.get('title'):
            text_parts.append(f"Title: {product['title']}")
        
        # Add description
        if product.get('description'):
            text_parts.append(f"Description: {product['description']}")
        
        # Add category
        if product.get('category'):
            text_parts.append(f"Category: {product['category']}")
        
        # Add brand
        if product.get('brand'):
            text_parts.append(f"Brand: {product['brand']}")
        
        # Add material
        if product.get('material'):
            text_parts.append(f"Material: {product['material']}")
        
        # Add color
        if product.get('color'):
            text_parts.append(f"Color: {product['color']}")
        
        # Add price info
        if product.get('price'):
            text_parts.append(f"Price: ${product['price']}")
        
        return " | ".join(text_parts)
    
    def upsert_products(self, products: List[Dict[str, Any]]) -> bool:
        """Upload products to Pinecone index"""
        if not self.index:
            logger.error("Pinecone index not available")
            return False
        
        try:
            vectors = []
            batch_size = 100
            
            logger.info(f"Creating embeddings for {len(products)} products...")
            
            for i, product in enumerate(products):
                # Create searchable text
                product_text = self.create_product_text(product)
                
                # Create embedding
                embedding = self.create_embedding(product_text)
                
                if embedding:
                    # Prepare vector for upsert (handle null values)
                    metadata = {
                        "title": product.get('title') or '',
                        "category": product.get('category') or '',
                        "brand": product.get('brand') or '',
                        "material": product.get('material') or '',
                        "color": product.get('color') or '',
                        "text": product_text[:1000]  # Limit metadata size
                    }
                    
                    # Only add price if it exists and is not None
                    if product.get('price') is not None:
                        metadata["price"] = float(product['price'])
                    else:
                        metadata["price"] = 0.0
                    
                    vector = {
                        "id": str(product['id']),
                        "values": embedding,
                        "metadata": metadata
                    }
                    vectors.append(vector)
                
                # Batch upload
                if len(vectors) >= batch_size or i == len(products) - 1:
                    if vectors:
                        logger.info(f"Uploading batch {len(vectors)} vectors...")
                        self.index.upsert(vectors=vectors)
                        vectors = []
            
            logger.info("Successfully uploaded all products to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload products to Pinecone: {e}")
            return False
    
    def semantic_search(self, query: str, max_results: int = 20, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Perform semantic search using Pinecone"""
        if not self.index:
            logger.error("Pinecone index not available")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            if not query_embedding:
                return []
            
            # Prepare filter
            pinecone_filter = {}
            if filters:
                if filters.get('max_price'):
                    pinecone_filter['price'] = {'$lte': filters['max_price']}
                if filters.get('min_price'):
                    if 'price' in pinecone_filter:
                        pinecone_filter['price']['$gte'] = filters['min_price']
                    else:
                        pinecone_filter['price'] = {'$gte': filters['min_price']}
                if filters.get('category'):
                    pinecone_filter['category'] = {'$eq': filters['category']}
            
            # Perform search
            search_results = self.index.query(
                vector=query_embedding,
                top_k=max_results,
                filter=pinecone_filter if pinecone_filter else None,
                include_metadata=True
            )
            
            # Format results
            results = []
            for match in search_results['matches']:
                metadata = match['metadata']
                result = {
                    'id': match['id'],
                    'title': metadata.get('title', ''),
                    'category': metadata.get('category', ''),
                    'brand': metadata.get('brand', ''),
                    'price': metadata.get('price', 0),
                    'material': metadata.get('material', ''),
                    'color': metadata.get('color', ''),
                    'similarity_score': round(match['score'] * 100, 2),  # Convert to percentage
                    'description': metadata.get('text', '')[:200] + '...'
                }
                results.append(result)
            
            logger.info(f"Semantic search returned {len(results)} results for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics"""
        if not self.index:
            return {"error": "Index not available"}
        
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": len(stats.namespaces) if stats.namespaces else 0
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"error": str(e)}