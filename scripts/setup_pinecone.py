#!/usr/bin/env python3
"""
Setup script to initialize Pinecone vector database with furniture data
Run this once to upload your furniture dataset to Pinecone for semantic search
"""

import sys
import os
from pathlib import Path

# Add backend to path
sys.path.append('backend')

from backend.main_server import load_furniture_dataset
from backend.services.pinecone_service import PineconeService

def main():
    """Initialize Pinecone with furniture dataset"""
    print("🚀 Setting up Pinecone Vector Database for Furniture Search")
    print("=" * 60)
    
    # Load furniture dataset
    print("📂 Loading furniture dataset...")
    dataset = load_furniture_dataset()
    
    if not dataset:
        print("❌ Failed to load dataset. Please check your CSV file.")
        return False
    
    print(f"✅ Loaded {len(dataset)} products from dataset")
    
    # Initialize Pinecone service
    print("\n🔧 Initializing Pinecone service...")
    try:
        pinecone_service = PineconeService()
        print("✅ Pinecone service initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Pinecone service: {e}")
        return False
    
    # Upload products to Pinecone
    print(f"\n📤 Uploading {len(dataset)} products to Pinecone...")
    print("⏳ This may take a few minutes to create embeddings...")
    
    success = pinecone_service.upsert_products(dataset)
    
    if success:
        print("✅ Successfully uploaded all products to Pinecone!")
        
        # Get index stats
        stats = pinecone_service.get_index_stats()
        print(f"\n📊 Index Statistics:")
        print(f"   • Total vectors: {stats.get('total_vectors', 'Unknown')}")
        print(f"   • Dimension: {stats.get('dimension', 'Unknown')}")
        print(f"   • Index fullness: {stats.get('index_fullness', 'Unknown')}")
        
        # Test search
        print(f"\n🔍 Testing semantic search...")
        test_results = pinecone_service.semantic_search("comfortable sofa", max_results=3)
        
        if test_results:
            print("✅ Search test successful! Sample results:")
            for i, result in enumerate(test_results[:3]):
                print(f"   {i+1}. {result['title'][:50]}... (Score: {result['similarity_score']})")
        else:
            print("⚠️  Search test returned no results")
        
        print(f"\n🎉 Pinecone setup complete!")
        print(f"   Your backend can now use semantic search with vector embeddings")
        
        return True
    else:
        print("❌ Failed to upload products to Pinecone")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)