#!/usr/bin/env python3
"""
Script to analyze the furniture dataset structure for analytics
"""

import sys
import json
from collections import Counter
from backend.main_server import load_furniture_dataset

def analyze_dataset():
    """Analyze the furniture dataset and print statistics"""
    print("Loading dataset...")
    data = load_furniture_dataset()
    print(f"Dataset Size: {len(data)}")
    
    if not data:
        print("No data loaded!")
        return
    
    # Sample product
    print("\nSample Product:")
    print(json.dumps(data[0], indent=2, default=str))
    
    # Categories analysis
    print("\nCategories Analysis:")
    cats = [p.get('category') for p in data if p.get('category')]
    cat_counts = Counter(cats)
    print(f"Total unique categories: {len(cat_counts)}")
    print("Top 15 categories:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {cat}: {count}")
    
    # Price analysis
    print("\nPrice Analysis:")
    prices = [p.get('price') for p in data if p.get('price') is not None]
    if prices:
        print(f"  Products with prices: {len(prices)}")
        print(f"  Min: ${min(prices):.2f}")
        print(f"  Max: ${max(prices):.2f}")
        print(f"  Average: ${sum(prices)/len(prices):.2f}")
        
        # Price ranges
        price_ranges = {
            "Under $50": len([p for p in prices if p < 50]),
            "$50-$100": len([p for p in prices if 50 <= p < 100]),
            "$100-$200": len([p for p in prices if 100 <= p < 200]),
            "$200-$500": len([p for p in prices if 200 <= p < 500]),
            "$500+": len([p for p in prices if p >= 500])
        }
        print("  Price distribution:")
        for range_name, count in price_ranges.items():
            print(f"    {range_name}: {count}")
    
    # Brand analysis
    print("\nBrand Analysis:")
    brands = [p.get('brand') for p in data if p.get('brand')]
    brand_counts = Counter(brands)
    print(f"Total unique brands: {len(brand_counts)}")
    print("Top 10 brands:")
    for brand, count in sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {brand}: {count}")
    
    # Material analysis
    print("\nMaterial Analysis:")
    materials = [p.get('material') for p in data if p.get('material')]
    material_counts = Counter(materials)
    print(f"Total unique materials: {len(material_counts)}")
    print("Top 10 materials:")
    for material, count in sorted(material_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {material}: {count}")
    
    # Color analysis
    print("\nColor Analysis:")
    colors = [p.get('color') for p in data if p.get('color')]
    color_counts = Counter(colors)
    print(f"Total unique colors: {len(color_counts)}")
    print("Top 10 colors:")
    for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {color}: {count}")

if __name__ == "__main__":
    analyze_dataset()