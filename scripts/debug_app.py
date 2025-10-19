#!/usr/bin/env python3
"""
Debug script to check FastAPI app routes
"""

from backend.main_server import app

def debug_routes():
    """Debug the available routes in the FastAPI app"""
    print("App loaded successfully")
    print("Available endpoints:")
    
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = list(route.methods)
            print(f"  {methods} {route.path}")
        else:
            print(f"  {route}")

if __name__ == "__main__":
    debug_routes()