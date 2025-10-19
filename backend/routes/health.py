"""
Health check routes for monitoring server status
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint
    
    Returns:
        Server health status
    """
    try:
        return {
            "status": "healthy",
            "message": "AI Furniture Recommendation Platform is running",
            "timestamp": asyncio.get_event_loop().time(),
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "message": "Service unavailable",
                "error": str(e)
            }
        )