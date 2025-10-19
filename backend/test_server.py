from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Test Server")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Test server is working!"}

@app.get("/api/health")
async def health():
    logger.info("Health endpoint called")
    return {"status": "healthy", "message": "Backend is running"}

@app.post("/api/search")
async def search(data: dict):
    logger.info(f"Search called with: {data}")
    return {
        "success": True,
        "message": "Search working",
        "results": [
            {
                "id": "test-1",
                "title": "Test Chair",
                "price": 199.99,
                "description": "A test chair"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting test server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")