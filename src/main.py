from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import time

from src.database import engine
from src.models import database_models
from src.api.routes import router
from src.config import cleanup_all_data


# Create database tables
database_models.Base.metadata.create_all(bind=engine)

# Clean up all existing data on startup for fresh assessment
cleanup_all_data()

# Initialize FastAPI app
app = FastAPI(
    title="Research Paper RAG Assistant API",
    description="A RAG system for querying and understanding academic papers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes with /api prefix
app.include_router(router, prefix="/api", tags=["api"])


# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to Research Paper RAG Assistant API",
        "version": "1.0.0",
        "status": "running"
    }


# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time()
    }
