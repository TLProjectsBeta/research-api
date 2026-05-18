from fastapi import APIRouter
from app.config import get_settings
import time

router = APIRouter()
start_time = time.time()

@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "version": settings.app_version,
        "uptime_seconds": round(time.time() - start_time, 2),
        "tracing_enabled": settings.langchain_tracing_v2 == "true"
    }

@router.get("/")
async def root():
    return {
        "message": "Research Assistant API",
        "docs": "/docs",
        "playground_research": "/research/playground",
        "playground_chat": "/chat/playground"
    }