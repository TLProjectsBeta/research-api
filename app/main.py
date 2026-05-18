import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from langserve import add_routes

from app.config import get_settings
from app.utils.logger import setup_logging, logger
from app.chains.research import research_chain
from app.chains.history import chat_chain
from app.middleware.auth import verify_api_key
from app.routers import health

settings = get_settings()

# Set LangSmith env vars before anything else
if settings.langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = settings.langchain_tracing_v2
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.debug)
    logger.info("Starting Research API", version=settings.app_version)
    yield
    logger.info("Shutting down Research API")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI Research Assistant - Powered by LangServe",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourfrontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health routes — no auth required
app.include_router(health.router)

# ── LangServe Routes ──────────────────────────────────────
# Registers: /research/invoke, /stream, /batch, /playground, /input_schema
add_routes(
    app,
    research_chain,
    path="/research",
    dependencies=[Depends(verify_api_key)],
    enabled_endpoints=["invoke", "stream", "batch", "playground", "input_schema"]
)

# Registers: /chat/invoke, /stream, /playground
from fastapi import Request
from langchain_core.messages import HumanMessage

@app.post("/chat/invoke", dependencies=[Depends(verify_api_key)])
async def chat_invoke(request: Request):
    body = await request.json()
    question = body["input"]["question"]
    session_id = body.get("config", {}).get("configurable", {}).get("session_id", "default")
    
    response = await chat_chain.ainvoke(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )
    return {"output": response}

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception", error=str(exc), path=str(request.url))
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1 if settings.debug else 4
    )