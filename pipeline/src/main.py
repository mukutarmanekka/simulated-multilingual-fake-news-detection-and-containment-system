"""FastAPI application — internal ML pipeline API."""

from __future__ import annotations

import time
import traceback

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router as api_router
from .utils.redaction import redact_secrets

logger = structlog.get_logger()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize ML models and connections on startup and clean up on shutdown."""
    logger.info("pipeline.startup", msg="Initializing pipeline services...")
    try:
        from .services.llm_service import llm_service
        from .services.vector_service import vector_service

        await vector_service.ensure_collection()
        logger.info("pipeline.startup", msg="Qdrant collection ready")
    except Exception as e:
        logger.warning("pipeline.startup.partial", error=redact_secrets(str(e)),
                       msg="Some services failed to init — will retry on first use")
    logger.info("pipeline.startup", msg="Pipeline ready")
    yield
    logger.info("pipeline.shutdown", msg="Shutting down pipeline")


app = FastAPI(
    title="Fake News Detection Pipeline",
    description="Real-time claim extraction and verification ML pipeline",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pipeline", "timestamp": time.time()}


# Mount API routes
app.include_router(api_router, prefix="/api")
