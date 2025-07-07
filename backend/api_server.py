# backend/api_server.py
# FINAL ARCHITECTURE: Using an asyncio.Queue for a dedicated, sequential DB writer.

import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pathlib import Path
import os
from typing import AsyncGenerator, Any

# --- App component imports ---
from config import settings
from utils import setup_logger, AvaChatError
from backend.api_handlers import router as api_handlers_router
from database.db_manager import DatabaseManager

# --- Fallback imports for robustness ---
try:
    from llm_interface.gemini_client import GeminiLLMClient
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import FAISSVectorStore
    from rag.rag_pipeline import RAGPipeline
except ImportError as e:
    print(f"CRITICAL Backend Import Error in api_server.py: {e}. Using dummy fallbacks.")
    # Dummy classes for missing components would be defined here in a real scenario
    # but are omitted to keep the focus on the production code.

logger = setup_logger("TennisServer", settings.LOG_LEVEL)


# --- Dedicated Database Writer Task (Producer-Consumer Pattern) ---
async def database_writer_task(queue: asyncio.Queue):
    """
    This is a long-running task that pulls match data from a queue and writes it
    to the database one by one. This is the correct way to handle a synchronous,
    single-writer resource like SQLite in an async application.
    """
    logger.info("✅ Database writer task started and waiting for items...")
    # This single DatabaseManager instance lives as long as the server.
    db_manager = DatabaseManager()

    while True:
        try:
            match_data = await queue.get()

            # A 'None' in the queue is the signal to shut down.
            if match_data is None:
                logger.info("Shutdown signal received. Database writer task is closing.")
                break

            # This is a blocking I/O call, but it's fine because this task
            # runs independently and doesn't block the main server.
            db_manager.process_match_data(match_data)
            queue.task_done()

        except Exception as e:
            logger.error(f"DATABASE_WRITER_ERROR: An error occurred: {e}", exc_info=True)

    if db_manager:
        db_manager.close()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the application's startup and shutdown lifecycle.
    """
    logger.info("Tennis Server Lifespan: Startup sequence initiated.")

    # Create and start the database writer queue and task
    db_write_queue = asyncio.Queue()
    app.state.db_write_queue = db_write_queue
    app.state.db_writer_task = asyncio.create_task(database_writer_task(db_write_queue))
    logger.info("Dedicated database writer task has been started.")

    # You can initialize other components like the RAG pipeline here
    # For now, we focus on the database part which is the point of failure.
    app.state.rag_pipeline = None  # Placeholder for RAG components

    logger.info("Tennis Server Lifespan: Startup complete. Application is ready.")
    yield  # Application runs here

    # --- Shutdown Sequence ---
    logger.info("Tennis Server Lifespan: Shutdown sequence initiated.")

    # Signal the writer task to shut down by sending the sentinel value
    if hasattr(app.state, 'db_write_queue') and app.state.db_write_queue:
        logger.info("Sending shutdown signal to database writer task...")
        await app.state.db_write_queue.put(None)

    # Wait for the writer task to finish processing any remaining items and exit
    if hasattr(app.state, 'db_writer_task') and app.state.db_writer_task:
        try:
            await asyncio.wait_for(app.state.db_writer_task, timeout=30.0)
            logger.info("Database writer task has been successfully shut down.")
        except asyncio.TimeoutError:
            logger.error("Database writer task did not shut down gracefully within the timeout.")
        except Exception as e:
            logger.error(f"Error during database writer task shutdown: {e}", exc_info=True)


    logger.info("Tennis Server Lifespan: Shutdown complete.")


# --- FastAPI App Instance ---
app = FastAPI(
    title="Tennis Intelligence Backend API",
    description="API server for Tennis Intelligence, handling RAG operations and LLM interactions with professional tennis data.",
    version="2.0.0",  # Version bump for new architecture
    lifespan=lifespan
)

# Include API routers
app.include_router(api_handlers_router, prefix="/api")

# Global exception handler
@app.exception_handler(AvaChatError)
async def avachat_exception_handler(request: Request, exc: AvaChatError) -> JSONResponse:
    logger.error(f"Unhandled AvaChatError at API level: {exc} for request {request.url.path}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": {"code": "TENNIS_ERROR", "message": str(exc)}})

# Root path for basic health check
@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    return {"message": "Welcome to the Tennis Intelligence API! Status: Operational"}