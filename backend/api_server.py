# backend/api_server.py
# FINAL ARCHITECTURE v3: Using asyncio.to_thread for true non-blocking DB writes.

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

logger = setup_logger("TennisServer", settings.LOG_LEVEL)


# --- Dedicated Database Writer Task (Producer-Consumer Pattern) ---
async def database_writer_task(queue: asyncio.Queue):
    """
    This is a long-running task that pulls match data from a queue and writes it
    to the database one by one. This is the correct way to handle a synchronous,
    single-writer resource like SQLite in an async application.
    """
    logger.info("✅ Database writer task started and waiting for items...")
    db_manager = DatabaseManager()

    while True:
        try:
            match_data = await queue.get()
            if match_data is None:
                logger.info("Shutdown signal received. Database writer task is closing.")
                break

            # --- THIS IS THE FINAL, CRITICAL FIX ---
            # We run the blocking database operation in a separate thread,
            # which prevents it from freezing the main asyncio event loop.
            # This is the definitive solution to the timeout problem.
            await asyncio.to_thread(db_manager.process_match_data, match_data)

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

    db_write_queue = asyncio.Queue()
    app.state.db_write_queue = db_write_queue
    app.state.db_writer_task = asyncio.create_task(database_writer_task(db_write_queue))
    logger.info("Dedicated database writer task has been started.")

    app.state.rag_pipeline = None  # Placeholder

    logger.info("Tennis Server Lifespan: Startup complete. Application is ready.")
    yield
    logger.info("Tennis Server Lifespan: Shutdown sequence initiated.")

    if hasattr(app.state, 'db_write_queue') and app.state.db_write_queue:
        logger.info("Sending shutdown signal to database writer task...")
        await app.state.db_write_queue.put(None)

    if hasattr(app.state, 'db_writer_task') and app.state.db_writer_task:
        try:
            await asyncio.wait_for(app.state.db_writer_task, timeout=10.0)
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
    version="3.0.0",  # Final version
    lifespan=lifespan
)

app.include_router(api_handlers_router, prefix="/api")

@app.exception_handler(AvaChatError)
async def avachat_exception_handler(request: Request, exc: AvaChatError) -> JSONResponse:
    logger.error(f"Unhandled AvaChatError at API level: {exc} for request {request.url.path}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": {"code": "TENNIS_ERROR", "message": str(exc)}})

@app.get("/", include_in_schema=False)
async def root() -> dict[str, Any]:
    return {"message": "Welcome to the Tennis Intelligence API! Status: Operational"}