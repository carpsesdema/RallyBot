# backend/api_server.py
# FINAL HARDENED VERSION - Starts the background monitor correctly.

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
from backend.background_tasks import monitor_live_matches # Import the monitor

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

            # Use asyncio.to_thread to run the blocking DB code without freezing the server.
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

    # --- THIS IS THE FIX ---
    # Start the background monitor task and pass it the application state
    # so it can access the shared queue.
    app.state.monitor_task = asyncio.create_task(monitor_live_matches(app.state))
    logger.info("Background match monitor task has been started.")

    yield  # The application runs here

    # --- Shutdown Sequence ---
    logger.info("Tennis Server Lifespan: Shutdown sequence initiated.")

    # Signal the monitor task to shut down (optional, but good practice)
    if hasattr(app.state, 'monitor_task') and app.state.monitor_task:
        app.state.monitor_task.cancel()

    # Signal the writer task to shut down
    if hasattr(app.state, 'db_write_queue') and app.state.db_write_queue:
        await app.state.db_write_queue.put(None)

    # Wait for the writer task to finish
    if hasattr(app.state, 'db_writer_task') and app.state.db_writer_task:
        await app.state.db_writer_task

    logger.info("Tennis Server Lifespan: Shutdown complete.")


# --- FastAPI App Instance ---
app = FastAPI(
    title="Tennis Intelligence Backend API",
    description="API server for Tennis Intelligence, handling RAG operations and LLM interactions with professional tennis data.",
    version="FINAL",
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