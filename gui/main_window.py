# gui/main_window.py
import sys
import asyncio
import logging
from typing import Optional, Union, List

# --- PySide6 Imports ---
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton,
        QVBoxLayout, QHBoxLayout, QWidget, QStatusBar, QFileDialog, QMessageBox
    )
    from PySide6.QtCore import QThread, QObject, Signal, Slot, Qt
except ImportError:
    print("CRITICAL: PySide6 is not installed. Please install it: pip install PySide6")


    # Define dummy classes for parsing if PySide6 is missing
    class QObject:
        pass


    class Signal:
        def __init__(self, *args, **kwargs):
            pass


    class Slot:
        def __init__(self, *args, **kwargs):
            pass


    class QMainWindow:
        pass  # Other Qt dummies would be needed for full parsing

# --- Application-specific Imports ---
# Attempt to import project components. Dummy fallbacks for incremental dev.
try:
    from config import Settings
    from models import QueryRequest, QueryResponse, IngestDirectoryRequest, IngestDirectoryResponse, ChatMessage, \
        ApiErrorResponse
    from gui.api_client import ApiClient  # apiClient from the same package
    from utils import ApiClientError  # For type checking in error handling
except ImportError as e:
    print(f"Import Error in gui/main_window.py: {e}. Ensure other modules are correctly defined.")


    class Settings:  # Dummy
        API_SERVER_HOST = "localhost";
        API_SERVER_PORT = 8000;
        LOG_LEVEL = "INFO"


    class ApiClient:  # Dummy
        async def post_chat_query(self, payload): return QueryResponse(answer="Dummy chat response",
                                                                       retrieved_chunks_details=[])

        async def post_ingest_directory(self, payload): return IngestDirectoryResponse(status="dummy success",
                                                                                       documents_processed=0,
                                                                                       chunks_created=0)

        async def close(self): pass


    class QueryRequest:
        def    __init__(self, query_text, **kwargs): self.query_text = query_text


    class QueryResponse:
        def __init__(self, answer, retrieved_chunks_details):
            self.answer = answer
            self.retrieved_chunks_details = retrieved_chunks_details


    class IngestDirectoryRequest:
        def __init__(self, directory_path): self.directory_path = directory_path


    class IngestDirectoryResponse:
        def __init__(self, status, documents_processed, chunks_created): self.status = status


    class ChatMessage:
        def __init__(self, role, content, sources=None):
            self.role = role
            self.content = content
            self.sources = sources


    class ApiErrorResponse:
        error: object = type('dummy', (), {'message': 'Dummy error'})()


    class ApiClientError(Exception):
        pass

logger = logging.getLogger(__name__)  # Will be configured by setup_logger in main.py


# --- API Worker ---
class ApiWorker(QObject):
    """
    Worker object to handle API calls in a separate thread.
    Emits signals with results or errors.
    """
    # Signals should emit types that PySide6 can handle. Python objects are generally fine.
    chat_response_received = Signal(object)  # Emits QueryResponse
    ingest_response_received = Signal(object)  # Emits IngestDirectoryResponse
    error_occurred = Signal(object)  # Emits ApiErrorResponse or str for other errors

    def __init__(self, api_client_instance: ApiClient):
        super().__init__()
        self.api_client = api_client_instance
        self._loop = None  # For managing asyncio event loop in this thread
        logger.info("ApiWorker initialized.")

    def _ensure_event_loop(self):
        """Ensures an asyncio event loop is running in the current thread."""
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:  # No event loop in this thread
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    @Slot(object)  # Takes QueryRequest object
    def call_chat_query(self, payload: QueryRequest):
        logger.debug(f"ApiWorker: Received call_chat_query for '{payload.query_text[:30]}...'")
        loop = self._ensure_event_loop()
        try:
            response = loop.run_until_complete(self.api_client.post_chat_query(payload))
            self.chat_response_received.emit(response)
        except ApiClientError as e:
            logger.error(f"ApiWorker: ApiClientError during chat query: {e.message}", exc_info=True)
            self.error_occurred.emit(
                e.error_response if hasattr(e, 'error_response') and e.error_response else str(e.message))
        except Exception as e:
            logger.error(f"ApiWorker: Unexpected error during chat query: {e}", exc_info=True)
            self.error_occurred.emit(f"Unexpected error in worker: {e}")

    @Slot(object)  # Takes IngestDirectoryRequest object
    def call_ingest_directory(self, payload: IngestDirectoryRequest):
        logger.debug(f"ApiWorker: Received call_ingest_directory for path '{payload.directory_path}'")
        loop = self._ensure_event_loop()
        try:
            response = loop.run_until_complete(self.api_client.post_ingest_directory(payload))
            self.ingest_response_received.emit(response)
        except ApiClientError as e:
            logger.error(f"ApiWorker: ApiClientError during ingest: {e.message}", exc_info=True)
            self.error_occurred.emit(
                e.error_response if hasattr(e, 'error_response') and e.error_response else str(e.message))
        except Exception as e:
            logger.error(f"ApiWorker: Unexpected error during ingest: {e}", exc_info=True)
            self.error_occurred.emit(f"Unexpected error in worker: {e}")

    def stop_loop_if_running(self):
        """Stops the event loop if it was created and is running."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            logger.info("ApiWorker: Requested asyncio event loop stop.")
        # It's tricky to fully manage and close the loop from here without more complex async thread management.
        # For this pattern, loop.run_until_complete() handles individual tasks.
        # If loop.run_forever() was used, stopping it and cleaning up is more involved.


# --- Main Window ---
class MainWindow(QMainWindow):
    # Define a signal that can be used to trigger the worker's methods from the main thread
    # This is an alternative to direct cross-thread method calls, often safer.
    # Parameter is the Pydantic model object.
    request_chat_query_signal = Signal(object)
    request_ingest_directory_signal = Signal(object)

    def __init__(self, settings_obj: Settings, api_client_obj: ApiClient, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.settings = settings_obj
        self.api_client = api_client_obj  # Keep a reference if needed, though worker uses it primarily
        self.logger = logging.getLogger(__name__ + ".MainWindow")  # Specific logger

        self.setWindowTitle("AvaChat - Advanced AI Assistant")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height

        self._init_ui()
        self._init_api_worker()

        self.logger.info("MainWindow initialized and UI setup complete.")
        self.update_status_bar("Welcome to AvaChat! Ready to assist you.")

    def _init_ui(self):
        # Central Widget and Layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Chat History Display (Read-only)
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Chat history will appear here...")
        main_layout.addWidget(self.chat_display, 1)  # Give more stretch factor

        # Input Area Layout (Horizontal)
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message to Ava...")
        self.input_field.returnPressed.connect(self._on_send_query)  # Send on Enter
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self._on_send_query)
        input_layout.addWidget(self.send_button)
        main_layout.addLayout(input_layout)

        # Ingest Button
        self.ingest_button = QPushButton("Ingest Documents from Directory")
        self.ingest_button.clicked.connect(self._on_ingest_directory)
        main_layout.addWidget(self.ingest_button)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.logger.debug("UI elements initialized.")

    def _init_api_worker(self):
        if not self.api_client:
            self.logger.error(
                "ApiClient is None. ApiWorker cannot be initialized. GUI will be non-functional for API calls.")
            self.update_status_bar("Error: API Client not available. Please restart.")
            # Disable buttons that need the worker
            self.send_button.setEnabled(False)
            self.ingest_button.setEnabled(False)
            return

        self.api_thread = QThread(self)  # Parent thread to main window for lifecycle management
        self.api_worker = ApiWorker(self.api_client)
        self.api_worker.moveToThread(self.api_thread)

        # Connect worker signals to MainWindow slots
        self.api_worker.chat_response_received.connect(self._handle_chat_response)
        self.api_worker.ingest_response_received.connect(self._handle_ingest_response)
        self.api_worker.error_occurred.connect(self._handle_api_error)

        # Connect MainWindow's request signals to ApiWorker's slots
        # This ensures the call to the worker's slot happens in the worker's thread.
        self.request_chat_query_signal.connect(self.api_worker.call_chat_query)
        self.request_ingest_directory_signal.connect(self.api_worker.call_ingest_directory)

        # Clean up thread when worker is destroyed (or when thread finishes)
        self.api_worker.destroyed.connect(self.api_thread.quit)  # If worker is deleted
        self.api_thread.finished.connect(self.api_thread.deleteLater)  # Schedule thread for deletion
        # self.api_thread.finished.connect(self.api_worker.deleteLater) # Also schedule worker for deletion

        self.api_thread.start()
        self.logger.info("ApiWorker thread started.")

    def _append_message_to_chat(self, role: str, text: str, sources: Optional[List[dict]] = None):
        """Appends a message to the chat display with basic HTML formatting."""
        color = "blue" if role == "user" else "green"
        if role == "error": color = "red"

        formatted_message = f"<div style='color:{color}; margin-bottom: 5px;'>"
        formatted_message += f"<b>{role.capitalize()}:</b> {text}"
        if sources:
            formatted_message += "<br><small><i>Sources:</i></small>"
            for src in sources:
                source_file = src.get('source_file', 'Unknown source')
                preview = src.get('chunk_text_preview', 'N/A')  # Or use 'text_preview' if that's the key
                # Make preview safe for HTML
                preview_escaped = preview.replace("<", "<").replace(">", ">")[:100] + "..."
                formatted_message += f"<br><small>  - {source_file} (<i>'{preview_escaped}'</i>)</small>"
        formatted_message += "</div>"

        self.chat_display.append(formatted_message)
        self.logger.debug(f"Appended '{role}' message to chat display: '{text[:50]}...'")

    def update_status_bar(self, message: str, timeout: int = 5000):
        """Displays a message on the status bar."""
        self.status_bar.showMessage(message, timeout)
        self.logger.info(f"Status bar updated: {message}")

    # --- Slots for UI Actions ---
    @Slot()
    def _on_send_query(self):
        query_text = self.input_field.text().strip()
        if not query_text:
            self.update_status_bar("Please type a message to send.", 3000)
            return

        self._append_message_to_chat("user", query_text)
        self.input_field.clear()

        # Disable input while waiting for response
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.update_status_bar("Sending query to Ava...")

        payload = QueryRequest(query_text=query_text, session_id="default_session")  # Add session_id if needed
        self.request_chat_query_signal.emit(payload)  # Emit signal to trigger worker
        self.logger.debug(f"User query sent: '{query_text}'")

    @Slot()
    def _on_ingest_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory Containing Documents for Ingestion")
        if dir_path:
            self.logger.info(f"Directory selected for ingestion: {dir_path}")
            self.update_status_bar(f"Starting ingestion for directory: {dir_path}...")
            self.ingest_button.setEnabled(False)  # Disable while ingesting

            payload = IngestDirectoryRequest(directory_path=str(dir_path))
            self.request_ingest_directory_signal.emit(payload)  # Emit signal
        else:
            self.update_status_bar("Ingestion cancelled by user.", 3000)

    # --- Slots for ApiWorker Signals ---
    @Slot(object)  # Expects QueryResponse
    def _handle_chat_response(self, response: QueryResponse):
        self.logger.info(f"Received chat response from ApiWorker: '{response.answer[:50]}...'")
        self._append_message_to_chat("assistant", response.answer, response.retrieved_chunks_details)
        self.update_status_bar("Ava responded.", 3000)
        # Re-enable input
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    @Slot(object)  # Expects IngestDirectoryResponse
    def _handle_ingest_response(self, response: IngestDirectoryResponse):
        self.logger.info(
            f"Received ingest response: Status '{response.status}', Docs: {response.documents_processed}, Chunks: {response.chunks_created}")
        msg = f"Ingestion successful: {response.documents_processed} documents processed, {response.chunks_created} chunks created."
        if response.status.lower() != "success":
            msg = f"Ingestion status: {response.status}. Docs: {response.documents_processed}, Chunks: {response.chunks_created}."

        self.update_status_bar(msg, 7000)
        QMessageBox.information(self, "Ingestion Complete", msg)
        self.ingest_button.setEnabled(True)  # Re-enable button

    @Slot(object)  # Expects ApiErrorResponse or str
    def _handle_api_error(self, error_data: Union[ApiErrorResponse, str]):
        error_message = "An API error occurred."
        if isinstance(error_data, ApiErrorResponse):
            error_message = f"API Error: {error_data.error.message}"
            if hasattr(error_data.error, 'code') and error_data.error.code:
                error_message = f"[{error_data.error.code}] {error_message}"
        elif isinstance(error_data, str):
            error_message = f"Error: {error_data}"

        self.logger.error(f"Handling API error: {error_message}")
        self._append_message_to_chat("error", error_message)
        self.update_status_bar(error_message, 7000)
        QMessageBox.warning(self, "API Error", error_message)

        # Re-enable input fields/buttons that might have been disabled
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.ingest_button.setEnabled(True)
        self.input_field.setFocus()

    def closeEvent(self, event):
        """Handle window close event to gracefully shut down the worker thread."""
        self.logger.info("Close event triggered for MainWindow. Shutting down ApiWorker thread...")
        if hasattr(self, 'api_worker') and self.api_worker:
            self.api_worker.stop_loop_if_running()  # Request asyncio loop stop if applicable

        if hasattr(self, 'api_thread') and self.api_thread and self.api_thread.isRunning():
            self.api_thread.quit()  # Tell the thread's event loop to exit
            if not self.api_thread.wait(5000):  # Wait up to 5 seconds for the thread to finish
                self.logger.warning("ApiWorker thread did not finish gracefully. Forcing termination (not ideal).")
                self.api_thread.terminate()  # Force terminate if it doesn't stop
                self.api_thread.wait()  # Wait again after terminate
            else:
                self.logger.info("ApiWorker thread finished gracefully.")

        # Close the ApiClient's httpx session if it's managed here (or by ApiWorker)
        if self.api_client and hasattr(self.api_client, 'close'):
            self.logger.info("Closing ApiClient session...")
            # ApiClient.close is async, need to run it.
            # This is tricky in closeEvent which is sync.
            # Best if ApiClient is closed by ApiWorker on thread finish, or main.py after app.exec().
            # For now, let's assume it's handled elsewhere or this is a best-effort.
            # asyncio.run(self.api_client.close()) # This would block closeEvent, not ideal.
            # A better pattern: ApiWorker's destructor or a slot connected to api_thread.finished
            # could close the api_client.
            # Or, ensure api_client.close() is called after app.exec() finishes in main.py

        self.logger.info("MainWindow closed.")
        super().closeEvent(event)


if __name__ == '__main__':
    # This basic test won't have a real backend, so API calls will likely fail or use dummies.
    # It's mainly to check if the GUI loads without immediate errors.

    # --- Dummy settings and api_client for standalone test ---
    class DummySettingsMain:
        API_SERVER_HOST = "localhost"
        API_SERVER_PORT = 80000  # Use a port unlikely to be in use for dummy test
        LOG_LEVEL = "DEBUG"


    class DummyApiClientMain:
        def __init__(self, settings_obj): self.settings = settings_obj

        async def post_chat_query(self, payload: QueryRequest) -> QueryResponse:
            await asyncio.sleep(1)  # Simulate network delay
            if "error" in payload.query_text.lower():
                raise ApiClientError("Simulated API error during chat.", error_response=ApiErrorResponse(
                    error=type('err', (), {'message': "Simulated detailed error"})()))
            return QueryResponse(answer=f"Dummy response to: {payload.query_text}", retrieved_chunks_details=[])

        async def post_ingest_directory(self, payload: IngestDirectoryRequest) -> IngestDirectoryResponse:
            await asyncio.sleep(2)
            return IngestDirectoryResponse(status="success", documents_processed=5, chunks_created=10)

        async def close(self): logger.info("DummyApiClientMain closed.")


    # Setup basic logging for the example run
    if not logging.getLogger().hasHandlers():  # Check if root logger has handlers
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Basic logging configured for MainWindow __main__ example.")

    app = QApplication(sys.argv)

    dummy_settings_instance = DummySettingsMain()
    dummy_api_client_instance = DummyApiClientMain(settings_obj=dummy_settings_instance)

    main_window = MainWindow(settings_obj=dummy_settings_instance, api_client_obj=dummy_api_client_instance)
    main_window.show()

    exit_code = app.exec()
    # Graceful shutdown of httpx client in ApiClient should happen via MainWindow.closeEvent typically
    # or after app.exec() if ApiClient is managed globally.
    # Here, we are relying on the test DummyApiClientMain to not need complex async shutdown
    # if the MainWindow's closeEvent doesn't fully handle async client closing correctly.
    sys.exit(exit_code)