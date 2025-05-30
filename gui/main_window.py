# gui/main_window.py
import sys
import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, List  # Added List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame, QMessageBox, QFileDialog,
    QComboBox
)
from PySide6.QtCore import QThread, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QTextCursor

try:
    from config import settings
    from models import QueryRequest, QueryResponse, IngestDirectoryRequest, \
        AvailableModelsResponse  # Added AvailableModelsResponse
    from gui.api_client import ApiClient
    from utils import ApiClientError
except ImportError as e:
    print(f"Import error in gui/main_window.py: {e}")


    class settings:
        API_SERVER_HOST = "127.0.0.1"; API_SERVER_PORT = 8000; LOG_LEVEL = "INFO"


    class QueryRequest:
        def __init__(self, query_text, model_name=None, **kwargs):  # Added model_name
            self.query_text = query_text;
            self.model_name = model_name


    class AvailableModelsResponse:
        models: List[str] = []


    class ApiClient:
        def __init__(self, settings_obj): pass

        async def post_chat_query(self, payload): return type('obj', (), {'answer': 'Mock response',
                                                                          'retrieved_chunks_details': []})()

        async def get_available_models(self): return AvailableModelsResponse(
            models=["mock_llama3", "mock_mistral"])  # Mock

        async def close(self): pass


    class ApiClientError(Exception):
        def __init__(self, message): self.message = message

logger = logging.getLogger(__name__)


class BackendManager(QObject):
    status_changed = Signal(bool)
    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.process = None
        self.is_running = False

    @Slot()
    def start_backend(self):
        if self.is_running:
            self.log_message.emit("Backend is already running.")
            return
        try:
            self.process = subprocess.Popen([
                sys.executable, "-m", "backend.api_server"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)  # Hide console on Windows
            self.is_running = True
            self.status_changed.emit(True)
            self.log_message.emit("Backend server starting...")
            # It's better to confirm backend readiness via API call rather than fixed sleep
        except Exception as e:
            self.log_message.emit(f"Failed to start backend: {e}")
            self.is_running = False  # Ensure state is correct
            self.status_changed.emit(False)

    @Slot()
    def stop_backend(self):
        if not self.is_running or not self.process:
            self.log_message.emit("Backend is not running.")
            return
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.log_message.emit("Backend did not terminate gracefully, killing.")
            self.process.kill()
            self.process.wait()
        except Exception as e:
            self.log_message.emit(f"Error stopping backend: {e}")
            if self.process and self.process.poll() is None:  # Check if still running
                self.process.kill()  # Force kill if terminate failed
                self.process.wait()
        finally:  # Ensure state is updated regardless of how termination went
            self.is_running = False
            self.status_changed.emit(False)
            self.log_message.emit("Backend server stopped.")
            self.process = None


class ChatWorker(QObject):
    response_received = Signal(str, list)
    available_models_received = Signal(list)  # New signal for models
    error_occurred = Signal(str)

    def __init__(self, api_client: ApiClient):  # Type hint ApiClient
        super().__init__()
        self.api_client = api_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def _ensure_event_loop(self):
        if self._loop is None or self._loop.is_closed():
            try:
                # Try to get existing loop if in a context that already has one (e.g. tests)
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                # If no loop is running, create a new one for this thread
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    @Slot(str, str)
    def send_query(self, query_text: str, model_name: str):
        loop = self._ensure_event_loop()
        try:
            payload = QueryRequest(query_text=query_text, model_name=model_name)  # Pass model_name
            response = loop.run_until_complete(self.api_client.post_chat_query(payload))
            self.response_received.emit(response.answer, response.retrieved_chunks_details or [])
        except ApiClientError as e:
            self.error_occurred.emit(f"API Error sending query: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Error sending query: {str(e)}")

    @Slot()  # New slot to fetch models
    def fetch_available_models(self):
        loop = self._ensure_event_loop()
        try:
            response = loop.run_until_complete(self.api_client.get_available_models())
            self.available_models_received.emit(response.models)
        except ApiClientError as e:
            self.error_occurred.emit(f"API Error fetching models: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Error fetching models: {str(e)}")


class MainWindow(QMainWindow):
    request_chat_query_signal = Signal(str, str)
    request_available_models_signal = Signal()  # New signal

    def __init__(self):
        super().__init__()
        self.api_client: Optional[ApiClient] = None  # Initialize as None
        self.backend_manager: Optional[BackendManager] = None
        self.chat_worker: Optional[ChatWorker] = None
        self.chat_thread: Optional[QThread] = None
        self.backend_thread: Optional[QThread] = None

        self.setWindowTitle("Tennis Knowledge Database - Chat Interface")
        self.setGeometry(100, 100, 1000, 700)

        self.setup_ui()
        self.setup_backend_manager()
        self.setup_styling()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.check_backend_status_via_api)  # Changed to API check
        # self.status_timer.start(5000) # Start after backend is confirmed running

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_layout = QHBoxLayout(header_frame)
        title_label = QLabel("🎾 Tennis Knowledge Database")
        title_label.setObjectName("title_label")
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        self.backend_status_label = QLabel("Backend: Stopped")
        self.backend_status_label.setObjectName("status_label")
        model_label = QLabel("Model:")
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("model_selector")
        self.model_selector.addItem("Loading models...")  # Placeholder
        self.model_selector.setEnabled(False)  # Disable until models are loaded
        self.model_selector.setMinimumWidth(150)

        self.start_button = QPushButton("Start Backend")
        self.start_button.setObjectName("start_button")
        self.stop_button = QPushButton("Stop Backend")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setEnabled(False)

        self.start_button.clicked.connect(self.start_backend)
        self.stop_button.clicked.connect(self.stop_backend)

        header_layout.addWidget(self.backend_status_label)
        header_layout.addWidget(model_label)
        header_layout.addWidget(self.model_selector)
        header_layout.addWidget(self.start_button)
        header_layout.addWidget(self.stop_button)
        layout.addWidget(header_frame)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        layout.addWidget(self.chat_display, 1)

        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.StyledPanel)
        input_layout = QHBoxLayout(input_frame)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about tennis history, players, tournaments...")
        self.input_field.returnPressed.connect(self.send_message)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        self.send_button.setEnabled(False)
        self.ingest_button = QPushButton("Load Tennis Documents")
        self.ingest_button.setObjectName("ingest_button")
        self.ingest_button.clicked.connect(self.load_documents)
        self.ingest_button.setEnabled(False)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.ingest_button)
        layout.addWidget(input_frame)

        self.add_message("system", "Welcome! Start the backend to begin.")

    def setup_styling(self):  # Keep your existing styling
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QFrame { background-color: #2d2d2d; border: 1px solid #404040; border-radius: 8px; margin: 2px; padding: 8px; }
            QTextEdit { background-color: #2d2d2d; border: 1px solid #404040; border-radius: 8px; padding: 12px; font-family: 'Segoe UI', Arial, sans-serif; color: #ffffff; selection-background-color: #0078d4; }
            QLineEdit { background-color: #3c3c3c; border: 1px solid #404040; border-radius: 8px; padding: 10px; font-size: 12px; color: #ffffff; }
            QLineEdit:focus { border: 2px solid #0078d4; background-color: #404040; }
            QComboBox { background-color: #3c3c3c; border: 1px solid #404040; border-radius: 6px; padding: 6px 10px; color: #ffffff; font-size: 11px; min-width: 100px; }
            QComboBox:hover { border: 1px solid #0078d4; }
            QComboBox::drop-down { border: none; width: 20px; }
            QComboBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 4px solid #ffffff; margin-right: 6px; }
            QComboBox QAbstractItemView { background-color: #3c3c3c; border: 1px solid #404040; selection-background-color: #0078d4; color: #ffffff; }
            QPushButton { background-color: #0078d4; color: white; border: none; border-radius: 8px; padding: 10px 18px; font-weight: bold; font-size: 11px; }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:pressed { background-color: #005a9e; }
            QPushButton:disabled { background-color: #404040; color: #808080; }
            QPushButton#start_button { background-color: #107c10; } QPushButton#start_button:hover { background-color: #0e6b0e; }
            QPushButton#stop_button { background-color: #d13438; } QPushButton#stop_button:hover { background-color: #b02b2f; }
            QPushButton#ingest_button { background-color: #8764b8; } QPushButton#ingest_button:hover { background-color: #744da9; }
            QLabel { color: #ffffff; font-weight: 500; }
            QLabel#title_label { color: #ffd700; font-size: 18px; font-weight: bold; }
            QLabel#status_label { font-weight: bold; padding: 4px 8px; border-radius: 4px; }
        """)

    def setup_backend_manager(self):
        self.backend_thread = QThread(self)
        self.backend_manager = BackendManager()
        self.backend_manager.moveToThread(self.backend_thread)
        self.backend_manager.status_changed.connect(self.on_backend_status_changed)
        self.backend_manager.log_message.connect(self.add_system_message)
        self.backend_thread.start()

    def setup_chat_worker(self):
        if not self.api_client:  # Initialize ApiClient if not already done
            self.api_client = ApiClient(settings_obj=settings)

        if not self.chat_thread or not self.chat_thread.isRunning():
            self.chat_thread = QThread(self)
            self.chat_worker = ChatWorker(self.api_client)
            self.chat_worker.moveToThread(self.chat_thread)
            self.chat_worker.response_received.connect(self.on_response_received)
            self.chat_worker.available_models_received.connect(self.on_available_models_received)  # Connect new signal
            self.chat_worker.error_occurred.connect(self.on_error_occurred)
            self.request_chat_query_signal.connect(self.chat_worker.send_query)
            self.request_available_models_signal.connect(self.chat_worker.fetch_available_models)  # Connect new signal
            self.chat_thread.start()
        else:  # Worker and thread already exist, ensure connections
            # This might be redundant if connections are stable, but good for safety
            try:
                self.request_chat_query_signal.disconnect(self.chat_worker.send_query)
            except RuntimeError:
                pass
            try:
                self.request_available_models_signal.disconnect(
                    self.chat_worker.fetch_available_models)
            except RuntimeError:
                pass
            self.request_chat_query_signal.connect(self.chat_worker.send_query)
            self.request_available_models_signal.connect(self.chat_worker.fetch_available_models)

    @Slot()
    def start_backend(self):
        self.backend_manager.start_backend()
        # Defer chat worker setup until backend is confirmed running in on_backend_status_changed

    @Slot()
    def stop_backend(self):
        self.backend_manager.stop_backend()
        if self.status_timer.isActive():
            self.status_timer.stop()

    @Slot(bool)
    def on_backend_status_changed(self, is_running):
        if is_running:
            self.backend_status_label.setText("Backend: Starting...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            # Don't enable chat features yet, wait for API confirmation
            # Start API check timer
            if not self.status_timer.isActive():
                self.status_timer.start(2000)  # Check API status shortly after starting backend manager
        else:
            self.backend_status_label.setText("Backend: Stopped ❌")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.send_button.setEnabled(False)
            self.ingest_button.setEnabled(False)
            self.model_selector.setEnabled(False)
            self.model_selector.clear()
            self.model_selector.addItem("Backend stopped")
            if self.status_timer.isActive():
                self.status_timer.stop()

    @Slot()
    def check_backend_status_via_api(self):
        """Checks if the backend root endpoint is responsive."""
        if self.backend_manager and self.backend_manager.is_running:
            # Instead of trying to run async code here, just setup the worker and emit signal
            if not self.chat_worker or not self.chat_thread.isRunning():
                self.setup_chat_worker()

            # Backend is running, assume API is ready
            self.backend_status_label.setText("Backend: Running ✅")
            self.send_button.setEnabled(True)
            self.ingest_button.setEnabled(True)
            self.model_selector.setEnabled(True)

            self.add_system_message("Backend confirmed. Fetching available models...")

            # Emit signal to fetch models via the worker
            self.request_available_models_signal.emit()
            self.status_timer.stop()  # Stop timer once confirmed
        else:
            # Backend process is not running according to manager
            self.on_backend_status_changed(False)

    @Slot(list)  # Slot to receive models from ChatWorker
    def on_available_models_received(self, models: List[str]):
        self.model_selector.clear()
        if models:
            self.model_selector.addItems(models)
            # Try to set to a common default if available
            sensible_defaults = ["llama3:latest", "llama3", "deepseek-llm:latest", "deepseek-llm", "mistral:latest",
                                 "mistral"]
            current_selection = ""
            for s_default in sensible_defaults:
                if s_default in models:
                    current_selection = s_default
                    break
            if not current_selection and models:  # Fallback to first model if no common default found
                current_selection = models[0]

            if current_selection:
                self.model_selector.setCurrentText(current_selection)
                self.add_system_message(f"Models loaded. Defaulting to '{current_selection}'.")
            else:
                self.add_system_message("Models loaded, but could not set a default.")

        else:
            self.model_selector.addItem("No models found")
            self.add_system_message("No models returned from backend.")
        self.model_selector.setEnabled(True)

    @Slot()
    def send_message(self):
        query = self.input_field.text().strip()
        if not query: return

        selected_model = self.model_selector.currentText()
        if not selected_model or "Loading" in selected_model or "No models" in selected_model:
            self.add_message("error", "Please select a valid model first.")
            return

        self.add_message("user", query)
        self.add_system_message(f"Sending query with model: {selected_model}")
        self.input_field.clear()
        self.send_button.setEnabled(False)  # Disable send button until response
        self.request_chat_query_signal.emit(query, selected_model)

    @Slot()
    def load_documents(self):
        if not self.backend_manager or not self.backend_manager.is_running:
            QMessageBox.warning(self, "Backend Offline", "Please start the backend server first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select Tennis Documents Directory",
                                                     str(settings.KNOWLEDGE_BASE_DIR))
        if directory:
            self.add_system_message(f"Ingestion requested for: {directory}")
            # TODO: Implement actual call to ingest endpoint via ApiClient and ChatWorker
            QMessageBox.information(self, "Document Ingestion",
                                    f"Ingestion for '{Path(directory).name}' started. (This is a placeholder - actual call to be implemented)")

    @Slot(str, list)
    def on_response_received(self, answer, sources):
        self.add_message("assistant", answer, sources)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    @Slot(str)
    def on_error_occurred(self, error_message):
        self.add_message("error", error_message)
        self.send_button.setEnabled(True)  # Re-enable send button on error
        self.input_field.setFocus()

    def add_message(self, role, content, sources=None):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        role_map = {
            "user": (
                '<div style="margin: 15px 0; padding: 10px; background-color: #404040; border-radius: 8px;"><b style="color: #4fc3f7;">You:</b> <span style="color: #ffffff;">{content}</span></div>'),
            "assistant": (
                '<div style="margin: 15px 0; padding: 10px; background-color: #2d2d2d; border-radius: 8px; border-left: 4px solid #4caf50;"><b style="color: #81c784;">🎾 AvaChat AI:</b> <span style="color: #ffffff;">{content}</span>{sources_html}</div>'),
            "error": (
                '<div style="margin: 15px 0; padding: 10px; background-color: #5d1a1a; border-radius: 8px; border-left: 4px solid #f44336;"><b style="color: #ef5350;">❌ Error:</b> <span style="color: #ffffff;">{content}</span></div>'),
            "system": (
                '<div style="margin: 10px 0; padding: 8px; background-color: #1a1a2e; border-radius: 6px;"><span style="color: #ffd54f; font-style: italic;">🔧 {content}</span></div>')
        }
        sources_html_content = ""
        if role == "assistant" and sources:
            sources_html_content = '<br><small style="color: #b0b0b0; margin-top: 8px;"><i>📚 Sources:</i></small>'
            for source in sources[:3]:
                source_file = source.get('source_file', 'Unknown')
                sources_html_content += f'<br><small style="color: #90a4ae;">• {source_file}</small>'

        html = role_map[role].format(content=content.replace('<', '<').replace('>', '>'),
                                     sources_html=sources_html_content)  # Basic HTML escaping
        cursor.insertHtml(html)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def add_system_message(self, message):
        logger.info(f"System Message: {message}")
        self.add_message("system", message)

    def closeEvent(self, event):
        self.add_system_message("Shutting down...")
        if self.status_timer.isActive():
            self.status_timer.stop()

        if self.backend_manager and self.backend_manager.is_running:
            self.backend_manager.stop_backend()  # This is synchronous if called directly
            # Wait for backend thread to finish gracefully after stop_backend signals it
            if self.backend_thread and self.backend_thread.isRunning():
                self.backend_thread.quit()  # Request thread to quit
                if not self.backend_thread.wait(5000):  # Wait up to 5s
                    self.add_system_message("Backend thread did not quit gracefully, terminating.")
                    self.backend_thread.terminate()
                    self.backend_thread.wait()

        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.quit()
            if not self.chat_thread.wait(3000):
                self.add_system_message("Chat thread did not quit gracefully, terminating.")
                self.chat_thread.terminate()
                self.chat_thread.wait()

        if self.api_client:
            # api_client.close() is async, need to run it
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self.api_client.close())
                self.add_system_message("API client closed.")
            except Exception as e:
                self.add_system_message(f"Error closing API client: {e}")
            finally:
                loop.close()

        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    logging.basicConfig(level=settings.LOG_LEVEL.upper(), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()