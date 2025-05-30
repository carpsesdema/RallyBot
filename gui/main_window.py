import sys
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional, List

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame, QMessageBox, QFileDialog
)
from PySide6.QtCore import QThread, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QTextCursor

try:
    from config import settings
    from models import QueryRequest, QueryResponse, IngestDirectoryRequest, \
        AvailableModelsResponse
    from gui.api_client import ApiClient
    from utils import ApiClientError
except ImportError as e:
    print(f"Import error in gui/main_window.py: {e}")


    class settings:
        API_SERVER_HOST = "127.0.0.1"
        API_SERVER_PORT = 8000
        LOG_LEVEL = "INFO"
        LLM_PROVIDER = "gemini"
        KNOWLEDGE_BASE_DIR = Path("./kb_docs_dummy")
        GEMINI_MODEL = "gemini-dummy-gui"


    class QueryRequest:
        def __init__(self, query_text, model_name=None, **kwargs):
            self.query_text = query_text
            self.model_name = model_name


    class AvailableModelsResponse:
        models: List[str] = [settings.GEMINI_MODEL]


    class ApiClientError(Exception):
        def __init__(self, message, status_code=None, error_response=None):
            super().__init__(message)
            self.message = message
            self.status_code = status_code
            self.error_response = error_response


    class ApiClient:
        def __init__(self, settings_obj): pass

        async def post_chat_query(self, payload): return type('obj', (), {'answer': 'Mock Gemini response',
                                                                          'retrieved_chunks_details': []})()

        async def get_available_models(self): return AvailableModelsResponse(models=[settings.GEMINI_MODEL])

        async def close(self): pass

logger = logging.getLogger(__name__)


class BackendManager(QObject):
    status_changed = Signal(bool)
    log_message = Signal(str)  # This signal will now primarily be for console logging via add_system_message

    def __init__(self):
        super().__init__()
        self.process = None
        self.is_running = False
        self.stdout_log_file = None
        self.stderr_log_file = None
        self.f_out = None
        self.f_err = None

    @Slot()
    def start_backend(self):
        if self.is_running: self.log_message.emit("Backend is already running."); return
        try:
            cmd = [sys.executable, "-u", "-m", "backend.api_server"]
            # Log to console via the signal/slot connected to add_system_message
            self.log_message.emit(f"Attempting to start backend with: {' '.join(cmd)}")
            project_root_dir = Path(__file__).parent.parent
            self.stdout_log_file = project_root_dir / "backend_stdout.log"
            self.stderr_log_file = project_root_dir / "backend_stderr.log"
            self.log_message.emit(f"Backend stdout will be logged to: {self.stdout_log_file}")
            self.log_message.emit(f"Backend stderr will be logged to: {self.stderr_log_file}")
            self.f_out = open(self.stdout_log_file, 'w')
            self.f_err = open(self.stderr_log_file, 'w')
            self.process = subprocess.Popen(cmd, stdout=self.f_out, stderr=self.f_err, text=True, bufsize=1,
                                            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                                            cwd=str(project_root_dir))
            self.is_running = True
            self.status_changed.emit(True)
            self.log_message.emit("Backend server process starting...")
        except Exception as e:
            self.log_message.emit(f"Failed to start backend: {e}")
            self.is_running = False
            self.status_changed.emit(False)
            if self.f_out: self.f_out.close()
            if self.f_err: self.f_err.close()

    @Slot()
    def stop_backend(self):
        if not self.is_running or not self.process:
            self.log_message.emit("Backend is not running.")
            if self.f_out: self.f_out.close();
            if self.f_err: self.f_err.close()
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
            if self.process and self.process.poll() is None: self.process.kill(); self.process.wait()
        finally:
            self.is_running = False
            self.status_changed.emit(False)
            self.log_message.emit("Backend server stopped.")
            self.process = None
            if self.f_out: self.f_out.close()
            if self.f_err: self.f_err.close()
            self.f_out = None
            self.f_err = None


class ChatWorker(QObject):
    response_received = Signal(str, list)
    available_models_received = Signal(list)
    error_occurred = Signal(str)

    def __init__(self, api_client: ApiClient):
        super().__init__()
        self.api_client = api_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        logger.info("ChatWorker initialized.")

    def _ensure_event_loop(self):
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop(); asyncio.set_event_loop(self._loop)
        return self._loop

    @Slot(str, str)
    def send_query(self, query_text: str, model_name: str):
        logger.info(f"ChatWorker: send_query slot called with query: '{query_text[:50]}...', model: {model_name}")
        loop = self._ensure_event_loop()
        try:
            payload = QueryRequest(query_text=query_text, model_name=model_name)
            logger.debug(
                f"ChatWorker: Calling api_client.post_chat_query with payload: {payload.model_dump() if hasattr(payload, 'model_dump') else payload.__dict__}")
            response = loop.run_until_complete(self.api_client.post_chat_query(payload))
            logger.info(
                f"ChatWorker: Received response from api_client: Answer length {len(response.answer if hasattr(response, 'answer') else '')}")
            self.response_received.emit(response.answer, response.retrieved_chunks_details or [])
        except ApiClientError as e:
            logger.error(f"ChatWorker: ApiClientError in send_query: {e}", exc_info=True)
            self.error_occurred.emit(f"API Error: {str(e.message if hasattr(e, 'message') else e)}")
        except Exception as e:
            logger.error(f"ChatWorker: Exception in send_query: {e}", exc_info=True)
            self.error_occurred.emit(f"Error: {str(e)}")

    @Slot()
    def fetch_available_models(self):
        logger.info("ChatWorker: fetch_available_models slot called.")
        loop = self._ensure_event_loop()
        try:
            response = loop.run_until_complete(self.api_client.get_available_models())
            logger.info(
                f"ChatWorker: Received models from api_client: {response.models if hasattr(response, 'models') else 'N/A'}")
            self.available_models_received.emit(response.models)
        except ApiClientError as e:
            logger.error(f"ChatWorker: ApiClientError in fetch_available_models: {e}", exc_info=True)
            detailed_error = str(e.message if hasattr(e, 'message') else e)
            if hasattr(e, 'status_code') and e.status_code: detailed_error += f" (Status: {e.status_code})"
            if hasattr(e, 'error_response') and e.error_response and hasattr(e.error_response, 'error') and hasattr(
                    e.error_response.error, 'message'):
                detailed_error += f" Detail: {e.error_response.error.message}"
            self.error_occurred.emit(f"API Error fetching models: {detailed_error}")
        except Exception as e:
            logger.error(f"ChatWorker: Exception in fetch_available_models: {e}", exc_info=True)
            self.error_occurred.emit(f"Error fetching models: {str(e)}")


class MainWindow(QMainWindow):
    request_chat_query_signal = Signal(str, str)
    request_available_models_signal = Signal()

    def __init__(self):
        super().__init__()
        self.api_client: Optional[ApiClient] = None
        self.backend_manager: Optional[BackendManager] = None
        self.chat_worker: Optional[ChatWorker] = None
        self.chat_thread: Optional[QThread] = None
        self.current_llm_model_name = settings.GEMINI_MODEL.split('/')[-1]

        self.setWindowTitle("Tennis Knowledge Database - Chat Interface (Gemini Hardcoded)")
        self.setGeometry(100, 100, 1000, 700)

        self.setup_ui()
        self.setup_backend_manager()
        self.setup_styling()

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.check_backend_status_via_api)
        logger.info("MainWindow initialized.")

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
        self.model_display_label = QLabel(f"Model: {self.current_llm_model_name} (Hardcoded)")
        self.model_display_label.setObjectName("model_display_label")
        self.start_button = QPushButton("Start Backend")
        self.start_button.setObjectName("start_button")
        self.stop_button = QPushButton("Stop Backend")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_backend)
        self.stop_button.clicked.connect(self.stop_backend)
        header_layout.addWidget(self.backend_status_label)
        header_layout.addWidget(self.model_display_label)
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
        # Initial welcome message still goes to GUI, but others can be console-only.
        self.add_message("system", "Welcome! System is hardcoded for Gemini. Start the backend to begin.")
        logger.info("UI setup complete.")

    def setup_styling(self):
        # System message color is now irrelevant for the chat display styling if not shown there.
        # The color #ffd54f was for the previous yellow system text.
        # If you want a general system message style for what IS shown (like initial welcome), keep it or adjust.
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #ffffff; }
            QFrame { background-color: #2d2d2d; border: 1px solid #404040; border-radius: 8px; margin: 2px; padding: 8px; }
            QTextEdit { background-color: #2d2d2d; border: 1px solid #404040; border-radius: 8px; padding: 12px; font-family: 'Segoe UI', Arial, sans-serif; color: #ffffff; selection-background-color: #0078d4; }
            QLineEdit { background-color: #3c3c3c; border: 1px solid #404040; border-radius: 8px; padding: 10px; font-size: 12px; color: #ffffff; }
            QLineEdit:focus { border: 2px solid #0078d4; background-color: #404040; }
            QPushButton { background-color: #0078d4; color: white; border: none; border-radius: 8px; padding: 10px 18px; font-weight: bold; font-size: 11px; }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:pressed { background-color: #005a9e; }
            QPushButton:disabled { background-color: #404040; color: #808080; }
            QPushButton#start_button { background-color: #107c10; } QPushButton#start_button:hover { background-color: #0e6b0e; }
            QPushButton#stop_button { background-color: #d13438; } QPushButton#stop_button:hover { background-color: #b02b2f; }
            QPushButton#ingest_button { background-color: #8764b8; } QPushButton#ingest_button:hover { background-color: #744da9; }
            QLabel { color: #ffffff; font-weight: 500; }
            QLabel#title_label { color: #ffd700; font-size: 18px; font-weight: bold; } /* Gold for title */
            QLabel#status_label { font-weight: bold; padding: 4px 8px; border-radius: 4px; }
            QLabel#model_display_label { font-weight: normal; padding: 4px 8px; color: #cccccc; }
            /* Style for the initial system message in chat if desired */
            QTextEdit div[style*="background-color: #1a1a2e;"] span { color: #a9d1ff !important; } /* Light blue for system messages */
        """)
        logger.info("Styling applied.")

    def setup_backend_manager(self):
        self.backend_thread = QThread(self)
        self.backend_manager = BackendManager()
        self.backend_manager.moveToThread(self.backend_thread)
        self.backend_manager.status_changed.connect(self.on_backend_status_changed)
        # This connection now primarily drives console logging for backend manager messages
        self.backend_manager.log_message.connect(self.add_system_message)
        self.backend_thread.start()
        logger.info("BackendManager setup and thread started.")

    def setup_chat_worker(self):
        if self.chat_thread and self.chat_thread.isRunning():
            logger.info("ChatWorker and thread already running. No re-setup needed.")
            return
        if not self.api_client:
            self.api_client = ApiClient(settings_obj=settings)
            logger.info("ApiClient initialized in setup_chat_worker.")
        self.chat_thread = QThread(self)
        self.chat_worker = ChatWorker(self.api_client)
        self.chat_worker.moveToThread(self.chat_thread)
        self.chat_worker.response_received.connect(self.on_response_received)
        self.chat_worker.available_models_received.connect(self.on_available_models_received)
        self.chat_worker.error_occurred.connect(self.on_error_occurred)
        self.request_chat_query_signal.connect(self.chat_worker.send_query)
        self.request_available_models_signal.connect(self.chat_worker.fetch_available_models)
        self.chat_thread.start()
        logger.info("ChatWorker setup complete, thread started, and signals connected.")

    @Slot()
    def start_backend(self):
        logger.info("Start Backend button clicked.")
        self.backend_manager.start_backend()

    @Slot()
    def stop_backend(self):
        logger.info("Stop Backend button clicked.")
        self.backend_manager.stop_backend()
        if self.status_timer.isActive():
            self.status_timer.stop()

    @Slot(bool)
    def on_backend_status_changed(self, is_running):
        logger.info(f"Backend status changed. Is running: {is_running}")
        if is_running:
            self.backend_status_label.setText("Backend: Starting...")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            if not self.status_timer.isActive():
                self.status_timer.start(3000)
        else:
            self.backend_status_label.setText("Backend: Stopped ❌")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.send_button.setEnabled(False)
            self.ingest_button.setEnabled(False)
            if self.status_timer.isActive():
                self.status_timer.stop()
            if self.chat_thread and self.chat_thread.isRunning():
                self.chat_thread.quit()
                self.chat_thread.wait(1000)
                logger.info("Chat thread stopped due to backend stop.")
            self.chat_thread = None
            self.chat_worker = None

    @Slot()
    def check_backend_status_via_api(self):
        logger.info("Checking backend status via API timer triggered.")
        self.setup_chat_worker()
        self.backend_status_label.setText("Backend: Running ✅")
        self.send_button.setEnabled(True)
        self.ingest_button.setEnabled(True)
        self.add_system_message(f"Backend confirmed. Using hardcoded Gemini model: {self.current_llm_model_name}")
        self.request_available_models_signal.emit()
        self.status_timer.stop()

    @Slot(list)
    def on_available_models_received(self, models: List[str]):
        logger.info(f"GUI on_available_models_received: {models}")
        if models and models[0] == self.current_llm_model_name:
            self.add_system_message(f"Model confirmation: Backend reports using '{models[0]}'.")  # Console Log
            self.model_display_label.setText(f"Model: {models[0]} (Hardcoded)")
        elif models:
            self.add_system_message(
                f"Warning: Backend model list {models} differs from expected '{self.current_llm_model_name}'.")  # Console Log
            self.model_display_label.setText(f"Model: {models[0]} (From Backend)")
        else:
            self.add_system_message("Warning: No models list received from backend.")  # Console Log
            self.model_display_label.setText(f"Model: {self.current_llm_model_name} (UI Default)")

    @Slot()
    def send_message(self):
        query = self.input_field.text().strip()
        if not query:
            logger.info("Send message: Empty query, doing nothing.")
            return
        self.add_message("user", query)  # Add user message to GUI
        self.add_system_message(
            f"Sending query (backend will use Gemini: {self.current_llm_model_name})")  # Console Log
        self.input_field.clear()
        self.send_button.setEnabled(False)
        self.request_chat_query_signal.emit(query, self.current_llm_model_name)

    @Slot()
    def load_documents(self):
        if not self.backend_manager or not self.backend_manager.is_running:
            QMessageBox.warning(self, "Backend Offline", "Please start the backend server first.")
            return
        start_dir = str(settings.KNOWLEDGE_BASE_DIR if settings.KNOWLEDGE_BASE_DIR.exists() else Path("."))
        directory = QFileDialog.getExistingDirectory(self, "Select Tennis Documents Directory", start_dir)
        if directory:
            self.add_system_message(f"Ingestion requested for: {directory}")  # Console Log
            QMessageBox.information(self, "Document Ingestion",
                                    f"Ingestion for '{Path(directory).name}' started. (Placeholder)")
            self.add_system_message(
                f"Note: Full ingestion functionality requires connecting this button to an API call.")  # Console Log

    @Slot(str, list)
    def on_response_received(self, answer, sources):
        logger.info(f"GUI on_response_received. Answer length: {len(answer)}")
        self.add_message("assistant", answer, sources)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    @Slot(str)
    def on_error_occurred(self, error_message):
        logger.error(f"GUI on_error_occurred: {error_message}")
        # Displaying errors in the chat window is still useful
        self.add_message("error", error_message)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    def add_message(self, role, content, sources=None):
        # This method now only adds "user", "assistant", and "error" messages to the GUI.
        # And the initial "system" welcome message.
        if role not in ["user", "assistant", "error", "system"]:  # Filter out other system messages from GUI
            logger.warning(f"add_message called with unexpected role '{role}'. Message: {content}")
            return

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
                '<div style="margin: 10px 0; padding: 8px; background-color: #1a1a2e; border-radius: 6px;"><span style="color: #a9d1ff !important; font-style: italic;">🔧 {content}</span></div>')
            # Light blue for initial welcome
        }
        sources_html_content = ""
        if role == "assistant" and sources:
            sources_html_content = '<br><small style="color: #b0b0b0; margin-top: 8px;"><i>📚 Sources:</i></small>'
            for source_item in sources[:3]:
                if isinstance(source_item, dict):
                    source_file = source_item.get('source_file', 'Unknown')
                    sources_html_content += f'<br><small style="color: #90a4ae;">• {source_file}</small>'
                else:
                    logger.warning(f"Unexpected source format: {source_item}")
        escaped_content = content.replace('&', '&').replace('<', '<').replace('>', '>')
        html_template = role_map.get(role)
        if html_template:
            html = html_template.format(content=escaped_content, sources_html=sources_html_content)
            cursor.insertHtml(html)
            self.chat_display.setTextCursor(cursor)
            self.chat_display.ensureCursorVisible()
        else:  # Should not happen with the role check above
            logger.error(f"Attempted to add_message with unknown role: {role}")

    def add_system_message(self, message: str):
        """
        Logs system messages to the console/logger.
        These messages will NOT appear in the chat GUI display, except for the initial welcome.
        """
        logger.info(f"System Info (Console Only): {message}")
        # The initial "Welcome!" message is added directly via self.add_message in setup_ui
        # All other calls to add_system_message (e.g., from BackendManager, status checks)
        # will now only log to the console.

    def closeEvent(self, event):
        logger.info("Close event triggered.")
        self.add_system_message("Shutting down...")  # This will now be a console log
        if self.status_timer.isActive(): self.status_timer.stop()
        if self.backend_manager:
            if self.backend_manager.is_running:
                self.backend_manager.stop_backend()
            else:
                if self.backend_manager.f_out: self.backend_manager.f_out.close()
                if self.backend_manager.f_err: self.backend_manager.f_err.close()
        if self.backend_thread and self.backend_thread.isRunning():
            self.backend_thread.quit()
            if not self.backend_thread.wait(5000):
                logger.warning("Backend thread did not quit gracefully, terminating.")
                self.backend_thread.terminate()
                self.backend_thread.wait()
            else:
                logger.info("Backend thread quit gracefully.")
        self.backend_thread = None
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.quit()
            if not self.chat_thread.wait(3000):
                logger.warning("Chat thread did not quit gracefully, terminating.")
                self.chat_thread.terminate()
                self.chat_thread.wait()
            else:
                logger.info("Chat thread quit gracefully.")
        self.chat_thread = None
        self.chat_worker = None
        if self.api_client:
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
                if hasattr(self.api_client, 'close') and asyncio.iscoroutinefunction(self.api_client.close):
                    loop.run_until_complete(self.api_client.close())
                    self.add_system_message("API client closed.")  # Console log
                    logger.info("API client closed.")
                else:
                    logger.info("API client does not have an async close method or it's not a coroutine.")
            except Exception as e:
                self.add_system_message(f"Error closing API client: {e}")  # Console log
                logger.error(f"Error closing API client: {e}", exc_info=True)
            finally:
                if 'loop' in locals() and not loop.is_closed() and not asyncio.get_event_loop_policy().get_event_loop().is_running():
                    loop.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    # Configure logging - this will make logger.info() from MainWindow appear in console
    logging.basicConfig(level=settings.LOG_LEVEL.upper(),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)  # Ensure logs go to stdout for console visibility

    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path: sys.path.insert(0, str(project_root)); logger.info(
        f"Added project root: {project_root}")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__": main()