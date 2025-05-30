# gui/main_window.py
import sys
import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QLabel, QFrame, QMessageBox, QFileDialog,
    QComboBox
)
from PySide6.QtCore import QThread, QObject, Signal, Slot, QTimer
from PySide6.QtGui import QFont, QTextCursor

try:
    from config import settings
    from models import QueryRequest, QueryResponse, IngestDirectoryRequest
    from gui.api_client import ApiClient
    from utils import ApiClientError
except ImportError as e:
    print(f"Import error: {e}")


    # Minimal fallbacks for development
    class settings:
        API_SERVER_HOST = "127.0.0.1"
        API_SERVER_PORT = 8000
        LOG_LEVEL = "INFO"


    class QueryRequest:
        def __init__(self, query_text, **kwargs):
            self.query_text = query_text


    class ApiClient:
        def __init__(self, settings_obj): pass

        async def post_chat_query(self, payload): return type('obj', (), {'answer': 'Mock response',
                                                                          'retrieved_chunks_details': []})()

        async def close(self): pass


    class ApiClientError(Exception):
        def __init__(self, message): self.message = message

logger = logging.getLogger(__name__)


class BackendManager(QObject):
    """Manages the FastAPI backend server"""
    status_changed = Signal(bool)  # True if running, False if stopped
    log_message = Signal(str)

    def __init__(self):
        super().__init__()
        self.process = None
        self.is_running = False

    @Slot()
    def start_backend(self):
        if self.is_running:
            self.log_message.emit("Backend is already running")
            return

        try:
            # Start the FastAPI server
            self.process = subprocess.Popen([
                sys.executable, "-m", "backend.api_server"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            self.is_running = True
            self.status_changed.emit(True)
            self.log_message.emit("Backend server starting...")

            # Give it a moment to start
            time.sleep(2)

        except Exception as e:
            self.log_message.emit(f"Failed to start backend: {e}")
            self.status_changed.emit(False)

    @Slot()
    def stop_backend(self):
        if not self.is_running or not self.process:
            self.log_message.emit("Backend is not running")
            return

        try:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.is_running = False
            self.status_changed.emit(False)
            self.log_message.emit("Backend server stopped")
        except Exception as e:
            self.log_message.emit(f"Error stopping backend: {e}")
            if self.process:
                self.process.kill()
                self.is_running = False
                self.status_changed.emit(False)


class ChatWorker(QObject):
    """Handles chat requests in a separate thread"""
    response_received = Signal(str, list)  # answer, sources
    error_occurred = Signal(str)

    def __init__(self, api_client):
        super().__init__()
        self.api_client = api_client
        self._loop = None

    def _ensure_event_loop(self):
        if self._loop is None:
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    @Slot(str, str)  # query_text, model_name
    def send_query(self, query_text, model_name="llama3"):
        loop = self._ensure_event_loop()
        try:
            payload = QueryRequest(query_text=query_text, model_name=model_name)
            response = loop.run_until_complete(self.api_client.post_chat_query(payload))
            self.response_received.emit(response.answer, response.retrieved_chunks_details or [])
        except ApiClientError as e:
            self.error_occurred.emit(f"API Error: {str(e)}")
        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")


class MainWindow(QMainWindow):
    # Define signals for worker communication
    request_chat_query_signal = Signal(str, str)  # query_text, model_name

    def __init__(self):
        super().__init__()
        self.api_client = None
        self.backend_manager = None
        self.chat_worker = None
        self.chat_thread = None
        self.backend_thread = None

        self.setWindowTitle("Tennis Knowledge Database - Chat Interface")
        self.setGeometry(100, 100, 1000, 700)

        self.setup_ui()
        self.setup_backend_manager()
        self.setup_styling()

        # Check backend status periodically
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_backend_status)
        self.status_timer.start(3000)  # Check every 3 seconds

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.StyledPanel)
        header_layout = QHBoxLayout(header_frame)

        title_label = QLabel("🎾 Tennis Knowledge Database")
        title_label.setObjectName("title_label")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        # Backend controls
        self.backend_status_label = QLabel("Backend: Stopped")
        self.backend_status_label.setObjectName("status_label")

        # Model selector
        model_label = QLabel("Model:")
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("model_selector")
        self.model_selector.addItems([
            "llama3",
            "llama3:8b",
            "llama3:70b",
            "llama3.1",
            "llama3.1:8b",
            "llama3.1:70b",
            "mistral",
            "gemma",
            "codellama",
            "phi3",
            "qwen2"
        ])
        self.model_selector.setCurrentText("llama3")
        self.model_selector.setMinimumWidth(120)

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

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Segoe UI", 10))
        layout.addWidget(self.chat_display, 1)

        # Input area
        input_frame = QFrame()
        input_frame.setFrameStyle(QFrame.StyledPanel)
        input_layout = QHBoxLayout(input_frame)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about tennis history, players, tournaments...")
        self.input_field.setFont(QFont("Segoe UI", 10))
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

        # Initial welcome message
        self.add_message("system", "Welcome to the Tennis Knowledge Database! Start the backend to begin chatting.")

    def setup_styling(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                margin: 2px;
                padding: 8px;
            }
            QTextEdit {
                background-color: #2d2d2d;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 12px;
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #ffffff;
                selection-background-color: #0078d4;
            }
            QLineEdit {
                background-color: #3c3c3c;
                border: 1px solid #404040;
                border-radius: 8px;
                padding: 10px;
                font-size: 12px;
                color: #ffffff;
            }
            QLineEdit:focus {
                border: 2px solid #0078d4;
                background-color: #404040;
            }
            QComboBox {
                background-color: #3c3c3c;
                border: 1px solid #404040;
                border-radius: 6px;
                padding: 6px 10px;
                color: #ffffff;
                font-size: 11px;
                min-width: 100px;
            }
            QComboBox:hover {
                border: 1px solid #0078d4;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 4px solid #ffffff;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #3c3c3c;
                border: 1px solid #404040;
                selection-background-color: #0078d4;
                color: #ffffff;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 18px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #404040;
                color: #808080;
            }
            QPushButton#start_button {
                background-color: #107c10;
            }
            QPushButton#start_button:hover {
                background-color: #0e6b0e;
            }
            QPushButton#stop_button {
                background-color: #d13438;
            }
            QPushButton#stop_button:hover {
                background-color: #b02b2f;
            }
            QPushButton#ingest_button {
                background-color: #8764b8;
            }
            QPushButton#ingest_button:hover {
                background-color: #744da9;
            }
            QLabel {
                color: #ffffff;
                font-weight: 500;
            }
            QLabel#title_label {
                color: #ffd700;
                font-size: 18px;
                font-weight: bold;
            }
            QLabel#status_label {
                font-weight: bold;
                padding: 4px 8px;
                border-radius: 4px;
            }
        """)

    def setup_backend_manager(self):
        self.backend_thread = QThread(self)
        self.backend_manager = BackendManager()
        self.backend_manager.moveToThread(self.backend_thread)

        self.backend_manager.status_changed.connect(self.on_backend_status_changed)
        self.backend_manager.log_message.connect(self.add_system_message)

        self.backend_thread.start()

    def setup_chat_worker(self):
        if not self.api_client:
            self.api_client = ApiClient(settings_obj=settings)

        self.chat_thread = QThread(self)
        self.chat_worker = ChatWorker(self.api_client)
        self.chat_worker.moveToThread(self.chat_thread)

        self.chat_worker.response_received.connect(self.on_response_received)
        self.chat_worker.error_occurred.connect(self.on_error_occurred)

        # Connect the signal properly
        self.request_chat_query_signal.connect(self.chat_worker.send_query)

        self.chat_thread.start()

    @Slot()
    def start_backend(self):
        self.backend_manager.start_backend()

    @Slot()
    def stop_backend(self):
        self.backend_manager.stop_backend()

    @Slot(bool)
    def on_backend_status_changed(self, is_running):
        if is_running:
            self.backend_status_label.setText("Backend: Running ✅")
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.send_button.setEnabled(True)
            self.ingest_button.setEnabled(True)

            # Setup chat worker when backend starts
            if not self.chat_worker:
                self.setup_chat_worker()
        else:
            self.backend_status_label.setText("Backend: Stopped ❌")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.send_button.setEnabled(False)
            self.ingest_button.setEnabled(False)

    def check_backend_status(self):
        # Simple check if backend is responsive
        if self.backend_manager and self.backend_manager.is_running:
            try:
                import requests
                response = requests.get(f"http://{settings.API_SERVER_HOST}:{settings.API_SERVER_PORT}/", timeout=1)
                if response.status_code != 200:
                    self.on_backend_status_changed(False)
            except:
                self.on_backend_status_changed(False)

    @Slot()
    def send_message(self):
        query = self.input_field.text().strip()
        if not query:
            return

        selected_model = self.model_selector.currentText()
        self.add_message("user", query)
        self.add_system_message(f"Using model: {selected_model}")
        self.input_field.clear()
        self.send_button.setEnabled(False)

        # Emit signal with both query and model
        self.request_chat_query_signal.emit(query, selected_model)

    @Slot()
    def load_documents(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Tennis Documents Directory"
        )
        if directory:
            self.add_system_message(f"Loading documents from: {directory}")
            # Here you would call the ingest endpoint
            # For now, just show a message
            QMessageBox.information(
                self, "Document Loading",
                f"Documents will be loaded from:\n{directory}\n\nThis feature connects to the /ingest endpoint."
            )

    @Slot(str, list)
    def on_response_received(self, answer, sources):
        self.add_message("assistant", answer, sources)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    @Slot(str)
    def on_error_occurred(self, error_message):
        self.add_message("error", error_message)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    def add_message(self, role, content, sources=None):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        if role == "user":
            cursor.insertHtml(
                f'<div style="margin: 15px 0; padding: 10px; background-color: #404040; border-radius: 8px;"><b style="color: #4fc3f7;">You:</b> <span style="color: #ffffff;">{content}</span></div>')
        elif role == "assistant":
            cursor.insertHtml(
                f'<div style="margin: 15px 0; padding: 10px; background-color: #2d2d2d; border-radius: 8px; border-left: 4px solid #4caf50;"><b style="color: #81c784;">🎾 Tennis AI:</b> <span style="color: #ffffff;">{content}</span>')
            if sources:
                cursor.insertHtml('<br><small style="color: #b0b0b0; margin-top: 8px;"><i>📚 Sources:</i></small>')
                for source in sources[:3]:  # Limit to 3 sources
                    source_file = source.get('source_file', 'Unknown')
                    cursor.insertHtml(f'<br><small style="color: #90a4ae;">• {source_file}</small>')
            cursor.insertHtml('</div>')
        elif role == "error":
            cursor.insertHtml(
                f'<div style="margin: 15px 0; padding: 10px; background-color: #5d1a1a; border-radius: 8px; border-left: 4px solid #f44336;"><b style="color: #ef5350;">❌ Error:</b> <span style="color: #ffffff;">{content}</span></div>')
        else:  # system
            cursor.insertHtml(
                f'<div style="margin: 10px 0; padding: 8px; background-color: #1a1a2e; border-radius: 6px;"><span style="color: #ffd54f; font-style: italic;">🔧 {content}</span></div>')

        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def add_system_message(self, message):
        self.add_message("system", message)

    def closeEvent(self, event):
        # Clean shutdown
        if self.backend_manager and self.backend_manager.is_running:
            self.backend_manager.stop_backend()

        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.quit()
            self.chat_thread.wait()

        if self.backend_thread and self.backend_thread.isRunning():
            self.backend_thread.quit()
            self.backend_thread.wait()

        if self.api_client:
            # Note: This is async, but we're in a sync context
            # In a real app, you'd handle this properly
            pass

        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)

    # Basic logging setup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()