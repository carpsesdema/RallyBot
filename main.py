# main.py
import sys
import logging
from pathlib import Path

# Ensure we can import our modules
sys.path.insert(0, str(Path(__file__).parent))

try:
    from PySide6.QtWidgets import QApplication
    from gui.main_window import MainWindow  # Import MainWindow from your existing file
    from config import settings
    from utils import setup_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install PySide6 fastapi uvicorn httpx pydantic faiss-cpu numpy python-dotenv")
    sys.exit(1)


def main():
    """Launch the Tennis Knowledge Database chat interface"""

    # Setup logging
    setup_logger("TennisKnowledgeDB", settings.LOG_LEVEL)
    logger = logging.getLogger("TennisLauncher")

    logger.info("Starting Tennis Knowledge Database...")

    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Tennis Knowledge Database")
    app.setApplicationVersion("1.0")

    # Create and show main window
    window = MainWindow()
    window.show()

    logger.info("Tennis Knowledge Database GUI started successfully")
    logger.info("Use the 'Start Backend' button to begin chatting")

    # Run the application
    exit_code = app.exec()
    logger.info(f"Application exited with code: {exit_code}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())