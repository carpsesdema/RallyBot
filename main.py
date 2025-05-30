# main.py
import sys
import logging  # Standard logging

# Attempt to import necessary components
# These will be fully available when the project is complete.
# For now, some might be placeholders if their files don't exist yet.
try:
    from PySide6.QtWidgets import QApplication
    # If config, utils, gui.main_window, gui.api_client are not yet created,
    # these imports will fail if this script is run directly before they exist.
    # This is expected during incremental development.
    from config import settings
    from utils import setup_logger, AvaChatError  # AvaChatError for global exception handling
    from gui.main_window import MainWindow  # This class will be created later
    from gui.api_client import ApiClient  # This class will be created later
except ImportError as e:
    # This initial print is for immediate feedback if core libraries are missing.
    # The logger might not be set up yet if utils.py itself fails to import.
    print(f"Critical Import Error in main.py: {e}. Ensure PySide6 is installed and project structure is correct.")
    # Define dummy classes if imports fail, so the script can be parsed,
    # though it won't run correctly. This helps in step-by-step creation.
    if "PySide6" in str(e):
        sys.exit(f"PySide6 is not installed. Please install it: pip install PySide6")


    class QApplication:  # Dummy
        def __init__(self, args): pass

        def exec(self): print("Dummy QApplication.exec() called. MainWindow not shown."); return 0


    class Settings:  # Dummy
        LOG_LEVEL = "INFO"
        API_SERVER_HOST = "127.0.0.1"
        API_SERVER_PORT = 8000


    settings = Settings()  # Instantiate dummy


    def setup_logger(name, level):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(name)
        logger.info(f"Dummy logger setup for {name} at level {level}")
        return logger


    class AvaChatError(Exception):
        pass  # Dummy


    class MainWindow:  # Dummy
        def __init__(self, settings_obj=None, api_client_obj=None):  # Adjusted to accept Renamed obj
            self.logger = logging.getLogger("DummyMainWindow")
            self.logger.info("Dummy MainWindow initialized.")

        def show(self): self.logger.info("Dummy MainWindow.show() called.")


    class ApiClient:  # Dummy
        def __init__(self, settings_obj=None):  # Adjusted to accept Renamed obj
            self.logger = logging.getLogger("DummyApiClient")
            self.logger.info("Dummy ApiClient initialized.")

# Global logger for main.py itself, configured after setup_logger is called
# This will be properly configured once setup_logger runs.
main_logger = None


def run_application():
    global main_logger

    # 1. Setup Logger
    # The actual settings.LOG_LEVEL will be used once config.py is fully integrated.
    # Using a default here in case settings object is a dummy.
    log_level = "INFO"
    if hasattr(settings, 'LOG_LEVEL'):
        log_level = settings.LOG_LEVEL

    # Call setup_logger from utils.py
    # This configures the root logger and returns a specific logger for "AvaChatApp"
    # For main.py specific logs, we can get a logger after setup.
    app_logger = setup_logger("AvaChatApp", log_level)  # Main application logger
    main_logger = logging.getLogger("main_entry_point")  # Logger for this file's scope

    main_logger.info("AvaChat application starting...")
    main_logger.info(f"Using log level: {log_level}")

    # 2. Create QApplication
    try:
        app = QApplication(sys.argv)
        main_logger.info("QApplication created.")
    except Exception as e:
        # This catch is for very early QApplication instantiation errors.
        # If QApplication itself can't be created (e.g., display server issues on Linux without headless mode)
        initial_error_msg = f"Failed to create QApplication: {e}. This can happen due to missing Qt plugins or display issues."
        if main_logger:
            main_logger.critical(initial_error_msg, exc_info=True)
        else:  # Logger not even available
            print(initial_error_msg)  # Fallback to print
        sys.exit(1)

    # 3. Instantiate ApiClient
    # The actual settings object will be passed from config.py
    try:
        api_client = ApiClient(settings_obj=settings)  # Pass settings object
        main_logger.info("ApiClient instantiated.")
    except Exception as e:
        main_logger.critical(f"Failed to instantiate ApiClient: {e}", exc_info=True)
        # Depending on severity, you might exit or try to run GUI in a degraded mode.
        # For now, we'll try to proceed so GUI might show an error.
        api_client = None  # Ensure it's None if instantiation fails

    # 4. Instantiate MainWindow
    # The actual settings object and the created api_client will be passed.
    try:
        if api_client is None and MainWindow.__name__ != "MainWindow":  # If using the real MainWindow and api_client failed
            main_logger.warning("Attempting to create MainWindow without a functional ApiClient.")

        window = MainWindow(settings_obj=settings, api_client_obj=api_client)  # Pass settings and api_client
        main_logger.info("MainWindow instantiated.")
    except Exception as e:
        main_logger.critical(f"Failed to instantiate MainWindow: {e}", exc_info=True)
        # If MainWindow fails, there's not much to do for a GUI app.
        sys.exit(1)

    # 5. Show MainWindow
    try:
        window.show()
        main_logger.info("MainWindow shown.")
    except Exception as e:
        main_logger.critical(f"Failed to show MainWindow: {e}", exc_info=True)
        sys.exit(1)

    # 6. Start Application Event Loop
    exit_code = app.exec()
    main_logger.info(f"Application exited with code {exit_code}.")
    sys.exit(exit_code)


if __name__ == "__main__":
    try:
        run_application()
    except AvaChatError as ace:
        # Catch custom application-specific errors that might bubble up if not handled deeper.
        # The main_logger might be None if setup_logger failed very early.
        if main_logger:
            main_logger.critical(f"Unhandled AvaChatError at top level: {ace}", exc_info=True)
        else:
            print(f"Unhandled AvaChatError (logger unavailable): {ace}")
        sys.exit(1)
    except Exception as e:
        # Catch any other unhandled exceptions at the very top level.
        if main_logger:
            main_logger.critical(f"Unhandled critical exception at top level: {e}", exc_info=True)
        else:
            print(f"Unhandled critical exception (logger unavailable): {e}")
        # Attempt to show an error message box if QApplication is available.
        try:
            from PySide6.QtWidgets import QMessageBox

            # Check if QApplication instance exists (it might not if error was before its creation)
            if QApplication.instance():
                error_box = QMessageBox()
                error_box.setIcon(QMessageBox.Critical)
                error_box.setWindowTitle("Critical Application Error")
                error_box.setText("AvaChat encountered an unhandled critical error and must close.")
                error_box.setDetailedText(str(e) + "\n\nCheck logs for more details.")
                error_box.setStandardButtons(QMessageBox.Ok)
                error_box.exec()
        except Exception as msg_e:
            # Fallback if even QMessageBox fails
            if main_logger:
                main_logger.error(f"Failed to show critical error QMessageBox: {msg_e}")
            else:
                print(f"Failed to show critical error QMessageBox: {msg_e}")
        sys.exit(1)