# utils.py
import logging
import sys


# --- Custom Exception Classes ---
class AvaChatError(Exception):
    """Base class for exceptions in AvaChat."""
    pass


class ConfigurationError(AvaChatError):
    """For errors related to application configuration."""
    pass


class LLMClientError(AvaChatError):
    """For errors originating from the LLM client (e.g., Ollama interaction)."""
    pass


class RAGPipelineError(AvaChatError):
    """For errors within the RAG pipeline processing."""
    pass


class VectorStoreError(AvaChatError):
    """For errors related to the vector store operations."""
    pass


class ApiClientError(AvaChatError):
    """For errors originating from the GUI's API client."""
    pass


class DocumentLoadingError(AvaChatError):
    """For errors during document loading."""
    pass


class TextSplittingError(AvaChatError):
    """For errors during text splitting/chunking."""
    pass


class EmbeddingGenerationError(AvaChatError):
    """For errors during the generation of embeddings."""
    pass


# --- Logger Setup ---
def setup_logger(app_name: str, log_level_str: str) -> logging.Logger:
    """
    Configures and returns a logger for the application.

    Args:
        app_name: The name of the application, used for the logger name.
        log_level_str: The desired logging level as a string (e.g., "INFO", "DEBUG").

    Returns:
        A configured logging.Logger instance.
    """
    numeric_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_level, int):
        logging.warning(f"Invalid log level: {log_level_str}. Defaulting to INFO.")
        numeric_level = logging.INFO

    # Get the root logger if app_name is not specific enough, or a named logger
    logger = logging.getLogger(app_name)
    logger.setLevel(numeric_level)

    # Prevent adding multiple handlers if already configured
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # Optional: File Handler (example, can be extended based on settings)
        # from config import settings # Be careful with circular imports if settings uses this logger
        # if settings.LOG_FILE_PATH: # Assuming a LOG_FILE_PATH setting exists
        #     file_handler = logging.FileHandler(settings.LOG_FILE_PATH)
        #     file_handler.setFormatter(formatter)
        #     logger.addHandler(file_handler)

        logger.info(f"Logger '{app_name}' configured with level {log_level_str.upper()}.")

    return logger


if __name__ == '__main__':
    # Example usage of setup_logger and custom exceptions
    # In a real app, settings would come from config.py
    test_logger = setup_logger("TestApp", "DEBUG")
    test_logger.debug("This is a debug message.")
    test_logger.info("This is an info message.")
    test_logger.warning("This is a warning message.")
    test_logger.error("This is an error message.")

    try:
        # Simulate an error
        raise LLMClientError("Failed to connect to LLM.")
    except AvaChatError as e:
        test_logger.exception(f"An AvaChat specific error occurred: {e}")
    except Exception as e:
        test_logger.exception(f"An unexpected error occurred: {e}")

    # Test logger idempotency (should not add handlers again)
    test_logger_again = setup_logger("TestApp", "INFO")
    test_logger_again.info("This info message should appear, but handlers shouldn't be duplicated.")

    another_logger = setup_logger("AnotherModule", "INFO")
    another_logger.info("Logging from another module.")