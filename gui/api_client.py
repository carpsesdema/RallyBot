# gui/api_client.py
import httpx
import logging
from typing import Optional

# Attempt to import settings, Pydantic models, and custom exceptions
# These will be fully available when the project is complete.
try:
    from config import Settings, settings
    from models import QueryRequest, QueryResponse, IngestDirectoryRequest, IngestDirectoryResponse, ApiErrorResponse, \
        ApiErrorDetail
    from utils import ApiClientError
except ImportError as e:
    # This print is for immediate feedback if core dependencies are missing during development.
    print(f"Import Error in gui/api_client.py: {e}. Ensure config.py, models.py, and utils.py exist and are correct.")


    # Define dummy classes if imports fail for standalone parsing/early dev
    class Settings:  # Dummy
        API_SERVER_HOST = "127.0.0.1"
        API_SERVER_PORT = 8000
        LOG_LEVEL = "INFO"  # For logger setup below


    class PydanticBaseModel:  # Dummy base for other Pydantic models
        def model_dump(self): return {}

        @classmethod
        def model_validate(cls, data): return cls()  # Simplified

        def model_dump_json(self, indent=None): return "{}"


    class QueryRequest(PydanticBaseModel):
        pass


    class QueryResponse(PydanticBaseModel):
        pass


    class IngestDirectoryRequest(PydanticBaseModel):
        pass


    class IngestDirectoryResponse(PydanticBaseModel):
        pass


    class ApiErrorDetail(PydanticBaseModel):
        pass


    class ApiErrorResponse(PydanticBaseModel):
        error: ApiErrorDetail = ApiErrorDetail()


    class ApiClientError(Exception):
        def __init__(self, message, status_code=None, error_response=None):
            super().__init__(message)
            self.status_code = status_code
            self.error_response = error_response

# Configure a logger for this module
# In a real app, utils.setup_logger would handle root config,
# and here we just get a named logger.
# For now, basic config if utils.py or settings isn't fully there.
try:
    from utils import setup_logger

    # Assuming settings might be a dummy here if config.py failed to import
    log_level_str = "INFO"
    if 'settings' in globals() and hasattr(settings, 'LOG_LEVEL'):
        log_level_str = settings.LOG_LEVEL
    elif isinstance(Settings, type) and hasattr(Settings, 'LOG_LEVEL'):  # Access class attr if dummy
        log_level_str = Settings.LOG_LEVEL

    # If setup_logger is available, it might configure the root logger.
    # We just need a logger instance for this module.
    # setup_logger("ApiClientModule", log_level_str) # Call it to ensure root is configured
    logger = logging.getLogger(__name__)  # Get a logger specific to this module
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.warning("Failed to import setup_logger from utils. Falling back to basic logging config for api_client.py.")


class ApiClient:
    """
    Asynchronous HTTP client for communicating with the AvaChat FastAPI backend.
    """

    def __init__(self, settings_obj: Settings):  # Renamed settings to settings_obj to avoid conflict
        self.settings = settings_obj
        self.base_url = f"http://{self.settings.API_SERVER_HOST}:{self.settings.API_SERVER_PORT}/api"
        # Initialize AsyncClient here. It's good practice to manage its lifecycle,
        # e.g., open on app start, close on app exit, if app-wide.
        # For this client, creating it per instance is okay, but if MainWindow creates/destroys
        # ApiClient often, consider passing a shared client or managing it in MainWindow.
        # For simplicity now, each ApiClient instance gets its own httpx.AsyncClient.
        # This client will be used by ApiWorker which should be long-lived with MainWindow.
        self._client = httpx.AsyncClient(base_url=self.base_url,
                                         timeout=60.0)  # Increased timeout for potentially long LLM/ingest tasks
        logger.info(f"ApiClient initialized for base URL: {self.base_url}")

    async def _request(self, method: str, endpoint: str, json_payload: Optional[dict] = None,
                       expected_response_model=None):
        """Helper method to make requests and handle responses."""
        try:
            logger.debug(
                f"Sending {method} request to {endpoint}. Payload: {json_payload if json_payload else 'No payload'}")
            response = await self._client.request(method, endpoint, json=json_payload)

            logger.debug(
                f"Received response from {endpoint}. Status: {response.status_code}. Response text: '{response.text[:200]}...'")

            if response.status_code >= 200 and response.status_code < 300:
                if expected_response_model:
                    try:
                        return expected_response_model.model_validate(response.json())
                    except Exception as e:  # Catch Pydantic validation errors or others
                        logger.error(
                            f"Failed to parse successful response for {endpoint} into {expected_response_model.__name__}: {e}. Response JSON: {response.json()}")
                        raise ApiClientError(
                            message=f"Failed to parse successful response: {e}",
                            status_code=response.status_code
                        )
                return response.json()  # Or return raw JSON if no model
            else:
                # Attempt to parse as ApiErrorResponse
                error_response_parsed = None
                try:
                    error_data = response.json()
                    error_response_parsed = ApiErrorResponse.model_validate(error_data)
                    error_message = error_response_parsed.error.message
                    if error_response_parsed.error.code:
                        error_message = f"[{error_response_parsed.error.code}] {error_message}"
                    logger.warning(f"API error from {endpoint} (Status {response.status_code}): {error_message}")
                except Exception:
                    error_message = response.text or f"HTTP Error {response.status_code}"
                    logger.warning(
                        f"API error from {endpoint} (Status {response.status_code}): {error_message}. Could not parse error response.")

                raise ApiClientError(
                    message=error_message,
                    status_code=response.status_code,
                    error_response=error_response_parsed
                )
        except httpx.HTTPStatusError as e:  # Should be caught by status code check above, but good to have
            logger.error(f"HTTPStatusError for {method} {endpoint}: {e.response.status_code} - {e.response.text}",
                         exc_info=True)
            raise ApiClientError(
                message=f"Server returned error: {e.response.status_code}",
                status_code=e.response.status_code
            ) from e
        except httpx.RequestError as e:  # Covers connection errors, timeouts etc.
            logger.error(f"RequestError for {method} {endpoint}: {e}", exc_info=True)
            raise ApiClientError(f"Request failed (e.g., connection error, timeout): {e}") from e
        except ApiClientError:  # Re-raise ApiClientErrors originating from parsing etc.
            raise
        except Exception as e:  # Catch-all for other unexpected errors during request
            logger.error(f"Unexpected error during {method} request to {endpoint}: {e}", exc_info=True)
            raise ApiClientError(f"An unexpected error occurred: {e}") from e

    async def post_chat_query(self, payload: QueryRequest) -> QueryResponse:
        """
        Sends a chat query to the backend.
        POST to /chat
        """
        logger.info(f"Posting chat query: '{payload.query_text[:50]}...'")
        return await self._request(
            method="POST",
            endpoint="/chat",
            json_payload=payload.model_dump(),  # Use model_dump for Pydantic V2
            expected_response_model=QueryResponse
        )

    async def post_ingest_directory(self, payload: IngestDirectoryRequest) -> IngestDirectoryResponse:
        """
        Requests the backend to ingest documents from a directory.
        POST to /ingest
        """
        logger.info(f"Posting ingest directory request for: {payload.directory_path}")
        return await self._request(
            method="POST",
            endpoint="/ingest",
            json_payload=payload.model_dump(),  # Use model_dump for Pydantic V2
            expected_response_model=IngestDirectoryResponse
        )

    async def close(self):
        """
        Closes the underlying httpx.AsyncClient.
        Should be called when the ApiClient is no longer needed (e.g., application shutdown).
        """
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("ApiClient's httpx.AsyncClient session closed.")


if __name__ == '__main__':
    import asyncio

    # Example Usage (requires a running FastAPI server with the specified endpoints)
    # This __main__ block is for demonstration and basic testing.
    # In the full app, ApiClient is used by ApiWorker in main_window.py.

    # Setup basic logging for the example run
    if not hasattr(logging.getLogger(__name__), 'handlers') or not logging.getLogger(__name__).handlers:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Basic logging configured for ApiClient __main__ example.")


    async def main_test():
        # Use dummy settings for this standalone example
        class DummySettings:
            API_SERVER_HOST = "127.0.0.1"  # Change if your server is elsewhere
            API_SERVER_PORT = 8000  # Change if your server uses a different port
            LOG_LEVEL = "DEBUG"

        dummy_settings = DummySettings()
        api_client = ApiClient(settings_obj=dummy_settings)

        logger.info("--- ApiClient Test ---")
        logger.info("NOTE: This test requires a compatible backend server running at "
                    f"http://{dummy_settings.API_SERVER_HOST}:{dummy_settings.API_SERVER_PORT}")

        # Test /chat endpoint
        try:
            logger.info("\nTesting POST /chat...")
            chat_payload = QueryRequest(query_text="Hello from ApiClient test!", session_id="test_session_123")
            # The following line will make a real HTTP request if a server is running.
            # query_response = await api_client.post_chat_query(chat_payload)
            # logger.info(f"Chat query response: {query_response.model_dump_json(indent=2)}")
            logger.info("Skipping actual call to /chat in this example. Uncomment to test with a live server.")
            logger.info("Expected: QueryResponse model or ApiClientError if server not running/misconfigured.")

        except ApiClientError as e:
            logger.error(f"ApiClientError during /chat test: {e.message} (Status: {e.status_code})")
            if e.error_response:
                logger.error(f"Parsed error response: {e.error_response.model_dump_json(indent=2)}")
        except Exception as e:
            logger.error(f"Unexpected error during /chat test: {e}", exc_info=True)

        # Test /ingest endpoint
        try:
            logger.info("\nTesting POST /ingest...")
            # Create a dummy directory for the test IF you were to run it for real.
            # For now, we just construct the payload.
            # from pathlib import Path
            # dummy_ingest_dir = Path("./_api_client_test_ingest_docs")
            # dummy_ingest_dir.mkdir(exist_ok=True)
            # (dummy_ingest_dir / "test_doc.txt").write_text("This is a test document for ingestion via API.")

            ingest_payload = IngestDirectoryRequest(
                directory_path="./some_docs_for_ingestion")  # Path is relative to server
            # ingest_response = await api_client.post_ingest_directory(ingest_payload)
            # logger.info(f"Ingest directory response: {ingest_response.model_dump_json(indent=2)}")
            logger.info("Skipping actual call to /ingest in this example. Uncomment to test with a live server.")
            logger.info("Expected: IngestDirectoryResponse model or ApiClientError.")

            # shutil.rmtree(dummy_ingest_dir) # Clean up dummy directory

        except ApiClientError as e:
            logger.error(f"ApiClientError during /ingest test: {e.message} (Status: {e.status_code})")
            if e.error_response:
                logger.error(f"Parsed error response: {e.error_response.model_dump_json(indent=2)}")
        except Exception as e:
            logger.error(f"Unexpected error during /ingest test: {e}", exc_info=True)
        finally:
            await api_client.close()  # Important to close the client session


    # asyncio.run(main_test()) # For Python 3.7+
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main_test())
    except KeyboardInterrupt:
        logger.info("API Client test run interrupted by user.")
    finally:
        # Ensure event loop is closed properly
        pending = asyncio.all_tasks(loop=loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending))
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)