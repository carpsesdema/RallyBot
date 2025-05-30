# gui/api_client.py
import httpx
import logging
from typing import Optional, List  # Added List

# Attempt to import settings, Pydantic models, and custom exceptions
try:
    from config import Settings  # Assuming Settings is appropriately available from your config.py
    from models import (
        QueryRequest, QueryResponse,
        IngestDirectoryRequest, IngestDirectoryResponse,
        AvailableModelsResponse,  # Added new response model
        ApiErrorResponse, ApiErrorDetail
    )
    from utils import ApiClientError  # Assuming ApiClientError is in your utils.py
except ImportError as e:
    # This print is for immediate feedback if core dependencies are missing.
    print(f"Import Error in gui/api_client.py: {e}. Ensure config.py, models.py, and utils.py exist and are correct.")


    # Define dummy classes if imports fail for standalone parsing/early dev
    class Settings:  # Dummy
        API_SERVER_HOST = "127.0.0.1"
        API_SERVER_PORT = 8000
        LOG_LEVEL = "INFO"


    class PydanticBaseModel:  # Dummy base for other Pydantic models
        def model_dump(self, **kwargs): return {}

        @classmethod
        def model_validate(cls, data, **kwargs): return cls()  # Simplified

        def model_dump_json(self, **kwargs): return "{}"  # Simplified for a string representation


    class QueryRequest(PydanticBaseModel):
        pass


    class QueryResponse(PydanticBaseModel):
        pass


    class IngestDirectoryRequest(PydanticBaseModel):
        pass


    class IngestDirectoryResponse(PydanticBaseModel):
        pass


    class AvailableModelsResponse(PydanticBaseModel):
        models: List[str] = []  # Dummy


    class ApiErrorDetail(PydanticBaseModel):
        pass


    class ApiErrorResponse(PydanticBaseModel):
        error: ApiErrorDetail = ApiErrorDetail()


    class ApiClientError(Exception):  # Dummy exception
        def __init__(self, message, status_code=None, error_response=None):
            super().__init__(message)
            self.message = message
            self.status_code = status_code
            self.error_response = error_response

# Configure a logger for this module
logger = logging.getLogger(__name__)
# Basic config if not already configured by a root logger elsewhere
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Basic logging configured for api_client.py as no handlers were found.")


class ApiClient:
    """
    Asynchronous HTTP client for communicating with the AvaChat FastAPI backend.
    """

    def __init__(self, settings_obj: Settings):
        self.settings = settings_obj
        self.base_url = f"http://{self.settings.API_SERVER_HOST}:{self.settings.API_SERVER_PORT}/api"
        # Initialize AsyncClient here. Manage its lifecycle (e.g., open/close with app).
        # For this client, creating it per instance is okay if ApiClient is long-lived with MainWindow.
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)  # Increased timeout
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

            if 200 <= response.status_code < 300:  # Successful response
                if expected_response_model:
                    try:
                        # Use model_validate for Pydantic V2
                        return expected_response_model.model_validate(response.json())
                    except Exception as e:  # Catch Pydantic validation errors or others
                        logger.error(
                            f"Failed to parse successful response for {endpoint} into {expected_response_model.__name__}: {e}. Response JSON: {response.json()}")
                        raise ApiClientError(
                            f"Failed to parse successful response: {e}",
                            response.status_code
                        )
                return response.json()  # Or return raw JSON if no model
            else:  # Error response
                error_response_parsed = None
                try:
                    error_data = response.json()
                    # Attempt to parse into ApiErrorResponse if it's structured that way
                    error_response_parsed = ApiErrorResponse.model_validate(error_data)
                    error_message = error_response_parsed.error.message
                    if error_response_parsed.error.code:
                        error_message = f"[{error_response_parsed.error.code}] {error_message}"
                except Exception:  # If parsing error detail fails, use raw text
                    error_message = response.text or f"HTTP Error {response.status_code}"
                logger.warning(f"API error from {endpoint} (Status {response.status_code}): {error_message}")
                raise ApiClientError(
                    error_message,
                    response.status_code,
                    error_response_parsed
                )
        except httpx.HTTPStatusError as e:  # Should be caught by status code check above, but good to have
            logger.error(f"HTTPStatusError for {method} {endpoint}: {e.response.status_code} - {e.response.text}",
                         exc_info=True)
            raise ApiClientError(
                f"Server returned error: {e.response.status_code}",
                e.response.status_code
            ) from e
        except httpx.RequestError as e:  # Covers connection errors, timeouts etc.
            logger.error(f"RequestError for {method} {endpoint}: {e}", exc_info=True)
            raise ApiClientError(f"Request failed (e.g., connection error, timeout): {e}") from e
        except ApiClientError:  # Re-raise ApiClientErrors originating from parsing etc.
            raise
        except Exception as e:  # Catch-all for other unexpected errors during request
            logger.error(f"Unexpected error during {method} request to {endpoint}: {e}", exc_info=True)
            raise ApiClientError(f"An unexpected error occurred: {e}") from e

    async def get_available_models(self) -> AvailableModelsResponse:
        """
        Fetches the list of available LLM models from the backend.
        GET to /models
        """
        logger.info("Fetching available models from backend via GET /models...")
        return await self._request(
            method="GET",
            endpoint="/models",
            expected_response_model=AvailableModelsResponse
        )

    async def post_chat_query(self, payload: QueryRequest) -> QueryResponse:
        """
        Sends a chat query to the backend.
        POST to /chat
        """
        logger.info(f"Posting chat query: '{payload.query_text[:50]}...', Model: {payload.model_name}")
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

    # This __main__ block is for demonstration and basic testing.
    # In the full app, ApiClient is used by a worker in main_window.py.

    # Setup basic logging for the example run if not already done
    if not logging.getLogger(__name__).handlers:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger.info("Basic logging configured for ApiClient __main__ example.")


    async def main_test():
        # Use dummy settings for this standalone example
        class TestSettings:  # More specific dummy for testing
            API_SERVER_HOST = "127.0.0.1"
            API_SERVER_PORT = 8000
            LOG_LEVEL = "DEBUG"  # Not directly used by ApiClient, but for consistency

        test_settings = TestSettings()
        api_client = ApiClient(settings_obj=test_settings)

        logger.info("--- ApiClient Test ---")
        logger.info("NOTE: This test requires a compatible backend server running at "
                    f"http://{test_settings.API_SERVER_HOST}:{test_settings.API_SERVER_PORT}")

        # Test GET /models endpoint
        try:
            logger.info("\nTesting GET /models...")
            # To run this, uncomment and ensure your backend is running and serves /models
            # models_response = await api_client.get_available_models()
            # logger.info(f"Available models response: {models_response.model_dump_json(indent=2)}")
            # if not models_response.models:
            #     logger.warning("No models listed by backend, or backend not reachable/configured for models.")
            logger.info("Skipping actual call to /models in this example. Uncomment to test with a live server.")
            logger.info("Expected: AvailableModelsResponse model or ApiClientError if server error.")

        except ApiClientError as e:
            logger.error(f"ApiClientError during /models test: {e.message} (Status: {e.status_code})")
            if e.error_response:
                logger.error(f"Parsed error response: {e.error_response.model_dump_json(indent=2)}")
        except Exception as e:
            logger.error(f"Unexpected error during /models test: {e}", exc_info=True)

        # Test POST /chat endpoint (example)
        try:
            logger.info("\nTesting POST /chat...")
            chat_payload = QueryRequest(query_text="Hello from ApiClient test!", model_name="default_test_model")
            # query_response = await api_client.post_chat_query(chat_payload)
            # logger.info(f"Chat query response: {query_response.model_dump_json(indent=2)}")
            logger.info("Skipping actual call to /chat in this example. Uncomment to test.")
        except ApiClientError as e:
            logger.error(f"ApiClientError during /chat test: {e.message}")
        except Exception as e:
            logger.error(f"Unexpected error during /chat test: {e}", exc_info=True)

        await api_client.close()  # Important to close the client session


    # asyncio.run(main_test()) # For Python 3.7+
    # Manual loop management for wider compatibility if needed, or if running in an env with existing loop
    loop = asyncio.get_event_loop()
    if loop.is_closed():  # Create new loop if default is closed (e.g. in some test runners or multiple runs)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main_test())
    except KeyboardInterrupt:
        logger.info("API Client test run interrupted by user.")
    finally:
        # Gracefully cancel all pending tasks and close loop
        pending_tasks = [task for task in asyncio.all_tasks(loop=loop) if not task.done()]
        if pending_tasks:
            for task in pending_tasks:
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
            except asyncio.CancelledError:
                logger.debug("Pending tasks cancelled during shutdown.")

        if hasattr(loop, 'shutdown_asyncgens'):  # Python 3.6+
            loop.run_until_complete(loop.shutdown_asyncgens())

        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)  # Clear the event loop