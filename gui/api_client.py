import httpx
import logging
from typing import Optional, List, Any, Dict  # Added Any
from pathlib import Path  # Added Path

try:
    from config import settings, Settings  # Now using the global settings instance
    from models import (
        QueryRequest, QueryResponse,
        IngestDirectoryRequest, IngestDirectoryResponse,
        AvailableModelsResponse,
        ApiErrorResponse, ApiErrorDetail
    )
    from utils import ApiClientError
except ImportError as e:
    print(f"Import Error in gui/api_client.py: {e}. Ensure config.py, models.py, and utils.py exist.")


    class SettingsClassFallback:  # More complete fallback
        REMOTE_BACKEND_URL = None
        LOCAL_API_SERVER_HOST = "127.0.0.1"
        LOCAL_API_SERVER_PORT = 8000
        LOG_LEVEL = "INFO"
        GEMINI_MODEL = "gemini-dummy-apiclient"  # For AvailableModelsResponse fallback


    settings = SettingsClassFallback()  # Use fallback instance


    class PydanticBaseModel:
        def model_dump(self, **kwargs): return {}

        @classmethod
        def model_validate(cls, data, **kwargs): return cls()

        def model_dump_json(self, **kwargs): return "{}"


    class QueryRequest(PydanticBaseModel):
        query_text: str; model_name: Optional[str]


    class QueryResponse(PydanticBaseModel):
        answer: str; retrieved_chunks_details: Optional[List[Dict[str, Any]]] = None


    class IngestDirectoryRequest(PydanticBaseModel):
        directory_path: str


    class IngestDirectoryResponse(PydanticBaseModel):
        status: str; documents_processed: int; chunks_created: int


    class AvailableModelsResponse(PydanticBaseModel):
        models: List[str] = [settings.GEMINI_MODEL]


    class ApiErrorDetail(PydanticBaseModel):
        code: Optional[str] = None; message: str = ""


    class ApiErrorResponse(PydanticBaseModel):
        error: ApiErrorDetail = ApiErrorDetail()


    class ApiClientError(Exception):
        def __init__(self, message, status_code=None, error_response=None):
            super().__init__(message);
            self.message = message
            self.status_code = status_code;
            self.error_response = error_response

logger = logging.getLogger(__name__)


class ApiClient:
    def __init__(self, app_settings: Settings):  # Pass the global settings instance
        self.settings = app_settings

        if self.settings.REMOTE_BACKEND_URL:
            # Ensure no trailing slash and append /api
            self.base_url = self.settings.REMOTE_BACKEND_URL.rstrip('/') + "/api"
            logger.info(f"ApiClient: Using REMOTE_BACKEND_URL: {self.base_url}")
        else:
            self.base_url = f"http://{self.settings.LOCAL_API_SERVER_HOST}:{self.settings.LOCAL_API_SERVER_PORT}/api"
            logger.info(f"ApiClient: Using LOCAL backend URL: {self.base_url}")

        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        logger.info(f"ApiClient initialized for effective base URL: {self.base_url}")

    async def _request(self, method: str, endpoint: str, json_payload: Optional[dict] = None,
                       expected_response_model=None):
        try:
            logger.debug(
                f"Sending {method} request to {self.base_url}{endpoint}. Payload: {json_payload if json_payload else 'No payload'}")
            response = await self._client.request(method, endpoint, json=json_payload)
            logger.debug(
                f"Received response from {endpoint}. Status: {response.status_code}. Response text: '{response.text[:200]}...'")
            if 200 <= response.status_code < 300:
                if expected_response_model:
                    try:
                        return expected_response_model.model_validate(response.json())
                    except Exception as e:
                        logger.error(
                            f"Failed to parse successful response for {endpoint} into {expected_response_model.__name__}: {e}. Response JSON: {response.json()}",
                            exc_info=True)
                        raise ApiClientError(f"Failed to parse successful response: {e}", response.status_code)
                return response.json()
            else:
                error_response_parsed = None;
                error_message = response.text or f"HTTP Error {response.status_code}"
                try:
                    error_data = response.json()
                    error_response_parsed = ApiErrorResponse.model_validate(error_data)
                    error_message = error_response_parsed.error.message
                    if error_response_parsed.error.code: error_message = f"[{error_response_parsed.error.code}] {error_message}"
                except Exception:
                    pass
                logger.warning(f"API error from {endpoint} (Status {response.status_code}): {error_message}")
                raise ApiClientError(error_message, response.status_code, error_response_parsed)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTPStatusError for {method} {endpoint}: {e.response.status_code} - {e.response.text}",
                         exc_info=True)
            raise ApiClientError(f"Server returned error: {e.response.status_code}", e.response.status_code) from e
        except httpx.RequestError as e:
            logger.error(f"RequestError for {method} {endpoint}: {e}", exc_info=True)
            raise ApiClientError(f"Request failed (e.g., connection error, timeout): {e}") from e
        except ApiClientError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during {method} request to {endpoint}: {e}", exc_info=True)
            raise ApiClientError(f"An unexpected error occurred: {e}") from e

    async def get_available_models(self) -> AvailableModelsResponse:
        logger.info("Fetching available models from backend...")
        return await self._request(method="GET", endpoint="/models", expected_response_model=AvailableModelsResponse)

    async def post_chat_query(self, payload: QueryRequest) -> QueryResponse:
        logger.info(
            f"Posting chat query: '{payload.query_text[:50]}...', Model (effectively ignored by backend): {payload.model_name}")
        return await self._request(method="POST", endpoint="/chat", json_payload=payload.model_dump(),
                                   expected_response_model=QueryResponse)

    async def post_ingest_directory(self, payload: IngestDirectoryRequest) -> IngestDirectoryResponse:
        logger.info(f"Posting ingest directory request for: {payload.directory_path}")
        return await self._request(method="POST", endpoint="/ingest", json_payload=payload.model_dump(),
                                   expected_response_model=IngestDirectoryResponse)

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("ApiClient's httpx.AsyncClient session closed.")


if __name__ == '__main__':
    import asyncio


    # Test requires global 'settings' to be the actual instance from config.py
    # For standalone test, ensure config.py is importable or mock settings adequately.

    # If running this file directly, settings might be the fallback.
    # For a real test, you'd run the main GUI application.

    async def main_test():
        # Use the global settings instance (which should be loaded from config.py)
        # If config.py uses dotenv, ensure .env is in the CWD or path is correct.

        # Forcing a remote URL for testing if you have one, otherwise it uses local.
        # settings.REMOTE_BACKEND_URL = "https://your-test-backend.up.railway.app" # Uncomment and set for remote test
        # settings.REMOTE_BACKEND_URL = None # Force local for this example

        # The ApiClient now takes the global settings object
        # The __init__ of ApiClient decides the base_url based on settings.REMOTE_BACKEND_URL
        api_client_instance = ApiClient(app_settings=settings)
        logger.info(f"--- ApiClient Test (using base_url: {api_client_instance.base_url}) ---")

        try:
            logger.info("\nTesting GET /models...")
            models_response = await api_client_instance.get_available_models()
            logger.info(f"Available models response: {models_response.model_dump_json(indent=2)}")
            assert models_response.models is not None
        except ApiClientError as e:
            logger.error(f"ApiClientError during /models test: {e.message} (Status: {e.status_code})")
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)

        await api_client_instance.close()


    if not isinstance(settings, SettingsClassFallback):  # Basic check if actual settings loaded
        asyncio.run(main_test())
    else:
        logger.warning(
            "Skipping ApiClient __main__ test as it seems to be using fallback settings. Run main GUI application for full test.")