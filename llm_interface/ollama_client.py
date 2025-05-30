# llm_interface/ollama_client.py
import httpx
import logging
from typing import List, Optional, Dict, Any

# Attempt to import settings and custom exceptions
# These might not be available in a standalone test of this file,
# but will be when the application runs.
try:
    from config import Settings
    from utils import LLMClientError, EmbeddingGenerationError
except ImportError:
    # Define fallbacks if running standalone or during isolated testing
    class Settings:  # Dummy for standalone testing
        OLLAMA_API_URL = "http://localhost:11434"
        OLLAMA_CHAT_MODEL = "llama3"
        OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


    class LLMClientError(Exception):
        pass


    class EmbeddingGenerationError(Exception):
        pass

logger = logging.getLogger(__name__)


class OllamaLLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.OLLAMA_API_URL
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)  # Sensible timeout
        logger.info(f"OllamaLLMClient initialized for base URL: {self.base_url}")

    async def list_available_models(self) -> List[str]:
        """
        Lists available models from the Ollama /api/tags endpoint.
        """
        try:
            logger.debug("Requesting list of available models from Ollama")
            response = await self._client.get("/api/tags")
            response.raise_for_status()

            response_data = response.json()
            if "models" not in response_data:
                logger.error(f"Ollama models response missing 'models' field. Data: {response_data}")
                raise LLMClientError(f"Ollama models response missing 'models' field. Full response: {response_data}")

            # Extract model names from the response
            model_names = [model.get("name", "unknown") for model in response_data["models"]]
            logger.info(f"Successfully retrieved {len(model_names)} models from Ollama.")
            return model_names

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during Ollama models request: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(f"Ollama API error (models): {e.response.status_code} - {e.response.text}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error during Ollama models request: {e}")
            raise LLMClientError(f"Ollama connection error (models): {e}") from e
        except Exception as e:
            logger.exception("Unexpected error in list_available_models")
            raise LLMClientError(f"Unexpected error communicating with Ollama (models): {e}") from e

    async def generate_response(self, prompt: str, model_name: Optional[str] = None) -> str:
        """
        Generates a response from the Ollama /api/generate endpoint.
        """
        model = model_name if model_name else self.settings.OLLAMA_CHAT_MODEL
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False  # As per requirements
        }
        try:
            logger.debug(f"Sending generation request to Ollama model {model}. Prompt: '{prompt[:50]}...'")
            response = await self._client.post("/api/generate", json=payload)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses

            response_data = response.json()
            if "response" not in response_data:
                logger.error(f"Ollama response missing 'response' field. Data: {response_data}")
                raise LLMClientError(f"Ollama response missing 'response' field. Full response: {response_data}")

            logger.info(f"Successfully received response from Ollama model {model}.")
            return response_data["response"]
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during Ollama generate request: {e.response.status_code} - {e.response.text}")
            raise LLMClientError(f"Ollama API error (generate): {e.response.status_code} - {e.response.text}") from e
        except httpx.RequestError as e:
            logger.error(f"Request error during Ollama generate request: {e}")
            raise LLMClientError(f"Ollama connection error (generate): {e}") from e
        except Exception as e:
            logger.exception(f"Unexpected error in generate_response for model {model}")
            raise LLMClientError(f"Unexpected error communicating with Ollama (generate): {e}") from e

    async def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        Generates embeddings from the Ollama /api/embeddings endpoint.
        Makes an individual POST request for each text.
        """
        model = model_name if model_name else self.settings.OLLAMA_EMBEDDING_MODEL
        embeddings_list: List[List[float]] = []

        logger.debug(f"Generating embeddings for {len(texts)} texts using Ollama model {model}.")
        for i, text_content in enumerate(texts):
            if not text_content.strip():  # Handle empty strings if necessary
                logger.warning(f"Skipping embedding generation for empty text at index {i}.")
                # Depending on strictness, you might want to append a zero vector or raise an error.
                # For now, we'll assume Ollama handles it or we skip. If Ollama errors, it'll be caught.
                # If skipping, ensure downstream code handles potential length mismatch if not appending anything.
                # A common practice is to embed a placeholder or a zero vector of correct dimension.
                # For simplicity, let's assume Ollama can process an empty string if it must.
                # If it errors, it'll be caught.
                pass

            payload = {
                "model": model,
                "prompt": text_content  # Ollama uses "prompt" for the text to embed
            }
            try:
                logger.debug(f"Requesting embedding for text (index {i}): '{text_content[:50]}...'")
                response = await self._client.post("/api/embeddings", json=payload)
                response.raise_for_status()

                response_data = response.json()
                if "embedding" not in response_data:
                    logger.error(
                        f"Ollama embedding response missing 'embedding' field for text '{text_content[:50]}...'. Data: {response_data}")
                    raise EmbeddingGenerationError(
                        f"Ollama embedding response missing 'embedding' field. Full response: {response_data}")

                embeddings_list.append(response_data["embedding"])
                logger.debug(f"Successfully received embedding for text (index {i}).")
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP error during Ollama embedding request for text '{text_content[:50]}...': {e.response.status_code} - {e.response.text}")
                raise EmbeddingGenerationError(
                    f"Ollama API error (embeddings): {e.response.status_code} - {e.response.text}") from e
            except httpx.RequestError as e:
                logger.error(f"Request error during Ollama embedding request for text '{text_content[:50]}...': {e}")
                raise EmbeddingGenerationError(f"Ollama connection error (embeddings): {e}") from e
            except Exception as e:
                logger.exception(
                    f"Unexpected error in generate_embeddings for text '{text_content[:50]}...' with model {model}")
                raise EmbeddingGenerationError(f"Unexpected error communicating with Ollama (embeddings): {e}") from e

        logger.info(f"Successfully generated {len(embeddings_list)} embeddings using Ollama model {model}.")
        return embeddings_list

    async def close_session(self) -> None:
        """
        Closes the httpx.AsyncClient session.
        """
        await self._client.aclose()
        logger.info("OllamaLLMClient session closed.")