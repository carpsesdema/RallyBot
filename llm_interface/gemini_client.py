# llm_interface/gemini_client.py
import logging
from typing import List, Optional
import google.generativeai as genai

try:
    from config import Settings
    from utils import LLMClientError, EmbeddingGenerationError
except ImportError:
    # Fallbacks for standalone testing
    class Settings:
        GOOGLE_API_KEY = ""
        GEMINI_MODEL = "gemini-1.5-flash"
        GEMINI_TEMPERATURE = 0.7


    class LLMClientError(Exception):
        pass


    class EmbeddingGenerationError(Exception):
        pass

logger = logging.getLogger(__name__)


class GeminiLLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

        if not settings.GOOGLE_API_KEY or settings.GOOGLE_API_KEY == "your_gemini_api_key_here":
            raise LLMClientError("Google API key not configured. Please set GOOGLE_API_KEY in .env file")

        # Configure Gemini
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)

        # For embeddings, we'll use text-embedding-004
        self.embedding_model = "text-embedding-004"

        logger.info(f"GeminiLLMClient initialized with model: {settings.GEMINI_MODEL}")

    async def generate_response(self, prompt: str, model_name: Optional[str] = None) -> str:
        """
        Generates a response using Google Gemini.
        """
        try:
            logger.debug(f"Sending request to Gemini model {model_name or self.settings.GEMINI_MODEL}")

            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=self.settings.GEMINI_TEMPERATURE,
                max_output_tokens=2048,
            )

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )

            if not response.text:
                raise LLMClientError("Gemini returned empty response")

            logger.info("Successfully received response from Gemini")
            return response.text

        except Exception as e:
            logger.error(f"Error generating response with Gemini: {e}", exc_info=True)
            raise LLMClientError(f"Gemini API error: {e}") from e

    async def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        """
        Generates embeddings using Google's text-embedding-004 model.
        """
        try:
            logger.debug(f"Generating embeddings for {len(texts)} texts using Gemini")

            embeddings = []
            for text in texts:
                if not text.strip():
                    logger.warning("Skipping empty text for embedding generation")
                    continue

                # Use Gemini's embedding API
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )

                embeddings.append(result['embedding'])

            logger.info(f"Successfully generated {len(embeddings)} embeddings using Gemini")
            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings with Gemini: {e}", exc_info=True)
            raise EmbeddingGenerationError(f"Gemini embedding error: {e}") from e

    async def close_session(self) -> None:
        """
        Gemini client doesn't need explicit session closing.
        """
        logger.info("GeminiLLMClient session closed (no-op)")


if __name__ == '__main__':
    import asyncio

    # Test the Gemini client
    logging.basicConfig(level=logging.DEBUG)


    async def test_gemini():
        # You'll need to set your API key
        class TestSettings:
            GOOGLE_API_KEY = "your_actual_api_key_here"  # Replace with real key
            GEMINI_MODEL = "gemini-1.5-flash"
            GEMINI_TEMPERATURE = 0.7

        try:
            client = GeminiLLMClient(TestSettings())

            # Test chat
            response = await client.generate_response("What is tennis?")
            print(f"Gemini response: {response}")

            # Test embeddings
            embeddings = await client.generate_embeddings(["Tennis is a racket sport", "Wimbledon is famous"])
            print(f"Generated {len(embeddings)} embeddings, first one has {len(embeddings[0])} dimensions")

        except Exception as e:
            print(f"Test failed: {e}")


    # asyncio.run(test_gemini())
    print("Gemini client ready. Set GOOGLE_API_KEY to test.")