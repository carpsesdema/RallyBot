import logging
from typing import List, Optional
import google.generativeai as genai

try:
    from config import Settings
    from utils import LLMClientError, EmbeddingGenerationError
except ImportError:
    class Settings:
        GOOGLE_API_KEY = ""
        GEMINI_MODEL = "gemini-1.5-flash"  # Default fallback for standalone
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
            logger.error("Google API key is missing or is a placeholder in settings.")
            raise LLMClientError("Google API key not configured. Please set GOOGLE_API_KEY.")

        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
        except Exception as e:
            logger.error(f"Failed to configure Google Generative AI: {e}", exc_info=True)
            raise LLMClientError(f"Failed to configure Google Generative AI: {e}") from e

        chat_model_name_from_settings = settings.GEMINI_MODEL
        effective_chat_model_name = chat_model_name_from_settings

        if chat_model_name_from_settings.startswith("models/"):
            effective_chat_model_name = chat_model_name_from_settings.split('/', 1)[1]
            logger.info(
                f"GEMINI_MODEL '{chat_model_name_from_settings}' in settings starts with 'models/'. "
                f"Using '{effective_chat_model_name}' for GenerativeModel initialization."
            )

        try:
            self.model = genai.GenerativeModel(effective_chat_model_name)
        except Exception as e:
            logger.error(
                f"Failed to initialize genai.GenerativeModel with model name '{effective_chat_model_name}': {e}",
                exc_info=True)
            raise LLMClientError(
                f"Failed to initialize Gemini GenerativeModel (name: {effective_chat_model_name}): {e}") from e

        self.embedding_model_name = "models/text-embedding-004"  # Standard model for embeddings
        logger.info(
            f"GeminiLLMClient initialized with chat model: {self.model.model_name} (from settings: {settings.GEMINI_MODEL}), embedding model: {self.embedding_model_name}")

    async def list_available_models(self) -> List[str]:
        """
        Returns a list containing the configured Gemini chat model name.
        For Gemini, we typically work with one primary configured chat model.
        """
        if self.model and self.model.model_name:
            # model.model_name might include "models/" prefix, ensure consistency
            # For the dropdown, it's often cleaner without "models/"
            name_to_return = self.model.model_name
            if name_to_return.startswith("models/"):
                name_to_return = name_to_return.split('/', 1)[1]
            return [name_to_return]
        logger.warning("GeminiLLMClient's model or model_name is not available for listing.")
        return [self.settings.GEMINI_MODEL.split('/', 1)[-1]]  # Fallback to settings

    async def generate_response(self, prompt: str, model_name: Optional[str] = None) -> str:
        if model_name and model_name != self.model.model_name.split('/', 1)[-1]:  # Compare against the clean name
            logger.warning(
                f"generate_response received model_name_override '{model_name}', "
                f"but GeminiLLMClient is initialized with model '{self.model.model_name}'. "
                "Override is not directly applied to the client's pre-loaded model instance. "
                "Ensure the correct model is selected via settings for this client."
            )
            # If strict model adherence is needed, one might re-initialize self.model here,
            # but current design uses one configured model.

        try:
            logger.debug(f"Sending request to Gemini model {self.model.model_name}")
            generation_config = genai.types.GenerationConfig(
                temperature=self.settings.GEMINI_TEMPERATURE,
                max_output_tokens=2048,
            )
            response = await self.model.generate_content_async(
                prompt,
                generation_config=generation_config
            )
            if not response.text:
                logger.warning(
                    f"Gemini model {self.model.model_name} returned response without text for prompt: '{prompt[:100]}...'")
                return ""

            logger.info(f"Successfully received response from Gemini model {self.model.model_name}.")
            return response.text
        except Exception as e:
            logger.error(f"Error generating response with Gemini model {self.model.model_name}: {e}", exc_info=True)
            raise LLMClientError(f"Gemini API error during generation with model {self.model.model_name}: {e}") from e

    async def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        if model_name:
            logger.warning(
                f"generate_embeddings received model_name_override '{model_name}', but GeminiLLMClient uses a fixed embedding model '{self.embedding_model_name}'. Override ignored.")

        logger.debug(
            f"Generating embeddings for {len(texts)} texts using Gemini embedding model '{self.embedding_model_name}'.")
        all_embeddings: List[List[float]] = []

        batch_size = 100

        try:
            for i in range(0, len(texts), batch_size):
                text_batch = texts[i:i + batch_size]
                valid_texts_in_batch = [t for t in text_batch if t and t.strip()]

                if not valid_texts_in_batch:
                    # Add empty lists for each original text in the empty batch
                    all_embeddings.extend([[] for _ in text_batch])
                    logger.warning(
                        f"Skipped an entire batch of {len(text_batch)} texts for embedding as all were empty/whitespace.")
                    continue

                result = await genai.embed_content_async(
                    model=self.embedding_model_name,
                    content=valid_texts_in_batch,
                    task_type="RETRIEVAL_DOCUMENT"
                )

                batch_embeddings_data = result.get('embedding')

                # Map results back to original batch structure, inserting [] for skipped texts
                processed_embeddings_iter = iter(batch_embeddings_data)
                for original_text in text_batch:
                    if original_text and original_text.strip():
                        all_embeddings.append(next(processed_embeddings_iter))
                    else:
                        all_embeddings.append([])

            if len(all_embeddings) != len(texts):
                # This should ideally not happen with the new mapping logic
                logger.error(
                    f"Critical error: Mismatch in final embeddings count ({len(all_embeddings)}) vs original texts count ({len(texts)}). This indicates a bug in batch processing logic.")
                # Fallback or raise error
                raise EmbeddingGenerationError(f"Embedding count mismatch: {len(all_embeddings)} vs {len(texts)}")

            logger.info(
                f"Generated embeddings for {len(texts)} original texts. Empty/whitespace texts result in empty embedding lists.")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings with Gemini model {self.embedding_model_name}: {e}",
                         exc_info=True)
            raise EmbeddingGenerationError(
                f"Gemini embedding API error with model {self.embedding_model_name}: {e}") from e

    async def close_session(self) -> None:
        logger.info("GeminiLLMClient session closed (no-op for google-generativeai library).")


if __name__ == '__main__':
    import asyncio

    logging.basicConfig(level=logging.DEBUG)


    class TestSettings:
        GOOGLE_API_KEY = "YOUR_REAL_GOOGLE_API_KEY_HERE"
        GEMINI_MODEL = "gemini-1.5-flash"
        GEMINI_TEMPERATURE = 0.7


    async def test_gemini_client():
        settings_instance = TestSettings()
        if settings_instance.GOOGLE_API_KEY == "YOUR_REAL_GOOGLE_API_KEY_HERE":
            logger.warning(
                "Cannot run test_gemini_client: GOOGLE_API_KEY is not set in the script. Please replace placeholder.")
            return

        logger.info(f"--- Testing GeminiLLMClient with model: {settings_instance.GEMINI_MODEL} ---")

        try:
            client = GeminiLLMClient(settings_instance)

            logger.info("Testing list_available_models...")
            try:
                models_list = await client.list_available_models()
                logger.info(f"Available models from Gemini client: {models_list}")
                assert len(models_list) >= 1
                assert settings_instance.GEMINI_MODEL.split('/')[-1] in models_list[0]
            except Exception as e:
                logger.error(f"list_available_models failed: {e}")

            logger.info("\nTesting chat generation...")
            prompt_text = "What is the main concept behind the game of tennis?"
            try:
                response_text = await client.generate_response(prompt_text)
                logger.info(f"Chat Response for '{prompt_text}':\n{response_text}")
                assert response_text is not None, "Chat response should not be None"
            except LLMClientError as e:
                logger.error(f"Chat generation failed: {e}")

            logger.info("\nTesting embedding generation...")
            texts_to_embed = [
                "Tennis is a sport played with a racket and a ball.",
                "The Wimbledon Championships is one of a kind.",
                "",
                "  ",
                "Rafael Nadal is a famous tennis player."
            ]
            try:
                embeddings = await client.generate_embeddings(texts_to_embed)
                logger.info(f"Generated {len(embeddings)} embeddings for {len(texts_to_embed)} input texts.")
                for i, emb in enumerate(embeddings):
                    original_text_preview = texts_to_embed[i][:30] + "..." if texts_to_embed[
                        i] else "[EMPTY/WHITESPACE STRING]"
                    if emb:
                        logger.info(
                            f"  Embedding for '{original_text_preview}': Dimensions={len(emb)}, Preview={emb[:3]}...")
                        assert len(emb) == 768, f"Expected embedding dimension 768, got {len(emb)}"
                    else:
                        logger.info(
                            f"  Empty embedding list for '{original_text_preview}' (as expected for empty/whitespace input).")

                assert len(embeddings) == len(texts_to_embed), \
                    f"Expected {len(texts_to_embed)} embeddings (with [] for empty), got {len(embeddings)}"

            except EmbeddingGenerationError as e:
                logger.error(f"Embedding generation failed: {e}")

            await client.close_session()

        except LLMClientError as e:
            logger.error(f"Failed to initialize or use GeminiLLMClient: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred during the test: {e}", exc_info=True)


    if TestSettings.GOOGLE_API_KEY == "YOUR_REAL_GOOGLE_API_KEY_HERE":
        print("\nNOTE: GeminiLLMClient __main__ test requires a valid GOOGLE_API_KEY to be set in the script.")
    else:
        print("\nAttempting to run GeminiLLMClient test (ensure GOOGLE_API_KEY is valid and has permissions)...")
        asyncio.run(test_gemini_client())