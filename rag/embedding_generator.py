# rag/embedding_generator.py
import logging
from typing import List, Tuple, TYPE_CHECKING, Optional

# Attempt to import necessary modules
try:
    from models import ChunkModel
    from utils import EmbeddingGenerationError

    if TYPE_CHECKING:  # To avoid circular dependency runtime error but allow type hinting
        # Assuming GeminiLLMClient is what's being used due to hardcoding in api_server.py
        from llm_interface.gemini_client import GeminiLLMClient
        # If you had an OllamaLLMClient for type hinting before:
        # from llm_interface.ollama_client import OllamaLLMClient
except ImportError:
    # Define fallbacks if running standalone or during isolated testing
    class ChunkModel:  # Dummy
        def __init__(self, id: str, document_id: str, text_content: str, metadata: dict):
            self.id = id
            self.document_id = document_id
            self.text_content = text_content
            self.metadata = metadata


    class EmbeddingGenerationError(Exception):
        pass


    # Dummy LLMClient for type hinting in standalone mode if needed
    if TYPE_CHECKING:
        class GeminiLLMClient: # Fallback type hint
            async def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[
                List[float]]:
                raise NotImplementedError

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    # Changed type hint to 'GeminiLLMClient' to match hardcoded usage in api_server.py
    # This assumes GeminiLLMClient is the actual type being passed from api_server.py
    def __init__(self, llm_client: 'GeminiLLMClient'):
        self.llm_client = llm_client
        logger.info(f"EmbeddingGenerator initialized with llm_client type: {type(llm_client)}")

    async def generate_embeddings_for_chunks(self, chunks: List[ChunkModel]) -> List[Tuple[ChunkModel, List[float]]]:
        """
        Generates embeddings for a list of ChunkModel objects.

        Args:
            chunks: A list of ChunkModel objects.

        Returns:
            A list of tuples, where each tuple contains the original ChunkModel
            and its corresponding embedding vector (List[float]).

        Raises:
            EmbeddingGenerationError: If no chunks are provided or if embedding generation fails.
        """
        if not chunks:
            logger.warning("No chunks provided to EmbeddingGenerator. Returning empty list.")
            return []

        logger.info(f"Starting embedding generation for {len(chunks)} chunks.")

        texts_to_embed: List[str] = []
        valid_chunks: List[ChunkModel] = []

        for chunk in chunks:
            if chunk.text_content and chunk.text_content.strip():
                texts_to_embed.append(chunk.text_content)
                valid_chunks.append(chunk)
            else:
                logger.warning(
                    f"Chunk {chunk.id} has empty text_content. Skipping embedding generation for this chunk.")

        if not texts_to_embed:
            logger.warning("No valid text content found in provided chunks to generate embeddings.")
            return []

        try:
            # The llm_client.generate_embeddings is expected to take List[str]
            # and return List[List[float]] corresponding to the input texts.
            # The model_name for embeddings is typically handled within the llm_client itself
            # (e.g., Gemini uses a specific embedding model, Ollama client would use settings.OLLAMA_EMBEDDING_MODEL)
            embedding_vectors = await self.llm_client.generate_embeddings(texts_to_embed)
        except Exception as e:
            logger.error(f"Error generating embeddings via LLM client: {e}", exc_info=True)
            raise EmbeddingGenerationError(f"Failed to generate embeddings: {e}") from e

        if len(embedding_vectors) != len(valid_chunks):
            msg = (f"Mismatch between number of embeddings received ({len(embedding_vectors)}) "
                   f"and number of valid chunks processed ({len(valid_chunks)}).")
            logger.error(msg)
            raise EmbeddingGenerationError(msg)

        chunk_embedding_pairs: List[Tuple[ChunkModel, List[float]]] = []
        for i, chunk_model in enumerate(valid_chunks): # Renamed 'chunk' to 'chunk_model' to avoid shadowing
            chunk_embedding_pairs.append((chunk_model, embedding_vectors[i]))

        logger.info(f"Successfully generated embeddings for {len(chunk_embedding_pairs)} chunks.")
        return chunk_embedding_pairs


if __name__ == '__main__':
    import asyncio
    # Basic logging for the example
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Dummy GeminiLLMClient for testing purposes (if not importing the real one)
    class MockGeminiLLMClient: # Changed to MockGeminiLLMClient
        def __init__(self, embedding_dim=5):
            self.embedding_dim = embedding_dim
            logger.info("MockGeminiLLMClient initialized.") # Changed name

        async def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
            logger.info(f"MockGeminiLLMClient generating embeddings for {len(texts)} texts (model: {model_name}).")
            if "error_trigger" in texts:
                raise EmbeddingGenerationError("Simulated LLM client error for 'error_trigger'")
            embeddings = []
            for i, text in enumerate(texts):
                embeddings.append([float(j + len(text) % 10) for j in range(self.embedding_dim)])
            return embeddings
        async def close_session(self):
            logger.info("MockGeminiLLMClient session closed.")

    async def main_test():
        mock_client = MockGeminiLLMClient(embedding_dim=3) # Use MockGeminiLLMClient
        embed_generator = EmbeddingGenerator(llm_client=mock_client) # Pass the mock client

        chunks_data = [
            ChunkModel(id="chunk1", document_id="doc1", text_content="Hello world", metadata={"source": "test.txt"}),
            ChunkModel(id="chunk2", document_id="doc1", text_content="Ava is helpful", metadata={"source": "test.txt"}),
            ChunkModel(id="chunk3", document_id="doc2", text_content="   ", metadata={"source": "empty.txt"}),
            ChunkModel(id="chunk4", document_id="doc2", text_content="Testing embeddings", metadata={"source": "another.txt"}),
        ]
        print("\n--- Testing EmbeddingGenerator ---")
        results = await embed_generator.generate_embeddings_for_chunks(chunks_data)
        print(f"\nGenerated {len(results)} embedding pairs:")
        for chunk_model_res, embedding in results: # Renamed to avoid conflict
            print(f"  Chunk ID: {chunk_model_res.id}, Text: '{chunk_model_res.text_content}', Embedding (first 3): {embedding[:3]}...")
            assert len(embedding) == mock_client.embedding_dim
        assert len(results) == 3
        # ... (rest of your __main__ test block) ...
        await mock_client.close_session()
        print("\n--- Embedding Generation Tests Complete ---")

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main_test())
    # ... (rest of your __main__ exception handling) ...
    finally:
        # ... (loop closing logic) ...
        if not loop.is_closed(): # type: ignore
            loop.close() # type: ignore
        asyncio.set_event_loop(None)