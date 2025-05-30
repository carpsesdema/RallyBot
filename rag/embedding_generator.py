# rag/embedding_generator.py
import logging
from typing import List, Tuple, TYPE_CHECKING, Optional

# Attempt to import necessary modules
try:
    from models import ChunkModel
    from utils import EmbeddingGenerationError

    if TYPE_CHECKING:  # To avoid circular dependency runtime error but allow type hinting
        from llm_interface.ollama_client import OllamaLLMClient
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


    # Dummy OllamaLLMClient for type hinting in standalone mode
    if TYPE_CHECKING:
        class OllamaLLMClient:
            async def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[
                List[float]]:
                raise NotImplementedError

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    def __init__(self, llm_client: 'OllamaLLMClient'):  # Use string literal for OllamaLLMClient for forward reference
        self.llm_client = llm_client
        logger.info("EmbeddingGenerator initialized.")

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
            embedding_vectors = await self.llm_client.generate_embeddings(texts_to_embed)
        except Exception as e:  # Catch specific errors from llm_client if possible, e.g., LLMClientError
            logger.error(f"Error generating embeddings via LLM client: {e}", exc_info=True)
            raise EmbeddingGenerationError(f"Failed to generate embeddings: {e}") from e

        if len(embedding_vectors) != len(valid_chunks):
            # This should ideally not happen if llm_client.generate_embeddings works correctly
            # or raises an error for partial failures.
            msg = (f"Mismatch between number of embeddings received ({len(embedding_vectors)}) "
                   f"and number of valid chunks processed ({len(valid_chunks)}).")
            logger.error(msg)
            raise EmbeddingGenerationError(msg)

        chunk_embedding_pairs: List[Tuple[ChunkModel, List[float]]] = []
        for i, chunk in enumerate(valid_chunks):
            chunk_embedding_pairs.append((chunk, embedding_vectors[i]))

        logger.info(f"Successfully generated embeddings for {len(chunk_embedding_pairs)} chunks.")
        return chunk_embedding_pairs


if __name__ == '__main__':
    import asyncio
    from typing import Optional  # Add this for the dummy client

    # Basic logging for the example
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    # Dummy OllamaLLMClient for testing purposes
    class MockOllamaLLMClient:
        def __init__(self, embedding_dim=5):  # Small embedding dim for test
            self.embedding_dim = embedding_dim
            logger.info("MockOllamaLLMClient initialized.")

        async def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
            logger.info(f"MockOllamaLLMClient generating embeddings for {len(texts)} texts (model: {model_name}).")
            if "error_trigger" in texts:
                raise EmbeddingGenerationError("Simulated LLM client error for 'error_trigger'")

            embeddings = []
            for i, text in enumerate(texts):
                # Simulate varying length embeddings for different texts if needed, or fixed.
                # For simplicity, using a fixed pattern based on text length.
                embeddings.append([float(j + len(text) % 10) for j in range(self.embedding_dim)])
            return embeddings

        async def close_session(self):  # Add dummy close_session
            logger.info("MockOllamaLLMClient session closed.")


    async def main_test():
        mock_client = MockOllamaLLMClient(embedding_dim=3)  # Use a small dimension for easy viewing
        embed_generator = EmbeddingGenerator(llm_client=mock_client)

        # Create some dummy ChunkModels
        chunks_data = [
            ChunkModel(id="chunk1", document_id="doc1", text_content="Hello world", metadata={"source": "test.txt"}),
            ChunkModel(id="chunk2", document_id="doc1", text_content="Ava is helpful", metadata={"source": "test.txt"}),
            ChunkModel(id="chunk3", document_id="doc2", text_content="   ", metadata={"source": "empty.txt"}),
            # Empty content
            ChunkModel(id="chunk4", document_id="doc2", text_content="Testing embeddings",
                       metadata={"source": "another.txt"}),
        ]

        print("\n--- Testing EmbeddingGenerator ---")
        results = await embed_generator.generate_embeddings_for_chunks(chunks_data)

        print(f"\nGenerated {len(results)} embedding pairs:")
        for chunk, embedding in results:
            print(f"  Chunk ID: {chunk.id}, Text: '{chunk.text_content}', Embedding (first 3): {embedding[:3]}...")
            assert len(embedding) == mock_client.embedding_dim

        assert len(results) == 3  # Chunk3 should be skipped

        print("\n--- Testing with no chunks ---")
        results_empty = await embed_generator.generate_embeddings_for_chunks([])
        assert len(results_empty) == 0
        print(f"Result for empty list: {results_empty}")

        print("\n--- Testing with error trigger ---")
        error_chunks = [
            ChunkModel(id="err_chunk1", document_id="err_doc", text_content="error_trigger", metadata={}),
            ChunkModel(id="err_chunk2", document_id="err_doc", text_content="some other content", metadata={}),
        ]
        try:
            await embed_generator.generate_embeddings_for_chunks(error_chunks)
        except EmbeddingGenerationError as e:
            print(f"Correctly caught EmbeddingGenerationError: {e}")

        await mock_client.close_session()  # Important to close if client manages resources
        print("\n--- Embedding Generation Tests Complete ---")


    # asyncio.run(main_test()) # Requires Python 3.7+
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main_test())
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user.")
    finally:
        # Ensure event loop is closed properly
        pending = asyncio.all_tasks(loop=loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending))
        # Only close the loop if it's not already closed, and set a new one to None
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)