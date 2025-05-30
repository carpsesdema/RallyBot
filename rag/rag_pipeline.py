# rag/rag_pipeline.py
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING

# Attempt to import necessary components
try:
    from config import Settings
    from models import DocumentModel, ChunkModel  # Assuming these are in global models.py
    from utils import RAGPipelineError, DocumentLoadingError, TextSplittingError, EmbeddingGenerationError, \
        LLMClientError

    # For type hinting RAG components
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter as TextSplitter  # Alias for clarity
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import VectorStoreInterface, VectorStoreError  # Use the interface

    if TYPE_CHECKING:  # For OllamaLLMClient to avoid runtime circular if it imports RAG later
        from llm_interface.ollama_client import OllamaLLMClient

except ImportError as e:
    print(f"Import error in rag_pipeline.py: {e}. Some features might not work if run standalone.")


    # Define fallbacks or raise error if critical dependencies are missing for standalone execution
    class Settings:  # Dummy
        OLLAMA_CHAT_MODEL = "dummy_chat_model"


    class DocumentModel:
        pass


    class ChunkModel:
        pass


    class RAGPipelineError(Exception):
        pass


    class DocumentLoadingError(Exception):
        pass


    class TextSplittingError(Exception):
        pass


    class EmbeddingGenerationError(Exception):
        pass


    class LLMClientError(Exception):
        pass


    class DocumentLoader:
        pass


    class TextSplitter:
        pass


    class EmbeddingGenerator:
        pass


    class VectorStoreInterface:
        pass


    if TYPE_CHECKING:
        class OllamaLLMClient: pass

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self,
                 settings: Settings,
                 llm_client: 'OllamaLLMClient',
                 document_loader: DocumentLoader,
                 text_splitter: TextSplitter,
                 embedding_generator: EmbeddingGenerator,
                 vector_store: VectorStoreInterface):
        self.settings = settings
        self.llm_client = llm_client
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        logger.info("RAGPipeline initialized with all components.")

    async def ingest_documents_from_directory(self, directory_path_str: str) -> Tuple[int, int]:
        """
        Loads documents from a directory, splits them into chunks,
        generates embeddings, and adds them to the vector store.

        Args:
            directory_path_str: Path to the directory containing documents.

        Returns:
            A tuple: (number_of_documents_processed, total_number_of_chunks_created).

        Raises:
            RAGPipelineError: If any step in the ingestion process fails.
        """
        logger.info(f"Starting ingestion process for directory: {directory_path_str}")

        try:
            # 1. Load documents
            documents: List[DocumentModel] = self.document_loader.load_documents_from_directory(directory_path_str)
            if not documents:
                logger.warning(f"No documents found or loaded from directory: {directory_path_str}")
                return 0, 0
            logger.info(f"Loaded {len(documents)} documents.")

            # 2. Split documents into chunks
            all_chunks: List[ChunkModel] = []
            for doc in documents:
                if not doc.content or not doc.content.strip():
                    logger.warning(
                        f"Document {doc.id} (source: {doc.metadata.get('source_filename')}) is empty or whitespace only, skipping splitting.")
                    continue
                doc_chunks = self.text_splitter.split_document(doc)
                all_chunks.extend(doc_chunks)

            if not all_chunks:
                logger.warning(f"No chunks were created from the loaded documents in {directory_path_str}.")
                return len(documents), 0
            logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")

            # 3. Generate embeddings for chunks
            # EmbeddingGenerator now takes the llm_client in its constructor.
            # OllamaLLMClient for embeddings is passed via EmbeddingGenerator instance.
            chunk_embedding_pairs: List[Tuple[ChunkModel, List[float]]] = \
                await self.embedding_generator.generate_embeddings_for_chunks(all_chunks)

            if not chunk_embedding_pairs:
                logger.warning("No embeddings were generated for the chunks.")
                # This might happen if all chunks were empty after splitting, though unlikely if previous checks passed.
                return len(documents), len(all_chunks)  # Return 0 embeddings added but chunks were made

            logger.info(f"Generated embeddings for {len(chunk_embedding_pairs)} chunks.")

            # 4. Add chunks and embeddings to vector store
            self.vector_store.add_documents(chunk_embedding_pairs)
            logger.info(f"Added {len(chunk_embedding_pairs)} chunk-embedding pairs to the vector store.")

            # 5. Persist vector store
            self.vector_store.save()
            logger.info("Vector store saved successfully after ingestion.")

            num_docs_processed = len(documents)
            num_chunks_created = len(all_chunks)  # or len(chunk_embedding_pairs) if only counting embedded ones

            logger.info(
                f"Ingestion complete for {directory_path_str}. Processed: {num_docs_processed} docs, Created: {num_chunks_created} chunks.")
            return num_docs_processed, num_chunks_created

        except DocumentLoadingError as e:
            logger.error(f"Error loading documents: {e}", exc_info=True)
            raise RAGPipelineError(f"Document loading failed: {e}") from e
        except TextSplittingError as e:
            logger.error(f"Error splitting text: {e}", exc_info=True)
            raise RAGPipelineError(f"Text splitting failed: {e}") from e
        except EmbeddingGenerationError as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise RAGPipelineError(f"Embedding generation failed: {e}") from e
        except Exception as e:  # Catch other potential errors (e.g., VectorStore errors)
            logger.error(f"An unexpected error occurred during ingestion pipeline: {e}", exc_info=True)
            raise RAGPipelineError(f"Ingestion pipeline failed: {e}") from e

    async def query_with_rag(self, query_text: str, top_k_chunks: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Processes a query using RAG:
        1. Generates embedding for the query.
        2. Searches vector store for similar chunks.
        3. Constructs a prompt with context from retrieved chunks.
        4. Generates an answer using the LLM.

        Args:
            query_text: The user's query.
            top_k_chunks: The number of relevant chunks to retrieve.

        Returns:
            A tuple: (llm_answer_text, list_of_retrieved_chunk_metadata).

        Raises:
            RAGPipelineError: If any step in the RAG query process fails.
        """
        logger.info(f"Processing RAG query: '{query_text[:100]}...', top_k={top_k_chunks}")

        if self.vector_store.is_empty():
            logger.warning("Vector store is empty. Performing query without RAG context.")
            # Fallback: directly query LLM without context
            try:
                answer = await self.llm_client.generate_response(prompt=query_text)
                return answer, []
            except LLMClientError as e:
                logger.error(f"LLM client error during no-RAG fallback query: {e}", exc_info=True)
                raise RAGPipelineError(f"LLM query failed (no RAG context): {e}") from e

        try:
            # 1. Generate embedding for the query
            # The ollama_client is part of self.llm_client or accessible via self.embedding_generator.llm_client
            # Using self.embedding_generator.llm_client as it's specifically for embeddings
            query_embedding_list = await self.embedding_generator.llm_client.generate_embeddings([query_text])
            if not query_embedding_list or not query_embedding_list[0]:
                raise RAGPipelineError("Failed to generate embedding for the query.")
            query_embedding = query_embedding_list[0]
            logger.debug("Generated query embedding.")

            # 2. Search vector store
            relevant_chunks: List[ChunkModel] = await self.vector_store.search_similar_chunks(query_embedding,
                                                                                              top_k=top_k_chunks)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks from vector store.")

            # 3. Construct context and prompt
            if not relevant_chunks:
                logger.info("No relevant chunks found. Querying LLM without additional context.")
                context_string = "No specific context found."
                retrieved_chunks_details = []
            else:
                # Sort chunks by any relevance score if available, or use as is.
                # For now, order from vector store is assumed to be by relevance.
                context_string = "\n\n---\n\n".join([chunk.text_content for chunk in relevant_chunks])
                retrieved_chunks_details = [
                    {
                        "source_file": chunk.metadata.get("source_filename", "Unknown"),
                        "chunk_id": chunk.id,
                        "text_preview": chunk.text_content[:150] + "..."  # Preview of the chunk text
                    } for chunk in relevant_chunks
                ]

            prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context_string}\n\nQuestion: {query_text}\n\nAnswer:"
            logger.debug(f"Constructed prompt for LLM (context length {len(context_string)} chars).")

            # 4. Generate answer using LLM
            llm_answer = await self.llm_client.generate_response(
                prompt=prompt,
                model_name=self.settings.OLLAMA_CHAT_MODEL  # Use chat model specified in settings
            )
            logger.info(f"LLM generated answer. Length: {len(llm_answer)}")

            return llm_answer, retrieved_chunks_details

        except EmbeddingGenerationError as e:
            logger.error(f"Query embedding generation failed: {e}", exc_info=True)
            raise RAGPipelineError(f"Query embedding failed: {e}") from e
        except VectorStoreError as e:
            logger.error(f"Vector store search failed: {e}", exc_info=True)
            raise RAGPipelineError(f"Vector store search failed: {e}") from e
        except LLMClientError as e:  # Catch errors from the final answer generation
            logger.error(f"LLM client error during RAG answer generation: {e}", exc_info=True)
            raise RAGPipelineError(f"LLM answer generation failed: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during RAG query: {e}", exc_info=True)
            raise RAGPipelineError(f"RAG query failed: {e}") from e


if __name__ == '__main__':
    import asyncio
    from pathlib import Path

    # --- Mock Components for Standalone Test ---
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


    class MockSettings:
        OLLAMA_CHAT_MODEL = "test-chat-model"
        OLLAMA_EMBEDDING_MODEL = "test-embed-model"  # Used by mock Ollama client
        OLLAMA_EMBEDDING_DIMENSION = 3  # For mock vector store
        VECTOR_STORE_PATH = Path("./_test_rag_pipeline/mock_faiss.index")
        VECTOR_STORE_METADATA_PATH = Path("./_test_rag_pipeline/mock_faiss_metadata.pkl")


    class MockOllamaLLMClient:
        def __init__(self, settings):
            self.settings = settings
            self.embeddings_called = 0
            self.responses_called = 0

        async def generate_embeddings(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
            self.embeddings_called += 1
            logger.info(
                f"MockOllamaLLMClient: generate_embeddings called for {len(texts)} texts with model {model_name or self.settings.OLLAMA_EMBEDDING_MODEL}.")
            # Use configured dimension
            return [[float(i) + 0.1 * idx for i in range(self.settings.OLLAMA_EMBEDDING_DIMENSION)] for idx, text in
                    enumerate(texts)]

        async def generate_response(self, prompt: str, model_name: Optional[str] = None) -> str:
            self.responses_called += 1
            logger.info(
                f"MockOllamaLLMClient: generate_response called with model {model_name or self.settings.OLLAMA_CHAT_MODEL}.")
            if "what is rag" in prompt.lower():
                return "RAG is Retrieval Augmented Generation, a technique to improve LLM answers with external knowledge."
            return f"Mocked LLM response to: '{prompt[:50]}...'"

        async def close_session(self): pass


    class MockDocumentLoader:
        def load_documents_from_directory(self, directory_path_str: str) -> List[DocumentModel]:
            logger.info(f"MockDocumentLoader: loading from {directory_path_str}")
            if "empty_dir" in directory_path_str: return []
            return [
                DocumentModel(id="doc1", content="LangChain is a framework. It helps build LLM apps.",
                              metadata={"source_filename": "doc1.txt"}),
                DocumentModel(id="doc2", content="FAISS is a library for vector search.",
                              metadata={"source_filename": "doc2.txt"})
            ]


    class MockTextSplitter:
        def split_document(self, document: DocumentModel) -> List[ChunkModel]:
            logger.info(f"MockTextSplitter: splitting doc {document.id}")
            # Simple split: one chunk per sentence (approx)
            parts = document.content.split(". ")
            chunks = []
            for i, part_text in enumerate(parts):
                if part_text.strip():
                    chunks.append(ChunkModel(id=f"{document.id}-c{i}", document_id=document.id,
                                             text_content=part_text.strip() + ".", metadata=document.metadata.copy()))
            return chunks


    class MockEmbeddingGenerator:  # Uses the llm_client passed to it
        def __init__(self, llm_client: MockOllamaLLMClient):
            self.llm_client = llm_client

        async def generate_embeddings_for_chunks(self, chunks: List[ChunkModel]) -> List[
            Tuple[ChunkModel, List[float]]]:
            logger.info(f"MockEmbeddingGenerator: generating embeddings for {len(chunks)} chunks.")
            texts = [c.text_content for c in chunks]
            if not texts: return []
            embeddings = await self.llm_client.generate_embeddings(texts)
            return list(zip(chunks, embeddings))


    class MockVectorStore(VectorStoreInterface):
        def __init__(self, settings):  # Add settings for file paths if needed for save/load
            self.settings = settings
            self.chunks_with_embeddings: List[Tuple[ChunkModel, List[float]]] = []
            self.save_called = 0
            self.load_called = 0
            logger.info("MockVectorStore initialized.")

        def add_documents(self, chunk_embeddings: List[Tuple[ChunkModel, List[float]]]) -> None:
            logger.info(f"MockVectorStore: adding {len(chunk_embeddings)} documents.")
            self.chunks_with_embeddings.extend(chunk_embeddings)

        async def search_similar_chunks(self, query_embedding: List[float], top_k: int) -> List[ChunkModel]:
            logger.info(f"MockVectorStore: searching with top_k={top_k}.")
            # Simple mock: return first top_k chunks, actual similarity not checked
            return [ce[0] for ce in self.chunks_with_embeddings[:top_k]]

        def save(self) -> None:
            self.save_called += 1
            logger.info("MockVectorStore: save called.")

        def load(self) -> None:
            self.load_called += 1
            logger.info("MockVectorStore: load called.")

        def is_empty(self) -> bool:
            is_e = not self.chunks_with_embeddings
            logger.info(f"MockVectorStore: is_empty called, result: {is_e}")
            return is_e


    async def test_rag_pipeline():
        mock_settings = MockSettings()

        # Ensure test directory exists for mock vector store if it tries to use paths
        mock_settings.VECTOR_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

        mock_llm_client = MockOllamaLLMClient(settings=mock_settings)
        mock_doc_loader = MockDocumentLoader()
        mock_text_splitter = MockTextSplitter()
        mock_embed_generator = MockEmbeddingGenerator(llm_client=mock_llm_client)
        mock_vector_store = MockVectorStore(settings=mock_settings)

        pipeline = RAGPipeline(
            settings=mock_settings,
            llm_client=mock_llm_client,
            document_loader=mock_doc_loader,
            text_splitter=mock_text_splitter,
            embedding_generator=mock_embed_generator,
            vector_store=mock_vector_store
        )

        print("\n--- Testing Ingestion ---")
        docs_processed, chunks_created = await pipeline.ingest_documents_from_directory("./dummy_docs")
        print(f"Ingestion result: Docs processed={docs_processed}, Chunks created={chunks_created}")
        assert docs_processed == 2
        assert chunks_created > 0  # Depends on mock splitter logic
        assert mock_vector_store.save_called == 1
        assert not mock_vector_store.is_empty()

        print("\n--- Testing Query with RAG ---")
        query = "What is RAG?"
        answer, sources = await pipeline.query_with_rag(query_text=query, top_k_chunks=2)
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"Sources: {sources}")
        assert "Retrieval Augmented Generation" in answer
        assert len(sources) <= 2  # Mock search might return fewer if fewer available

        print("\n--- Testing Query on Empty Store (after clearing mock store) ---")
        mock_vector_store.chunks_with_embeddings = []  # Manually empty it
        assert mock_vector_store.is_empty()
        answer_no_rag, sources_no_rag = await pipeline.query_with_rag(query_text="Tell me a story.", top_k_chunks=2)
        print(f"Query (no RAG): Tell me a story.")
        print(f"Answer (no RAG): {answer_no_rag}")
        assert not sources_no_rag
        assert "Mocked LLM response" in answer_no_rag  # Should hit the fallback

        print("\n--- RAG Pipeline Tests Complete ---")


    # Run the test
    # asyncio.run(test_rag_pipeline()) # For Python 3.7+
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(test_rag_pipeline())
    except KeyboardInterrupt:
        logger.info("Test run interrupted by user.")
    finally:
        pending = asyncio.all_tasks(loop=loop)
        if pending:
            loop.run_until_complete(asyncio.gather(*pending))
        if not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)