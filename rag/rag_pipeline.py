# rag/rag_pipeline.py
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING

try:
    from config import Settings, settings
    from models import DocumentModel, ChunkModel
    from utils import RAGPipelineError, DocumentLoadingError, TextSplittingError, EmbeddingGenerationError, LLMClientError, VectorStoreError
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import VectorStoreInterface
    if TYPE_CHECKING:
        from llm_interface.ollama_client import OllamaLLMClient
        from llm_interface.gemini_client import GeminiLLMClient # If you use it directly
except ImportError as e:
    print(f"Import error in rag_pipeline.py: {e}. Some features might not work if run standalone.")
    class Settings: OLLAMA_CHAT_MODEL = "dummy_chat_model"; LLM_PROVIDER = "ollama" # Added LLM_PROVIDER
    class DocumentModel: pass
    class ChunkModel: pass
    class RAGPipelineError(Exception): pass
    class DocumentLoadingError(Exception): pass
    class TextSplittingError(Exception): pass
    class EmbeddingGenerationError(Exception): pass
    class LLMClientError(Exception): pass
    class VectorStoreError(Exception): pass # Added
    class DocumentLoader: pass
    class TextSplitter: pass
    class EmbeddingGenerator: pass
    class VectorStoreInterface:
        def is_empty(self): return True
        async def search_similar_chunks(self, qe, tk): return []
    if TYPE_CHECKING:
        class OllamaLLMClient: pass
        class GeminiLLMClient: pass

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self,
                 settings: Settings,
                 llm_client, # Can be OllamaLLMClient or GeminiLLMClient instance
                 document_loader: DocumentLoader,
                 text_splitter: TextSplitter,
                 embedding_generator: EmbeddingGenerator,
                 vector_store: VectorStoreInterface):
        self.settings = settings
        self.llm_client = llm_client # This will be the client chosen at startup
        self.document_loader = document_loader
        self.text_splitter = text_splitter
        self.embedding_generator = embedding_generator # This uses the llm_client for embeddings
        self.vector_store = vector_store
        logger.info(f"RAGPipeline initialized with LLM provider: {settings.LLM_PROVIDER}.")

    async def ingest_documents_from_directory(self, directory_path_str: str) -> Tuple[int, int]:
        logger.info(f"Starting ingestion process for directory: {directory_path_str}")
        try:
            documents: List[DocumentModel] = self.document_loader.load_documents_from_directory(directory_path_str)
            if not documents:
                logger.warning(f"No documents found or loaded from directory: {directory_path_str}")
                return 0, 0
            logger.info(f"Loaded {len(documents)} documents.")

            all_chunks: List[ChunkModel] = []
            for doc in documents:
                if not doc.content or not doc.content.strip():
                    logger.warning(f"Document {doc.id} (source: {doc.metadata.get('source_filename')}) is empty, skipping.")
                    continue
                doc_chunks = self.text_splitter.split_document(doc)
                all_chunks.extend(doc_chunks)

            if not all_chunks:
                logger.warning(f"No chunks created from documents in {directory_path_str}.")
                return len(documents), 0
            logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks.")

            chunk_embedding_pairs: List[Tuple[ChunkModel, List[float]]] = \
                await self.embedding_generator.generate_embeddings_for_chunks(all_chunks)

            if not chunk_embedding_pairs:
                logger.warning("No embeddings generated for chunks.")
                return len(documents), len(all_chunks)
            logger.info(f"Generated embeddings for {len(chunk_embedding_pairs)} chunks.")

            self.vector_store.add_documents(chunk_embedding_pairs)
            logger.info(f"Added {len(chunk_embedding_pairs)} pairs to vector store.")
            self.vector_store.save()
            logger.info("Vector store saved after ingestion.")
            return len(documents), len(all_chunks)
        except DocumentLoadingError as e:
            logger.error(f"Error loading documents: {e}", exc_info=True)
            raise RAGPipelineError(f"Document loading failed: {e}") from e
        except TextSplittingError as e:
            logger.error(f"Error splitting text: {e}", exc_info=True)
            raise RAGPipelineError(f"Text splitting failed: {e}") from e
        except EmbeddingGenerationError as e:
            logger.error(f"Error generating embeddings: {e}", exc_info=True)
            raise RAGPipelineError(f"Embedding generation failed: {e}") from e
        except VectorStoreError as e: # Catch VectorStore specific errors
            logger.error(f"Vector store error during ingestion: {e}", exc_info=True)
            raise RAGPipelineError(f"Vector store operation failed during ingestion: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}", exc_info=True)
            raise RAGPipelineError(f"Ingestion pipeline failed: {e}") from e

    async def query_with_rag(self, query_text: str, top_k_chunks: int = 3, model_name_override: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        logger.info(f"Processing RAG query: '{query_text[:100]}...', top_k={top_k_chunks}, model_override='{model_name_override}'")

        final_model_name = model_name_override # Use override if provided
        if not final_model_name: # If no override, use the default based on provider
            if self.settings.LLM_PROVIDER == "gemini":
                final_model_name = self.settings.GEMINI_MODEL
            else: # Default to Ollama
                final_model_name = self.settings.OLLAMA_CHAT_MODEL
        logger.info(f"Effective model for LLM generation: {final_model_name}")


        if self.vector_store.is_empty():
            logger.warning("Vector store is empty. Querying LLM directly without RAG context.")
            try:
                answer = await self.llm_client.generate_response(prompt=query_text, model_name=final_model_name)
                return answer, []
            except LLMClientError as e:
                logger.error(f"LLM client error during no-RAG fallback: {e}", exc_info=True)
                raise RAGPipelineError(f"LLM query failed (no RAG context): {e}") from e

        try:
            # 1. Generate query embedding (uses the llm_client's embedding capability)
            # The embedding model is typically fixed per provider or configured in llm_client itself
            query_embedding_list = await self.embedding_generator.llm_client.generate_embeddings([query_text])
            if not query_embedding_list or not query_embedding_list[0]:
                raise RAGPipelineError("Failed to generate embedding for the query.")
            query_embedding = query_embedding_list[0]
            logger.debug("Generated query embedding.")

            # 2. Search vector store
            relevant_chunks: List[ChunkModel] = await self.vector_store.search_similar_chunks(query_embedding, top_k=top_k_chunks)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")

            # 3. Construct context and prompt
            context_string = "No specific context found."
            retrieved_chunks_details = []
            if relevant_chunks:
                context_string = "\n\n---\n\n".join([chunk.text_content for chunk in relevant_chunks])
                retrieved_chunks_details = [
                    {
                        "source_file": chunk.metadata.get("source_filename", "Unknown"),
                        "chunk_id": chunk.id,
                        "text_preview": chunk.text_content[:150] + "..."
                    } for chunk in relevant_chunks
                ]

            prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context_string}\n\nQuestion: {query_text}\n\nAnswer:"
            logger.debug(f"Constructed prompt for LLM (context length {len(context_string)} chars).")

            # 4. Generate answer using LLM with the determined model name
            llm_answer = await self.llm_client.generate_response(prompt=prompt, model_name=final_model_name)
            logger.info(f"LLM generated answer. Length: {len(llm_answer)}")
            return llm_answer, retrieved_chunks_details

        except EmbeddingGenerationError as e:
            logger.error(f"Query embedding generation failed: {e}", exc_info=True)
            raise RAGPipelineError(f"Query embedding failed: {e}") from e
        except VectorStoreError as e:
            logger.error(f"Vector store search failed: {e}", exc_info=True)
            raise RAGPipelineError(f"Vector store search failed: {e}") from e
        except LLMClientError as e:
            logger.error(f"LLM client error during RAG answer generation: {e}", exc_info=True)
            raise RAGPipelineError(f"LLM answer generation failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during RAG query: {e}", exc_info=True)
            raise RAGPipelineError(f"RAG query failed: {e}") from e

if __name__ == '__main__':
    import asyncio
    # Mock components (simplified for brevity, assume they exist from previous examples)
    logging.basicConfig(level=logging.DEBUG)
    class MockSettings:
        LLM_PROVIDER = "ollama"
        OLLAMA_CHAT_MODEL = "test-chat-model"
        GEMINI_MODEL = "test-gemini-model"
        OLLAMA_EMBEDDING_MODEL = "test-embed-model"
        OLLAMA_EMBEDDING_DIMENSION = 3
        VECTOR_STORE_PATH = Path("./_test_rag_pipeline/mock_faiss.index")
        VECTOR_STORE_METADATA_PATH = Path("./_test_rag_pipeline/mock_faiss_metadata.pkl")
    class MockOllamaLLMClient:
        def __init__(self, settings): self.settings = settings
        async def generate_embeddings(self, texts, model_name=None):
            return [[0.1]*settings.OLLAMA_EMBEDDING_DIMENSION for _ in texts]
        async def generate_response(self, prompt, model_name=None):
            return f"Mock response for '{prompt[:30]}...' with model {model_name}"
    class MockDocumentLoader:
        def load_documents_from_directory(self,p): return [DocumentModel(id="d1",content="Test doc.",metadata={})]
    class MockTextSplitter:
        def split_document(self,d): return [ChunkModel(id="c1",document_id=d.id,text_content=d.content,metadata=d.metadata)]
    class MockEmbeddingGenerator:
        def __init__(self, llm_client): self.llm_client = llm_client
        async def generate_embeddings_for_chunks(self, chunks):
            texts = [c.text_content for c in chunks]
            if not texts: return []
            embeddings = await self.llm_client.generate_embeddings(texts)
            return list(zip(chunks, embeddings))
    class MockVectorStore(VectorStoreInterface):
        def __init__(self,s): self.chunks_with_embeddings=[]; self.is_empty_val=True
        def add_documents(self,ce): self.chunks_with_embeddings.extend(ce); self.is_empty_val=False
        async def search_similar_chunks(self,qe,tk): return [c for c,e in self.chunks_with_embeddings[:tk]]
        def save(self): pass
        def load(self): pass
        def is_empty(self): return self.is_empty_val

    async def test_rag_pipeline_model_override():
        mock_settings = MockSettings()
        mock_llm = MockOllamaLLMClient(mock_settings)
        pipeline = RAGPipeline(
            settings=mock_settings, llm_client=mock_llm,
            document_loader=MockDocumentLoader(), text_splitter=MockTextSplitter(),
            embedding_generator=MockEmbeddingGenerator(llm_client=mock_llm),
            vector_store=MockVectorStore(mock_settings)
        )
        # Ingest something so vector store is not empty
        await pipeline.ingest_documents_from_directory("./dummy_docs_for_ingest")

        query = "What is tennis?"
        print("\n--- Testing Query with RAG (default model) ---")
        answer, sources = await pipeline.query_with_rag(query_text=query)
        print(f"Answer (default): {answer}")
        assert mock_settings.OLLAMA_CHAT_MODEL in answer # Check if default Ollama model was used

        print("\n--- Testing Query with RAG (model_name_override) ---")
        override_model = "super-special-model:latest"
        answer_override, sources_override = await pipeline.query_with_rag(query_text=query, model_name_override=override_model)
        print(f"Answer (override): {answer_override}")
        assert override_model in answer_override

        # Test Gemini default if provider was gemini
        mock_settings.LLM_PROVIDER = "gemini"
        pipeline_gemini_default = RAGPipeline(
            settings=mock_settings, llm_client=mock_llm, # Still using mock ollama client for mechanics
            document_loader=MockDocumentLoader(), text_splitter=MockTextSplitter(),
            embedding_generator=MockEmbeddingGenerator(llm_client=mock_llm),
            vector_store=MockVectorStore(mock_settings)
        )
        await pipeline_gemini_default.ingest_documents_from_directory("./dummy_docs_for_ingest2")
        answer_gemini, _ = await pipeline_gemini_default.query_with_rag(query_text=query)
        print(f"Answer (Gemini default): {answer_gemini}")
        assert mock_settings.GEMINI_MODEL in answer_gemini


    loop = asyncio.get_event_loop()
    try: loop.run_until_complete(test_rag_pipeline_model_override())
    except KeyboardInterrupt: logger.info("Test interrupted.")
    finally:
        pending = asyncio.all_tasks(loop=loop)
        if pending: loop.run_until_complete(asyncio.gather(*pending))
        if not loop.is_closed(): loop.close()
        asyncio.set_event_loop(None)