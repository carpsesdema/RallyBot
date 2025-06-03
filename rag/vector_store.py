# rag/vector_store.py
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Protocol, runtime_checkable, Dict

import numpy as np

# FAISS import
try:
    import faiss
except ImportError:
    faiss = None

# Attempt to import models and custom exceptions
try:
    from models import ChunkModel
    from utils import VectorStoreError
    # config import removed as app_settings is not directly used in this file anymore
    # If it was used, ensure it's correctly sourced.
except ImportError:
    class ChunkModel:
        def __init__(self, id: str, document_id: str, text_content: str, metadata: dict):
            self.id = id; self.document_id = document_id; self.text_content = text_content; self.metadata = metadata
    class VectorStoreError(Exception): pass
    if faiss is None: print("WARNING: FAISS library not found. FAISSVectorStore will not be functional.")

logger = logging.getLogger(__name__)

@runtime_checkable
class VectorStoreInterface(Protocol):
    """Interface for vector store operations."""
    def add_documents(self, chunk_embeddings: List[Tuple[ChunkModel, List[float]]]) -> None: ...
    async def search_similar_chunks(self, query_embedding: List[float], top_k: int) -> List[ChunkModel]: ...
    def save(self) -> None: ...
    def load(self) -> None: ...
    def is_empty(self) -> bool: ...

# FAISSVectorStore now explicitly inherits from VectorStoreInterface
class FAISSVectorStore(VectorStoreInterface):
    """A vector store implementation using FAISS."""
    def __init__(self,
                 embedding_dimension: int,
                 index_file_path: Path,
                 metadata_file_path: Path):
        if faiss is None:
            raise ImportError("FAISS library is not installed. Please install faiss-cpu or faiss-gpu.")

        self.embedding_dimension: int = embedding_dimension
        self.index_file_path: Path = index_file_path
        self.metadata_file_path: Path = metadata_file_path

        self.index: Optional[faiss.Index] = None
        # This list will store ChunkModels in the order they are added to FAISS.
        # This is CRUCIAL for correctly mapping FAISS integer indices back to ChunkModels.
        self.stored_chunk_models_ordered: List[ChunkModel] = []
        # This dictionary is for quick metadata lookup if needed, but not primary for search retrieval.
        self.chunk_id_to_metadata: Dict[str, Dict] = {} # Example: could store just metadata if ChunkModel is heavy

        self._initialize_store()
        logger.info(
            f"FAISSVectorStore initialized. Index path: {index_file_path}, Metadata path: {metadata_file_path}, Dimension: {embedding_dimension}")

    def _initialize_store(self):
        """Initializes an empty FAISS index and clears metadata lists."""
        self.index = faiss.IndexFlatL2(self.embedding_dimension)
        self.stored_chunk_models_ordered = []
        self.chunk_id_to_metadata = {} # Reset this too
        logger.info(f"Initialized a new empty FAISS IndexFlatL2 with dimension {self.embedding_dimension}.")

    def add_documents(self, chunk_embeddings: List[Tuple[ChunkModel, List[float]]]) -> None:
        if not chunk_embeddings:
            logger.warning("No chunk embeddings provided to add_documents.")
            return
        if self.index is None:
            logger.error("FAISS index is not initialized. Cannot add documents.")
            raise VectorStoreError("FAISS index not initialized.")

        embeddings_to_add_np_list = []
        new_chunks_to_store_ordered = []

        for chunk_model, embedding_vector in chunk_embeddings:
            if len(embedding_vector) != self.embedding_dimension:
                msg = (f"Embedding dimension mismatch for chunk {chunk_model.id}. "
                       f"Expected {self.embedding_dimension}, got {len(embedding_vector)}.")
                logger.error(msg); raise VectorStoreError(msg)

            # Optional: Check for duplicates if chunk_model.id should be unique in the store
            # if chunk_model.id in self.chunk_id_to_metadata:
            #     logger.warning(f"Chunk ID {chunk_model.id} already exists. Skipping re-add. Consider updating logic if updates are needed.")
            #     continue

            embeddings_to_add_np_list.append(np.array(embedding_vector, dtype=np.float32))
            new_chunks_to_store_ordered.append(chunk_model)
            self.chunk_id_to_metadata[chunk_model.id] = chunk_model.metadata # Store metadata by ID

        if not embeddings_to_add_np_list:
            logger.info("No new, valid embeddings to add after filtering.")
            return

        try:
            embeddings_np = np.array(embeddings_to_add_np_list).astype(np.float32)
            if embeddings_np.ndim == 1 and len(embeddings_to_add_np_list) == 1: # Single embedding
                 embeddings_np = np.expand_dims(embeddings_np, axis=0)
            elif embeddings_np.ndim == 1 and len(embeddings_to_add_np_list) > 1 : # Should not happen if list of lists
                 logger.error("Unexpected shape for embeddings_np before adding to FAISS.")
                 raise VectorStoreError("Embedding array shape error.")


            self.index.add(embeddings_np)
            self.stored_chunk_models_ordered.extend(new_chunks_to_store_ordered) # Add to ordered list
            logger.info(
                f"Added {len(embeddings_to_add_np_list)} new embeddings. "
                f"Total index size: {self.index.ntotal}, "
                f"Total ordered chunks stored: {len(self.stored_chunk_models_ordered)}")
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS index: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to add embeddings to FAISS: {e}") from e

    async def search_similar_chunks(self, query_embedding: List[float], top_k: int) -> List[ChunkModel]:
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search attempted on an uninitialized or empty FAISS index.")
            return []
        if len(query_embedding) != self.embedding_dimension:
            msg = (f"Query embedding dimension mismatch. Expected {self.embedding_dimension}, got {len(query_embedding)}.")
            logger.error(msg); raise VectorStoreError(msg)

        query_np = np.array([query_embedding]).astype(np.float32)
        try:
            actual_k = min(top_k, self.index.ntotal) # Cannot request more neighbors than exist
            if actual_k == 0: return [] # No items in index to search

            logger.debug(f"Searching FAISS index with top_k={actual_k}. Index size: {self.index.ntotal}")
            distances, indices = self.index.search(query_np, k=actual_k)
        except Exception as e:
            logger.error(f"Error during FAISS search: {e}", exc_info=True)
            raise VectorStoreError(f"FAISS search failed: {e}") from e

        retrieved_chunks: List[ChunkModel] = []
        if indices.size > 0:
            for faiss_idx in indices[0]: # indices[0] contains the list of indices for the first query vector
                if 0 <= faiss_idx < len(self.stored_chunk_models_ordered):
                    retrieved_chunks.append(self.stored_chunk_models_ordered[faiss_idx])
                else:
                    # This should ideally not happen if FAISS returns valid indices relative to ntotal
                    logger.warning(f"FAISS returned index {faiss_idx} which is out of bounds for "
                                   f"stored_chunk_models_ordered (len {len(self.stored_chunk_models_ordered)}).")
        logger.info(f"FAISS search returned {len(retrieved_chunks)} chunks.")
        return retrieved_chunks

    def save(self) -> None:
        if self.index is None:
            logger.warning("Attempted to save an uninitialized FAISS index. Nothing to save."); return
        try:
            self.index_file_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_file_path))
            logger.info(f"FAISS index saved to {self.index_file_path}")

            self.metadata_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Save both the ordered list of ChunkModels and the ID-to-metadata dict
            data_to_save = {
                "stored_chunk_models_ordered": self.stored_chunk_models_ordered,
                "chunk_id_to_metadata": self.chunk_id_to_metadata
            }
            with open(self.metadata_file_path, "wb") as f:
                pickle.dump(data_to_save, f)
            logger.info(f"Vector store metadata (ordered chunks and ID map) saved to {self.metadata_file_path}")
        except Exception as e:
            logger.error(f"Error saving FAISSVectorStore state: {e}", exc_info=True)
            raise VectorStoreError(f"Failed to save FAISSVectorStore: {e}") from e

    def load(self) -> None:
        if not self.index_file_path.exists():
            logger.warning(f"FAISS index file not found at {self.index_file_path}. Initializing new store.")
            self._initialize_store(); return
        if not self.metadata_file_path.exists():
            logger.warning(f"Metadata file not found at {self.metadata_file_path}. Initializing new store.")
            self._initialize_store(); return
        try:
            self.index = faiss.read_index(str(self.index_file_path))
            logger.info(f"FAISS index loaded from {self.index_file_path}. Index size: {self.index.ntotal if self.index else 'None'}")

            with open(self.metadata_file_path, "rb") as f:
                loaded_data = pickle.load(f)
            self.stored_chunk_models_ordered = loaded_data.get("stored_chunk_models_ordered", [])
            self.chunk_id_to_metadata = loaded_data.get("chunk_id_to_metadata", {})
            logger.info(
                f"Vector store metadata loaded. {len(self.stored_chunk_models_ordered)} ordered chunks, "
                f"{len(self.chunk_id_to_metadata)} ID metadata entries.")

            if self.index and self.index.d != self.embedding_dimension:
                msg = (f"Loaded FAISS index dimension ({self.index.d}) does not match "
                       f"configured embedding dimension ({self.embedding_dimension}).")
                logger.error(msg); self._initialize_store(); raise VectorStoreError(msg)

            # Sanity check: number of items in index vs. loaded ordered chunks
            if self.index and self.index.ntotal != len(self.stored_chunk_models_ordered):
                logger.warning(
                    f"Mismatch after load: FAISS index size ({self.index.ntotal}) and "
                    f"number of stored_chunk_models_ordered ({len(self.stored_chunk_models_ordered)}). "
                    "Store may be inconsistent. Consider re-ingesting if issues arise."
                )
        except Exception as e:
            logger.error(f"Error loading FAISSVectorStore state: {e}. Initializing a new empty store.", exc_info=True)
            self._initialize_store()

    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if faiss is None: logger.error("FAISS library not available. Skipping FAISSVectorStore tests.")
    else:
        DIMENSION = 3
        TEST_INDEX_FILE = Path("./_test_vector_store/test_faiss.index")
        TEST_METADATA_FILE = Path("./_test_vector_store/test_faiss_metadata.pkl")
        TEST_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
        if TEST_INDEX_FILE.exists(): TEST_INDEX_FILE.unlink()
        if TEST_METADATA_FILE.exists(): TEST_METADATA_FILE.unlink()

        async def run_tests():
            store = FAISSVectorStore(DIMENSION, TEST_INDEX_FILE, TEST_METADATA_FILE)
            print(f"Store initially empty: {store.is_empty()}"); assert store.is_empty()

            chunk1 = ChunkModel(id="c1", document_id="d1", text_content="Hello there", metadata={"src": "doc1.txt"})
            emb1 = [0.1, 0.2, 0.3]
            chunk2 = ChunkModel(id="c2", document_id="d1", text_content="General Kenobi", metadata={"src": "doc1.txt"})
            emb2 = [0.7, 0.8, 0.9]
            chunk3 = ChunkModel(id="c3", document_id="d2", text_content="Ava is an AI", metadata={"src": "doc2.txt"})
            emb3 = [0.15, 0.25, 0.35] # Similar to chunk1

            store.add_documents([(chunk1, emb1), (chunk2, emb2)])
            print(f"Store empty after adding 2 docs: {store.is_empty()}"); assert not store.is_empty()
            assert store.index.ntotal == 2
            assert len(store.stored_chunk_models_ordered) == 2
            assert len(store.chunk_id_to_metadata) == 2

            store.add_documents([(chunk3, emb3)])
            assert store.index.ntotal == 3
            assert len(store.stored_chunk_models_ordered) == 3

            print("\nSearching for vector similar to emb1 (chunk1)...")
            query_emb_similar_to_c1 = [0.11, 0.21, 0.31]
            results = await store.search_similar_chunks(query_emb_similar_to_c1, top_k=2)
            print(f"Search results: {[(r.id, r.text_content) for r in results]}")
            assert len(results) == 2
            assert results[0].id == "c1" or results[0].id == "c3" # Order depends on exact similarity
            if len(results) > 1: assert results[1].id == "c3" or results[1].id == "c1"

            store.save()
            assert TEST_INDEX_FILE.exists(); assert TEST_METADATA_FILE.exists()

            print("\nLoading store from files...")
            store_loaded = FAISSVectorStore(DIMENSION, TEST_INDEX_FILE, TEST_METADATA_FILE)
            store_loaded.load()
            assert not store_loaded.is_empty()
            assert store_loaded.index.ntotal == 3
            assert len(store_loaded.stored_chunk_models_ordered) == 3
            assert "c1" in store_loaded.chunk_id_to_metadata

            print("\nSearching loaded store...")
            results_loaded = await store_loaded.search_similar_chunks(query_emb_similar_to_c1, top_k=1)
            print(f"Search results from loaded store: {[(r.id, r.text_content) for r in results_loaded]}")
            assert len(results_loaded) == 1
            assert results_loaded[0].id == "c1" or results_loaded[0].id == "c3"

            print("\nFAISSVectorStore tests complete.")
            # Clean up
            if TEST_INDEX_FILE.exists(): TEST_INDEX_FILE.unlink()
            if TEST_METADATA_FILE.exists(): TEST_METADATA_FILE.unlink()
            # if TEST_INDEX_FILE.parent.exists(): TEST_INDEX_FILE.parent.rmdir() # careful with rmdir

        import asyncio
        asyncio.run(run_tests())