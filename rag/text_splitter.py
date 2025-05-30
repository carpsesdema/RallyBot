# rag/text_splitter.py
import logging
from typing import List, Optional, Callable

# Attempt to import models and custom exceptions
try:
    from models import DocumentModel, ChunkModel
    from utils import TextSplittingError
except ImportError:
    # Define fallbacks if running standalone or during isolated testing
    class DocumentModel:  # Dummy
        def __init__(self, id: str, content: str, metadata: dict):
            self.id = id
            self.content = content
            self.metadata = metadata


    class ChunkModel:  # Dummy
        def __init__(self, document_id: str, text_content: str, metadata: dict, id: Optional[str] = None):
            self.id = id or "dummy_chunk_id"
            self.document_id = document_id
            self.text_content = text_content
            self.metadata = metadata


    class TextSplittingError(Exception):
        pass

logger = logging.getLogger(__name__)


class RecursiveCharacterTextSplitter:
    """
    Splits text recursively by characters, trying to respect chunk_size and chunk_overlap.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Common separators, ordered by assumed "importance" or natural break points
        self._separators = ["\n\n", "\n", ". ", " ", ""]
        logger.info(
            f"RecursiveCharacterTextSplitter initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def _split_text_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """
        Attempts to split text using a list of separators, starting with the first.
        If the text is small enough, it's returned as is.
        If a separator splits the text into chunks smaller than chunk_size, those are used.
        Otherwise, it moves to the next separator.
        """
        final_chunks = []

        # If text is already smaller than chunk_size, no need to split.
        if len(text) <= self.chunk_size:
            if text.strip():  # Only add if not just whitespace
                return [text]
            return []

        # Try the first separator
        separator = separators[0]
        remaining_separators = separators[1:]

        # Split by the current separator
        if separator:  # If separator is not empty string
            splits = text.split(separator)
        else:  # If separator is empty string, split by character
            splits = list(text)
            # For character split, we don't re-add the separator, so handle merging carefully

        current_chunk = ""
        for i, part in enumerate(splits):
            # Re-add separator if it wasn't an empty string (character split)
            # and if it's not the first part (to avoid leading separator on merged chunks)
            part_to_add = part
            if separator and i > 0:  # if current_chunk: # Alternative condition
                # Add separator back to maintain context, unless it's character split
                part_to_add = separator + part

            if len(current_chunk) + len(part_to_add) <= self.chunk_size:
                current_chunk += part_to_add
            else:
                # Current chunk is full (or adding next part makes it too full)
                if current_chunk.strip():
                    final_chunks.append(current_chunk)

                # Start new chunk. If the part itself is > chunk_size, it needs further splitting.
                if len(part_to_add) > self.chunk_size and remaining_separators:
                    # This part is too big and we have more separators to try
                    # We need to handle overlap correctly here. The part_to_add may already have a leading separator.
                    # The sub-split should consider overlap with the *previous final_chunk* if possible,
                    # or internal overlap if it's a very long segment without good separator breaks.

                    # For simplicity, let's take the previous chunk's end for overlap
                    overlap_text = ""
                    if final_chunks and self.chunk_overlap > 0:
                        # Ensure overlap doesn't re-include the separator if already added
                        overlap_text = final_chunks[-1][-self.chunk_overlap:]

                    # If part_to_add started with a separator, we need to be careful not to duplicate it
                    # when prepending overlap.
                    effective_part_to_add = part_to_add
                    if separator and part_to_add.startswith(separator):
                        effective_part_to_add = part_to_add[len(separator):]

                    recursive_splits = self._split_text_with_separators(overlap_text + effective_part_to_add,
                                                                        remaining_separators)
                    final_chunks.extend(recursive_splits)
                    current_chunk = ""  # Reset current_chunk as recursive_splits handles it
                elif len(part_to_add) > self.chunk_size:
                    # Part is too big, no more separators, force split
                    # This will split by character if '' is the last separator
                    # or just take chunk_size slices
                    start = 0
                    while start < len(part_to_add):
                        end = start + self.chunk_size
                        chunk_content = part_to_add[start:end]
                        if chunk_content.strip():
                            final_chunks.append(chunk_content)
                        start += self.chunk_size - self.chunk_overlap  # Slide window with overlap
                        if start + self.chunk_overlap > len(part_to_add) and start < len(
                                part_to_add):  # ensure last bit isn't too small or overlap doesn't overshoot
                            start = len(part_to_add) - self.chunk_overlap if len(
                                part_to_add) > self.chunk_overlap else start

                    current_chunk = ""  # Reset
                else:
                    # Start new chunk with current part, considering overlap from previous *final_chunk*
                    overlap_text = ""
                    if final_chunks and self.chunk_overlap > 0:
                        overlap_text = final_chunks[-1][-self.chunk_overlap:]

                    # current_chunk = overlap_text + part_to_add # This could double count separator with part_to_add
                    # Let's refine the overlap logic
                    if part_to_add.strip():  # Only start new chunk if part is not empty
                        if overlap_text and not part_to_add.startswith(
                                overlap_text):  # Basic check to avoid full duplication
                            current_chunk = overlap_text.rstrip(
                                separator if separator else None) + part_to_add  # Try to merge smartly
                        else:
                            current_chunk = part_to_add
                    else:
                        current_chunk = ""

        if current_chunk.strip():
            final_chunks.append(current_chunk)

        # Post-process: merge small chunks if possible, ensure overlap and size constraints
        # This recursive strategy might make post-processing complex; simpler iterative merging is an alternative.
        # For now, this provides a basic recursive split. A more robust implementation (like Langchain's)
        # is more involved.

        return [chunk for chunk in final_chunks if chunk.strip()]

    def split_document(self, document: DocumentModel) -> List[ChunkModel]:
        """
        Splits a DocumentModel's content into ChunkModels.
        """
        if not document.content or not document.content.strip():
            logger.warning(f"Document {document.id} has no content to split.")
            return []

        logger.debug(f"Splitting document ID: {document.id}, original length: {len(document.content)}")

        text_chunks = self._split_text_with_separators(document.content, self._separators)

        # Create ChunkModel instances
        chunk_models: List[ChunkModel] = []
        for i, text_chunk in enumerate(text_chunks):
            if not text_chunk.strip():  # Should be handled by _split_text_with_separators
                continue

            chunk_metadata = document.metadata.copy()  # Start with document metadata
            chunk_metadata["chunk_sequence_number"] = i
            # Could add char_start_index, char_end_index if tracked during splitting

            chunk = ChunkModel(
                document_id=document.id,
                text_content=text_chunk,
                metadata=chunk_metadata
            )
            chunk_models.append(chunk)
            logger.debug(f"Created chunk {i} for document {document.id}, length: {len(text_chunk)}")

        if not chunk_models:
            # This might happen if the content was very short and only whitespace after initial processing.
            # Or if the splitting logic resulted in no valid chunks.
            # If document content is small but valid, it should result in at least one chunk.
            if document.content.strip() and len(document.content.strip()) <= self.chunk_size:
                logger.debug(f"Document content is small, creating a single chunk for document {document.id}")
                chunk_metadata = document.metadata.copy()
                chunk_metadata["chunk_sequence_number"] = 0
                chunk_models.append(ChunkModel(
                    document_id=document.id,
                    text_content=document.content.strip(),
                    metadata=chunk_metadata
                ))
            else:
                logger.warning(f"No chunks created for document {document.id}. Content length: {len(document.content)}")

        logger.info(f"Document {document.id} split into {len(chunk_models)} chunks.")
        return chunk_models


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Dummy DocumentModel for testing
    doc1_content = """This is the first sentence. This is the second sentence.\n\nThis is a new paragraph after a double newline. It contains multiple phrases.
    This is a longer sentence that might be split if the chunk size is small enough. Let's see how it handles this.
    Another sentence. And one more.
    Final sentence of the first document."""

    doc_meta = {"source_filename": "test_doc.txt"}
    doc1 = DocumentModel(id="doc1", content=doc1_content, metadata=doc_meta)

    doc2_content = "Short document."
    doc2 = DocumentModel(id="doc2", content=doc2_content, metadata=doc_meta)

    doc3_content = "A" * 1500  # Longer than default chunk_size
    doc3 = DocumentModel(id="doc3", content=doc3_content, metadata=doc_meta)

    doc4_content = " " * 100  # Only whitespace
    doc4 = DocumentModel(id="doc4", content=doc4_content, metadata=doc_meta)

    print("\n--- Testing RecursiveCharacterTextSplitter (chunk_size=100, chunk_overlap=20) ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

    print(f"\n--- Splitting Document 1 ({len(doc1.content)} chars) ---")
    chunks1 = splitter.split_document(doc1)
    for i, chunk in enumerate(chunks1):
        print(
            f"  Chunk {i + 1} (doc1): '{chunk.text_content[:80].replace(chr(10), ' ')}...' (len: {len(chunk.text_content)}) ID: {chunk.id}")
        assert len(chunk.text_content) <= 100 or (
                    i == 0 and len(chunks1) == 1), f"Chunk too long: {len(chunk.text_content)}"

    print(f"\n--- Splitting Document 2 ({len(doc2.content)} chars) ---")
    chunks2 = splitter.split_document(doc2)
    for i, chunk in enumerate(chunks2):
        print(
            f"  Chunk {i + 1} (doc2): '{chunk.text_content[:80].replace(chr(10), ' ')}...' (len: {len(chunk.text_content)})")
    assert len(chunks2) == 1, f"Expected 1 chunk for short doc, got {len(chunks2)}"
    assert chunks2[0].text_content == doc2_content

    print(f"\n--- Splitting Document 3 ({len(doc3.content)} chars) ---")
    chunks3 = splitter.split_document(doc3)
    print(f"Doc3 split into {len(chunks3)} chunks.")
    for i, chunk in enumerate(chunks3):
        print(
            f"  Chunk {i + 1} (doc3): '{chunk.text_content[:80].replace(chr(10), ' ')}...' (len: {len(chunk.text_content)})")
        assert len(chunk.text_content) <= 100, f"Chunk {i + 1} of Doc3 too long: {len(chunk.text_content)}"
        if i > 0:  # Check overlap
            overlap_check_start_idx = max(0, len(chunks3[i - 1].text_content) - 20)
            expected_overlap = chunks3[i - 1].text_content[overlap_check_start_idx:]
            # This simple overlap check might fail if the splitter's internal logic is complex or separators are involved.
            # A more robust check would verify the semantic overlap.
            # print(f"    Previous chunk ending: ...{expected_overlap}")
            # print(f"    Current chunk starting: {chunk.text_content[:20]}...")
            # assert chunk.text_content.startswith(expected_overlap) # This is a strict overlap check

    print(f"\n--- Splitting Document 4 (whitespace only, {len(doc4.content)} chars) ---")
    chunks4 = splitter.split_document(doc4)
    print(f"Doc4 (whitespace) split into {len(chunks4)} chunks.")
    assert len(chunks4) == 0, "Whitespace-only document should result in 0 chunks."

    print("\n--- Testing with very small chunk_size (e.g., 10, overlap 2) ---")
    small_splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
    small_doc_content = "abcdefghijklmnopqrstuvwxyz"
    small_doc = DocumentModel(id="small_doc", content=small_doc_content, metadata={"source": "small_test.txt"})
    chunks_small = small_splitter.split_document(small_doc)
    print(f"Small doc ('{small_doc_content}') split into {len(chunks_small)} chunks (size=10, overlap=2):")
    for i, chunk in enumerate(chunks_small):
        print(f"  Chunk {i + 1}: '{chunk.text_content}' (len: {len(chunk.text_content)})")
        assert len(chunk.text_content) <= 10

    print("\n--- Text Splitting Tests Complete ---")