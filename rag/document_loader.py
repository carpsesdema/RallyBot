# rag/document_loader.py
import logging
from pathlib import Path
from typing import List, Set, Optional

# Attempt to import models and custom exceptions
try:
    from models import DocumentModel
    from utils import DocumentLoadingError
except ImportError:
    # Define fallbacks if running standalone or during isolated testing
    class DocumentModel:  # Dummy
        def __init__(self, content: str, metadata: dict):
            self.id = "dummy_id"
            self.content = content
            self.metadata = metadata


    class DocumentLoadingError(Exception):
        pass

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Loads documents from a specified directory.
    Supports .txt and .md files by default.
    """

    def __init__(self, supported_extensions: Optional[Set[str]] = None):
        self.supported_extensions = supported_extensions or {".txt", ".md"}
        logger.info(f"DocumentLoader initialized with supported extensions: {self.supported_extensions}")

    def load_documents_from_directory(self, directory_path_str: str) -> List[DocumentModel]:
        """
        Recursively finds and loads documents from the given directory path.

        Args:
            directory_path_str: The string path to the directory.

        Returns:
            A list of DocumentModel objects.

        Raises:
            DocumentLoadingError: If the directory does not exist or other loading issues occur.
        """
        directory_path = Path(directory_path_str)
        if not directory_path.is_dir():
            msg = f"Directory not found: {directory_path}"
            logger.error(msg)
            raise DocumentLoadingError(msg)

        loaded_documents: List[DocumentModel] = []
        logger.info(f"Starting document loading from directory: {directory_path}")

        for file_path in directory_path.rglob("*"):  # rglob for recursive
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                logger.debug(f"Attempting to load document: {file_path}")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not content.strip():
                        logger.warning(f"Document is empty, skipping: {file_path}")
                        continue

                    metadata = {"source_filename": file_path.name, "full_path": str(file_path)}
                    document = DocumentModel(content=content, metadata=metadata)
                    loaded_documents.append(document)
                    logger.info(f"Successfully loaded document: {file_path.name} ({len(content)} chars)")

                except Exception as e:
                    logger.error(f"Failed to load or read document {file_path}: {e}", exc_info=True)
                    # Optionally, re-raise as DocumentLoadingError or collect errors to report later
                    # For now, we log and skip the faulty file.
                    # raise DocumentLoadingError(f"Failed to process file {file_path}: {e}") from e
            elif file_path.is_file():
                logger.debug(f"Skipping file with unsupported extension: {file_path}")

        if not loaded_documents:
            logger.warning(f"No documents loaded from directory: {directory_path}. Check extensions or content.")
        else:
            logger.info(f"Finished loading. Total documents loaded: {len(loaded_documents)} from {directory_path}")

        return loaded_documents


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create dummy files and directory for testing
    test_dir = Path("./_test_docs_temp")
    test_dir.mkdir(exist_ok=True)

    sub_dir = test_dir / "subdir"
    sub_dir.mkdir(exist_ok=True)

    (test_dir / "doc1.txt").write_text("This is document 1. It's a text file.")
    (test_dir / "doc2.md").write_text("# Markdown Document\nThis is document 2. It's a markdown file.")
    (test_dir / "notes.txt").write_text("Some brief notes.")
    (sub_dir / "sub_doc.txt").write_text("Document in a subdirectory.")
    (test_dir / "empty.txt").write_text("")  # Empty file
    (test_dir / "image.jpg").write_text("dummy image content")  # Unsupported

    loader = DocumentLoader()

    print(f"\n--- Testing Document Loading from: {test_dir.resolve()} ---")
    try:
        documents = loader.load_documents_from_directory(str(test_dir))
        print(f"\nSuccessfully loaded {len(documents)} documents:")
        for doc in documents:
            print(
                f"  ID: {doc.id}, Source: {doc.metadata.get('source_filename')}, Content preview: '{doc.content[:30]}...'")

        # Test with a non-existent directory
        print("\n--- Testing non-existent directory ---")
        try:
            loader.load_documents_from_directory("./non_existent_dir")
        except DocumentLoadingError as e:
            print(f"Correctly caught error for non-existent directory: {e}")

    finally:
        # Clean up dummy files and directory
        import shutil

        try:
            (test_dir / "doc1.txt").unlink(missing_ok=True)
            (test_dir / "doc2.md").unlink(missing_ok=True)
            (test_dir / "notes.txt").unlink(missing_ok=True)
            (sub_dir / "sub_doc.txt").unlink(missing_ok=True)
            (test_dir / "empty.txt").unlink(missing_ok=True)
            (test_dir / "image.jpg").unlink(missing_ok=True)
            sub_dir.rmdir()
            test_dir.rmdir()
            print("\nCleaned up test directory and files.")
        except Exception as e:
            print(f"Error during cleanup: {e}")