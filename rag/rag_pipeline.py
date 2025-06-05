# rag/rag_pipeline.py
import logging
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, TYPE_CHECKING

try:
    from config import Settings, settings
    from models import DocumentModel, ChunkModel
    from utils import RAGPipelineError, DocumentLoadingError, TextSplittingError, EmbeddingGenerationError, \
        LLMClientError, VectorStoreError
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import VectorStoreInterface
    from llm_interface.tennis_api_client import TennisAPIClient  # NEW IMPORT

    if TYPE_CHECKING:
        from llm_interface.ollama_client import OllamaLLMClient
        from llm_interface.gemini_client import GeminiLLMClient
except ImportError as e:
    print(f"Import error in rag_pipeline.py: {e}. Some features might not work if run standalone.")


    class Settings:
        OLLAMA_CHAT_MODEL = "dummy_chat_model"; LLM_PROVIDER = "ollama"


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


    class VectorStoreError(Exception):
        pass


    class DocumentLoader:
        pass


    class TextSplitter:
        pass


    class EmbeddingGenerator:
        pass


    class VectorStoreInterface:
        def is_empty(self): return True

        async def search_similar_chunks(self, qe, tk): return []


    class TennisAPIClient:
        pass  # NEW DUMMY


    if TYPE_CHECKING:
        class OllamaLLMClient: pass


        class GeminiLLMClient: pass

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self,
                 settings: Settings,
                 llm_client,
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
                    logger.warning(
                        f"Document {doc.id} (source: {doc.metadata.get('source_filename')}) is empty, skipping.")
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
        except VectorStoreError as e:
            logger.error(f"Vector store error during ingestion: {e}", exc_info=True)
            raise RAGPipelineError(f"Vector store operation failed during ingestion: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}", exc_info=True)
            raise RAGPipelineError(f"Ingestion pipeline failed: {e}") from e

    async def query_with_rag(self, query_text: str, top_k_chunks: int = 3, model_name_override: Optional[str] = None) -> \
    Tuple[str, List[Dict[str, Any]]]:
        logger.info(
            f"Processing RAG query: '{query_text[:100]}...', top_k={top_k_chunks}, model_override='{model_name_override}'")

        final_model_name = model_name_override
        if not final_model_name:
            if self.settings.LLM_PROVIDER == "gemini":
                final_model_name = self.settings.GEMINI_MODEL
            else:
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
            query_embedding_list = await self.embedding_generator.llm_client.generate_embeddings([query_text])
            if not query_embedding_list or not query_embedding_list[0]:
                raise RAGPipelineError("Failed to generate embedding for the query.")
            query_embedding = query_embedding_list[0]
            logger.debug("Generated query embedding.")

            relevant_chunks: List[ChunkModel] = await self.vector_store.search_similar_chunks(query_embedding,
                                                                                              top_k=top_k_chunks)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks.")

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

    # NEW TENNIS INTELLIGENCE METHODS
    async def query_with_tennis_intelligence(self, query_text: str, top_k_chunks: int = 3,
                                             model_name_override: Optional[str] = None) -> Tuple[
        str, List[Dict[str, Any]]]:
        """
        Enhanced query method that detects tennis betting queries and provides live analysis
        """
        logger.info(f"🎾 TENNIS QUERY: Processing '{query_text[:50]}...'")

        # Check if this is a tennis-related query
        is_tennis_query = self._is_tennis_query(query_text)
        player_names = self._extract_player_names(query_text)

        final_model_name = model_name_override or (
            self.settings.GEMINI_MODEL if self.settings.LLM_PROVIDER == "gemini"
            else self.settings.OLLAMA_CHAT_MODEL
        )

        # If it's a tennis query with specific players, use live analysis
        if is_tennis_query and len(player_names) >= 2:
            return await self._tennis_matchup_analysis(query_text, player_names, final_model_name)
        elif is_tennis_query:
            return await self._tennis_general_analysis(query_text, final_model_name)
        else:
            # Fall back to regular RAG
            return await self.query_with_rag(query_text, top_k_chunks, model_name_override)

    def _is_tennis_query(self, query_text: str) -> bool:
        """Detect if query is tennis-related"""
        tennis_keywords = [
            'tennis', 'match', 'tournament', 'atp', 'wta', 'grand slam',
            'wimbledon', 'french open', 'us open', 'australian open',
            'clay', 'grass', 'hard court', 'serve', 'ace', 'break point',
            'set', 'game', 'deuce', 'tiebreak', 'ranking', 'seed',
            'bet', 'odds', 'favorite', 'underdog', 'spread', 'total',
            'h2h', 'head to head', 'analysis', 'prediction'
        ]

        betting_keywords = [
            'bet', 'betting', 'odds', 'line', 'spread', 'total', 'over', 'under',
            'favorite', 'underdog', 'value', 'pick', 'prediction', 'lean'
        ]

        query_lower = query_text.lower()
        has_tennis = any(keyword in query_lower for keyword in tennis_keywords)
        has_betting = any(keyword in query_lower for keyword in betting_keywords)

        return has_tennis or has_betting

    def _extract_player_names(self, query_text: str) -> List[str]:
        """Extract potential player names from query"""
        known_players = [
            'djokovic', 'alcaraz', 'federer', 'nadal', 'sinner', 'medvedev',
            'zverev', 'rublev', 'tsitsipas', 'ruud', 'fritz', 'hurkacz',
            'swiatek', 'sabalenka', 'gauff', 'rybakina', 'jabeur', 'vondrousova',
            'pegula', 'ostapenko', 'krejcikova', 'keys', 'badosa', 'collins'
        ]

        found_players = []
        query_lower = query_text.lower()

        for player in known_players:
            if player in query_lower:
                found_players.append(player)

        # Also look for "X vs Y" pattern
        vs_pattern = r'(\w+)\s+(?:vs|versus|v\.?)\s+(\w+)'
        vs_match = re.search(vs_pattern, query_text, re.IGNORECASE)
        if vs_match:
            found_players.extend([vs_match.group(1), vs_match.group(2)])

        return list(set(found_players))

    async def _tennis_matchup_analysis(self, query_text: str, player_names: List[str],
                                       model_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Perform live tennis matchup analysis"""
        try:
            tennis_client = TennisAPIClient()

            player1, player2 = player_names[0], player_names[1]
            logger.info(f"🎾 MATCHUP: Analyzing {player1} vs {player2}")

            analysis = await tennis_client.analyze_matchup(player1, player2)

            context = self._format_tennis_analysis(analysis, player1, player2)

            prompt = f"""You are a professional tennis betting analyst. Based on the following live data and analysis, provide detailed betting insights and recommendations.

LIVE TENNIS DATA:
{context}

USER QUERY: {query_text}

Provide a comprehensive analysis including:
1. Head-to-head breakdown highlighting key patterns
2. Surface-specific performance insights  
3. Recent form analysis
4. Key statistical advantages for each player
5. BETTING IMPLICATIONS with specific recommendations (e.g., "Lean Player X -1.5 sets")
6. Risk factors and value opportunities

Format your response like a professional betting analysis with clear, actionable insights."""

            answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)

            sources = [
                {
                    "source_file": "Live Tennis API",
                    "chunk_id": f"matchup_{player1}_{player2}",
                    "text_preview": f"Live analysis: {player1} vs {player2}",
                    "source_type": "live_tennis_data"
                }
            ]

            await tennis_client.close()
            return f"🎾 {answer}", sources

        except Exception as e:
            logger.error(f"Tennis matchup analysis failed: {e}")
            return await self.query_with_rag(query_text, 3, model_name)

    async def _tennis_general_analysis(self, query_text: str, model_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Handle general tennis queries with live data context"""
        try:
            tennis_client = TennisAPIClient()

            live_events = await tennis_client.get_live_events()

            live_context = "CURRENT LIVE TENNIS EVENTS:\n"
            if live_events:
                for i, event in enumerate(live_events[:5]):
                    live_context += f"• {event}\n"
            else:
                live_context += "No live events currently available.\n"

            rag_answer, rag_sources = await self.query_with_rag(query_text, 3, model_name)

            enhanced_prompt = f"""Based on the following information, provide a comprehensive tennis analysis with betting insights.

EXISTING KNOWLEDGE:
{rag_answer}

{live_context}

USER QUERY: {query_text}

Provide analysis with focus on:
1. Current tennis landscape and live events
2. Betting opportunities and market insights
3. Key players and matchups to watch
4. Statistical trends and patterns
5. Actionable betting recommendations

Keep the tone professional but accessible."""

            enhanced_answer = await self.llm_client.generate_response(prompt=enhanced_prompt, model_name=model_name)

            enhanced_sources = rag_sources + [
                {
                    "source_file": "Live Tennis Events",
                    "chunk_id": "live_events_context",
                    "text_preview": "Current live tennis events and market data",
                    "source_type": "live_tennis_data"
                }
            ]

            await tennis_client.close()
            return f"🎾 {enhanced_answer}", enhanced_sources

        except Exception as e:
            logger.error(f"Tennis general analysis failed: {e}")
            return await self.query_with_rag(query_text, 3, model_name)

    def _format_tennis_analysis(self, analysis: Dict[str, Any], player1: str, player2: str) -> str:
        """Format tennis analysis data for LLM consumption"""
        formatted = f"MATCHUP ANALYSIS: {player1.upper()} vs {player2.upper()}\n\n"

        if analysis.get("h2h_record"):
            formatted += f"HEAD-TO-HEAD DATA:\n{analysis['h2h_record']}\n\n"
        else:
            formatted += "HEAD-TO-HEAD: No prior matches found\n\n"

        if analysis.get("odds"):
            formatted += f"CURRENT BETTING ODDS:\n{analysis['odds']}\n\n"

        if analysis.get("surface_stats"):
            formatted += f"MATCH STATISTICS:\n{analysis['surface_stats']}\n\n"

        if analysis.get("recent_form"):
            formatted += f"RECENT FORM:\n{analysis['recent_form']}\n\n"

        if analysis.get("betting_insights"):
            formatted += f"INITIAL INSIGHTS:\n"
            for insight in analysis["betting_insights"]:
                formatted += f"• {insight}\n"

        return formatted


if __name__ == '__main__':
    import asyncio

    logging.basicConfig(level=logging.DEBUG)

    # Your existing test code would go here
    # I kept it minimal to avoid conflicts