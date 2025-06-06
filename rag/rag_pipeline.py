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
    from llm_interface.tennis_api_client import ProfessionalTennisAPIClient as TennisAPIClient
    from rag.document_loader import DocumentLoader
    from rag.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
    from rag.embedding_generator import EmbeddingGenerator
    from rag.vector_store import VectorStoreInterface
    from .web_search import WebSearchFallback # NEW: Import WebSearchFallback

    if TYPE_CHECKING:
        from llm_interface.ollama_client import OllamaLLMClient
        from llm_interface.gemini_client import GeminiLLMClient
except ImportError as e:
    print(f"Import error in rag_pipeline.py: {e}. Some features might not work if run standalone.")
    # Fallback classes remain the same...

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
        self.web_searcher = WebSearchFallback()  # NEW: Initialize web searcher
        logger.info(f"RAGPipeline initialized with enhanced tennis intelligence and web search fallback.")

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
        except Exception as e:
            logger.error(f"Unexpected error during ingestion: {e}", exc_info=True)
            raise RAGPipelineError(f"Ingestion pipeline failed: {e}") from e

    async def query_with_rag(self, query_text: str, top_k_chunks: int = 3, model_name_override: Optional[str] = None) -> \
    Tuple[str, List[Dict[str, Any]]]:
        """Enhanced RAG query with Google search fallback"""
        logger.info(f"Processing RAG query with fallback: '{query_text[:100]}...', top_k={top_k_chunks}")

        final_model_name = model_name_override or (
            self.settings.GEMINI_MODEL if self.settings.LLM_PROVIDER == "gemini"
            else self.settings.OLLAMA_CHAT_MODEL
        )

        # Try RAG first
        try:
            if self.vector_store.is_empty():
                logger.info("Vector store is empty, proceeding directly to web search fallback.")
                return await self._web_search_fallback(query_text, final_model_name)

            query_embedding_list = await self.embedding_generator.llm_client.generate_embeddings([query_text])
            if not query_embedding_list or not query_embedding_list[0]:
                raise RAGPipelineError("Failed to generate embedding for the query.")
            query_embedding = query_embedding_list[0]

            relevant_chunks: List[ChunkModel] = await self.vector_store.search_similar_chunks(
                query_embedding, top_k=top_k_chunks
            )

            if not relevant_chunks:
                logger.info("No relevant chunks found, using web search fallback.")
                return await self._web_search_fallback(query_text, final_model_name)

            context_string = "\n\n---\n\n".join([chunk.text_content for chunk in relevant_chunks])
            retrieved_chunks_details = [
                {
                    "source_file": chunk.metadata.get("source_filename", "Unknown"),
                    "chunk_id": chunk.id,
                    "text_preview": chunk.text_content[:150] + "...",
                    "source_type": "knowledge_base"
                } for chunk in relevant_chunks
            ]

            prompt_with_fallback_check = f"""Based on the following context, please answer the question.

Context:
{context_string}

Question: {query_text}

IMPORTANT: If the context does not contain sufficient or current information to answer the question properly, respond with exactly "INSUFFICIENT_CONTEXT" and nothing else. Otherwise, provide a complete answer based on the context.

Answer:"""

            llm_answer = await self.llm_client.generate_response(
                prompt=prompt_with_fallback_check, model_name=final_model_name
            )

            if "INSUFFICIENT_CONTEXT" in llm_answer.strip().upper():
                logger.info("LLM determined context is insufficient, using web search fallback.")
                return await self._web_search_fallback(query_text, final_model_name, rag_context=context_string)

            logger.info(f"RAG answer generated successfully. Length: {len(llm_answer)}")
            return llm_answer, retrieved_chunks_details

        except Exception as e:
            logger.error(f"RAG pipeline failed, falling back to web search: {e}", exc_info=True)
            return await self._web_search_fallback(query_text, final_model_name)

    async def _web_search_fallback(self, query_text: str, model_name: str, rag_context: Optional[str] = None) -> Tuple[
        str, List[Dict[str, Any]]]:
        """Perform web search fallback"""
        try:
            web_results = await self.web_searcher.search_google(query_text)

            if not web_results:
                prompt = f"Please answer this tennis-related question as best you can: {query_text}"
                answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)
                return f"⚠️ No specific information found in my knowledge base or on the web. Here is a general response: {answer}", []

            web_context = "\n\n---\n\n".join([
                f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}"
                for result in web_results
            ])

            combined_context = ""
            if rag_context:
                combined_context = f"Internal Knowledge Base Context (may be outdated):\n{rag_context}\n\nRecent Web Information:\n{web_context}"
            else:
                combined_context = f"Recent Web Information:\n{web_context}"

            prompt = f"""Based on the following information, please answer the question.

{combined_context}

Question: {query_text}

Please provide a comprehensive answer. Prioritize information from the 'Recent Web Information' section for any time-sensitive questions.

Answer:"""

            web_answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)

            sources_details = [
                {
                    "source_file": result['title'],
                    "url": result.get('url', ''),
                    "text_preview": result['content'][:200] + "...",
                    "source_type": "web_search"
                } for result in web_results
            ]

            logger.info(f"Web search fallback completed. Found {len(web_results)} sources.")
            return f"🌐 {web_answer}", sources_details

        except Exception as e:
            logger.error(f"Web search fallback failed: {e}", exc_info=True)
            prompt = f"Please answer this tennis question as best you can: {query_text}"
            answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)
            return f"⚠️ An error occurred during web search. Here is a general response: {answer}", []

    async def query_with_tennis_intelligence(self, query_text: str, top_k_chunks: int = 3,
                                             model_name_override: Optional[str] = None) -> Tuple[
        str, List[Dict[str, Any]]]:
        """
        Enhanced query method with comprehensive tennis analysis and database-ready insights
        """
        logger.info(f"🎾 ENHANCED TENNIS QUERY: Processing '{query_text[:50]}...'")

        is_tennis_query = self._is_tennis_query(query_text)
        player_names = self._extract_player_names(query_text)

        final_model_name = model_name_override or (
            self.settings.GEMINI_MODEL if self.settings.LLM_PROVIDER == "gemini"
            else self.settings.OLLAMA_CHAT_MODEL
        )

        if is_tennis_query and len(player_names) >= 2:
            return await self._comprehensive_matchup_analysis(query_text, player_names, final_model_name)
        elif is_tennis_query and len(player_names) == 1:
            return await self._single_player_analysis(query_text, player_names[0], final_model_name)
        elif is_tennis_query:
            return await self._tennis_market_analysis(query_text, final_model_name)
        else:
            logger.info(f"Query not identified as tennis-specific. Using RAG with web search fallback.")
            return await self.query_with_rag(query_text, top_k_chunks, model_name_override)

    def _is_tennis_query(self, query_text: str) -> bool:
        """Enhanced tennis query detection"""
        tennis_keywords = [
            'tennis', 'match', 'tournament', 'atp', 'wta', 'grand slam',
            'wimbledon', 'french open', 'us open', 'australian open',
            'clay', 'grass', 'hard court', 'serve', 'ace', 'break point',
            'set', 'game', 'deuce', 'tiebreak', 'ranking', 'seed',
            'bet', 'odds', 'favorite', 'underdog', 'spread', 'total',
            'h2h', 'head to head', 'analysis', 'prediction', 'form',
            'surface', 'baseline', 'volley', 'return', 'forehand', 'backhand'
        ]

        betting_keywords = [
            'bet', 'betting', 'odds', 'line', 'spread', 'total', 'over', 'under',
            'favorite', 'underdog', 'value', 'pick', 'prediction', 'lean',
            'moneyline', 'prop', 'futures', 'live', 'in-play'
        ]

        query_lower = query_text.lower()
        has_tennis = any(keyword in query_lower for keyword in tennis_keywords)
        has_betting = any(keyword in query_lower for keyword in betting_keywords)

        return has_tennis or has_betting

    def _extract_player_names(self, query_text: str) -> List[str]:
        """Enhanced player name extraction with more comprehensive list"""
        known_players = [
            # Men's Top Players
            'djokovic', 'alcaraz', 'sinner', 'medvedev', 'zverev', 'rublev',
            'tsitsipas', 'ruud', 'fritz', 'hurkacz', 'de minaur', 'paul',
            'dimitrov', 'rune', 'khachanov', 'shelton', 'auger-aliassime',
            'federer', 'nadal', 'murray',  # Legends

            # Women's Top Players
            'swiatek', 'sabalenka', 'gauff', 'rybakina', 'jabeur', 'vondrousova',
            'pegula', 'ostapenko', 'krejcikova', 'keys', 'badosa', 'collins',
            'garcia', 'kasatkina', 'muchova', 'haddad maia', 'vekic',
            'williams', 'osaka', 'halep'  # Notable players
        ]

        found_players = []
        query_lower = query_text.lower()

        for player in known_players:
            if player in query_lower:
                found_players.append(player)

        vs_patterns = [
            r'(\w+)\s+(?:vs|versus|v\.?|against)\s+(\w+)',
            r'(\w+)\s+(?:-|–|—)\s+(\w+)',  # Different dash types
            r'(\w+)\s+(?:plays|faces|meets)\s+(\w+)'
        ]

        for pattern in vs_patterns:
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                found_players.extend([match.group(1).lower(), match.group(2).lower()])

        return list(set(found_players))

    async def _comprehensive_matchup_analysis(self, query_text: str, player_names: List[str],
                                              model_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Comprehensive head-to-head analysis with database-ready insights"""
        tennis_client = None
        try:
            tennis_client = TennisAPIClient()
            player1, player2 = player_names[0], player_names[1]
            logger.info(f"🎾 COMPREHENSIVE MATCHUP: {player1} vs {player2}")
            h2h_analysis = await tennis_client.analyze_head_to_head(player1, player2)
            context = self._format_comprehensive_analysis(h2h_analysis, query_text)
            prompt = f"""You are a professional tennis betting analyst with access to comprehensive player data and statistics. Provide detailed betting insights and recommendations.

COMPREHENSIVE TENNIS DATA:
{context}

USER QUERY: {query_text}

Provide a professional analysis including:
1. **Head-to-Head Overview**: Current rankings, form, and key statistics
2. **Statistical Advantages**: Serve percentages, return games, surface preferences
3. **Recent Form Analysis**: Last 5-10 matches performance trends
4. **Betting Implications**:
   - Moneyline recommendation with confidence level
   - Set betting opportunities (over/under 3.5 sets, etc.)
   - Game handicap suggestions
   - Prop bet values (aces, double faults, break points)
5. **Risk Assessment**: Injury concerns, fatigue factors, motivation levels
6. **Value Opportunities**: Market inefficiencies and contrarian plays
7. **Recommended Bet Sizing**: Based on confidence and edge

Format your response as a professional betting report with clear, actionable insights and specific bet recommendations."""
            answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)
            sources = [{"source_file": "Live Tennis Intelligence API", "chunk_id": f"h2h_{player1}_{player2}", "text_preview": f"Comprehensive H2H: {player1} vs {player2} with betting analysis", "source_type": "live_tennis_data"}]
            return f"🎾 {answer}", sources
        except Exception as e:
            logger.error(f"Comprehensive matchup analysis failed: {e}", exc_info=True)
            return await self.query_with_rag(query_text, 3, model_name)
        finally:
            if tennis_client: await tennis_client.close()

    async def _single_player_analysis(self, query_text: str, player_name: str,
                                      model_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Comprehensive single player analysis"""
        tennis_client = None
        try:
            tennis_client = TennisAPIClient()
            logger.info(f"🎾 SINGLE PLAYER ANALYSIS: {player_name}")
            player_analysis = await tennis_client.get_comprehensive_player_analysis(player_name)
            context = self._format_single_player_analysis(player_analysis, query_text)
            prompt = f"""You are a professional tennis analyst. Provide comprehensive insights about this player based on current data.

PLAYER INTELLIGENCE DATA:
{context}

USER QUERY: {query_text}

Provide analysis covering:
1. **Current Form & Ranking**: Position, recent trajectory, points
2. **Playing Style & Strengths**: Technical analysis and tactical preferences
3. **Surface Performance**: Clay, grass, hard court win rates and preferences
4. **Recent Match Analysis**: Form trends, notable wins/losses
5. **Betting Profile**: Reliability as favorite/underdog, value patterns
6. **Upcoming Opportunities**: Tournament schedule and betting angles
7. **Long-term Outlook**: Career trajectory and investment potential

Focus on actionable insights for betting and fantasy tennis applications."""
            answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)
            sources = [{"source_file": "Player Intelligence Database", "chunk_id": f"player_{player_name}", "text_preview": f"Comprehensive analysis: {player_name}", "source_type": "live_tennis_data"}]
            return f"🎾 {answer}", sources
        except Exception as e:
            logger.error(f"Single player analysis failed: {e}", exc_info=True)
            return await self.query_with_rag(query_text, 3, model_name)
        finally:
            if tennis_client: await tennis_client.close()

    async def _tennis_market_analysis(self, query_text: str, model_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """General tennis market and tournament analysis"""
        tennis_client = None
        try:
            tennis_client = TennisAPIClient()
            logger.info(f"🎾 TENNIS MARKET ANALYSIS")
            tournament_data = await tennis_client.get_current_tournaments()
            context = f"""CURRENT TENNIS MARKET OVERVIEW:
ATP TOP 10 RANKINGS:
{self._format_rankings_data(tournament_data.get('atp_top_10', []))}
WTA TOP 10 RANKINGS:
{self._format_rankings_data(tournament_data.get('wta_top_10', []))}
MARKET CONTEXT:
- Data last updated: {tournament_data.get('last_updated', 'Unknown')}
- Analysis type: General tennis market overview
"""
            prompt = f"""You are a tennis market analyst. Provide insights about the current tennis landscape and betting opportunities.

{context}

USER QUERY: {query_text}

Provide analysis covering:
1. **Current Market Leaders**: Top players and their dominance patterns
2. **Emerging Threats**: Rising players creating betting value
3. **Tournament Outlook**: Upcoming events and betting angles
4. **Market Inefficiencies**: Overvalued/undervalued players
5. **Seasonal Trends**: Surface transitions and performance patterns
6. **Betting Strategies**: Current profitable approaches and angles

Focus on actionable market insights and betting opportunities."""
            answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)
            sources = [{"source_file": "Tennis Market Intelligence", "chunk_id": "market_overview", "text_preview": "Current tennis market analysis and betting opportunities", "source_type": "live_tennis_data"}]
            return f"🎾 {answer}", sources
        except Exception as e:
            logger.error(f"Tennis market analysis failed: {e}", exc_info=True)
            return await self.query_with_rag(query_text, 3, model_name)
        finally:
            if tennis_client: await tennis_client.close()

    def _format_comprehensive_analysis(self, h2h_data: Dict[str, Any], query: str) -> str:
        if not h2h_data or "player1_profile" not in h2h_data or "player2_profile" not in h2h_data: return "Incomplete H2H data."
        p1_name = h2h_data['player1_profile'].get('name', 'Player 1')
        p2_name = h2h_data['player2_profile'].get('name', 'Player 2')
        formatted = f"HEAD-TO-HEAD ANALYSIS: {p1_name.upper()} vs {p2_name.upper()}\n\n"
        p1_rank = h2h_data['player1_profile'].get('ranking', 'N/A')
        p2_rank = h2h_data['player2_profile'].get('ranking', 'N/A')
        formatted += f"RANKING: {p1_name} (#{p1_rank}) vs {p2_name} (#{p2_rank})\n\n"
        # Add more details if available in the dictionary
        return formatted

    def _format_single_player_analysis(self, player_data: Dict[str, Any], query: str) -> str:
        if not player_data or 'profile' not in player_data: return "Incomplete player data."
        profile = player_data['profile']
        player_name = profile.get("name", "Unknown")
        formatted = f"PLAYER ANALYSIS: {player_name.upper()}\n\n"
        stats = player_data.get("player_details", {}).get("player", {})
        if stats:
            formatted += f"• Country: {stats.get('country', {}).get('name', 'N/A')}\n"
            formatted += f"• Plays: {stats.get('plays', 'N/A')}\n"
            formatted += f"• Turned Pro: {stats.get('turnedPro', 'N/A')}\n"
        form = player_data.get('recent_form_string', 'N/A')
        formatted += f"• Recent Form (W/L): {form}\n"
        return formatted

    def _format_rankings_data(self, rankings: List[Dict[str, Any]]) -> str:
        if not rankings: return "No ranking data available"
        formatted = ""
        for i, player_data in enumerate(rankings, 1):
            player = player_data.get("player", {})
            name = player.get("name", "Unknown")
            points = player_data.get("points", 0)
            formatted += f"{i}. {name} - {points} pts\n"
        return formatted