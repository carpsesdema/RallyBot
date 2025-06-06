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
    from llm_interface.tennis_api_client import TennisAPIClient

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
        logger.info(f"RAGPipeline initialized with enhanced tennis intelligence.")

    async def ingest_documents_from_directory(self, directory_path_str: str) -> Tuple[int, int]:
        # Keep existing implementation
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
        # Keep existing RAG implementation
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

        except Exception as e:
            logger.error(f"Unexpected error during RAG query: {e}", exc_info=True)
            raise RAGPipelineError(f"RAG query failed: {e}") from e

    # ENHANCED TENNIS INTELLIGENCE WITH REAL DATA
    async def query_with_tennis_intelligence(self, query_text: str, top_k_chunks: int = 3,
                                             model_name_override: Optional[str] = None) -> Tuple[
        str, List[Dict[str, Any]]]:
        """
        Enhanced query method with comprehensive tennis analysis and database-ready insights
        """
        logger.info(f"🎾 ENHANCED TENNIS QUERY: Processing '{query_text[:50]}...'")

        # Check if this is a tennis-related query
        is_tennis_query = self._is_tennis_query(query_text)
        player_names = self._extract_player_names(query_text)

        final_model_name = model_name_override or (
            self.settings.GEMINI_MODEL if self.settings.LLM_PROVIDER == "gemini"
            else self.settings.OLLAMA_CHAT_MODEL
        )

        # Enhanced tennis analysis with real data
        if is_tennis_query and len(player_names) >= 2:
            return await self._comprehensive_matchup_analysis(query_text, player_names, final_model_name)
        elif is_tennis_query and len(player_names) == 1:
            return await self._single_player_analysis(query_text, player_names[0], final_model_name)
        elif is_tennis_query:
            return await self._tennis_market_analysis(query_text, final_model_name)
        else:
            # Fall back to regular RAG
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

        # Enhanced pattern matching for "X vs Y" variations
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
        try:
            tennis_client = TennisAPIClient()

            player1, player2 = player_names[0], player_names[1]
            logger.info(f"🎾 COMPREHENSIVE MATCHUP: {player1} vs {player2}")

            # Get comprehensive H2H analysis
            h2h_analysis = await tennis_client.analyze_head_to_head(player1, player2)

            # Format comprehensive data for LLM analysis
            context = self._format_comprehensive_analysis(h2h_analysis, query_text)

            # Enhanced prompt for professional betting analysis
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

            # Enhanced source metadata for database storage
            sources = [
                {
                    "source_file": "Live Tennis Intelligence API",
                    "chunk_id": f"h2h_{player1}_{player2}",
                    "text_preview": f"Comprehensive H2H: {player1} vs {player2} with betting analysis",
                    "source_type": "live_tennis_data",
                    "data_quality": "high",
                    "analysis_type": "head_to_head",
                    "players": [player1, player2],
                    "betting_insights_count": len(h2h_analysis.get("betting_implications", [])),
                    "ranking_gap": h2h_analysis.get("ranking_comparison", {}).get("ranking_gap", 0)
                }
            ]

            await tennis_client.close()
            return f"🎾 {answer}", sources

        except Exception as e:
            logger.error(f"Comprehensive matchup analysis failed: {e}")
            return await self.query_with_rag(query_text, 3, model_name)

    async def _single_player_analysis(self, query_text: str, player_name: str,
                                      model_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Comprehensive single player analysis"""
        try:
            tennis_client = TennisAPIClient()

            logger.info(f"🎾 SINGLE PLAYER ANALYSIS: {player_name}")

            # Get comprehensive player analysis
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

            sources = [
                {
                    "source_file": "Player Intelligence Database",
                    "chunk_id": f"player_{player_name}",
                    "text_preview": f"Comprehensive analysis: {player_name}",
                    "source_type": "live_tennis_data",
                    "analysis_type": "single_player",
                    "player": player_name,
                    "ranking": player_analysis.get("statistical_summary", {}).get("current_ranking", 0),
                    "betting_tier": player_analysis.get("betting_profile", {}).get("betting_tier", "unknown")
                }
            ]

            await tennis_client.close()
            return f"🎾 {answer}", sources

        except Exception as e:
            logger.error(f"Single player analysis failed: {e}")
            return await self.query_with_rag(query_text, 3, model_name)

    async def _tennis_market_analysis(self, query_text: str, model_name: str) -> Tuple[str, List[Dict[str, Any]]]:
        """General tennis market and tournament analysis"""
        try:
            tennis_client = TennisAPIClient()

            logger.info(f"🎾 TENNIS MARKET ANALYSIS")

            # Get current tournament context
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

            sources = [
                {
                    "source_file": "Tennis Market Intelligence",
                    "chunk_id": "market_overview",
                    "text_preview": "Current tennis market analysis and betting opportunities",
                    "source_type": "live_tennis_data",
                    "analysis_type": "market_overview",
                    "atp_players_analyzed": len(tournament_data.get('atp_top_10', [])),
                    "wta_players_analyzed": len(tournament_data.get('wta_top_10', []))
                }
            ]

            await tennis_client.close()
            return f"🎾 {answer}", sources

        except Exception as e:
            logger.error(f"Tennis market analysis failed: {e}")
            return await self.query_with_rag(query_text, 3, model_name)

    def _format_comprehensive_analysis(self, h2h_data: Dict[str, Any], query: str) -> str:
        """Format comprehensive H2H data for LLM analysis"""
        formatted = f"HEAD-TO-HEAD ANALYSIS: {h2h_data['player1'].upper()} vs {h2h_data['player2'].upper()}\n\n"

        # Ranking comparison
        ranking_comp = h2h_data.get("ranking_comparison", {})
        if ranking_comp:
            formatted += f"RANKING COMPARISON:\n"
            formatted += f"• {h2h_data['player1']}: #{ranking_comp.get('player1_ranking', 'N/A')}\n"
            formatted += f"• {h2h_data['player2']}: #{ranking_comp.get('player2_ranking', 'N/A')}\n"
            formatted += f"• Ranking Gap: {ranking_comp.get('ranking_gap', 'N/A')} positions\n"
            formatted += f"• Current Favorite: {ranking_comp.get('favorite', 'N/A')}\n\n"

        # Player 1 detailed data
        p1_data = h2h_data.get("player1_data", {})
        if p1_data.get("statistical_summary"):
            formatted += f"{h2h_data['player1'].upper()} PROFILE:\n"
            stats = p1_data["statistical_summary"]
            formatted += f"• Ranking: #{stats.get('current_ranking', 'N/A')}\n"
            formatted += f"• Points: {stats.get('ranking_points', 'N/A')}\n"
            formatted += f"• Country: {stats.get('country', 'N/A')}\n"
            formatted += f"• Betting Tier: {p1_data.get('betting_profile', {}).get('betting_tier', 'N/A')}\n\n"

        # Player 2 detailed data
        p2_data = h2h_data.get("player2_data", {})
        if p2_data.get("statistical_summary"):
            formatted += f"{h2h_data['player2'].upper()} PROFILE:\n"
            stats = p2_data["statistical_summary"]
            formatted += f"• Ranking: #{stats.get('current_ranking', 'N/A')}\n"
            formatted += f"• Points: {stats.get('ranking_points', 'N/A')}\n"
            formatted += f"• Country: {stats.get('country', 'N/A')}\n"
            formatted += f"• Betting Tier: {p2_data.get('betting_profile', {}).get('betting_tier', 'N/A')}\n\n"

        # Betting implications
        implications = h2h_data.get("betting_implications", [])
        if implications:
            formatted += f"BETTING IMPLICATIONS:\n"
            for implication in implications:
                formatted += f"• {implication}\n"
            formatted += "\n"

        formatted += f"RISK ASSESSMENT: {h2h_data.get('risk_assessment', 'Medium')}\n"

        return formatted

    def _format_single_player_analysis(self, player_data: Dict[str, Any], query: str) -> str:
        """Format single player data for analysis"""
        player_name = player_data.get("player_name", "Unknown")
        formatted = f"PLAYER ANALYSIS: {player_name.upper()}\n\n"

        # Statistical summary
        stats = player_data.get("statistical_summary", {})
        if stats:
            formatted += f"CURRENT STATISTICS:\n"
            formatted += f"• Ranking: #{stats.get('current_ranking', 'N/A')}\n"
            formatted += f"• Points: {stats.get('ranking_points', 'N/A')}\n"
            formatted += f"• Country: {stats.get('country', 'N/A')}\n"
            formatted += f"• Recent Matches: {stats.get('recent_matches_count', 0)}\n\n"

        # Betting profile
        betting = player_data.get("betting_profile", {})
        if betting:
            formatted += f"BETTING PROFILE:\n"
            formatted += f"• Tier: {betting.get('betting_tier', 'Unknown')}\n"
            formatted += f"• Form: {betting.get('form_indicator', 'Neutral')}\n"
            formatted += f"• Value Assessment: {betting.get('value_assessment', 'Monitor')}\n\n"

        formatted += f"Data Last Updated: {player_data.get('last_updated', 'Unknown')}\n"

        return formatted

    def _format_rankings_data(self, rankings: List[Dict[str, Any]]) -> str:
        """Format rankings data for market analysis"""
        if not rankings:
            return "No ranking data available"

        formatted = ""
        for i, player in enumerate(rankings[:10], 1):
            team = player.get("team", {})
            name = team.get("name", "Unknown")
            country = player.get("country", {}).get("alpha3", "")
            points = player.get("userCount", 0)
            formatted += f"{i}. {name} ({country}) - {points} pts\n"

        return formatted


if __name__ == '__main__':
    import asyncio

    logging.basicConfig(level=logging.DEBUG)

    # Test the enhanced tennis intelligence
    print("Enhanced RAG Pipeline with fixed tennis intelligence ready for deployment")