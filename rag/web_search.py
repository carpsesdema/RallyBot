# Add to requirements.txt:
# googlesearch-python==1.2.3
# beautifulsoup4==4.12.2
# requests==2.31.0

# Create new file: rag/web_search.py
import logging
import requests
from typing import List, Dict, Any
from googlesearch import search
from bs4 import BeautifulSoup
import re

logger = logging.getLogger(__name__)


class WebSearchFallback:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results

    async def search_google(self, query: str) -> List[Dict[str, Any]]:
        """Search Google and extract content from top results"""
        try:
            # Add tennis context to the search query
            enhanced_query = f"{query} tennis"
            logger.info(f"Performing Google search for: {enhanced_query}")

            # Get top search results
            search_results = []
            urls = list(search(enhanced_query, num_results=self.max_results, stop=self.max_results))

            for url in urls:
                try:
                    # Fetch and parse the webpage
                    response = requests.get(url, timeout=10, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')

                        # Extract title and content
                        title = soup.find('title')
                        title_text = title.get_text().strip() if title else "No title"

                        # Extract main content (remove scripts, styles, etc.)
                        for script in soup(["script", "style", "nav", "footer", "header"]):
                            script.decompose()

                        # Get text content
                        content = soup.get_text()
                        # Clean up whitespace
                        content = re.sub(r'\s+', ' ', content).strip()

                        # Take first 1000 characters to avoid too much content
                        content_preview = content[:1000] + "..." if len(content) > 1000 else content

                        search_results.append({
                            'title': title_text,
                            'url': url,
                            'content': content_preview,
                            'source_type': 'web_search'
                        })

                except Exception as e:
                    logger.warning(f"Failed to fetch content from {url}: {e}")
                    continue

            logger.info(f"Successfully retrieved {len(search_results)} web search results")
            return search_results

        except Exception as e:
            logger.error(f"Google search failed: {e}")
            return []


# Update rag/rag_pipeline.py - add this method to RAGPipeline class:

async def query_with_rag_and_fallback(self, query_text: str, top_k_chunks: int = 3,
                                      model_name_override: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    """Enhanced RAG query with Google search fallback"""
    logger.info(f"Processing RAG query with fallback: '{query_text[:100]}...', top_k={top_k_chunks}")

    final_model_name = model_name_override or (
        self.settings.GEMINI_MODEL if self.settings.LLM_PROVIDER == "gemini"
        else self.settings.OLLAMA_CHAT_MODEL
    )

    # Try RAG first
    try:
        # Check if vector store has content
        if self.vector_store.is_empty():
            logger.info("Vector store is empty, proceeding directly to web search fallback")
            return await self._web_search_fallback(query_text, final_model_name)

        # Generate query embedding
        query_embedding_list = await self.embedding_generator.llm_client.generate_embeddings([query_text])
        if not query_embedding_list or not query_embedding_list[0]:
            raise RAGPipelineError("Failed to generate embedding for the query.")
        query_embedding = query_embedding_list[0]

        # Search vector store
        relevant_chunks: List[ChunkModel] = await self.vector_store.search_similar_chunks(
            query_embedding, top_k=top_k_chunks
        )

        # Check if we have good results
        if not relevant_chunks:
            logger.info("No relevant chunks found, using web search fallback")
            return await self._web_search_fallback(query_text, final_model_name)

        # Try RAG with retrieved chunks
        context_string = "\n\n---\n\n".join([chunk.text_content for chunk in relevant_chunks])
        retrieved_chunks_details = [
            {
                "source_file": chunk.metadata.get("source_filename", "Unknown"),
                "chunk_id": chunk.id,
                "text_preview": chunk.text_content[:150] + "...",
                "source_type": "knowledge_base"
            } for chunk in relevant_chunks
        ]

        # Check if context seems relevant/recent enough
        prompt_with_fallback_check = f"""Based on the following context, please answer the question. 

Context:
{context_string}

Question: {query_text}

IMPORTANT: If the context does not contain sufficient or current information to answer the question properly, respond with exactly "INSUFFICIENT_CONTEXT" and nothing else.

Otherwise, provide a complete answer based on the context.

Answer:"""

        llm_answer = await self.llm_client.generate_response(
            prompt=prompt_with_fallback_check,
            model_name=final_model_name
        )

        # Check if LLM says context is insufficient
        if "INSUFFICIENT_CONTEXT" in llm_answer.strip():
            logger.info("LLM determined context is insufficient, using web search fallback")
            return await self._web_search_fallback(query_text, final_model_name, rag_context=context_string)

        logger.info(f"RAG answer generated successfully. Length: {len(llm_answer)}")
        return llm_answer, retrieved_chunks_details

    except Exception as e:
        logger.error(f"RAG pipeline failed, falling back to web search: {e}")
        return await self._web_search_fallback(query_text, final_model_name)


async def _web_search_fallback(self, query_text: str, model_name: str, rag_context: str = None) -> Tuple[
    str, List[Dict[str, Any]]]:
    """Perform web search fallback"""
    try:
        # Initialize web search if not already done
        if not hasattr(self, 'web_searcher'):
            from rag.web_search import WebSearchFallback
            self.web_searcher = WebSearchFallback()

        # Perform web search
        web_results = await self.web_searcher.search_google(query_text)

        if not web_results:
            # Final fallback - just use LLM without context
            prompt = f"Please answer this tennis-related question as best you can: {query_text}"
            answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)
            return f"⚠️ No specific information found. General response: {answer}", []

        # Combine web search results
        web_context = "\n\n---\n\n".join([
            f"Title: {result['title']}\nURL: {result['url']}\nContent: {result['content']}"
            for result in web_results
        ])

        # Create prompt with web context
        combined_context = ""
        if rag_context:
            combined_context = f"Knowledge Base Context:\n{rag_context}\n\nRecent Web Information:\n{web_context}"
        else:
            combined_context = f"Recent Web Information:\n{web_context}"

        prompt = f"""Based on the following information, please answer the question.

{combined_context}

Question: {query_text}

Please provide a comprehensive answer. If you found recent/current information from web sources, prioritize that for time-sensitive questions.

Answer:"""

        web_answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)

        # Format sources for return
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
        logger.error(f"Web search fallback failed: {e}")
        # Ultimate fallback
        prompt = f"Please answer this tennis question as best you can: {query_text}"
        answer = await self.llm_client.generate_response(prompt=prompt, model_name=model_name)
        return f"⚠️ Search unavailable. General response: {answer}", []