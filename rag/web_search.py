# rag/web_search.py
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
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    async def search_google(self, query: str) -> List[Dict[str, Any]]:
        """Search Google and extract content from top results"""
        try:
            # Add tennis context to the search query if it seems to be missing
            enhanced_query = query
            if "tennis" not in query.lower():
                enhanced_query = f"{query} tennis"

            logger.info(f"Performing Google search for: '{enhanced_query}'")

            # Get top search results
            search_results = []
            # Adding a pause to be a good web citizen
            urls = list(search(enhanced_query, num_results=self.max_results, stop=self.max_results, pause=2.0))

            for url in urls:
                try:
                    # Fetch and parse the webpage
                    response = requests.get(url, timeout=10, headers=self.headers)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.content, 'html.parser')

                    title = soup.find('title')
                    title_text = title.get_text().strip() if title else "No title found"

                    # Decompose irrelevant tags
                    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                        element.decompose()

                    # Get text content, separated by spaces, and clean it up
                    content = soup.get_text(separator=' ', strip=True)
                    content = re.sub(r'\s+', ' ', content).strip()

                    # Take a reasonable preview
                    content_preview = content[:1500] + "..." if len(content) > 1500 else content

                    if content_preview:  # Only add if we managed to extract some content
                        search_results.append({
                            'title': title_text,
                            'url': url,
                            'content': content_preview,
                            'source_type': 'web_search'
                        })

                except requests.RequestException as e:
                    logger.warning(f"Request failed for URL {url}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to process content from {url}: {e}")
                    continue

            logger.info(f"Successfully retrieved content from {len(search_results)} web search results")
            return search_results

        except Exception as e:
            logger.error(f"Google search process failed: {e}", exc_info=True)
            return []