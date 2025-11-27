"""Web search module using duckduckgo-search."""

from duckduckgo_search import DDGS
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search and return formatted results.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        Formatted string with search results
    """
    try:
        results = []
        with DDGS() as ddgs:
            # Use text search with 'us-en' region for better relevance
            search_results = list(ddgs.text(query, region='us-en', max_results=max_results))
            
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'No Title')
                href = result.get('href', '#')
                body = result.get('body', 'No description available.')
                
                results.append(f"Result {i}:\nTitle: {title}\nURL: {href}\nSummary: {body}")
        
        if not results:
            return "No web search results found."
            
        return "\n\n".join(results)
        
    except Exception as e:
        logger.error(f"Error performing web search: {str(e)}")
        # Graceful degradation: return explicit message so models know search failed
        return "[System Note: Web search was attempted but failed. Please answer based on your internal knowledge.]"
