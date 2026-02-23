"""
Web Breakout Agent — Ultima_RAG Live Web Search Tool

Uses DuckDuckGo (no API key) and Trafilatura (clean text extraction).
Designed for 6GB VRAM systems: hard-truncates each source to 1000 chars.
Called by the Metacognitive Brain's `evaluate_knowledge` node when:
  1. Local LanceDB evidence is insufficient (LLM votes 'NO').
  2. The user has enabled the Web Search toggle on the frontend.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def fallback_web_search(query: str, max_results: int = 2) -> str:
    """
    Search the live web using DuckDuckGo and extract clean text via Trafilatura.

    Args:
        query: The user's search query (or an optimized version of it).
        max_results: Maximum number of search results to scrape (default: 2).

    Returns:
        A list of dictionaries with search result data: [{'title': str, 'url': str, 'text': str}, ...].
        If search fails, returns an empty list or a list with an error entry.
    """
    logger.info(f"[WebBreakoutAgent] Initiating live web search for: '{query}'")

    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.error("[WebBreakoutAgent] 'duckduckgo-search' is not installed.")
        return []

    try:
        import trafilatura
    except ImportError:
        logger.error("[WebBreakoutAgent] 'trafilatura' is not installed.")
        return []

    try:
        # SOTA: Improved browser-like headers to prevent 403 Forbidden errors
        custom_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }

        with DDGS() as ddgs:
            # SOTA: ddgs.text returns a generator, convert to list
            # We use text search which is better for broad news queries than snippet search
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            logger.info(f"[WebBreakoutAgent] DuckDuckGo returned no results for query: '{query}'")
            return []

        structured_results = []

        for res in results:
            url = res.get("href", "")
            title = res.get("title", "Untitled")
            snippet = res.get("body", "")

            if not url:
                continue

            logger.info(f"[WebBreakoutAgent] Fetching: {url}")

            scraped_text = ""
            try:
                # SOTA: Trafilatura config handling (ensure sections exist to prevent RuntimeWarnings/Errors)
                from trafilatura.settings import use_config
                config = use_config()
                if not config.has_section("network"):
                    config.add_section("network")
                
                config.set("network", "USER_AGENT", custom_headers["User-Agent"])
                config.set("network", "MAX_REDIRECTS", "5")
                
                downloaded = trafilatura.fetch_url(url, config=config) 
                
                if downloaded:
                    clean_text = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False,
                        config=config
                    )
                    if clean_text:
                        # VRAM Guard: Hard-cap at 1000 chars per source
                        scraped_text = clean_text[:1000]

            except Exception as fetch_err:
                logger.warning(f"[WebBreakoutAgent] Failed to scrape {url}: {fetch_err}")

            # Fallback: Use the DuckDuckGo snippet if scraping fails or is blocked
            final_text = scraped_text if scraped_text else f"[Scraping Blocked - Snippet Only]: {snippet[:500]}..."
            
            structured_results.append({
                "title": title,
                "url": url,
                "text": final_text
            })

        if not structured_results:
            return []

        logger.info(f"[WebBreakoutAgent] Successfully retrieved {len(structured_results)} web source(s).")
        return structured_results

    except Exception as e:
        logger.error(f"[WebBreakoutAgent] Web search failed with exception: {e}")
        return []

