"""
Utility functions for the Ultima_RAG API.
Handles identity queries and streaming simulations.
Cache-based duplicate query retrieval has been removed (intentionally).
"""

import json
import asyncio
from typing import Optional, List, Dict, Any, Callable
from ..core.utils import logger


# =============================================================================
# IDENTITY LOGIC
# =============================================================================

IDENTITY_KEYWORDS = [
    "who created you", "who made you", "who is your owner",
    "who built you", "who developed you", "who designed you",
    "who is your creator", "who is your master", "your owner",
    "your creator", "your developer", "who are you",
    "tell me about yourself", "what are you", "introduce yourself"
]

IDENTITY_RESPONSE = (
    "I am **Ultima_RAG**, an elite metacognitive intelligence system created by "
    "**Pankaj Varma**.\n\n"
    "If you wish to connect with my creator, here are his contact details:\n\n"
    "📱 **WhatsApp / Telegram**: +91 9372123700\n\n"
    "📧 **Email**: Pv43770@gmail.com\n\n"
    "I am here to process, analyze, and reason over your documents and queries "
    "with precision. How can I assist you today?"
)

def is_identity_query(query: str) -> bool:
    """Check if query is asking about the creator/owner of Ultima_RAG."""
    q = query.strip().lower()
    return any(kw in q for kw in IDENTITY_KEYWORDS)


# =============================================================================
# STREAMING HELPERS
# =============================================================================

async def simulate_streaming(text: str, delay: float = 0.02):
    """Simulate word-by-word streaming for hardcoded responses."""
    words = text.split(' ')
    for i, word in enumerate(words):
        token = word if i == 0 else ' ' + word
        yield f"data: {json.dumps({'stage': 'streaming', 'token': token}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(delay)

