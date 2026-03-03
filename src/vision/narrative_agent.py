# UltimaRAG — Multi-Agent RAG System
# Copyright (C) 2026 Pankaj Varma

"""
Narrative Agent for UltimaRAG.
Provides high-level LLM enrichment for structured multimodal extractions.
"""

from ..core.ollama_client import get_ollama_client
from ..core.utils import logger

class NarrativeAgent:
    """Agent specialized in turning structured extraction context into humanized narratives."""
    
    def __init__(self):
        self.client = get_ollama_client()

    async def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Generates a narrative based on the provided prompt."""
        try:
            result = await self.client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            # OllamaClient.generate returns {'response': str, ...} or a string
            if isinstance(result, dict):
                return result.get("response", "")
            return str(result)
        except Exception as e:
            logger.error(f"NarrativeAgent generation failed: {e}")
            return f"Error: {str(e)}"

def get_narrative_agent():
    """Factory function for NarrativeAgent."""
    return NarrativeAgent()
