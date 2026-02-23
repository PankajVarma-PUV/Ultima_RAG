"""
Humanizer Agent for Ultima_RAG
Rewrites dry synthesized responses to sound natural and human-written.
Uses local Ollama with configured model for lightweight rewriting.
"""

from typing import Dict, List, Optional
import re

from ..core.ollama_client import OllamaClient, get_ollama_client, OllamaConfig
from ..core.config import Config
from ..core.utils import logger, Timer


# =============================================================================
# HUMANIZER PROMPT
# =============================================================================

HUMANIZER_PROMPT = """
<role>
You are the Ultima_RAG Cinematic Stylist. Your mission is to transform robotic AI outputs into elite, natural, and high-fidelity human narratives.
</role>

<task>
Rewrite the provided answer to sound natural and immersive while preserving 100% of the factual grounding and citations.
</task>

<nlg_protocols>
1. BURSTINESS: Vary your sentence length. Mix short, impactful statements with longer, flowing descriptions.
2. ACTIVE VOICE: Use strong verbs. Avoid "There is", "It appears that", or "One can see".
3. NO FILLERS: Strictly avoid "Based on the documents", "In conclusion", "As mentioned above", or "I found that".
4. CITATION LOGIC: Keep all [[FileName]] citations exactly where they are in the source text.
5. PERSONA: You are a professional, warm, and highly-attuned partner, not a search engine.
</nlg_protocols>

<input_data>
Question: {user_question}
Raw Answer: {raw_answer}
</input_data>

REHABILITATED NARRATIVE:"""


# =============================================================================
# HUMANIZER AGENT
# =============================================================================

class HumanizerAgent:
    """
    Agent: Humanizer (Response Rewriter)
    
    Responsibilities:
    - Rewrite dry/robotic synthesized responses to sound natural
    - Preserve all factual content and citations
    - Fix source naming (filename or "Context")
    """
    
    def __init__(self):
        """
        Initialize Humanizer Agent with Ollama client.
        """
        self.client = get_ollama_client()
        
        if not self.client.is_available():
            logger.warning("Ollama not available - humanizer will use passthrough")
        else:
            logger.info(f"HumanizerAgent initialized with Ollama: {OllamaConfig.MODEL_NAME}")
    
    def humanize(
        self,
        query: str,
        raw_answer: str,
        citations: Optional[List[Dict]] = None,
        skip_humanization: bool = False
    ) -> Dict:
        """
        Rewrite a synthesized answer to sound more natural.
        
        Args:
            query: Original user question
            raw_answer: Dry synthesized answer from SynthesizerAgent
            citations: List of citation dictionaries for source fixing
            skip_humanization: If True, only fix sources without rewriting
        
        Returns:
            Humanized result dictionary
        """
        if not raw_answer or len(raw_answer.strip()) < 10:
            return {
                "humanized_answer": raw_answer,
                "original_answer": raw_answer,
                "was_humanized": False,
                "citations": citations or []
            }
        
        # Fix source naming in citations
        fixed_citations = self._fix_source_names(citations or [])
        
        if skip_humanization or not self.client.is_available():
            return {
                "humanized_answer": raw_answer,
                "original_answer": raw_answer,
                "was_humanized": False,
                "citations": fixed_citations
            }
        
        with Timer("Humanization"):
            try:
                prompt = HUMANIZER_PROMPT.format(
                    user_question=query,
                    raw_answer=raw_answer
                )
                
                result = self.client.generate(
                    prompt,
                    temperature=0.3,  # Slight creativity for natural tone
                    max_tokens=Config.ollama_multi_model.LIGHTWEIGHT_MAX_TOKENS
                )
                
                humanized = result.get("response", "") if isinstance(result, dict) else result
                humanized = humanized.strip()
                
                # Validate output
                if not humanized or len(humanized) < 10:
                    humanized = raw_answer
                    was_humanized = False
                else:
                    was_humanized = True
                
            except Exception as e:
                logger.warning(f"Humanization failed: {e}. Using original answer.")
                humanized = raw_answer
                was_humanized = False
        
        logger.info(f"Humanization complete: was_humanized={was_humanized}")
        
        return {
            "humanized_answer": humanized,
            "original_answer": raw_answer,
            "was_humanized": was_humanized,
            "citations": fixed_citations
        }
    
    def _fix_source_names(self, citations: List[Dict]) -> List[Dict]:
        """
        Fix source names in citations:
        - If chunk has metadata.filename → use filename
        - If source is 'pasted_context' or similar → "Context"
        - Otherwise → keep original source
        """
        fixed = []
        
        for citation in citations:
            fixed_citation = {**citation}
            source = citation.get("source", "")
            
            # Check for pasted context indicators
            if source in ["pasted_context", "context", "user_context", "direct_input"]:
                fixed_citation["source"] = "Context"
            elif source and ("/" in source or "\\" in source):
                # Extract filename from path
                import os
                fixed_citation["source"] = os.path.basename(source)
            
            fixed.append(fixed_citation)
        
        return fixed


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_humanizer() -> HumanizerAgent:
    """Get a configured HumanizerAgent instance"""
    return HumanizerAgent()


def humanize_response(
    query: str,
    raw_answer: str,
    citations: Optional[List[Dict]] = None
) -> Dict:
    """Convenience function to humanize a response"""
    humanizer = get_humanizer()
    return humanizer.humanize(query, raw_answer, citations)

