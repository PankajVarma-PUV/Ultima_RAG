from typing import Dict, List, Optional
import asyncio
import re
from ..core.utils import logger
from ..core.models import UnifiedResponse, QualityMetrics

class HallucinationHealer:
    """
    SOTA Hallucination Healer.
    Rewrites responses that fail the Triad Verification Pass.
    """
    
    def __init__(self):
        from ..core.ollama_client import OllamaClient
        from ..core.config import Config
        # Enforce minimum timeout for Healer (10 minutes) given the complexity of the task
        healer_timeout = max(Config.ollama.TIMEOUT, 600)
        
        self.llm = OllamaClient(
            model_name=Config.ollama.MODEL_NAME,
            timeout=healer_timeout
        )
        logger.info(f"Hallucination Healer initialized | Model: {self.llm.model_name} | Timeout: {self.llm.timeout}s | URL: {self.llm.base_url}")

    async def heal(self, query: str, flawed_response: str, gaps: List[str], evidence: str) -> tuple:
        """
        Rewrites the response to fix hallucinations or missing evidence.
        Returns (response, reasoning)
        """
        logger.info(f"Healing response for query: {query}")
        
        # SOTA Corrective Prompt: Ensures both text and visual evidence are treated as the source of truth.
        prompt = f"""
<role>
You are Ultima_RAG, the Elite Hallucination-Correction Intelligence. Your purpose is to surgically remove inaccuracies and bridge factual gaps using causal reasoning.
</role>

<investigation_report>
Query: {query}
Draft: {flawed_response}
Detected Discrepancies: {", ".join(gaps)}
</investigation_report>

<source_of_truth>
{evidence}
</source_of_truth>

<correction_protocols>
1. CAUSAL REASONING: Step-by-step, inside a <thinking> block, explain WHY the draft failed and HOW the <source_of_truth> provides the correction.
2. PRECISION: Remove every claim not explicitly supported by the <source_of_truth>.
3. CITATION: Maintain suffix [[FileName]] citations.
4. NARRATIVE REHABILITATION: Generate a clean, immersive response that resolves the discrepancies.
</correction_protocols>

Ultima_RAG REWRITTEN RESPONSE:"""
        
        try:
            # OllamaClient.generate() is synchronous — run in thread to avoid blocking
            logger.info(f"Healer generating correction with prompt length: {len(prompt)} characters")
            response = await asyncio.to_thread(self.llm.generate, prompt)
            
            # SOTA: Strip <thinking> tags from final response
            if response:
                clean_response = re.sub(r'<thinking>[\s\S]*?</thinking>', '', response).strip()
            else:
                clean_response = flawed_response
                
            corrected = clean_response.replace("Ultima_RAG REWRITTEN RESPONSE:", "").strip()
            reasoning = f"Corrected response to address gaps: {', '.join(gaps)} using provided evidence."
            return corrected, reasoning
        except Exception as e:
            logger.error(f"Healing Error: {e}")
            return flawed_response, f"Healing failed due to error: {e}"


