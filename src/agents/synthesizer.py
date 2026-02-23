"""
Synthesis Agent for Ultima_RAG
Generates answers strictly from provided context with citations.
Uses local Ollama with Config-defined model for LLM operations.
"""

import json
import re
from typing import Dict, List, Optional

from ..core.config import Config, OllamaConfig
from ..core.ollama_client import OllamaClient, get_ollama_client
from ..core.prompts import SYNTHESIS_PROMPT, format_context_for_synthesis
from ..core.utils import logger, Timer


# =============================================================================
# SYNTHESIS AGENT
# =============================================================================

class SynthesizerAgent:
    """
    Agent 4: Synthesis Agent
    
    Responsibilities:
    - Construct answer from top-ranked chunks
    - Use direct quotes where possible
    - Cite sources inline (e.g., [Source: doc_23.pdf, p.5])
    - Flag uncertainty if context is ambiguous
    - Return "INSUFFICIENT_EVIDENCE" if context doesn't support answer
    """
    
    def __init__(self):
        """
        Initialize Synthesis Agent with Ollama client.
        """
        self.client = get_ollama_client()
        
        if not self.client.is_available():
            logger.warning("Ollama not available - synthesis will fail")
        else:
            logger.info(f"SynthesisAgent initialized with Ollama: {OllamaConfig.MODEL_NAME}")
    
    def synthesize(
        self,
        query: str,
        chunks: List[Dict],
        query_analysis: Optional[Dict] = None
    ) -> Dict:
        """
        Synthesize an answer from retrieved chunks.
        """
        if not chunks:
            return {
                "answer": "No context information was provided to answer this question. Please provide some text or upload a document.",
                "status": "NO_CONTEXT",
                "confidence": 0.1,
                "citations": [],
                "metadata": {"reason": "No chunks provided"}
            }
        
        # Check for multimodal content
        has_mm_summary = any(
            c.get('metadata', {}).get('type') in ['visual_summary', 'audio_summary'] 
            for c in chunks
        )
        has_mm_raw = any(
            c.get('metadata', {}).get('type') in ['scene', 'ocr', 'audio_transcript'] 
            for c in chunks
        )
        
        is_multimodal = has_mm_summary or has_mm_raw
        
        with Timer("Synthesis"):
            # Format context
            context = format_context_for_synthesis(chunks)
            
            # Select model and prompt
            model = OllamaConfig.MODEL_NAME
            if is_multimodal:
                logger.info("Multimodal content detected. Switching to Reasoning Model.")
                model = OllamaConfig.MODEL_NAME
                from ..core.prompts import MULTIMODAL_REASONING_PROMPT
                prompt = MULTIMODAL_REASONING_PROMPT.format(
                    context=context,
                    query=query
                )
            else:
                prompt = SYNTHESIS_PROMPT.format(
                    context=context,
                    query=query
                )
            
            # Generate answer using Ollama
            try:
                # Use a specific model if needed
                client = self.client if model == OllamaConfig.MODEL_NAME else OllamaClient(model_name=model)
                
                response_text = client.generate(
                    prompt,
                    temperature=0.2 if is_multimodal else OllamaConfig.TEMPERATURE,
                    max_tokens=Config.ollama_multi_model.HEAVY_MAX_TOKENS,
                    repeat_penalty=1.1
                )
            except Exception as e:
                logger.error(f"Ollama generation failed ({model}): {e}")
                # Still provide the context as a fallback
                fallback_header = "The screenshot contains text as: " if is_multimodal else "I was unable to process this query fully, but here is the relevant context: "
                fallback = f"{fallback_header}{chunks[0].get('text', '')[:500]}..."
                return {
                    "answer": fallback,
                    "status": "ERROR",
                    "confidence": 0.2,
                    "citations": [],
                    "metadata": {"error": str(e)}
                }
            
            response_text = response_text.strip()
        
        # Parse response
        result = self._parse_response(response_text, chunks)
        if is_multimodal:
            result["metadata"]["multimodal"] = True
            result["metadata"]["model"] = model
        
        logger.info(f"Synthesis complete: status={result['status']}, confidence={result['confidence']:.2f}")
        return result
    
    def _parse_response(self, response: str, chunks: List[Dict]) -> Dict:
        """Parse the synthesis response and extract structure - ALWAYS returns an answer."""
        
        # SOTA: Strip internal <thinking> blocks before showing to user
        if "<thinking>" in response:
            # Extract thinking for logs/metadata if needed, then strip
            thinking_match = re.search(r'<thinking>([\s\S]*?)</thinking>', response)
            if thinking_match:
                logger.debug(f"Synthesizer Thought: {thinking_match.group(1).strip()[:200]}...")
            response = re.sub(r'<thinking>[\s\S]*?</thinking>', '', response).strip()

        # Default status
        status = "ANSWERED"
        confidence_modifier = 1.0
        
        # Check for special status markers but still extract content
        if "INSUFFICIENT_EVIDENCE" in response:
            status = "PARTIAL"
            confidence_modifier = 0.5
            # Try to extract any useful content after the marker
            clean_response = re.sub(r'INSUFFICIENT_EVIDENCE[:\s]*', '', response).strip()
            if clean_response:
                response = clean_response
        
        if "AMBIGUOUS_EVIDENCE" in response:
            status = "AMBIGUOUS"
            confidence_modifier = 0.7
            # Extract explanation if present
            match = re.search(r'AMBIGUOUS_EVIDENCE[:\s]*(.+)', response, re.DOTALL)
            if match:
                response = match.group(1).strip()
        
        # Ensure we always have a response
        if not response or len(response.strip()) < 10:
            # Use chunk content as fallback
            if chunks:
                response = f"Based on the provided information: {chunks[0].get('text', '')[:500]}..."
            else:
                response = "The provided context was processed but no specific answer could be formulated."
        
        # Normal answer
        citations = self._extract_citations(response, chunks)
        confidence = self._calculate_confidence(response, chunks, citations) * confidence_modifier
        
        return {
            "answer": response,  # NEVER None
            "status": status,
            "confidence": confidence,
            "citations": citations,
            "metadata": {
                "citation_count": len(citations),
                "has_quotes": '"' in response
            }
        }
    
    def _extract_citations(self, answer: str, chunks: List[Dict]) -> List[Dict]:
        """Extract [Source: docname] citations from the answer"""
        # Find [Source: docname] or [Source: doc1, doc2] patterns
        citation_pattern = r'\[Source:\s*([^\]]+)\]'
        matches = re.findall(citation_pattern, answer, re.IGNORECASE)
        
        citations = []
        seen_sources = set()
        
        # Build a map of docname -> chunk for quick lookup
        docname_to_chunk = {}
        for chunk in chunks:
            source = chunk.get("source", "unknown")
            if not source or source in ['Unknown', 'Context']:
                # Try fallback to file_name metadata
                source = chunk.get('metadata', {}).get('file_name', source)
                
            # Full name matching
            source_lower = source.lower()
            docname_to_chunk[source_lower] = chunk
            
            # Base name matching (e.g. "cat" matches "cat.jpg")
            if '.' in source:
                base_name = source.rsplit('.', 1)[0].lower()
                if base_name not in docname_to_chunk:
                    docname_to_chunk[base_name] = chunk
        
        for match in matches:
            # Split by comma for multiple sources
            source_names = [s.strip().lower() for s in match.split(',')]
            
            for source_name in source_names:
                if source_name in seen_sources:
                    continue
                
                # SOTA Phase 20: Strict Source Validation
                # Reject if source_name is overly descriptive or purely numeric
                if len(source_name) > 50 or source_name.isdigit():
                    logger.warning(f"Synthesizer: Rejecting pseudo-source: {source_name}")
                    continue

                seen_sources.add(source_name)
                
                # Find matching chunk
                chunk = docname_to_chunk.get(source_name)
                if chunk:
                    citations.append({
                        "source_name": source_name,
                        "source": chunk.get("source", "unknown"),
                        "page": chunk.get("metadata", {}).get("page"),
                        "chunk_id": chunk.get("chunk_id"),
                        "text_snippet": chunk.get("text", "")[:200] + "..."
                    })
                else:
                    logger.warning(f"Synthesizer: Citation mismatch - {source_name} not in context.")
        
        return citations
    
    def _calculate_confidence(
        self, 
        answer: str, 
        chunks: List[Dict],
        citations: List[Dict]
    ) -> float:
        """
        Calculate confidence score based on answer quality indicators.
        
        Factors:
        - Has citations (0.3)
        - Multiple sources cited (0.2)
        - Direct quotes present (0.2)
        - Answer length appropriate (0.15)
        - High-scoring chunks used (0.15)
        """
        score = 0.0
        
        # Has citations
        if citations:
            score += 0.3
        
        # Multiple sources
        unique_sources = len(set(c.get("source") for c in citations))
        if unique_sources > 1:
            score += 0.2
        elif unique_sources == 1:
            score += 0.1
        
        # Direct quotes
        if '"' in answer:
            score += 0.2
        
        # Answer length (not too short, not too long)
        word_count = len(answer.split())
        if 30 <= word_count <= 500:
            score += 0.15
        elif 10 <= word_count < 30 or 500 < word_count <= 800:
            score += 0.07
        
        # High-scoring chunks
        if chunks:
            avg_score = sum(
                c.get('rerank_score', c.get('score', 0)) 
                for c in chunks
            ) / len(chunks)
            if avg_score > 0.7:
                score += 0.15
            elif avg_score > 0.5:
                score += 0.1
        
        return min(score, 1.0)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_synthesizer() -> SynthesizerAgent:
    """Get a configured SynthesizerAgent instance"""
    return SynthesizerAgent()


def synthesize_answer(
    query: str,
    chunks: List[Dict]
) -> Dict:
    """Convenience function to synthesize an answer"""
    synthesizer = get_synthesizer()
    return synthesizer.synthesize(query, chunks)

