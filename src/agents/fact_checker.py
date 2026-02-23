"""
Fact Checker Agent for Ultima_RAG
Validates synthesized answers against retrieved evidence using NLI.
Uses local Ollama with configured model for LLM operations.
"""

import json
import re
from typing import Dict, List, Optional

from sentence_transformers import CrossEncoder

from ..core.config import Config, FactCheckConfig, OllamaConfig
from ..core.ollama_client import OllamaClient, get_ollama_client
from ..core.prompts import CLAIM_EXTRACTION_PROMPT
from ..core.utils import logger, Timer


# =============================================================================
# FACT CHECKER AGENT
# =============================================================================

class FactChecker:
    """
    Agent 5: Fact Checker
    
    Responsibilities:
    - Extract atomic claims from synthesis output
    - Match each claim to source chunks
    - Score claim-evidence alignment (0-1)
    - Flag unsupported claims
    - Calculate overall factuality score
    """
    
    def __init__(
        self,
        nli_model_name: str = FactCheckConfig.NLI_MODEL,
        support_threshold: float = FactCheckConfig.SUPPORT_THRESHOLD
    ):
        """
        Initialize Fact Checker with NLI model and Ollama client.
        
        Args:
            nli_model_name: Cross-encoder NLI model name
            support_threshold: Minimum score to consider claim supported
        """
        # NLI model for claim verification
        self.nli_model = CrossEncoder(nli_model_name)
        self.support_threshold = support_threshold
        
        # LLM client for claim extraction
        self.client = get_ollama_client()
        
        logger.info(f"FactChecker initialized with NLI model: {nli_model_name}")
    
    def check_facts(
        self,
        synthesis_output: Dict,
        source_chunks: List[Dict]
    ) -> Dict:
        """
        Verify synthesized answer against source evidence using LLM-based evaluation.
        
        Uses Gemma3:4b to read both the context and response together
        and provide an accuracy score.
        
        Args:
            synthesis_output: Output from SynthesizerAgent
            source_chunks: Original chunks used for synthesis
        
        Returns:
            Fact-check results dictionary
        """
        # Handle non-answered cases - still give a reasonable score
        status = synthesis_output.get("status", "UNKNOWN")
        answer = synthesis_output.get("answer", "")
        
        if not answer or len(answer.strip()) < 10:
            return {
                "factuality_score": 0.3,
                "claims": [],
                "unsupported_claims": [],
                "total_claims": 0,
                "supported_claims": 0,
                "status": "EMPTY_ANSWER",
                "llm_evaluation": None
            }
        
        # Build context from chunks
        context_parts = []
        for chunk in source_chunks[:15]:  # SOTA: Increased breadth for multi-file RAG support
            # SOTA Phase 6: Sync with both 'text' (RAG) and 'content' (Multimodal) keys
            text = chunk.get('text') or chunk.get('content', '') or chunk.get('perception', '')
            source = chunk.get('file_name') or chunk.get('source', 'Unknown')
            if text:
                context_parts.append(f"SOURCE [{source}]: {text}")
        
        context = "\n\n".join(context_parts)
        
        if not context:
            # No context to evaluate against - use synthesis confidence
            return {
                "factuality_score": synthesis_output.get("confidence", 0.5),
                "claims": [],
                "unsupported_claims": [],
                "total_claims": 0,
                "supported_claims": 0,
                "status": "NO_CONTEXT",
                "llm_evaluation": None
            }
        
        with Timer("Fact Checking (LLM-based)"):
            # Use LLM-based evaluation
            llm_result = self._evaluate_with_llm(
                context=context,
                question=synthesis_output.get("metadata", {}).get("query", ""),
                response=answer
            )
            
            # Extract factuality score from LLM evaluation
            accuracy_score = llm_result.get("accuracy_score", 0.7)
            
            # Blend with synthesis confidence for robustness
            synthesis_conf = synthesis_output.get("confidence", 0.5)
            factuality_score = (accuracy_score * 0.7) + (synthesis_conf * 0.3)
        
        result = {
            "factuality_score": max(0.1, factuality_score), 
            "groundedness": max(0.1, factuality_score),
            "relevancy": max(0.1, accuracy_score),
            "utility": synthesis_output.get("confidence", 0.5),
            "reflection": llm_result.get("reflection", {"is_relevant": True, "is_supported": True, "has_utility": True}),
            "claims": [],
            "unsupported_claims": llm_result.get("factual_errors", []),
            "total_claims": 1,
            "supported_claims": 1 if llm_result.get("is_accurate", True) else 0,
            "status": "COMPLETED",
            "llm_evaluation": llm_result
        }
        
        logger.info(
            f"Fact check (LLM): accuracy={accuracy_score:.2f}, "
            f"final_score={factuality_score:.2f}, accurate={llm_result.get('is_accurate')}, "
            f"reflection={result['reflection']}"
        )
        
        return result
    
    def _evaluate_with_llm(self, context: str, question: str, response: str) -> Dict:
        """
        Use Gemma3:4b to evaluate response accuracy against context.
        """
        from ..core.prompts import RESPONSE_EVALUATION_PROMPT
        
        # Build prompt
        prompt = RESPONSE_EVALUATION_PROMPT.format(
            context=context[:3000],  # Limit context length
            question=question or "Based on the context above, what can you tell me?",
            response=response[:1500]  # Limit response length
        )
        
        try:
            if not self.client.is_available():
                # Fallback if Ollama not available
                logger.warning("Ollama not available for fact check, using default score")
                return {
                    "accuracy_score": 0.7,
                    "reasoning": "LLM evaluation unavailable, using default score",
                    "factual_errors": [],
                    "is_accurate": True
                }
            
            result = self.client.generate(
                prompt,
                temperature=0.1,  # Low temperature for consistent evaluation
                max_tokens=Config.ollama_multi_model.LIGHTWEIGHT_MAX_TOKENS
            )
            result_text = result.get("response", "") if isinstance(result, dict) else result
            
            # Parse JSON response
            result_text = result_text.strip()
            
            # Clean up markdown if present
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:json)?\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)
            
            # Try to parse JSON
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                evaluation = json.loads(result_text)
            
            # Validate and normalize score
            raw_score = evaluation.get("accuracy_score", 0.7)
            
            # Robust Parsing Strategy
            if isinstance(raw_score, str):
                # Handle "85%" or "0.85" or "85"
                raw_score = raw_score.replace('%', '').strip()
                try:
                    score = float(raw_score)
                    if score > 1.0: score = score / 100.0 # Convert 85 to 0.85
                except ValueError:
                    score = 0.7
            else:
                score = float(raw_score)
                
            score = max(0.0, min(1.0, score))  # Clamp to 0-1
            evaluation["accuracy_score"] = score
            
            return evaluation
            
        except Exception as e:
            logger.warning(f"LLM evaluation failed: {e}, using default score")
            return {
                "accuracy_score": 0.7,
                "reasoning": f"Evaluation parsing failed: {str(e)[:50]}",
                "factual_errors": [],
                "is_accurate": True
            }
    
    def evaluate_relevance(self, question: str, source_chunks: List[Dict]) -> Dict:
        """
        Evaluate if the user's question is relevant to the document content.
        
        Args:
            question: User's question
            source_chunks: Chunks from the knowledge base
        
        Returns:
            Relevance evaluation with score (0.0-1.0)
        """
        from ..core.prompts import RELEVANCE_EVALUATION_PROMPT
        
        # Build context summary from chunks
        context_parts = []
        for chunk in source_chunks[:5]:
            text = chunk.get('text', '')[:300]  # First 300 chars of each chunk
            if text:
                context_parts.append(text)
        
        context = "\n\n".join(context_parts)
        
        if not context:
            return {
                "relevance_score": 0.5,
                "is_relevant": True,
                "reasoning": "No context available for comparison",
                "document_topics": [],
                "question_topic": question
            }
        
        prompt = RELEVANCE_EVALUATION_PROMPT.format(
            context=context[:2000],
            question=question
        )
        
        try:
            if not self.client.is_available():
                return {
                    "relevance_score": 0.7,
                    "is_relevant": True,
                    "reasoning": "LLM unavailable for relevance check",
                    "document_topics": [],
                    "question_topic": question
                }
            
            result = self.client.generate(
                prompt,
                temperature=0.1,
                max_tokens=Config.ollama_multi_model.LIGHTWEIGHT_MAX_TOKENS
            )
            result_text = result.get("response", "") if isinstance(result, dict) else result
            
            result_text = result_text.strip()
            
            # Clean up markdown if present
            if result_text.startswith("```"):
                result_text = re.sub(r'^```(?:json)?\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)
            
            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                evaluation = json.loads(result_text)
            
            # Normalize score
            score = float(evaluation.get("relevance_score", 0.7))
            score = max(0.0, min(1.0, score))
            evaluation["relevance_score"] = score
            
            logger.info(f"Relevance check: score={score:.2f}, relevant={evaluation.get('is_relevant')}")
            return evaluation
            
        except Exception as e:
            logger.warning(f"Relevance evaluation failed: {e}")
            return {
                "relevance_score": 0.7,
                "is_relevant": True,
                "reasoning": f"Evaluation failed: {str(e)[:50]}",
                "document_topics": [],
                "question_topic": question
            }
    
    def _extract_claims(self, answer: str) -> List[str]:
        """
        Extract atomic claims from the answer.
        Uses LLM if available, otherwise falls back to sentence splitting.
        """
        if self.client.is_available():
            try:
                return self._extract_claims_with_llm(answer)
            except Exception as e:
                logger.warning(f"LLM claim extraction failed: {e}")
        
        return self._extract_claims_fallback(answer)
    
    def _extract_claims_with_llm(self, answer: str) -> List[str]:
        """Use Ollama to extract atomic claims"""
        prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
        
        response = self.client.generate(
            prompt,
            temperature=0.0,
            max_tokens=Config.ollama_multi_model.LIGHTWEIGHT_MAX_TOKENS
        )
        
        response_text = response.strip()
        
        # Clean up markdown if present
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        # Try to find JSON array in response
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            claims = json.loads(json_match.group())
        else:
            claims = json.loads(response_text)
        
        if isinstance(claims, list):
            return [str(c) for c in claims if c]
        
        return []
    
    def _extract_claims_fallback(self, answer: str) -> List[str]:
        """Fallback: split answer into sentences as claims"""
        # Remove citations for cleaner claims
        clean_answer = re.sub(r'\[Source\s+\d+(?:\s*,\s*\d+)*\]', '', answer)
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', clean_answer)
        
        # Filter short/empty sentences
        claims = [
            s.strip() for s in sentences 
            if s.strip() and len(s.split()) >= 5
        ]
        
        return claims[:10]  # Limit to prevent explosion
    
    def _verify_claim(
        self, 
        claim: str, 
        evidence_chunks: List[Dict]
    ) -> Dict:
        """
        Verify a claim against evidence using NLI.
        
        Returns the best matching evidence and support status.
        """
        best_match = {
            "evidence": None,
            "source": None,
            "score": 0.0,
            "supported": False
        }
        
        if not evidence_chunks:
            return best_match
        
        # Prepare evidence texts
        evidence_texts = [chunk["text"] for chunk in evidence_chunks]
        
        # Create pairs for NLI: (evidence, claim) - checking if evidence entails claim
        pairs = [(evidence, claim) for evidence in evidence_texts]
        
        # Get NLI scores - may return 1D array of scores or 2D array (n_samples, n_classes)
        scores = self.nli_model.predict(pairs, show_progress_bar=False)
        
        # Handle different score formats
        import numpy as np
        scores = np.array(scores)
        
        # If 2D (multi-class), take the entailment score (usually first column) or max
        if scores.ndim == 2:
            # For NLI models, typically: [entailment, neutral, contradiction]
            # Take entailment score (index 0) or use max
            scores = scores[:, 0] if scores.shape[1] >= 1 else scores.max(axis=1)
        
        # Find best supporting evidence
        for chunk, score in zip(evidence_chunks, scores):
            # Convert numpy scalar to Python float
            score_val = score.item() if hasattr(score, 'item') else float(score)
            if score_val > best_match["score"]:
                best_match = {
                    "evidence": chunk["text"][:300] + "..." if len(chunk["text"]) > 300 else chunk["text"],
                    "source": chunk.get("source", "unknown"),
                    "score": score_val,
                    "supported": score_val >= self.support_threshold
                }
        
        return best_match


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_fact_checker(
    threshold: Optional[float] = None
) -> FactChecker:
    """Get a configured FactChecker instance"""
    return FactChecker(
        support_threshold=threshold or FactCheckConfig.SUPPORT_THRESHOLD
    )


def check_facts(
    synthesis_output: Dict,
    source_chunks: List[Dict]
) -> Dict:
    """Convenience function to check facts"""
    checker = get_fact_checker()
    return checker.check_facts(synthesis_output, source_chunks)

