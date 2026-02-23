"""
Refusal Gate Agent for Ultima_RAG
Now operates as a Quality Indicator agent - always returns answers with confidence scores.
Changed from "reject policy" to "always show with fact score" policy.
"""

from typing import Dict, List, Optional

from ..core.config import RefusalGateConfig
from ..core.prompts import CONFIDENCE_WARNINGS
from ..core.utils import logger


# =============================================================================
# CONFIDENCE LEVELS
# =============================================================================

class ConfidenceLevel:
    """Confidence level classifications for UI display"""
    HIGH = "HIGH_CONFIDENCE"           # Green - 85%+
    MEDIUM = "MEDIUM_CONFIDENCE"       # Yellow - 65-84%
    LOW = "LOW_CONFIDENCE"             # Orange - 45-64%
    VERY_LOW = "VERY_LOW_CONFIDENCE"   # Red - <45%


# =============================================================================
# REFUSAL GATE AGENT (Now Quality Indicator)
# =============================================================================

class RefusalGate:
    """
    Agent 6: Quality Indicator (formerly Refusal Gate)
    
    Changed Behavior:
    - Previously: Would BLOCK responses below thresholds
    - Now: ALWAYS returns responses with quality indicators
    
    Responsibilities:
    - Classify response confidence level (HIGH/MEDIUM/LOW/VERY_LOW)
    - Generate quality warnings for transparency
    - Always return the answer with fact score
    - Suggest alternative queries if quality is low
    
    Philosophy: "Show everything, but be transparent about confidence."
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = RefusalGateConfig.HIGH_CONFIDENCE_THRESHOLD,
        medium_confidence_threshold: float = RefusalGateConfig.MEDIUM_CONFIDENCE_THRESHOLD,
        low_confidence_threshold: float = RefusalGateConfig.LOW_CONFIDENCE_THRESHOLD
    ):
        """
        Initialize Quality Indicator with classification thresholds.
        
        Args:
            high_confidence_threshold: Threshold for HIGH confidence (green)
            medium_confidence_threshold: Threshold for MEDIUM confidence (yellow)
            low_confidence_threshold: Threshold for LOW confidence (orange)
            Below low_confidence_threshold = VERY_LOW (red)
        """
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.low_threshold = low_confidence_threshold
        
        logger.info(
            f"RefusalGate initialized (always-show mode): "
            f"thresholds=[{high_confidence_threshold}, {medium_confidence_threshold}, {low_confidence_threshold}]"
        )
    
    def decide(
        self,
        synthesis_output: Dict,
        fact_check_output: Dict,
        query_analysis: Dict,
        relevance_result: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze quality and ALWAYS return the answer with confidence indicators.
        
        Args:
            synthesis_output: Output from SynthesizerAgent
            fact_check_output: Output from FactChecker
            query_analysis: Output from QueryAnalyzer
            relevance_result: Output from relevance evaluation (optional)
        
        Returns:
            Response dictionary with answer and quality indicators
        """
        # Calculate the overall fact score
        fact_score = self._calculate_fact_score(synthesis_output, fact_check_output)
        
        # Classify confidence level
        confidence_level = self._classify_confidence(fact_score)
        
        # Collect quality warnings
        warnings = self._collect_warnings(synthesis_output, fact_check_output)
        
        # Generate suggestions for low confidence responses
        suggestions = []
        if confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            suggestions = self._suggest_alternatives(query_analysis, synthesis_output)
        
        # Extract relevance info if provided
        relevance_score = relevance_result.get("relevance_score", 1.0) if relevance_result else 1.0
        is_relevant = relevance_result.get("is_relevant", True) if relevance_result else True
        
        # Add relevance warning if question is out of scope
        if relevance_score < 0.4:
            warnings.insert(0, "⚠️ Out of scope: Your question may not be covered by the uploaded documents.")
        elif relevance_score < 0.6:
            warnings.insert(0, "⚠️ Partial coverage: Your question may be only partially covered by the documents.")
        
        # ALWAYS return the answer (core change from reject policy)
        result = self._generate_response(
            synthesis=synthesis_output,
            fact_check=fact_check_output,
            fact_score=fact_score,
            confidence_level=confidence_level,
            warnings=warnings,
            suggestions=suggestions,
            relevance_score=relevance_score,
            relevance_info=relevance_result
        )
        
        return result
    
    def _calculate_fact_score(
        self, 
        synthesis: Dict, 
        fact_check: Dict
    ) -> float:
        """
        Calculate overall QualityScore = (0.5*Grounded) + (0.3*Relevancy) + (0.2*Utility)
        """
        g = fact_check.get("groundedness", 0.5)
        r = fact_check.get("relevancy", 0.5)
        u = fact_check.get("utility", 0.5)
        
        combined_score = (g * 0.5) + (r * 0.3) + (u * 0.2)
        
        return max(0.0, min(1.0, combined_score))
    
    def _classify_confidence(self, fact_score: float) -> str:
        """Classify the confidence level based on fact score."""
        if fact_score >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif fact_score >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif fact_score >= self.low_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _collect_warnings(
        self, 
        synthesis: Dict, 
        fact_check: Dict
    ) -> List[str]:
        """Collect relevant quality warnings for the response."""
        warnings = []
        
        # Check for insufficient evidence
        if synthesis.get("status") == "INSUFFICIENT_EVIDENCE":
            warnings.append(CONFIDENCE_WARNINGS["LOW_EVIDENCE"])
        
        # Check for contradictory sources
        if synthesis.get("status") == "AMBIGUOUS":
            warnings.append(CONFIDENCE_WARNINGS["CONTRADICTORY_SOURCES"])
        
        # Check for low factuality (handle None values)
        factuality_score = fact_check.get("factuality_score")
        if factuality_score is not None and factuality_score < 0.6:
            warnings.append(CONFIDENCE_WARNINGS["LOW_FACTUALITY"])
        
        # Check for unsupported claims
        unsupported = fact_check.get("unsupported_claims") or []
        if len(unsupported) > 1:
            warnings.append(CONFIDENCE_WARNINGS["UNSUPPORTED_CLAIMS"])
        
        # Check for low synthesis confidence (handle None values)
        synthesis_confidence = synthesis.get("confidence")
        if synthesis_confidence is not None and synthesis_confidence < 0.5:
            warnings.append(CONFIDENCE_WARNINGS["LOW_SYNTHESIS_CONFIDENCE"])
        
        return warnings
    
    def _generate_response(
        self,
        synthesis: Dict,
        fact_check: Dict,
        fact_score: float,
        confidence_level: str,
        warnings: List[str],
        suggestions: List[str],
        relevance_score: float = 1.0,
        relevance_info: Optional[Dict] = None
    ) -> Dict:
        """Generate the response with quality indicators - ALWAYS shows an answer."""
        
        # Get the answer - should never be None after our fixes
        answer = synthesis.get("answer")
        if not answer or answer.strip() == "":
            # Ultimate fallback
            answer = "Based on the context provided, I processed the information but could not formulate a specific response. Please try rephrasing your question."
        
        result = {
            # ALWAYS answer (core policy)
            "should_answer": True,
            
            # The actual response
            "final_response": answer,
            
            # Quality indicators (prominently displayed)
            "fact_score": fact_score,
            "confidence_score": fact_score,  # Alias for API compatibility
            "confidence_level": confidence_level,
            
            # Relevance indicator (NEW)
            "relevance_score": relevance_score,
            "is_relevant": relevance_score >= 0.5,
            
            # Transparency information
            "quality_warnings": warnings,
            "suggestions": suggestions,
            
            # Keep citations
            "citations": synthesis.get("citations", []),
            
            # Debug info
            "debug_info": {
                "synthesis_status": synthesis.get("status"),
                "synthesis_confidence": synthesis.get("confidence"),
                "factuality_score": fact_check.get("factuality_score"),
                "unsupported_claims_count": len(fact_check.get("unsupported_claims", [])),
                "total_claims": fact_check.get("total_claims", 0),
                "supported_claims": fact_check.get("supported_claims", 0),
                "relevance_info": relevance_info
            }
        }
        
        logger.info(
            f"Response generated: fact_score={fact_score:.2f}, relevance={relevance_score:.2f}, "
            f"confidence_level={confidence_level}, warnings={len(warnings)}"
        )
        return result
    
    def _suggest_alternatives(
        self,
        query_analysis: Dict,
        synthesis: Dict
    ) -> List[str]:
        """Generate alternative query suggestions for low-confidence responses."""
        suggestions = []
        
        # Suggest breaking down complex queries
        if query_analysis.get("difficulty") == "complex":
            sub_queries = query_analysis.get("sub_queries", [])
            for sub in sub_queries[:2]:
                if sub != query_analysis.get("original_query"):
                    suggestions.append(f"Try asking: \"{sub}\"")
        
        # Suggest focusing on entities
        entities = query_analysis.get("entities", [])
        if entities:
            entity = entities[0]
            suggestions.append(f"Try a broader question about: \"{entity}\"")
        
        # Generic suggestions based on intent
        intent = query_analysis.get("intent", "factual")
        if intent == "comparative":
            suggestions.append("Try asking about each item separately")
        elif intent == "multi-hop":
            suggestions.append("Try breaking this into simpler questions")
        
        return suggestions[:3]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_refusal_gate(
    high_threshold: Optional[float] = None,
    medium_threshold: Optional[float] = None,
    low_threshold: Optional[float] = None
) -> RefusalGate:
    """Get a configured RefusalGate (Quality Indicator) instance"""
    return RefusalGate(
        high_confidence_threshold=high_threshold or RefusalGateConfig.HIGH_CONFIDENCE_THRESHOLD,
        medium_confidence_threshold=medium_threshold or RefusalGateConfig.MEDIUM_CONFIDENCE_THRESHOLD,
        low_confidence_threshold=low_threshold or RefusalGateConfig.LOW_CONFIDENCE_THRESHOLD
    )


def make_decision(
    synthesis_output: Dict,
    fact_check_output: Dict,
    query_analysis: Dict
) -> Dict:
    """Convenience function to analyze quality and return response with indicators"""
    gate = get_refusal_gate()
    return gate.decide(synthesis_output, fact_check_output, query_analysis)

