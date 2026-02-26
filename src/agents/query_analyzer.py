# UltimaRAG — Multi-Agent RAG System
# Copyright (C) 2026 Pankaj Varma
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Query Analyzer Agent for UltimaRAG
Transforms user queries into retrieval-optimized formats with intent detection.
Uses local Ollama with configured model for LLM operations.
"""

import json
import re
from typing import Dict, List, Optional

from ..core.config import Config, OllamaConfig
from ..core.ollama_client import OllamaClient, get_ollama_client
from ..core.prompts import QUERY_ANALYSIS_PROMPT
from ..core.utils import logger, Timer


# =============================================================================
# QUERY ANALYZER AGENT
# =============================================================================

class QueryAnalyzer:
    """
    Agent 1: Query Analyzer
    
    Responsibilities:
    - Detect query intent (factual, comparative, procedural, multi-hop)
    - Decompose complex questions into sub-queries
    - Generate retrieval-optimized query variations
    - Extract entities and temporal constraints
    - Classify query difficulty
    """
    
    def __init__(self):
        """
        Initialize Query Analyzer with Ollama client.
        """
        model = Config.ollama_multi_model.AGENT_MODELS.get("query_analyzer", OllamaConfig.MODEL_NAME)
        self.client = OllamaClient(model_name=model)
        
        if not self.client.is_available():
            logger.warning("Ollama not available - will use fallback analysis")
        else:
            logger.info(f"QueryAnalyzer initialized with Ollama: {model}")
    
    def analyze(self, query: str) -> Dict:
        """
        Analyze a user query and return structured analysis.
        
        Args:
            query: Raw user query string
        
        Returns:
            Query analysis dictionary with intent, sub-queries, etc.
        """
        with Timer("Query Analysis"):
            try:
                # Try LLM-based analysis first
                if self.client.is_available():
                    result = self._analyze_with_llm(query)
                else:
                    result = self._analyze_fallback(query)
            except Exception as e:
                logger.warning(f"LLM analysis failed: {e}. Using fallback.")
                result = self._analyze_fallback(query)
        
        # Ensure all required fields exist
        result = self._validate_and_complete(result, query)
        
        logger.info(f"Query analyzed: intent={result['intent']}, difficulty={result['difficulty']}")
        return result
    
    def _analyze_with_llm(self, query: str) -> Dict:
        """Use Ollama to analyze the query"""
        prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
        
        response = self.client.generate(
            prompt,
            temperature=OllamaConfig.TEMPERATURE,
            max_tokens=1024
        )
        
        # Parse JSON response
        response_text = response.strip()
        
        # Clean up markdown code blocks if present
        if response_text.startswith("```"):
            response_text = re.sub(r'^```(?:json)?\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)
        
        # Try to find JSON in response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        
        return json.loads(response_text)
    
    def _analyze_fallback(self, query: str) -> Dict:
        """
        Fallback rule-based analysis when LLM fails.
        Uses heuristics for intent detection and entity extraction.
        """
        query_lower = query.lower().strip()
        
        # Intent detection heuristics
        intent = "factual"
        
        if any(w in query_lower for w in ["compare", "difference", "vs", "versus", "better"]):
            intent = "comparative"
        elif any(w in query_lower for w in ["how to", "how do", "steps to", "guide to"]):
            intent = "procedural"
        elif query_lower.count("?") > 1 or " and " in query_lower:
            intent = "multi-hop"
        
        # Simple entity extraction (capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities = list(set(entities))[:5]
        
        # Generate retrieval queries
        retrieval_queries = [
            query,
            ' '.join(w for w in query.split() if len(w) > 3),  # Keywords only
        ]
        if entities:
            retrieval_queries.append(' '.join(entities))
        
        # Difficulty based on length and complexity
        word_count = len(query.split())
        if word_count < 10 and intent == "factual":
            difficulty = "simple"
        elif word_count > 25 or intent == "multi-hop":
            difficulty = "complex"
        else:
            difficulty = "medium"
        
        return {
            "original_query": query,
            "intent": intent,
            "sub_queries": [query],
            "retrieval_queries": retrieval_queries,
            "entities": entities,
            "temporal_constraint": None,
            "difficulty": difficulty
        }
    
    def _validate_and_complete(self, result: Dict, original_query: str) -> Dict:
        """Ensure all required fields exist with valid values"""
        default = {
            "original_query": original_query,
            "intent": "factual",
            "sub_queries": [original_query],
            "retrieval_queries": [original_query],
            "entities": [],
            "temporal_constraint": None,
            "difficulty": "simple"
        }
        
        for key, default_value in default.items():
            if key not in result or result[key] is None:
                result[key] = default_value
        
        # Ensure original_query is correct
        result["original_query"] = original_query
        
        # Ensure at least one retrieval query
        if not result["retrieval_queries"]:
            result["retrieval_queries"] = [original_query]
        
        # Validate intent
        valid_intents = ["factual", "comparative", "procedural", "multi-hop"]
        if result["intent"] not in valid_intents:
            result["intent"] = "factual"
        
        # Validate difficulty
        valid_difficulties = ["simple", "medium", "complex"]
        if result["difficulty"] not in valid_difficulties:
            result["difficulty"] = "medium"
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_query_analyzer() -> QueryAnalyzer:
    """Get a configured QueryAnalyzer instance"""
    return QueryAnalyzer()


def analyze_query(query: str) -> Dict:
    """Convenience function to analyze a query"""
    analyzer = get_query_analyzer()
    return analyzer.analyze(query)

