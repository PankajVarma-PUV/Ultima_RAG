"""
Reranker Module for Ultima_RAG
Cross-encoder reranking for precision filtering.
"""

from typing import Dict, List, Optional
import numpy as np

from sentence_transformers import CrossEncoder

from ..core.config import RerankerConfig
from ..core.utils import logger, Timer


# =============================================================================
# RERANKER MODULE
# =============================================================================

class RerankerModule:
    """
    Agent 3: Reranker Module
    
    Responsibilities:
    - Score query-chunk relevance using cross-encoder
    - Filter chunks below threshold
    - Re-order chunks by relevance
    - Preserve top-N for synthesis
    """
    
    def __init__(
        self,
        model_name: str = RerankerConfig.MODEL_NAME,
        threshold: float = RerankerConfig.RERANK_THRESHOLD,
        top_k: int = RerankerConfig.RERANK_TOP_K
    ):
        """
        Initialize Reranker with cross-encoder model.
        
        Args:
            model_name: Cross-encoder model name
            threshold: Minimum score threshold for filtering
            top_k: Number of top results to return
        """
        self.model = CrossEncoder(model_name)
        self.threshold = threshold
        self.top_k = top_k
        
        logger.info(f"RerankerModule initialized with model: {model_name}")
        logger.info(f"Threshold: {threshold}, Top-K: {top_k}")
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> Dict:
        """
        Rerank chunks using cross-encoder scoring.
        
        Args:
            query: Original user query
            chunks: Retrieved chunks from Retriever Agent
            threshold: Override default threshold
            top_k: Override default top_k
        
        Returns:
            Reranking results dictionary
        """
        threshold = threshold if threshold is not None else self.threshold
        top_k = top_k or self.top_k
        
        if not chunks:
            return {
                "reranked_chunks": [],
                "filtered_count": 0,
                "dropped_count": 0
            }
        
        with Timer("Reranking"):
            # Prepare query-chunk pairs
            pairs = [(query, chunk["text"]) for chunk in chunks]
            
            # Get cross-encoder scores
            scores = self.model.predict(
                pairs,
                batch_size=RerankerConfig.BATCH_SIZE,
                show_progress_bar=False
            )
            
            # Ensure scores is numpy array
            scores = np.array(scores)
            
            # Attach scores to chunks  
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                chunk_copy = {**chunk}
                # Use .item() to convert numpy scalar to Python float
                chunk_copy['rerank_score'] = score.item() if hasattr(score, 'item') else float(score)
                chunk_copy['retrieval_score'] = chunk.get('score', 0.0)
                scored_chunks.append(chunk_copy)
            
            # Sort by rerank score
            scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            # Filter by threshold (PROTECT multimodal evidence)
            filtered = [
                chunk for chunk in scored_chunks 
                if chunk['rerank_score'] >= threshold or chunk.get('metadata', {}).get('content_type') == 'multimodal_evidence'
            ]
            
            # ALWAYS keep at least MIN_CHUNKS_TO_KEEP chunks even if below threshold
            min_chunks = getattr(RerankerConfig, 'MIN_CHUNKS_TO_KEEP', 3)
            if len(filtered) < min_chunks:
                # Add more from scored_chunks until we have minimum
                filtered = scored_chunks[:max(min_chunks, len(filtered))]
                logger.info(f"Keeping {len(filtered)} chunks (forced minimum)")
            
            # Get top-k
            reranked = filtered[:top_k]
            
            # Add rank
            for i, chunk in enumerate(reranked, 1):
                chunk['rank'] = i
        
        dropped_count = len(chunks) - len(filtered)
        
        result = {
            "reranked_chunks": reranked,
            "filtered_count": len(reranked),
            "dropped_count": dropped_count,
            "original_count": len(chunks),
            "threshold_used": threshold,
            "scores": {
                "max": float(np.max(scores)) if len(scores) > 0 else 0.0,
                "min": float(np.min(scores)) if len(scores) > 0 else 0.0,
                "mean": float(np.mean(scores)) if len(scores) > 0 else 0.0
            }
        }
        
        logger.info(
            f"Reranked: {len(chunks)} → {len(reranked)} chunks "
            f"(dropped {dropped_count} below threshold {threshold})"
        )
        
        return result
    
    def score_single(self, query: str, text: str) -> float:
        """
        Score a single query-text pair.
        
        Args:
            query: Query string
            text: Text to score against query
        
        Returns:
            Relevance score
        """
        scores = self.model.predict([(query, text)])
        score = scores[0]
        return score.item() if hasattr(score, 'item') else float(score)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_reranker(
    threshold: Optional[float] = None,
    top_k: Optional[int] = None
) -> RerankerModule:
    """Get a configured RerankerModule instance"""
    return RerankerModule(
        threshold=threshold or RerankerConfig.RERANK_THRESHOLD,
        top_k=top_k or RerankerConfig.RERANK_TOP_K
    )


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    threshold: Optional[float] = None,
    top_k: Optional[int] = None
) -> List[Dict]:
    """Convenience function to rerank chunks"""
    reranker = get_reranker(threshold, top_k)
    result = reranker.rerank(query, chunks, threshold, top_k)
    return result["reranked_chunks"]

