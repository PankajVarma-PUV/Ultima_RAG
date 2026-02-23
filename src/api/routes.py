"""
API Routes for Ultima_RAG
Additional route handlers and utilities.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core.utils import logger


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(prefix="/api/v1", tags=["RAG API"])


# =============================================================================
# MODELS
# =============================================================================

class BatchQueryRequest(BaseModel):
    """Request for batch query processing"""
    queries: List[str]
    include_debug: bool = False


class BatchQueryResponse(BaseModel):
    """Response for batch query processing"""
    results: List[dict]
    total: int
    successful: int


class AnalyzeRequest(BaseModel):
    """Request for query analysis only"""
    query: str


class SearchRequest(BaseModel):
    """Request for search only (no synthesis)"""
    query: str
    mode: str = "hybrid"
    top_k: int = 10


# =============================================================================
# ROUTES
# =============================================================================

@router.post("/analyze")
async def analyze_query(request: AnalyzeRequest):
    """
    Analyze a query without running full pipeline.
    Useful for debugging query decomposition.
    """
    from ..agents.query_analyzer import QueryAnalyzer
    
    try:
        analyzer = QueryAnalyzer()
        result = analyzer.analyze(request.query)
        return {"success": True, "analysis": result}
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_only(request: SearchRequest):
    """
    Search without synthesis.
    Returns raw retrieval results.
    """
    # This would need access to the app_state
    # For now, return a placeholder
    return {
        "message": "Use /query endpoint for full functionality",
        "note": "Search-only mode coming soon"
    }


@router.post("/batch")
async def batch_query(request: BatchQueryRequest):
    """
    Process multiple queries in batch.
    """
    results = []
    successful = 0
    
    for query in request.queries:
        try:
            # Would call orchestrator here
            results.append({
                "query": query,
                "status": "pending",
                "message": "Batch processing coming soon"
            })
        except Exception as e:
            results.append({
                "query": query,
                "status": "error",
                "error": str(e)
            })
    
    return BatchQueryResponse(
        results=results,
        total=len(request.queries),
        successful=successful
    )


@router.get("/config")
async def get_config():
    """
    Get current system configuration (non-sensitive).
    """
    from ..core.config import Config
    
    return {
        "retrieval": {
            "dense_top_k": Config.retrieval.DENSE_TOP_K,
            "sparse_top_k": Config.retrieval.SPARSE_TOP_K,
            "hybrid_alpha": Config.retrieval.HYBRID_ALPHA,
            "final_top_k": Config.retrieval.FINAL_TOP_K
        },
        "reranker": {
            "threshold": Config.reranker.RERANK_THRESHOLD,
            "top_k": Config.reranker.RERANK_TOP_K
        },
        "fact_check": {
            "support_threshold": Config.fact_check.SUPPORT_THRESHOLD,
            "min_factuality": Config.fact_check.MIN_FACTUALITY_SCORE
        },
        "refusal_gate": {
            "min_factuality": Config.refusal_gate.MIN_FACTUALITY,
            "max_unsupported": Config.refusal_gate.MAX_UNSUPPORTED_CLAIMS
        }
    }


@router.get("/metrics")
async def get_metrics():
    """
    Get system metrics (placeholder).
    """
    return {
        "queries_processed": 0,
        "avg_latency_ms": 0,
        "refusal_rate": 0
    }

