from typing import Dict, List, Optional, Any
import numpy as np
import json
from datetime import datetime

from ..core.utils import logger, Timer
from ..core.database import Ultima_RAGDatabase
from ..data.embedder import DeterministicEmbedder, get_embedder

class RetrieverAgent:
    """
    SOTA Retriever Agent for Ultima_RAG.
    Utilizes LanceDB for project-aware, vector-native retrieval.
    """
    
    def __init__(
        self,
        db: Ultima_RAGDatabase,
        embedder: Optional[DeterministicEmbedder] = None
    ):
        from ..core.config import RerankerConfig
        from sentence_transformers import CrossEncoder
        
        self.db = db
        self.embedder = embedder or get_embedder()
        
        # SOTA Optimization: Initialize Reranker once to prevent latency on every query
        try:
            logger.info(f"Loading Reranker: {RerankerConfig.MODEL_NAME}...")
            self.reranker = CrossEncoder(RerankerConfig.MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to initialize reranker {RerankerConfig.MODEL_NAME}: {e}")
            self.reranker = None
            
        logger.info(f"RetrieverAgent (Ultima_RAG Edition) initialized | Reranker: {RerankerConfig.MODEL_NAME}")
    
    async def retrieve(
        self,
        query: str,
        project_id: str = "default",
        file_names: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        include_multimodal: bool = True,
        conversation_id: str = "default"
    ) -> Dict:
        """
        Execute unified retrieval for a query, with SOTA Reranking.
        """
        from ..core.config import RerankerConfig, Config
        from sentence_transformers import CrossEncoder

        # SOTA: Respect .env-driven global chunk limit if not overridden
        top_k = top_k or Config.retrieval.FINAL_TOP_K

        with Timer("Ultima_RAG Retrieval + Rerank"):
            # 1. Generate Query Embedding
            query_vector = self.embedder.encode(query).tolist()
            
            # 2. Perform SOTA Hybrid Search (LanceDB Native Dense + BM25)
            initial_pool = RerankerConfig.RERANK_TOP_K
            
            # A. Search main knowledge base
            results = self.db.search_knowledge(
                query_vector=query_vector,
                query_text=query,
                project_id=project_id,
                conversation_id=conversation_id,
                file_names=file_names,
                limit=initial_pool
            )
            
            # B. SOTA Phase 23: Search persistent web cache (Scoped to conversation)
            # This allows future queries to reference previous web scrapes without re-searching.
            # We skip this if specific files are @mentioned (focus on docs only).
            if not file_names:
                web_results = self.db.search_web_knowledge(
                    query_vector=query_vector,
                    conversation_id=conversation_id,
                    limit=initial_pool // 2
                )
                if web_results:
                    logger.info(f"Retriever: Found {len(web_results)} relevant chunks in Web Cache.")
                    results.extend(web_results)
            
            if not results:
                return {"evidence": [], "query_used": query, "total_retrieved": 0}

            # 3. SOTA Reranking Pass (Cross-Encoder)
            try:
                if getattr(self, "reranker", None):
                    pairs = [[query, r.get('text', '')] for r in results]
                    scores = self.reranker.predict(pairs)
                    
                    for i, score in enumerate(scores):
                        results[i]['rerank_score'] = float(score)
                    
                    # Sort by rerank score descending
                    results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
                    logger.info(f"Retriever: Reranked {len(results)} chunks. Selected top {top_k}.")
                else:
                    logger.warning("Reranker not loaded. Skipping reranking pass.")
                    
                # Keep top_k after reranking or base vector search
                final_results = results[:top_k]
            except Exception as e:
                logger.error(f"Reranking failed: {e}. Falling back to vector ranking.")
                final_results = results[:top_k]
            
        return {
            "evidence": final_results,
            "query_used": query,
            "file_names_filtered": file_names,
            "total_retrieved": len(final_results)
        }

    def retrieve_multimodal(self, conversation_id: str, query: Optional[str] = None) -> List[Dict]:
        """
        SOTA: Fetch multimodal evidence for ALL files in a conversation.
        For each file, prioritizes High-Fidelity Enriched Narratives, falling back 
        to Raw Scraped Clips if enrichment is not yet available.
        """
        logger.info(f"Retriever: Fetching multimodal evidence for chat {conversation_id}")
        
        # 1. Get all assets to ensure we account for every file
        assets = self.db.get_assets(conversation_id)
        if not assets:
            logger.info("Retriever: No assets found for this conversation.")
            return []
            
        evidence = []
        for asset in assets:
            asset_id = asset['id']
            file_name = asset.get('file_name', 'Unknown')
            
            # A. Try Enriched Content first
            enriched = self.db.get_enriched_content_by_file_id(asset_id)
            if enriched and enriched.get('enriched_content'):
                evidence.append({
                    "chunk_id": f"mm_enriched_{asset_id[:8]}",
                    "text": enriched['enriched_content'],
                    "source": file_name,
                    "file_type": enriched.get('content_type', 'media'),
                    "score": 1.0, # High confidence
                    "metadata": enriched.get('metadata', {})
                })
                continue
            
            # B. Fallback to Scraped Content
            scraped = self.db.get_scraped_content(asset_id)
            if scraped:
                for item in scraped:
                    evidence.append({
                        "chunk_id": f"mm_scraped_{item['id'][:8]}",
                        "text": item['content'],
                        "source": file_name,
                        "file_type": asset.get('file_type', 'media'),
                        "score": 0.9, # Raw confidence
                        "metadata": item.get('metadata', {})
                    })
        
        logger.info(f"Retriever: Found evidence for {len(assets)} files. Total items: {len(evidence)}")
        return evidence

