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
Core Configuration Module for UltimaRAG
Centralized configuration management with environment variable support.

ALL performance-sensitive values are driven by environment variables.
To update hardware, simply update the .env file — no code changes needed.

Quick reference for a high-end GPU machine (e.g. RTX 4090 / A100):
  MAX_INPUT_TOKENS=32768
  MAX_OUTPUT_TOKENS=4096
  LIGHTWEIGHT_MAX_TOKENS=2048
  HEAVY_MAX_TOKENS=4096
  CHUNK_SIZE=1500
  CHUNK_OVERLAP=200
  FINAL_TOP_K=10
"""

import os
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables from Credentials folder
CREDENTIALS_PATH = Path(__file__).parent.parent.parent / "Credentials" / ".env"
load_dotenv(CREDENTIALS_PATH)

from .utils import logger

# =============================================================================
# LLM CONFIGURATION (OLLAMA - LOCAL)
# =============================================================================

class OllamaConfig:
    """
    Configuration for LLM Provider (Default: Ollama).
    All values are loaded from .env — safe defaults tuned for Gemma3:12b on 6GB VRAM.
    """
    # Connection
    BASE_URL: str = os.getenv("LLM_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    MODEL_NAME: str = os.getenv("MODEL_NAME", "gemma3:12b")

    # Provider
    PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    API_KEY: str = os.getenv("LLM_API_KEY", "")

    # -------------------------------------------------------------------------
    # Context Window Budget
    # These are the MASTER limits — all other token caps derive from these.
    # Gemma3:12b default: 2048 total context (input + output combined)
    # A100 / RTX4090 example: MAX_INPUT_TOKENS=32768, MAX_OUTPUT_TOKENS=4096
    # -------------------------------------------------------------------------
    MAX_INPUT_TOKENS: int = int(os.getenv("MAX_INPUT_TOKENS", "2048"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "512"))

    # Generation behavior
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "600"))


class OllamaMultiModelConfig:
    """
    Per-task token budgets. Lightweight = fast tasks; Heavy = complex tasks.
    Both default to the same model (single-GPU) but can be split on multi-GPU setups.

    ENV keys:
      LIGHTWEIGHT_MODEL         — model for fast tasks (default: MODEL_NAME)
      LIGHTWEIGHT_MAX_TOKENS    — max generated tokens for lightweight tasks (default: 512)
      HEAVY_MODEL               — model for heavy tasks (default: MODEL_NAME)
      HEAVY_MAX_TOKENS          — max generated tokens for heavy tasks (default: 1024)

    Hardware upgrade guide:
      RTX 3090 / 4090:  LIGHTWEIGHT_MAX_TOKENS=1024, HEAVY_MAX_TOKENS=2048
      A100 / H100:      LIGHTWEIGHT_MAX_TOKENS=2048, HEAVY_MAX_TOKENS=4096
    """
    target_model: str = os.getenv("MODEL_NAME", "gemma3:12b")

    # Lightweight model (intent classification, query analysis, humanization)
    LIGHTWEIGHT_MODEL: str = os.getenv("LIGHTWEIGHT_MODEL", target_model)
    LIGHTWEIGHT_MAX_TOKENS: int = int(os.getenv("LIGHTWEIGHT_MAX_TOKENS", "512"))

    # Heavy model (summarization, translation, synthesis, fact-checking)
    HEAVY_MODEL: str = os.getenv("HEAVY_MODEL", target_model)
    HEAVY_MAX_TOKENS: int = int(os.getenv("HEAVY_MAX_TOKENS", "1024"))

    # Per-agent model assignments — all default to target_model, overridable individually
    AGENT_MODELS = {
        "query_analyzer": os.getenv("AGENT_MODEL_QUERY_ANALYZER", target_model),
        "synthesizer":    os.getenv("AGENT_MODEL_SYNTHESIZER",    target_model),
        "humanizer":      os.getenv("AGENT_MODEL_HUMANIZER",      target_model),
        "fact_checker":   os.getenv("AGENT_MODEL_FACT_CHECKER",   target_model),
        "translator":     os.getenv("AGENT_MODEL_TRANSLATOR",     target_model),
        "rewriter":       os.getenv("AGENT_MODEL_REWRITER",       target_model),
        "chat":           os.getenv("AGENT_MODEL_CHAT",           target_model),
    }


# =============================================================================
# MEMGPT CONFIGURATION
# =============================================================================

class MemGPTConfig:
    """
    MemGPT sliding-window configuration.
    Controls how history is paged in/out of the active context window.

    ENV keys:
      MEMGPT_OVERFLOW_RATIO   — fraction of MAX_INPUT_TOKENS to use for history (default: 0.80)
                                e.g. 0.80 means 80% of 2048 = 1638 tokens for history
      MEMGPT_SUMMARY_TOKENS   — max tokens for the archival turn summary (default: 128)
      MEMGPT_CONTENT_TRUNCATE — max chars per message side before summarization (default: 400)

    The THRESHOLD is always derived from OllamaConfig.MAX_INPUT_TOKENS × OVERFLOW_RATIO,
    so upgrading the context window (MAX_INPUT_TOKENS) automatically extends memory too.
    """
    OVERFLOW_RATIO: float = float(os.getenv("MEMGPT_OVERFLOW_RATIO", "0.80"))
    SUMMARY_MAX_TOKENS: int = int(os.getenv("MEMGPT_SUMMARY_TOKENS", "128"))
    CONTENT_TRUNCATE_CHARS: int = int(os.getenv("MEMGPT_CONTENT_TRUNCATE", "400"))

    @classmethod
    def get_threshold(cls) -> int:
        """
        Dynamically computed overflow threshold.
        Always = MAX_INPUT_TOKENS × OVERFLOW_RATIO.
        Updates automatically when MAX_INPUT_TOKENS changes in .env.
        """
        return int(OllamaConfig.MAX_INPUT_TOKENS * cls.OVERFLOW_RATIO)


# =============================================================================
# RETRIEVAL CONFIGURATION
# =============================================================================

class RetrievalConfig:
    """
    Hybrid retrieval configuration.

    ENV keys:
      RETRIEVAL_DENSE_TOP_K   — candidates from dense (vector) search (default: 50)
      RETRIEVAL_SPARSE_TOP_K  — candidates from sparse (keyword) search (default: 50)
      RETRIEVAL_HYBRID_ALPHA  — weight for dense vs sparse, 1.0 = pure dense (default: 0.7)
      RETRIEVAL_MMR_LAMBDA    — diversity weight in MMR, 1.0 = pure relevance (default: 0.5)
      RETRIEVAL_FINAL_TOP_K   — chunks sent to LLM after reranking (default: 3)
                                Gemma3:12b: keep at 3. RTX4090: 8–10. A100+: 15–20
      RETRIEVAL_MIN_SCORE     — minimum similarity score to include a chunk (default: 0.3)
    """
    DENSE_TOP_K: int = int(os.getenv("RETRIEVAL_DENSE_TOP_K", "50"))
    SPARSE_TOP_K: int = int(os.getenv("RETRIEVAL_SPARSE_TOP_K", "50"))
    HYBRID_ALPHA: float = float(os.getenv("RETRIEVAL_HYBRID_ALPHA", "0.7"))
    MMR_LAMBDA: float = float(os.getenv("RETRIEVAL_MMR_LAMBDA", "0.5"))
    FINAL_TOP_K: int = int(os.getenv("RETRIEVAL_FINAL_TOP_K", "3"))
    MIN_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.3"))
    MAX_CHUNKS: int = int(os.getenv("RETRIEVAL_MAX_CHUNKS", str(FINAL_TOP_K)))


class RerankerConfig:
    """
    Cross-encoder reranker configuration.

    ENV keys:
      RERANKER_MODEL          — HuggingFace reranker model (default: BAAI/bge-reranker-v2-m3)
      RERANKER_THRESHOLD      — min reranker score to keep a chunk (default: 0.1)
      RERANKER_TOP_K          — how many candidates to rerank (default: 50)
      RERANKER_BATCH_SIZE     — batch size for inference (default: 32)
      RERANKER_MIN_KEEP       — always keep at least this many many chunks (default: 3)
    """
    MODEL_NAME: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANK_THRESHOLD: float = float(os.getenv("RERANKER_THRESHOLD", "0.1"))
    RERANK_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "50"))
    BATCH_SIZE: int = int(os.getenv("RERANKER_BATCH_SIZE", "32"))
    MIN_CHUNKS_TO_KEEP: int = int(os.getenv("RERANKER_MIN_KEEP", "3"))


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

class FactCheckConfig:
    """Fact checker configuration"""
    NLI_MODEL: str = os.getenv("FACT_CHECK_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    SUPPORT_THRESHOLD: float = float(os.getenv("FACT_CHECK_SUPPORT_THRESHOLD", "0.7"))
    MIN_FACTUALITY_SCORE: float = float(os.getenv("FACT_CHECK_MIN_FACTUALITY", "0.75"))


class RefusalGateConfig:
    """Confidence level classification configuration (no longer blocking)"""
    # Thresholds for confidence level classification (for UI display)
    HIGH_CONFIDENCE_THRESHOLD: float = float(os.getenv("REFUSAL_HIGH_CONF", "0.85"))
    MEDIUM_CONFIDENCE_THRESHOLD: float = float(os.getenv("REFUSAL_MED_CONF", "0.65"))
    LOW_CONFIDENCE_THRESHOLD: float = float(os.getenv("REFUSAL_LOW_CONF", "0.45"))
    MIN_FACTUALITY: float = float(os.getenv("REFUSAL_MIN_FACTUALITY", "0.75"))
    MAX_UNSUPPORTED_CLAIMS: int = int(os.getenv("REFUSAL_MAX_UNSUPPORTED", "2"))
    MIN_SYNTHESIS_CONFIDENCE: float = float(os.getenv("REFUSAL_MIN_SYNTHESIS_CONF", "0.6"))



# SynthesisConfig has been intentionally REMOVED.
# The iterative continuation loop (CONTINUE_PROMPT / MAX_CONTINUATION_LOOPS)
# caused Context Inflation crashes on low-resource hardware (2048-ctx).
# Ultima_RAG now uses Single-Shot generation with a prompt Token Budget Guard.
# Response quality is maintained via the Healer node in metacognitive_brain.py.


# =============================================================================
# EMBEDDINGS CONFIGURATION
# =============================================================================

class EmbeddingConfig:
    """
    Embedding model configuration.

    ENV keys:
      EMBEDDING_MODEL     — sentence-transformer model (default: BAAI/bge-m3)
      EMBEDDING_DIMENSION — output vector dimension; MUST match the model (default: 1024)
      EMBEDDING_BATCH     — batch size for encoding (default: 32)
      EMBEDDING_CACHE     — whether to cache embeddings (default: true)

    WARNING: Changing EMBEDDING_MODEL requires changing EMBEDDING_DIMENSION too.
    The Dimension Mismatch Guardian in database.py will auto-recreate incompatible tables.
    """
    MODEL_NAME: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH", "32"))
    CACHE_EMBEDDINGS: bool = os.getenv("EMBEDDING_CACHE", "true").lower() == "true"


# =============================================================================
# CHUNKING CONFIGURATION
# =============================================================================

class ChunkingConfig:
    """
    Document chunking configuration (token-based).

    ENV keys:
      CHUNK_SIZE        — target tokens per chunk (default: 500)
      CHUNK_OVERLAP     — token overlap between chunks (default: 100)
      CHUNK_MIN_SIZE    — merge chunks smaller than this (default: 150)
      CHUNK_MAX_SIZE    — hard guardrail, never exceed this (default: 1000)

    Hardware upgrade guide:
      Gemma3:12b (2048 ctx):   CHUNK_SIZE=500,  CHUNK_OVERLAP=100, CHUNK_MAX_SIZE=1000
      RTX 4090  (8192 ctx):   CHUNK_SIZE=1000, CHUNK_OVERLAP=150, CHUNK_MAX_SIZE=2000
      A100      (32768 ctx):  CHUNK_SIZE=2000, CHUNK_OVERLAP=200, CHUNK_MAX_SIZE=4000
    """
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    MIN_CHUNK_SIZE: int = int(os.getenv("CHUNK_MIN_SIZE", "150"))
    MAX_CHUNK_SIZE: int = int(os.getenv("CHUNK_MAX_SIZE", "1000"))


class LongT5Config:
    """Google LongT5 deduplication configuration for video frames"""
    MODEL_NAME: str = os.getenv("LONGT5_MODEL", "google/long-t5-tglobal-base")
    MAX_INPUT_TOKENS: int = int(os.getenv("LONGT5_MAX_INPUT", "2048"))
    MAX_OUTPUT_TOKENS: int = int(os.getenv("LONGT5_MAX_OUTPUT", "1024"))
    USE_FP16: bool = os.getenv("LONGT5_FP16", "true").lower() == "true"
    DEVICE: str = os.getenv("LONGT5_DEVICE", "cuda")


# =============================================================================
# DETERMINISTIC MODE
# =============================================================================

class DeterministicConfig:
    """Configuration for reproducible outputs"""
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    TORCH_DETERMINISTIC: bool = os.getenv("TORCH_DETERMINISTIC", "true").lower() == "true"
    FAISS_NPROBE: int = int(os.getenv("FAISS_NPROBE", "10"))
    MMR_SEED: int = int(os.getenv("MMR_SEED", "42"))
    CACHE_EMBEDDINGS: bool = os.getenv("EMBEDDING_CACHE", "true").lower() == "true"


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

class DatabaseConfig:
    """Database configuration"""
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "5"))
    POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))


# =============================================================================
# PATHS CONFIGURATION
# =============================================================================

class PathConfig:
    """File and directory paths"""
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    MODELS_DIR: Path = BASE_DIR / "models"
    INDEXES_DIR: Path = MODELS_DIR / "indexes"
    CACHE_DIR: Path = BASE_DIR / "cache"
    Ultima_DB_DIR: Path = DATA_DIR / "Ultima_db"

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories"""
        dirs = [
            cls.DATA_DIR, cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR,
            cls.EMBEDDINGS_DIR, cls.MODELS_DIR, cls.INDEXES_DIR, cls.CACHE_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# UNIFIED CONFIG ACCESS
# =============================================================================

class Config:
    """
    Unified configuration access point.
    All values are environment-variable driven.
    See Credentials/.env for the full list of configurable keys.
    """
    ollama = OllamaConfig
    retrieval = RetrievalConfig
    reranker = RerankerConfig
    fact_check = FactCheckConfig
    refusal_gate = RefusalGateConfig
    embedding = EmbeddingConfig
    chunking = ChunkingConfig
    deterministic = DeterministicConfig
    database = DatabaseConfig
    paths = PathConfig
    ollama_multi_model = OllamaMultiModelConfig
    memgpt = MemGPTConfig
    # synthesis config removed — continuation loop was deprecated.

    @classmethod
    def validate_context_budget(cls):
        """
        SOTA: Industrial Context Guardian.
        Validates the synergy between CHUNK_SIZE, FINAL_TOP_K and MAX_INPUT_TOKENS.
        Prevents 'Context Inflation' crashes on low-resource (2048ctx) hardware.
        """
        max_ctx = cls.ollama.MAX_INPUT_TOKENS
        chunk_size = cls.chunking.CHUNK_SIZE
        top_k = cls.retrieval.FINAL_TOP_K
        
        rag_load = chunk_size * top_k
        safety_threshold = int(max_ctx * 0.7) # Leave 30% for system prompt & response
        
        logger.info(f"🛡️ Context Guardian: RAG Load={rag_load} tokens (Budget={safety_threshold}/{max_ctx})")
        
        if rag_load > safety_threshold:
            logger.warning("⚠️ CRITICAL CONTEXT DENSITY WARNING ⚠️")
            logger.warning(f"Settings: CHUNK_SIZE={chunk_size}, TOP_K={top_k} => Total={rag_load}")
            logger.warning(f"Hardware Limit: {max_ctx} tokens.")
            logger.warning(f"RISK: 'Context Inflation'. Short prompts will likely exceed {max_ctx} and crash.")
            
            # Suggest calibration
            suggested_top_k = max(1, safety_threshold // chunk_size)
            suggested_chunk = max(100, safety_threshold // max(1, top_k))
            
            logger.warning(f"FIX 1: Reduce RETRIEVAL_FINAL_TOP_K to {suggested_top_k}")
            logger.warning(f"FIX 2: Reduce CHUNK_SIZE to {suggested_chunk}")
            logger.warning("---------------------------------------------")
        else:
            logger.info("✅ Context Budget: STABLE for current hardware.")

    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration - check Ollama connectivity"""
        import requests
        try:
            response = requests.get(f"{cls.ollama.BASE_URL}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ValueError(f"Ollama not responding at {cls.ollama.BASE_URL}")
            return True
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"Cannot connect to Ollama at {cls.ollama.BASE_URL}. "
                "Please ensure Ollama is running: `ollama serve`"
            )


# Initialize paths on import
PathConfig.ensure_dirs()

