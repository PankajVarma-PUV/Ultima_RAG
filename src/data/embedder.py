"""
Embedding Module for Ultima_RAG
Deterministic embedder with caching for reproducible retrieval.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Union, Optional
import numpy as np

from sentence_transformers import SentenceTransformer

from ..core.config import EmbeddingConfig, PathConfig, DeterministicConfig
from ..core.utils import logger, set_seed, get_deterministic_hash


# =============================================================================
# DETERMINISTIC EMBEDDER
# =============================================================================

class DeterministicEmbedder:
    """
    Embedding generator with caching for deterministic outputs.
    Uses sentence-transformers with optional GPU acceleration.
    """
    
    def __init__(
        self,
        model_name: str = EmbeddingConfig.MODEL_NAME,
        cache_path: Optional[Path] = None,
        use_cache: bool = EmbeddingConfig.CACHE_EMBEDDINGS,
        device: Optional[str] = None
    ):
        """
        Initialize embedder with model and cache.
        
        Args:
            model_name: Sentence-transformers model name or path
            cache_path: Path to embedding cache file
            use_cache: Whether to cache embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        # Set seed for determinism
        set_seed(DeterministicConfig.RANDOM_SEED)
        
        # Initialize model
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        
        # Setup caching
        self.use_cache = use_cache
        self.cache_path = cache_path or (PathConfig.CACHE_DIR / "embeddings_cache.pkl")
        self.cache: Dict[str, np.ndarray] = self._load_cache()
        
        logger.info(f"DeterministicEmbedder initialized with model: {model_name}")
        logger.info(f"Cache enabled: {use_cache}, Cache size: {len(self.cache)}")
    
    def _load_cache(self) -> Dict[str, np.ndarray]:
        """Load embedding cache from disk"""
        if self.use_cache and self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)
                    logger.info(f"Loaded embedding cache with {len(cache)} entries")
                    return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save embedding cache to disk"""
        if self.use_cache:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, 'wb') as f:
                    pickle.dump(self.cache, f)
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate deterministic cache key for text"""
        # Normalize and hash for consistent keys
        normalized = text.strip().lower()
        return get_deterministic_hash(normalized)
    
    def encode(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings with caching.
        
        Args:
            text: Single text or list of texts to encode
            normalize: Whether to L2 normalize embeddings
            show_progress: Show progress bar for batch encoding
        
        Returns:
            Embedding vector(s) as numpy array
        """
        # Handle single text
        if isinstance(text, str):
            return self._encode_single(text, normalize)
        
        # Handle batch
        return self._encode_batch(text, normalize, show_progress)
    
    def _encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode single text with cache lookup"""
        cache_key = self._get_cache_key(text)
        
        # Check cache
        if self.use_cache and cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate embedding
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        # Cache result
        if self.use_cache:
            self.cache[cache_key] = embedding
        
        return embedding
    
    def _encode_batch(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """Encode batch of texts with partial cache lookup"""
        embeddings = []
        texts_to_encode = []
        text_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if self.use_cache and cache_key in self.cache:
                embeddings.append((i, self.cache[cache_key]))
            else:
                texts_to_encode.append(text)
                text_indices.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            new_embeddings = self.model.encode(
                texts_to_encode,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                batch_size=EmbeddingConfig.BATCH_SIZE
            )
            
            # Cache new embeddings
            for text, embedding in zip(texts_to_encode, new_embeddings):
                cache_key = self._get_cache_key(text)
                if self.use_cache:
                    self.cache[cache_key] = embedding
                embeddings.append((text_indices[texts_to_encode.index(text)], embedding))
        
        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([e[1] for e in embeddings])
        
        return result
    
    def save_cache(self) -> None:
        """Manually save cache to disk"""
        self._save_cache()
        logger.info(f"Cache saved with {len(self.cache)} entries")
    
    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self.cache = {}
        if self.cache_path.exists():
            self.cache_path.unlink()
        logger.info("Embedding cache cleared")
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_embedder(
    model_name: Optional[str] = None,
    use_cache: bool = True
) -> DeterministicEmbedder:
    """
    Get a configured embedder instance.
    
    Args:
        model_name: Model to use (default from config)
        use_cache: Whether to use caching
    
    Returns:
        Configured DeterministicEmbedder instance
    """
    return DeterministicEmbedder(
        model_name=model_name or EmbeddingConfig.MODEL_NAME,
        use_cache=use_cache
    )


def embed_chunks(
    chunks: List[Dict],
    embedder: Optional[DeterministicEmbedder] = None
) -> List[Dict]:
    """
    Add embeddings to chunk dictionaries.
    
    Args:
        chunks: List of chunk dicts with 'text' field
        embedder: Optional embedder instance
    
    Returns:
        Chunks with 'embedding' field added
    """
    if embedder is None:
        embedder = get_embedder()
    
    texts = [chunk['text'] for chunk in chunks]
    embeddings = embedder.encode(texts, show_progress=True)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk['embedding'] = embedding
    
    # Save cache
    embedder.save_cache()
    
    logger.info(f"Added embeddings to {len(chunks)} chunks")
    return chunks

