"""
Utility Functions for Ultima_RAG
Common utilities including deterministic mode, logging, and text processing.
"""

import random
import logging
import hashlib
from typing import List, Optional, Any
from datetime import datetime

import numpy as np

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging for the application"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Silence noisy third-party libraries
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger('Ultima_RAG')


logger = setup_logging()


# =============================================================================
# DETERMINISTIC MODE
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.
    Call this at application startup for deterministic behavior.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Seeds set to {seed} (including PyTorch)")
    except ImportError:
        logger.info(f"Seeds set to {seed} (PyTorch not available)")


def get_deterministic_hash(text: str) -> str:
    """Generate deterministic hash for caching"""
    return hashlib.md5(text.encode()).hexdigest()


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for consistent processing"""
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace tokenization with lowercasing"""
    return text.lower().split()


def calculate_word_overlap(text1: str, text2: str) -> float:
    """Calculate word overlap ratio between two texts"""
    words1 = set(tokenize_simple(text1))
    words2 = set(tokenize_simple(text2))
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """Truncate text to max length with suffix"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


# =============================================================================
# SIMILARITY UTILITIES
# =============================================================================

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize scores to [0, 1] range"""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    return [(s - min_score) / (max_score - min_score) for s in scores]


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_chunks(chunks: List[dict]) -> bool:
    """Validate chunk structure"""
    required_keys = {'text', 'chunk_id'}
    
    for chunk in chunks:
        if not isinstance(chunk, dict):
            return False
        if not required_keys.issubset(chunk.keys()):
            return False
        if not chunk.get('text'):
            return False
    
    return True


def validate_query_analysis(analysis: dict) -> bool:
    """Validate query analysis output structure"""
    required_keys = {'original_query', 'intent', 'retrieval_queries'}
    return required_keys.issubset(analysis.keys())


def get_file_category(extension: str) -> str:
    """
    Map file extension to database-allowed categories:
    'pdf', 'txt', 'image', 'word', 'csv', 'xls', 'audio', 'video'
    """
    ext = extension.lower().strip('.')
    
    mapping = {
        'pdf': 'pdf',
        'txt': 'txt', 'md': 'txt',
        'png': 'image', 'jpg': 'image', 'jpeg': 'image', 'webp': 'image', 'gif': 'image',
        'image': 'image', # Identity mapping for already categorized inputs
        'docx': 'word', 'doc': 'word',
        'csv': 'csv',
        'xls': 'xls', 'xlsx': 'xls',
        'mp3': 'audio', 'wav': 'audio', 'flac': 'audio', 'm4a': 'audio',
        'audio': 'audio', # Identity mapping
        'mp4': 'video', 'mov': 'video', 'webm': 'video', 'mkv': 'video', 'avi': 'video',
        'video': 'video'  # Identity mapping
    }
    
    return mapping.get(ext, 'txt') # Fallback to txt


# =============================================================================
# TIMING UTILITIES
# =============================================================================

class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time: Optional[datetime] = None
        self.elapsed_ms: float = 0.0
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        elapsed = datetime.now() - self.start_time
        self.elapsed_ms = elapsed.total_seconds() * 1000
        logger.debug(f"{self.name} completed in {self.elapsed_ms:.2f}ms")


# =============================================================================
# DETERMINISM VERIFICATION
# =============================================================================

def test_determinism() -> bool:
    """
    Test that deterministic mode is working correctly.
    Returns True if outputs are reproducible.
    """
    set_seed(42)
    
    # Test random
    r1 = [random.random() for _ in range(5)]
    set_seed(42)
    r2 = [random.random() for _ in range(5)]
    assert r1 == r2, "Python random not deterministic"
    
    # Test numpy
    set_seed(42)
    n1 = np.random.rand(5).tolist()
    set_seed(42)
    n2 = np.random.rand(5).tolist()
    assert n1 == n2, "NumPy random not deterministic"
    
    logger.info("✓ Determinism verification passed")
    return True

