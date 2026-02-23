"""
Document Chunking Module for Ultima_RAG
Implements semantic, fixed-size, and paragraph-based chunking strategies using tokens.
"""

import re
from typing import List, Dict, Optional

try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

from transformers import AutoTokenizer

from ..core.config import ChunkingConfig, EmbeddingConfig
from ..core.utils import logger, normalize_text

# Initialize global tokenizer for token counting
# Aligned with all-MiniLM-L6-v2 by default
try:
    _tokenizer = AutoTokenizer.from_pretrained(EmbeddingConfig.MODEL_NAME)
    TOKENIZER_AVAILABLE = True
except Exception as e:
    logger.warning(f"Failed to load tokenizer: {e}. Falling back to word splitting.")
    TOKENIZER_AVAILABLE = False


def get_token_count(text: str) -> int:
    """Get number of tokens in text."""
    if TOKENIZER_AVAILABLE:
        return len(_tokenizer.encode(text, add_special_tokens=False))
    return len(text.split())


# =============================================================================
# CHUNKING STRATEGIES
# =============================================================================

def fixed_chunk(
    text: str, 
    chunk_size: int = ChunkingConfig.CHUNK_SIZE,
    overlap: int = ChunkingConfig.CHUNK_OVERLAP
) -> List[str]:
    """
    Fixed-size chunking with overlap based on tokens.
    """
    if not TOKENIZER_AVAILABLE:
        return _fixed_chunk_words(text, chunk_size, overlap)

    tokens = _tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    if len(tokens) <= chunk_size:
        return [text] if text.strip() else []
    
    step = chunk_size - overlap
    if step <= 0: step = 1
    
    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = _tokenizer.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)
    
    return chunks

def _fixed_chunk_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fallback word-based fixed chunking."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    if step <= 0: step = 1
    for i in range(0, len(words), step):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def semantic_chunk(
    text: str, 
    max_chunk_size: int = ChunkingConfig.CHUNK_SIZE,
    min_chunk_size: int = ChunkingConfig.MIN_CHUNK_SIZE
) -> List[str]:
    """
    Semantic chunking that respects sentence boundaries and token limits.
    """
    # Use NLTK if available, otherwise fall back to regex
    if NLTK_AVAILABLE:
        sentences = sent_tokenize(text)
    else:
        # Simple regex-based sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_tokens = get_token_count(sentence)
        
        # If adding this sentence exceeds max size, save current chunk
        if current_tokens + sentence_tokens > max_chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    # Merge small chunks with previous ones
    merged_chunks = []
    for chunk in chunks:
        if merged_chunks and get_token_count(chunk) < min_chunk_size:
            merged_chunks[-1] = merged_chunks[-1] + ' ' + chunk
        else:
            merged_chunks.append(chunk)
    
    return merged_chunks


def paragraph_chunk(
    text: str, 
    max_chunk_size: int = ChunkingConfig.CHUNK_SIZE
) -> List[str]:
    """
    Paragraph-based chunking that splits on double newlines and respects token limits.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        para_tokens = get_token_count(para)
        
        if current_tokens + para_tokens > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens
    
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks


# =============================================================================
# CHUNK ENRICHMENT
# =============================================================================

def enrich_chunk(
    chunk_text: str,
    chunk_index: int,
    doc_metadata: Dict,
    doc_id: str
) -> Dict:
    """
    Enrich a chunk with metadata for retrieval and attribution.
    
    Args:
        chunk_text: The text content of the chunk
        chunk_index: Index of this chunk in the document
        doc_metadata: Metadata from the source document
        doc_id: Unique document identifier
    
    Returns:
        Enriched chunk dictionary
    """
    chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
    
    return {
        "chunk_id": chunk_id,
        "text": normalize_text(chunk_text),
        "source": doc_metadata.get("source") or doc_metadata.get("file_name") or doc_metadata.get("filename") or "Document",
        "page": doc_metadata.get("page", None),
        "chunk_index": chunk_index,
        "token_count": get_token_count(chunk_text),
        "char_count": len(chunk_text),
        "metadata": {
            **doc_metadata,
            "chunk_id": chunk_id
        }
    }


# =============================================================================
# DOCUMENT CHUNKING PIPELINE
# =============================================================================

class DocumentChunker:
    """
    Complete document chunking pipeline with configurable strategy.
    """
    
    def __init__(
        self, 
        strategy: str = "semantic",
        chunk_size: int = ChunkingConfig.CHUNK_SIZE,
        overlap: int = ChunkingConfig.CHUNK_OVERLAP
    ):
        """
        Initialize chunker with strategy.
        
        Args:
            strategy: One of 'semantic', 'fixed', 'paragraph'
            chunk_size: Max tokens per chunk
            overlap: Token overlap for fixed chunking
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        self.strategy_map = {
            "semantic": lambda t: semantic_chunk(t, self.chunk_size),
            "fixed": lambda t: fixed_chunk(t, self.chunk_size, self.overlap),
            "paragraph": lambda t: paragraph_chunk(t, self.chunk_size)
        }
        
        if strategy not in self.strategy_map:
            raise ValueError(f"Unknown strategy: {strategy}. Use: {list(self.strategy_map.keys())}")
        
        logger.info(f"DocumentChunker initialized with '{strategy}' strategy")
    
    def chunk_document(
        self, 
        text: str, 
        doc_id: str,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Chunk a document and enrich with metadata.
        
        Args:
            text: Document text content
            doc_id: Unique document identifier
            metadata: Optional document metadata
        
        Returns:
            List of enriched chunk dictionaries
        """
        if not text or not text.strip():
            logger.warning(f"Empty document: {doc_id}")
            return []
        
        metadata = metadata or {}
        chunk_func = self.strategy_map[self.strategy]
        
        # Get raw chunks
        raw_chunks = chunk_func(text)
        
        # Apply MAX_CHUNK_SIZE hard guardrail
        final_chunks = []
        for chunk in raw_chunks:
            tokens = get_token_count(chunk)
            if tokens > ChunkingConfig.MAX_CHUNK_SIZE and TOKENIZER_AVAILABLE:
                # Force truncate
                logger.warning(f"Chunk exceeded MAX_CHUNK_SIZE ({tokens} > {ChunkingConfig.MAX_CHUNK_SIZE}). Truncating.")
                token_ids = _tokenizer.encode(chunk, add_special_tokens=False)[:ChunkingConfig.MAX_CHUNK_SIZE]
                chunk = _tokenizer.decode(token_ids)
            final_chunks.append(chunk)

        # Enrich each chunk
        enriched_chunks = []
        for i, chunk_text in enumerate(final_chunks):
            enriched = enrich_chunk(chunk_text, i, metadata, doc_id)
            enriched_chunks.append(enriched)
        
        logger.info(f"Document '{doc_id}' chunked into {len(enriched_chunks)} chunks")
        return enriched_chunks
    
    def chunk_documents(
        self, 
        documents: List[Dict]
    ) -> List[Dict]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of dicts with 'text', 'doc_id', and optional 'metadata'
        
        Returns:
            All chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(
                text=doc.get("text", ""),
                doc_id=doc.get("doc_id", f"doc_{len(all_chunks)}"),
                metadata=doc.get("metadata", {})
            )
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks from {len(documents)} documents: {len(all_chunks)}")
        return all_chunks

