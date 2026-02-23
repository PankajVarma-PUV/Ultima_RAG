"""
Manager for Multimodal processing in Ultima_RAG.
Orchestrates between Image, Audio, and Video pipelines.
"""

from typing import Dict, Optional
import hashlib
from pathlib import Path

from ..core.utils import logger
from ..core.file_manager import get_upload_path
from ..core.database import get_database
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from ..agents.content_enricher import ContentEnricher

class MultimodalManager:
    """Manager to orchestrate multimodal file scraping and persistence."""
    
    def __init__(self):
        self.db = get_database()
        self.image_proc = ImageProcessor()
        self.audio_proc = AudioProcessor()
        self.video_proc = VideoProcessor()
        self.enricher = ContentEnricher()

    def get_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    async def process_file(self, conversation_id: str, file_path: str, file_type: str, file_name: str) -> Dict:
        """
        Main entry point for processing a multimodal file.
        Checks registry first to skip processing if possible.
        """
        file_hash = self.get_file_hash(file_path)
        
        # 1. Check Registry (Strictly scoped to same conversation)
        # Goal: If same file in SAME chat, return cached. If same file in NEW chat, re-process.
        existing_doc = self.db.get_document_by_hash(file_hash, conversation_id=conversation_id)
        
        if existing_doc:
            file_id = existing_doc['file_id']
            # Get existing content
            content_items = self.db.get_scraped_content(file_id)
            # Check for enriched content (Unified Pipeline)
            enriched = self.db.get_enriched_content_by_file_id(file_id)
            
            # Check if existing content is just failure message
            is_failure_cache = False
            if len(content_items) == 1:
                text = content_items[0].get('content', '').lower()
                if "no evidence was found" in text or "error during vision" in text:
                    is_failure_cache = True
            
            if not is_failure_cache:
                logger.info(f"File {file_name} already processed in this chat session. Returning cached.")
                return {
                    "status": "cached",
                    "file_id": file_id,
                    "content": content_items,
                    "enriched_content": enriched.get('enriched_content') if enriched else None
                }
            else:
                logger.info(f"File {file_name} has a failed/empty cache entry in this chat. FORCING RE-PROCESS.")

        # 1.5. GLOBAL CACHE CHECK (DISABLED for SOTA Compliance)
        # The user requires fresh LLM enrichment and perception for every asset to ensure
        # 100% accuracy and 'Knowledge Base' structure alignment.
        # logger.info(f"Global enrichment hit check disabled for {file_name} to ensure elite processing.")


        # 2. Process based on type
        from ..core.utils import get_file_category
        normalized_type = get_file_category(file_type)
        
        logger.info(f"Processing new {file_type} ({normalized_type}) file: {file_name}")
        scraped_items = []
        
        if normalized_type == 'image':
            scraped_items = await self.image_proc.process(file_path)
        elif normalized_type == 'audio':
            scraped_items = await self.audio_proc.transcribe(file_path)
        elif normalized_type == 'video':
            scraped_items = await self.video_proc.process(file_path)
        else:
            logger.warning(f"Unsupported multimodal file type: {file_type} (mapped to {normalized_type})")
            return {"status": "unsupported", "file_name": file_name}

        # 3. Register and Save
        file_id = self.db.register_document(
            file_name=file_name,
            file_hash=file_hash,
            file_type=normalized_type,
            conversation_id=conversation_id,
            file_path=file_path
        )
        
        # 3a. Save Legacy Scraped Content
        raw_full_content = []
        for idx, item in enumerate(scraped_items):
            content_text = item['content']
            raw_full_content.append(content_text)
            self.db.add_scraped_content(
                file_id=file_id,
                content=content_text,
                sub_type=item.get('sub_type', 'text'),
                chunk_index=idx,
                timestamp=item.get('timestamp'),
                page_number=item.get('page_number'),
                metadata={'type': item.get('type', item.get('sub_type', 'general'))}
            )
        
        # 3b. Unified Enrichment Pipeline (New Architecture: Background Task)
        enrichment_task = None
        try:
            # Defensive check: if enriched content already exists for this file_id, skip re-enrichment
            existing_enriched = self.db.get_enriched_content_by_file_id(file_id)
            if existing_enriched:
                logger.info(f"Enriched content already exists for {file_name}, skipping redundant re-enrichment.")
            else:
                import asyncio
                full_raw_text = "\n\n".join(raw_full_content)
                
                # Launch background enrichment and store the task
                logger.info(f"🚀 Launching background enrichment for: {file_name}")
                enrichment_task = asyncio.create_task(self._background_enrichment(
                    file_id=file_id,
                    conversation_id=conversation_id,
                    full_raw_text=full_raw_text,
                    normalized_type=normalized_type,
                    file_name=file_name
                ))
        except Exception as e:
            logger.error(f"Enrichment background trigger failed for {file_name}: {e}")

        return {
            "status": "success",
            "file_id": file_id,
            "content": scraped_items,
            "processing_status": "enriching_in_background",
            "enrichment_task": enrichment_task # SOTA: Allow caller to await or track
        }

    async def _background_enrichment(self, file_id: str, conversation_id: str, full_raw_text: str, normalized_type: str, file_name: str):
        """Helper to run enrichment in the background and persist to DB."""
        try:
            enriched_text = await self.enricher.enrich_content(
                raw_content=full_raw_text,
                content_type=normalized_type,
                file_name=file_name
            )
            
            # Store in Enriched Content table
            self.db.add_enriched_content(
                file_id=file_id,
                conversation_id=conversation_id,
                original_content=full_raw_text,
                enriched_content=enriched_text,
                content_type=normalized_type,
                file_name=file_name,
                metadata={'source': 'MultimodalManager', 'version': 'SOTA-Unified-v1-BG'}
            )
            
            # SOTA: Index Enriched Content for RAG
            # This turns the elite narrative into searchable chunks in the knowledge_base
            logger.info(f"Indexing fresh enrichment for {file_name} in chat {conversation_id}")
            self.db.add_knowledge_from_text(
                text=enriched_text,
                file_name=file_name,
                conversation_id=conversation_id,
                metadata={'source': 'MultimodalEnrichment'}
            )

            logger.info(f"✅ Background Enrichment and Indexing complete for {file_name}")


        except Exception as e:
            logger.error(f"❌ Background Enrichment failed for {file_name}: {e}")

