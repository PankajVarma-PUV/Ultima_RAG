"""
Image Processor for SentinelRAG (Upgraded to SentinelRAG Elite).
Handles image preprocessing, OCR, and SOTA vision perception.

SOTA Features:
- Naive Dynamic Resolution (1280px limit)
- Multi-Agent Fusion (Config-defined NLP + Qwen2-VL + EasyOCR)
"""

import os
import json
import asyncio
from typing import Dict, List
from PIL import Image
import torch

from ..core.utils import logger
from .qwen_agent import get_vision_agent


class ImageProcessor:
    """Agent: Image Pipeline Orchestrator (Preprocessing + OCR + Vision)"""
    
    def __init__(self):
        self.vision_agent = get_vision_agent()
        self.ocr_reader = None  # Lazy load

    async def warm_up(self):
        """Pre-load vision and OCR models to eliminate cold-start latency."""
        logger.info("🔥 Warming up ImageProcessor (Vision + OCR)...")
        # Load Qwen2-VL
        if hasattr(self.vision_agent, '_lazy_load'):
            await asyncio.to_thread(self.vision_agent._lazy_load)
        # Load EasyOCR
        self._get_ocr_reader()
        logger.info("✅ ImageProcessor warm-up complete.")

    def _get_ocr_reader(self):
        if self.ocr_reader is None:
            try:
                import easyocr
                logger.info("Initializing EasyOCR reader...")
                # Enable GPU acceleration only if CUDA is available
                cuda_available = torch.cuda.is_available()
                self.ocr_reader = easyocr.Reader(['en'], gpu=cuda_available)
                logger.info(f"EasyOCR reader initialized (GPU={cuda_available})")
            except ImportError:
                logger.warning("easyocr not installed. OCR stage will be skipped.")
                return None
        return self.ocr_reader

    async def _process_tiled(self, img: Image.Image) -> str:
        """
        SOTA Tiled Perception (SentinelRAG Guard).
        Splits image into 2x2 grid and processes tiles SEQUENTIALLY to save VRAM.
        """
        w, h = img.size
        mid_w, mid_h = w // 2, h // 2
        
        # Quadrants: Top-Left, Top-Right, Bottom-Left, Bottom-Right
        quads = [
            (0, 0, mid_w, mid_h),
            (mid_w, 0, w, mid_h),
            (0, mid_h, mid_w, h),
            (mid_w, mid_h, w, h)
        ]
        
        tile_descriptions = []
        quad_labels = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right"]
        
        for i, box in enumerate(quads):
            tile = img.crop(box)
            logger.info(f"Vision-HD: Processing {quad_labels[i]} tile...")
            
            # Process tile sequentially
            desc = await self.vision_agent.describe_image(tile, prompt="Describe the details and text in this specific quadrant of the image.")
            if desc and not desc.startswith("Error"):
                tile_descriptions.append(f"[{quad_labels[i]}]: {desc}")
            
            # Explicit cache clear between tiles to fit 6GB VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not tile_descriptions:
            return "Error: All tiles failed processing."
            
        return " | ".join(tile_descriptions)

    async def process(self, file_path: str) -> List[Dict]:
        """
        Multimodal image processing flow:
        1. Preprocessing (Resize to max 1280px long-edge)
        2. OCR (EasyOCR)
        4. Merge Stage (Config-defined LLM) - Creates processed_description
        
        Returns:
            List of dicts with 'content', 'sub_type'
        """
        logger.info(f"Starting production vision pipeline for: {os.path.basename(file_path)}")
        
        try:
            # 1. Preprocessing Stage
            with Image.open(file_path) as img:
                img = img.convert("RGB")
                w, h = img.size
                
                # Enforce max(W, H) <= 1280 with aspect ratio preserved
                # This is the SOTA '1280 Guard' for 6GB VRAM
                max_side = 1280
                scale = min(max_side / w, max_side / h, 1.0)
                
                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    logger.info(f"Preprocessing: Resized {w}x{h} to {new_w}x{new_h}")
                else:
                    img_resized = img.copy()

            scraped_items = []
            ocr_content = ""
            vision_content = ""

            # 2 & 3. Parallel Perception Stage (OCR + Vision)
            
            async def run_ocr():
                reader = self._get_ocr_reader()
                if not reader:
                    return None
                try:
                    import numpy as np
                    img_np = np.array(img_resized)
                    # Run CPU/GPU OCR in thread to avoid blocking loop
                    results = await asyncio.to_thread(reader.readtext, img_np, detail=1)
                    texts = [text for (_, text, prob) in results if prob > 0.2]
                    return " ".join(texts) if texts else ""
                except Exception as e:
                    logger.error(f"OCR Agent failed: {e}")
                    return None

            async def run_vision():
                try:
                    # Trigger "Vision-HD" for complex docs
                    if w > 2000 or h > 2000:
                        return await self._process_tiled(img_resized)
                    else:
                        return await self.vision_agent.describe_image(img_resized)
                except Exception as e:
                    logger.error(f"Vision Agent failed: {e}")
                    return None

            # Execute OCR and Vision concurrently
            ocr_task = run_ocr()
            vision_task = run_vision()
            
            ocr_result, vision_result = await asyncio.gather(ocr_task, vision_task)

            # 4. Final Aggregation
            if ocr_result:
                scraped_items.append({"content": ocr_result, "sub_type": "ocr"})
            if vision_result:
                scraped_items.append({"content": vision_result, "sub_type": "vision"})

            return scraped_items
            
        except Exception as e:
            logger.error(f"Image Pipeline failed: {e}")
            return [{"content": f"Critical Error: {str(e)}", "sub_type": "image"}]
