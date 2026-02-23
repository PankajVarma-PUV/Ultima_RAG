"""
Video Processor for Ultima_RAG.
Orchestrates frame extraction, OCR, vision perception, audio transcription,
and LongT5 deduplication for video content.

Output structure:
- sub_type='video_visual': Deduplicated visual descriptions
- sub_type='video_audio': Audio transcript segments
"""

import os
import tempfile
import base64
from typing import Dict, List
import numpy as np
import cv2
from PIL import Image

from .audio_processor import AudioProcessor
from .qwen_agent import get_vision_agent
from ..core.utils import logger, Timer



class VideoProcessor:
    """Agent: Video Pipeline Orchestrator (Sampling + OCR + Vision + Audio + Deduplication)"""
    
    def __init__(self):
        self.audio_proc = AudioProcessor()
        self.vision_agent = get_vision_agent()
        self._ocr_reader = None  # Lazy load

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            try:
                import easyocr
                logger.info("Initializing EasyOCR reader...")
                # English only, CPU bound as per 6GB constraint
                self._ocr_reader = easyocr.Reader(['en'], gpu=False)
            except ImportError:
                logger.warning("easyocr not installed. OCR stage will be skipped.")
        return self._ocr_reader

    def _is_frame_significant(self, prev_frame, curr_frame, threshold=0.15) -> bool:
        """Simple visual difference check to skip redundant frames (SSIM-lite)."""
        if prev_frame is None: return True
        
        # Resize to small gray for fast comparison
        prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (100, 100)), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(cv2.resize(curr_frame, (100, 100)), cv2.COLOR_BGR2GRAY)
        
        # Calculate mean absolute difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        avg_diff = np.mean(diff) / 255.0
        
        return avg_diff > threshold

    def _is_text_quality_sufficient(self, text: str) -> bool:
        """Filter out garbage OCR noise (mostly symbols/random chars)."""
        if not text or len(text.strip()) < 3: return False
        
        # Rule: At least 40% of characters should be alphanumeric
        alnum_count = sum(1 for c in text if c.isalnum())
        ratio = alnum_count / len(text)
        
        # Rule: Filter out strings with excessive backslashes or repeated symbols
        if text.count('\\') > 2 or text.count('"') > 2: return False
        
        return ratio > 0.4

    async def process(self, file_path: str) -> List[Dict]:
        """
        Enhanced Multimodal Video Pipeline:
        1. Audio Transcription (Async)
        2. Intelligent Frame Sampling (SSIM-lite visual difference check)
        3. OCR & Vision Perception on Significant Frames
        4. Unified Audio-Visual Narrative Fusion (Config-defined LLM)
        """
        logger.info(f"🚀 Ultima_RAG Video Pipeline: {os.path.basename(file_path)}")
        scraped_items = []
        
        try:
            # 1. Audio Processing
            audio_text = ""
            try:
                audio_results = await self.audio_proc.transcribe(file_path)
                if audio_results:
                    audio_text = " ".join([item.get('content', '') for item in audio_results])
                    scraped_items.append({"content": audio_text, "sub_type": "video_audio"})
            except Exception as ae:
                logger.error(f"Audio stage failed: {ae}")

            # 2. Visual Processing (Intelligent Sampling)
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened(): raise ValueError("Could not open video file")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            all_frame_descriptions = []
            ocr_texts = []
            prev_processed_frame = None
            processed_count = 0

            # Sample 1 frame per second, but only if significant
            for sec in range(int(duration)):
                frame_idx = int(sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret: break
                
                timestamp = f"{sec//60:02d}:{sec%60:02d}"
                
                # Intelligent Skip: If visually similar to last processed frame, skip expensive vision
                if prev_processed_frame is not None and not self._is_frame_significant(prev_processed_frame, frame):
                    continue

                processed_count += 1
                prev_processed_frame = frame.copy()

                # Preprocessing
                h, w = frame.shape[:2]
                scale = min(448 / w, 448 / h, 1.0)
                frame_resized = cv2.resize(frame, (int(w*scale), int(h*scale))) if scale < 1.0 else frame

                # OCR - only on significant frames
                reader = self._get_ocr_reader()
                if reader:
                    try:
                        ocr_results = reader.readtext(frame_resized, detail=0)
                        valid_ocr = [t for t in ocr_results if self._is_text_quality_sufficient(t)]
                        if valid_ocr: ocr_texts.append(f"[{timestamp}] {' '.join(valid_ocr)}")
                    except Exception: pass

                # Vision - only on significant frames
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                    vision_desc = await self.vision_agent.describe_image(pil_img, "Describe exactly what is happening in this video scene.")
                    if vision_desc and not vision_desc.startswith("Error"):
                        all_frame_descriptions.append(f"At {timestamp}: {vision_desc.strip()}")
                except Exception: pass

            cap.release()
            logger.info(f"Visual sampling complete: {processed_count} unique scenes analyzed from {int(duration)}s video")

            # 3. Consolidate Context
            full_visual = ". ".join(all_frame_descriptions)
            if ocr_texts: full_visual += "\n\nTEXT EXTRACTED FROM VIDEO:\n" + " | ".join(ocr_texts)
            
            if full_visual:
                scraped_items.append({"content": full_visual, "sub_type": "video_visual"})

            return scraped_items

        except Exception as e:
            logger.error(f"Video pipeline failed: {e}")
            return [{"content": f"Video Error: {str(e)}", "sub_type": "video_visual"}]

