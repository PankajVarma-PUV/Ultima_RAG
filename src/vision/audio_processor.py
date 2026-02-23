"""
Audio Processor for Ultima_RAG.
Uses Faster-Whisper for high-speed local transcription.

Output structure:
- sub_type='audio': Transcription segments and summaries
"""

from typing import Dict, List
import os

from ..core.utils import logger
from ..core.ollama_client import get_ollama_client


class AudioProcessor:
    """Handles audio transcription logic."""
    
    def __init__(self, model_size: str = "small"):
        self.model_size = model_size
        self.model = None  # Lazy load

    def _load_model(self):
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            except ImportError:
                logger.warning("faster-whisper not installed. Audio processing will be stubbed.")

    async def transcribe(self, file_path: str) -> List[Dict]:
        """
        Transcribe audio file and generate a semantic summary.
        
        Returns:
            List of dicts with 'content', 'sub_type', and optional 'timestamp'
        """
        logger.info(f"Transcribing audio: {file_path}")
        self._load_model()
        
        scraped_items = []
        full_transcript = []
        
        if self.model:
            try:
                segments, info = self.model.transcribe(file_path, beam_size=5)
                for segment in segments:
                    timestamp = f"{int(segment.start) // 60}:{int(segment.start) % 60:02d}"
                    content = segment.text.strip()
                    if content:
                        scraped_items.append({
                            "content": content,
                            "sub_type": "audio",
                            "timestamp": timestamp
                        })
                        full_transcript.append(content)
            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
        else:
            logger.warning("Using placeholder transcription (faster-whisper not loaded)")
            scraped_items.append({
                "content": "Audio transcript unavailable: faster-whisper not installed.",
                "sub_type": "audio"
            })

        # Audio Summary Compression
        if full_transcript:
            try:
                client = get_ollama_client()
                all_text = " ".join(full_transcript)
                
                summary_prompt = f"Summarize the semantic meaning of this audio transcript in one or two concise sentences:\n\n{all_text}"
                summary = client.generate(summary_prompt, temperature=0.1)
                if summary:
                    scraped_items.append({
                        "content": summary.strip(),
                        "sub_type": "audio"
                    })
            except Exception as se:
                logger.error(f"Audio summary compression failed: {se}")

        logger.info(f"Audio processing complete: {len(scraped_items)} items")
        return scraped_items

