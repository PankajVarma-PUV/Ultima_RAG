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
Audio Processor for UltimaRAG.
Uses Faster-Whisper for high-speed local transcription.

Output structure:
- sub_type='audio': Transcription segments and summaries
"""

from typing import Dict, List, Optional, Callable
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

    def _build_structured_context(self, segments: List[Dict]) -> str:
        """
        Merges transcription segments into a time-aligned structured text block.
        Ensures the Narrative LLM receives a professional intelligence report.
        """
        lines = []
        for seg in segments:
            t = seg.get('timestamp', '0:00')
            content = seg.get('content', '').strip()
            if content:
                lines.append(f"[{t}] SPEECH: {content}")
        
        return "\n".join(lines)

    async def transcribe(self, file_path: str, check_abort_fn: Optional[Callable] = None) -> List[Dict]:
        """
        Transcribe audio file and generate a high-fidelity structured transcript.
        Designed for SOTA enrichment by the ContentEnricher agent.
        """
        if check_abort_fn and check_abort_fn():
            logger.info("Abort signaled before audio transcription. Skipping.")
            return []

        logger.info(f"🚀 Transcribing audio: {os.path.basename(file_path)}")
        self._load_model()
        
        raw_segments = []
        
        if self.model:
            try:
                # beam_size 5 for high-fidelity decoding
                segments, info = self.model.transcribe(file_path, beam_size=5)
                for segment in segments:
                    if check_abort_fn and check_abort_fn():
                        logger.info("Abort signaled during audio transcription loop.")
                        break
                        
                    timestamp = f"{int(segment.start) // 60}:{int(segment.start) % 60:02d}"
                    content = segment.text.strip()
                    if content:
                        raw_segments.append({
                            "content": content,
                            "timestamp": timestamp
                        })
            except Exception as e:
                logger.error(f"Whisper transcription error: {e}")
        else:
            logger.warning("Using placeholder transcription (faster-whisper not loaded)")
            raw_segments.append({
                "content": "Audio transcript unavailable: faster-whisper not installed.",
                "timestamp": "0:00"
            })

        # ── STRICT CHECK: Before Structured Build ──
        if check_abort_fn and check_abort_fn(): return []

        structured_transcript = self._build_structured_context(raw_segments)
        
        if not structured_transcript.strip():
            logger.warning(f"No speech detected in audio: {os.path.basename(file_path)}")
            return []

        # Return a single item with the structured transcript.
        # This will be passed to the ContentEnricher for the final 'Beautiful Description'.
        logger.info(f"Audio processing complete: {len(raw_segments)} segments transcribed.")
        return [{
            "content": structured_transcript,
            "sub_type": "audio",
            "metadata": {"type": "audio_transcript", "segments": len(raw_segments)}
        }]

