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
Video Processor for UltimaRAG.
Orchestrates frame extraction, OCR, vision perception, audio transcription,
and narrative enrichment for video content.

Output structure (unchanged):
- sub_type='video_visual': Temporally fused + enriched visual description
- sub_type='video_audio':  Chronologically ordered audio transcript

Pipeline:
    Stage 1  — Audio transcription with timestamp-bucket alignment
    Stage 2  — Reel-aware frame sampling (2fps ≤60s / 1fps >60s)
    Stage 3  — SSIM-lite significant-frame detection (threshold 0.12)
    Stage 4  — EasyOCR + Qwen-Vision per significant frame → stored per bucket
    Stage 5  — Temporal fusion: audio + OCR + vision aligned by 5s windows
    Stage 6  — Per-bucket OCR deduplication via Jaccard token similarity
    Stage 7  — Narrative LLM enrichment prompt → humanized description
    Stage 8  — Graceful fallback: raw structured context if LLM unavailable
"""

import os
import math
from typing import Dict, List, Optional, Callable
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image

from .audio_processor import AudioProcessor
from .qwen_agent import get_vision_agent
from ..core.utils import logger


# ── Pipeline Tuning Constants ─────────────────────────────────────────────────
# All tunable values live here. Never buried inside logic.

REEL_THRESHOLD        = 60     # seconds: videos ≤ this are treated as reels
REEL_FPS_SAMPLE       = 2      # candidate frames/sec in reel mode
LONG_FPS_SAMPLE       = 1      # candidate frames/sec for longer videos
BUCKET_SIZE_SEC       = 5      # seconds per time-alignment window
SSIM_THRESHOLD        = 0.12   # mean-abs-diff threshold; 0.12 catches fast reel cuts
OCR_MIN_LEN           = 3      # discard OCR strings shorter than this
OCR_ALNUM_RATIO       = 0.40   # minimum alphanumeric character ratio
OCR_JACCARD_THRESH    = 0.75   # Jaccard similarity threshold for OCR dedup
VISION_DESC_MAX_CHARS = 600    # cap per-frame Qwen description length
NARRATIVE_MAX_TOKENS  = 1024   # max tokens for narrative LLM output
NARRATIVE_TEMPERATURE = 0.65   # narrative LLM sampling temperature


class VideoProcessor:
    """Agent: Video Pipeline Orchestrator (Sampling + OCR + Vision + Audio + Deduplication)

    Drop-in replacement for the original VideoProcessor.
    Public interface unchanged: VideoProcessor().process(file_path) -> List[Dict]
    """

    def __init__(self):
        self.audio_proc    = AudioProcessor()
        self.vision_agent  = get_vision_agent()        # Qwen-Vision: frame description
        self.narrative_llm = self._load_narrative_llm() # Text LLM: narrative enrichment
        self._ocr_reader   = None                       # Lazy-loaded EasyOCR

    # ─────────────────────────────────────────────────────────────────────────
    # Narrative LLM — optional, graceful fallback
    # Fully decoupled from vision_agent so any text-generation model can be
    # swapped by updating narrative_agent.py without touching this file.
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_narrative_llm():
        """
        Attempts to load the configured Narrative LLM agent.
        Returns None silently if narrative_agent is not yet wired up.
        process() gracefully falls back to returning raw structured context.

        Contract for narrative_agent.py:
            from .narrative_agent import get_narrative_agent
            # returned object must expose:
            # await agent.generate(prompt: str, max_tokens: int, temperature: float) -> str
        """
        try:
            from .narrative_agent import get_narrative_agent
            agent = get_narrative_agent()
            logger.info("Narrative LLM agent loaded successfully.")
            return agent
        except (ImportError, Exception) as e:
            logger.warning(
                f"Narrative LLM not available ({e}). "
                "video_visual will contain raw structured context until "
                "narrative_agent.py is wired up."
            )
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Lazy OCR (unchanged from original)
    # ─────────────────────────────────────────────────────────────────────────

    def _get_ocr_reader(self):
        if self._ocr_reader is None:
            try:
                import easyocr
                logger.info("Initializing EasyOCR reader...")
                # SOTA: Multi-language 'Global Core' (CPU bound as per 6GB constraint)
                self._ocr_reader = easyocr.Reader(['en', 'ch_tra'], gpu=False)
            except ImportError:
                logger.warning("easyocr not installed. OCR stage will be skipped.")
        return self._ocr_reader

    # ─────────────────────────────────────────────────────────────────────────
    # Frame quality filters
    # ─────────────────────────────────────────────────────────────────────────

    def _is_frame_significant(
        self,
        prev_frame: Optional[np.ndarray],
        curr_frame:  np.ndarray,
        threshold:   float = SSIM_THRESHOLD
    ) -> bool:
        """
        SSIM-lite mean-absolute-difference check.
        Threshold 0.12 (was 0.15) catches fast reel scene-cuts the original missed.
        """
        if prev_frame is None:
            return True
        prev_gray = cv2.cvtColor(cv2.resize(prev_frame, (100, 100)), cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(cv2.resize(curr_frame, (100, 100)), cv2.COLOR_BGR2GRAY)
        avg_diff  = np.mean(cv2.absdiff(prev_gray, curr_gray)) / 255.0
        return avg_diff > threshold

    def _is_text_quality_sufficient(self, text: str) -> bool:
        """Filter garbage OCR noise. Logic unchanged from original."""
        if not text or len(text.strip()) < OCR_MIN_LEN:
            return False
        alnum_count = sum(1 for c in text if c.isalnum())
        if text.count('\\') > 2 or text.count('"') > 2:
            return False
        return (alnum_count / len(text)) > OCR_ALNUM_RATIO

    # ─────────────────────────────────────────────────────────────────────────
    # OCR deduplication — Jaccard token similarity
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """
        Token-level Jaccard similarity between two OCR strings.
        Reels show the same overlay text across many consecutive frames —
        this collapses them before they pollute the LLM context.
        """
        set_a, set_b = set(a.lower().split()), set(b.lower().split())
        if not set_a and not set_b:
            return 1.0
        return len(set_a & set_b) / len(set_a | set_b)

    def _deduplicate_ocr(self, texts: List[str]) -> List[str]:
        """Remove near-duplicate OCR strings within a single time bucket."""
        unique: List[str] = []
        for candidate in texts:
            if not any(
                self._jaccard_similarity(candidate, e) >= OCR_JACCARD_THRESH
                for e in unique
            ):
                unique.append(candidate)
        return unique

    # ─────────────────────────────────────────────────────────────────────────
    # Audio → timestamp-preserving bucket dict
    # ─────────────────────────────────────────────────────────────────────────

    async def _get_audio_segments(self, file_path: str, check_abort_fn: Optional[Callable] = None) -> Dict[int, str]:
        """
        Transcribes audio via AudioProcessor (Faster-Whisper) and maps each
        segment to its 5-second time bucket.

        Returns:
            { bucket_key (int) → speech_text (str) }
            bucket_key = int(segment_start_sec // BUCKET_SIZE_SEC)

        Expected segment format from AudioProcessor.transcribe():
            { 'content': str, 'start': float (optional), 'end': float (optional) }

        Fallback: if 'start' is absent, all speech is bucketed at 0.
        Sorted output guarantees chronological flat transcript.
        """
        timestamped: Dict[int, List[str]] = defaultdict(list)
        try:
            audio_results = await self.audio_proc.transcribe(file_path, check_abort_fn=check_abort_fn)
            if not audio_results:
                return {}
            for seg in audio_results:
                text = seg.get('content', '').strip()
                if not text:
                    continue
                start_sec = seg.get('start', None)
                bucket    = (
                    int(float(start_sec) // BUCKET_SIZE_SEC)
                    if start_sec is not None else 0
                )
                timestamped[bucket].append(text)

            # Sort by key — guarantees chronological order in flat transcript
            return {b: " ".join(txts) for b, txts in sorted(timestamped.items())}

        except Exception as ae:
            logger.error(f"Audio stage failed: {ae}")
            return {}

    # ─────────────────────────────────────────────────────────────────────────
    # Time-bucket utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_buckets(duration: float) -> Dict[int, Dict]:
        """
        Pre-allocate one bucket dict per 5s window for the full video.
        math.ceil avoids the phantom empty final bucket of the previous +1 formula.
        """
        n = max(1, math.ceil(duration / BUCKET_SIZE_SEC))
        return {i: {"ocr": [], "visual": []} for i in range(n)}

    @staticmethod
    def _sec_to_bucket(sec: float) -> int:
        return int(sec // BUCKET_SIZE_SEC)

    # ─────────────────────────────────────────────────────────────────────────
    # Temporal fusion — structured context builder
    # ─────────────────────────────────────────────────────────────────────────

    def _build_structured_context(
        self,
        buckets:        Dict[int, Dict],
        audio_segments: Dict[int, str],
        duration:       float
    ) -> str:
        """
        Merges all three streams into time-aligned structured text blocks.
        This is what the Narrative LLM receives instead of raw noise.

        Output format per window:
            [0s-5s]
            SPEECH    : <Whisper transcription>
            ON-SCREEN : <deduped OCR text>
            VISUAL    : <Qwen scene description(s)>

        Iterates the UNION of visual + audio bucket keys so audio segments
        that Faster-Whisper maps beyond the video's pre-allocated windows
        are never silently dropped.
        """
        all_keys = sorted(set(buckets.keys()) | set(audio_segments.keys()))
        lines    = []

        for key in all_keys:
            t_start = key * BUCKET_SIZE_SEC
            t_end   = min(t_start + BUCKET_SIZE_SEC, duration)

            speech = audio_segments.get(key, "").strip()
            bucket = buckets.get(key, {"ocr": [], "visual": []})
            ocr    = " | ".join(self._deduplicate_ocr(bucket["ocr"]))
            visual = " -> ".join(bucket["visual"])

            if not any([speech, ocr, visual]):
                continue  # skip fully silent/static windows

            lines.append(
                f"[{t_start:.0f}s-{t_end:.0f}s]\n"
                f"SPEECH    : {speech or 'N/A'}\n"
                f"ON-SCREEN : {ocr    or 'N/A'}\n"
                f"VISUAL    : {visual or 'N/A'}"
            )

        return "\n\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Narrative enrichment prompt — LLM-agnostic
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _build_narrative_prompt(structured_context: str, duration: float) -> str:
        """
        Wraps structured context in a model-agnostic enrichment prompt.
        Works with any instruction-following text LLM.
        Swap the model via narrative_agent.py — this method never needs changing.
        """
        return (
            f"You are an expert multimodal video analyst for UltimaRAG Intelligence System.\n"
            f"Below is a structured, time-aligned extraction from a video ({duration:.1f}s long).\n\n"
            f"Each time segment has three aligned streams:\n"
            f"  SPEECH    = spoken words transcribed from audio (Whisper)\n"
            f"  ON-SCREEN = text visible in the video frame (OCR)\n"
            f"  VISUAL    = scene description from a vision model (Qwen)\n\n"
            f"Your task:\n"
            f"1. Write a rich, detailed, naturally flowing description of the entire video.\n"
            f"2. Integrate all three streams cohesively — do NOT list them separately.\n"
            f"3. Surface: main topic, key actions, scene transitions, on-screen information,\n"
            f"   any product or brand mentions, and the overall tone and intent of the content.\n"
            f"4. If this is short-form social content (Reel / Short / TikTok), note the format,\n"
            f"   pacing, creator style, and intended audience.\n"
            f"5. Output ONLY the enriched description. No headers, no bullets, no metadata.\n\n"
            f"--- STRUCTURED VIDEO EXTRACTION ---\n"
            f"{structured_context}\n"
            f"--- END OF EXTRACTION ---\n\n"
            f"Enriched video description:"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main entry point — public signature UNCHANGED
    # ─────────────────────────────────────────────────────────────────────────

    async def process(self, file_path: str, check_abort_fn: Optional[Callable] = None) -> List[Dict]:
        """
        Enhanced Multimodal Video Pipeline.
        Signature and return contract identical to original — safe drop-in.

        Returns:
            [
              {"content": "<chronological transcript>", "sub_type": "video_audio" },
              {"content": "<enriched description>",     "sub_type": "video_visual"},
            ]
        """
        logger.info(f"🚀 UltimaRAG Video Pipeline: {os.path.basename(file_path)}")
        scraped_items: List[Dict] = []

        try:
            # ── Stage 1: Audio with timestamp-bucket alignment ────────────────
            if check_abort_fn and check_abort_fn(): return []
            audio_segments: Dict[int, str] = await self._get_audio_segments(file_path, check_abort_fn=check_abort_fn)
            if check_abort_fn and check_abort_fn(): return []

            if audio_segments:
                flat_audio = " ".join(audio_segments[k] for k in sorted(audio_segments))
                scraped_items.append({"content": flat_audio, "sub_type": "video_audio"})
                logger.info(f"Audio: {len(audio_segments)} time bucket(s) transcribed")

            # ── Stage 2: Video metadata + reel-aware sampling config ──────────
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                raise ValueError("Could not open video file")

            fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration     = total_frames / fps if fps > 0 else 0
            is_reel      = duration <= REEL_THRESHOLD
            sample_rate  = REEL_FPS_SAMPLE if is_reel else LONG_FPS_SAMPLE
            sample_step  = 1.0 / sample_rate
            total_candidates = math.ceil(duration * sample_rate)

            logger.info(
                f"Video: {duration:.1f}s | {fps:.1f}fps native | "
                f"sampling at {sample_rate}fps "
                f"({'reel' if is_reel else 'standard'} mode) | "
                f"{total_candidates} candidate frames"
            )

            # ── Stage 3: Pre-allocate time buckets ────────────────────────────
            buckets = self._make_buckets(duration)

            # ── Stage 4: Frame loop — OCR + Qwen-Vision ───────────────────────
            prev_processed_frame: Optional[np.ndarray] = None
            processed_count = 0

            for i in range(total_candidates):
                # ── STRICT ABORT CHECK: check between EVERY frame ──
                if check_abort_fn and check_abort_fn():
                    logger.info("Abort signaled during video frame sampling loop. Breaking.")
                    break

                # Integer index prevents IEEE 754 float accumulation drift
                current_sec = round(i * sample_step, 6)
                if current_sec >= duration:
                    break

                frame_idx = int(current_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                timestamp  = f"{int(current_sec)//60:02d}:{int(current_sec)%60:02d}"
                bucket_key = self._sec_to_bucket(current_sec)

                # Intelligent skip: visually identical to last processed frame
                if not self._is_frame_significant(prev_processed_frame, frame):
                    continue

                processed_count      += 1
                prev_processed_frame  = frame.copy()

                # Preprocessing (unchanged from original)
                h, w  = frame.shape[:2]
                scale = min(448 / w, 448 / h, 1.0)
                frame_resized = (
                    cv2.resize(frame, (int(w * scale), int(h * scale)))
                    if scale < 1.0 else frame
                )

                # OCR — raw valid results per bucket; dedup deferred to fusion stage
                reader = self._get_ocr_reader()
                if reader:
                    try:
                        ocr_results = reader.readtext(frame_resized, detail=0)
                        valid_ocr   = [t for t in ocr_results if self._is_text_quality_sufficient(t)]
                        if valid_ocr:
                            buckets[bucket_key]["ocr"].extend(valid_ocr)
                    except Exception:
                        pass

                # Vision — original high-fidelity scene prompt preserved (no regression)
                try:
                    pil_img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
                    scene_prompt = (
                        "You are **UltimaRAG's Intelligence Fusion Agent**.\n"
                        "Describe this specific video scene with absolute technical precision. "
                        "Extract all factual entities, actions, and temporal context.\n"
                        "Output ONLY the enriched intelligence segment for this timestamp."
                    )
                    vision_desc = await self.vision_agent.describe_image(pil_img, scene_prompt)
                    if vision_desc and not vision_desc.startswith("Error"):
                        buckets[bucket_key]["visual"].append(
                            vision_desc.strip()[:VISION_DESC_MAX_CHARS]
                        )
                except Exception:
                    pass

            cap.release()
            active_buckets = sum(1 for b in buckets.values() if b["visual"] or b["ocr"])
            logger.info(
                f"Visual sampling complete: {processed_count} unique scenes analyzed | "
                f"{active_buckets}/{len(buckets)} active buckets | "
                f"{duration:.1f}s video"
            )

            # ── Stage 5 + 6: Temporal fusion ──────────────────────────────────
            #
            # Replaces original flat concatenation with time-aligned structure.
            # Example output for a 20s reel:
            #
            #   [0s-5s]
            #   SPEECH    : Hey everyone, welcome to today's recipe
            #   ON-SCREEN : @creator | Save this recipe!
            #   VISUAL    : Person gesturing at ingredients on counter
            #
            #   [5s-10s]
            #   SPEECH    : First we're going to chop the onions
            #   ON-SCREEN : N/A
            #   VISUAL    : Close-up of knife chopping onions -> overhead pan shot
            #
            structured_context = self._build_structured_context(
                buckets, audio_segments, duration
            )

            if not structured_context.strip():
                logger.warning("No structured context generated — silent/static video.")
                return scraped_items

            # ── Stage 7: Narrative LLM enrichment ────────────────────────────
            if check_abort_fn and check_abort_fn():
                logger.info("Abort signaled before Narrative enrichment. Returning structured context.")
                scraped_items.append({"content": structured_context, "sub_type": "video_visual"})
                return scraped_items

            narrative_prompt = self._build_narrative_prompt(structured_context, duration)
            final_visual     = structured_context  # safe fallback

            if self.narrative_llm is not None:
                try:
                    enriched = await self.narrative_llm.generate(
                        prompt      = narrative_prompt,
                        max_tokens  = NARRATIVE_MAX_TOKENS,
                        temperature = NARRATIVE_TEMPERATURE
                    )
                    if enriched and not enriched.startswith("Error"):
                        final_visual = enriched.strip()
                        logger.info("Narrative enrichment: success")
                    else:
                        logger.warning(
                            "Narrative LLM returned empty/error — "
                            "falling back to structured context."
                        )
                except Exception as ne:
                    logger.error(
                        f"Narrative enrichment failed: {ne} — "
                        "falling back to structured context."
                    )
            else:
                logger.info(
                    "Narrative LLM not wired — "
                    "returning structured context as video_visual."
                )

            # ── Output — return contract UNCHANGED ───────────────────────────
            scraped_items.append({
                "content":  final_visual,
                "sub_type": "video_visual"
            })

            return scraped_items

        except Exception as e:
            logger.error(f"Video pipeline failed: {e}")
            return [{"content": f"Video Error: {str(e)}", "sub_type": "video_visual"}]