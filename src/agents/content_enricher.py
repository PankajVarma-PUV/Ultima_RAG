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
Content Enricher Agent for UltimaRAG.
Transforms raw extraction data into high-fidelity enriched narratives.
Single Source of Truth (SSOT) Implementation.
"""

from typing import Dict, Any, Optional
import json
from ..core.ollama_client import OllamaClient
from ..core.utils import logger, Timer
from ..core.config import Config

ENRICHMENT_PROMPTS = {
    "image": """You are **UltimaRAG's Knowledge Base Architect**.
    
RAW PERCEPTION DATA (OCR + VISION):
{raw_content}

TASK: Systematically transform this raw perception data into a high-fidelity Knowledge Base segment.

INSTRUCTIONS:
1. **Fact Extraction**: Precisely identify and extract all factual data, textual elements, and visual entities.
2. **Contextualization**: Explain the relationship between the extracted text (OCR) and the visual scene (Vision).
3. **Structured Narrative**: Organize the information into clear, descriptive paragraphs that function as a standalone knowledge entry.
4. **Tone**: Authoritative, cinematic, and professional.
5. **No Technical Noise**: Output ONLY the enriched knowledge base content. Do NOT include technical markers like "OCR result" or "Vision labels".

OUTPUT: A definitive, well-structured knowledge segment capturing the totality of the asset.""",

    "video": """You are **UltimaRAG's Multi-Modal Intelligence Fusion Agent**.

RAW MULTIMODAL DATA (WHISPER AUDIO + QWEN VISUAL + OCR):
{raw_content}

TASK: Combine all provided audio and visual insights to create a comprehensive, high-fidelity Knowledge Base dossier.

INSTRUCTIONS:
1. **Intelligence Fusion**: Properly combine the Whisper-extracted audio description, the OCR-extracted text, and the Qwen Vision-based scene descriptions.
2. **Chronological Reconstruction**: Map the visual actions to the corresponding audio events to tell a cohesive story.
3. **Knowledge Deep-Dive**: Extract core themes, names, technical terms, and semantic meaning.
4. **Structure**: Use clear paragraph breaks. Maintain a cinematic and authoritative narrative.
5. **Sanitization**: Output ONLY the final enriched intelligence. Do NOT use labels like "Audio:" or "Visual:".

OUTPUT: A multi-page level knowledge dossier that captures the complete value of this video.""",

    "document": """You are a **Knowledge Extraction Specialist**.

RAW DOCUMENT CONTENT:
{raw_content}

TASK: Reconstruct the extracted document text into a high-fidelity, structured Knowledge Base entry.

INSTRUCTIONS:
1. **Cleanup**: Fix all extraction artifacts, line breaks, and formatting issues.
2. **Structural Integrity**: Restore logical hierarchy and paragraph flow.
3. **Factual Preservation**: Ensure all names, dates, and technical data are preserved with 100% accuracy.
4. **Readability**: Enhance the flow for AI reasoning and human collaboration.

OUTPUT: The definitive, enriched, and structured version of the document.""",

    "audio": """You are **UltimaRAG's Senior Audio Intelligence Analyst**.

RAW TRANSCRIPTION INSIGHTS:
{raw_content}

TASK: Transform these raw transcription insights into a polished, high-fidelity Knowledge Base segment.

INSTRUCTIONS:
1. **Semantic Enhancement**: Turn fragmented transcriptions into a cohesive, meaningful narrative.
2. **Thematic Organization**: Identify and categorize the main topics or directives discussed.
3. **Clarity**: Ensure the speaker's intent and tone are captured with authoritative clarity.
4. **Professional Polish**: Add proper punctuation and logical paragraph breaks.

OUTPUT: The enriched, high-fidelity knowledge segment derived from the audio transcription."""
}

class ContentEnricher:
    """Agent: Content Enrichment Specialist (Configured Model)"""
    
    def __init__(self, model_name: str = None):
        from ..core.config import OllamaConfig, Config
        self.model_name = model_name or Config.ollama_multi_model.HEAVY_MODEL
        
        # Enforce minimum timeout for Enrichment (10 minutes) given the complexity of the task
        enrichment_timeout = max(OllamaConfig.TIMEOUT, 600)
        
        self.client = OllamaClient(
            model_name=self.model_name,
            timeout=enrichment_timeout
        )
        logger.info(f"ContentEnricher initialized with {self.model_name} | Timeout: {enrichment_timeout}s")

    async def enrich_content(self, raw_content: str, content_type: str, file_name: str) -> str:
        """
        Enrich raw extracted content into a high-fidelity narrative.
        
        Args:
            raw_content: The merged raw extraction data.
            content_type: Type of content ('image', 'video', 'pdf', 'audio', etc.)
            file_name: Name of the original file for context.
            
        Returns:
            Enriched narrative string.
        """
        if not raw_content or len(raw_content.strip()) < 20:
            logger.warning(f"Raw content for {file_name} too short to enrich. Using original.")
            return raw_content

        # Map internal types to prompt categories
        category_map = {
            "image": "image",
            "photo": "image",
            "screenshot": "image",
            "video": "video",
            "audio": "audio",
            "pdf": "document",
            "txt": "document",
            "document": "document"
        }
        category = category_map.get(content_type.lower(), "document")
        
        if category == "document":
            logger.info(f"⏭️ Skipping LLM enrichment for document/text: {file_name} (Using raw content)")
            return raw_content

        prompt_template = ENRICHMENT_PROMPTS.get(category, ENRICHMENT_PROMPTS["document"])
        prompt = prompt_template.format(raw_content=raw_content)
        
        if category == "image":
            target_model = Config.ollama_multi_model.LIGHTWEIGHT_MODEL
            target_tokens = Config.ollama_multi_model.LIGHTWEIGHT_MAX_TOKENS
        else:
            target_model = Config.ollama_multi_model.HEAVY_MODEL
            target_tokens = Config.ollama_multi_model.HEAVY_MAX_TOKENS

        logger.info(f"✨ Enriching {category} content for: {file_name} [Model: {target_model}]")
        
        with Timer(f"Content Enrichment ({file_name})"):
            try:
                # SOTA: OllamaClient is native async HTTPX now
                result = await self.client.generate(
                    prompt, 
                    temperature=0.4, 
                    max_tokens=target_tokens,
                    model=target_model
                )
                
                enriched = result.get("response", "") if isinstance(result, dict) else result
                
                if not enriched or len(enriched.strip()) < 10:
                    logger.warning("Enrichment produced empty result. Falling back to raw.")
                    return raw_content
                    
                return enriched.strip()
                
            except Exception as e:
                logger.error(f"Enrichment failed for {file_name}: {e}")
                return raw_content # Fallback to original content on failure

