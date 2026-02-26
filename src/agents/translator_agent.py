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
Specialized agent for translation tasks using Config-defined model.
"""

from typing import Dict, Optional

from ..core.ollama_client import OllamaClient, OllamaConfig
from ..core.config import Config
from ..core.utils import logger, Timer


# =============================================================================
# TRANSLATION PROMPT
# =============================================================================

TRANSLATION_PROMPT = """You are an expert {target_language} translator.

Task: Translate the following text into {target_language}.

STRICT RULES (MUST FOLLOW):
1. Output ONLY the translated text in {target_language} - nothing else.
2. Do NOT include:
   - Transliterations (English spellings of translated words)
   - Explanations or notes
   - Original English text in parentheses
   - Any meta-commentary like "Here's the translation..."
3. Preserve all facts, numbers, dates, and proper nouns exactly.
4. Use natural, native {target_language} phrasing.
5. Maintain the original tone and formatting (headings, bullet points, etc.).
6. For technical terms with no direct translation, use the accepted {target_language} equivalent or keep the original term.
7. STERN RULE: NEVER include the original English text alongside the translation. 
8. Do NOT provide "English - {target_language}" pairs. Provide ONLY {target_language}.

Text to translate:
{source_text}

{target_language} translation:"""


# =============================================================================
# TRANSLATOR AGENT
# =============================================================================

class TranslatorAgent:
    """
    Specialized agent for translation tasks.
    Uses Config-defined model for quality translations.
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Translator Agent.
        
        Args:
            model_name: Ollama model to use (default: from config)
        """
        self.model_name = model_name or Config.ollama_multi_model.AGENT_MODELS.get("translator", OllamaConfig.MODEL_NAME)
        self.client = OllamaClient(model_name=self.model_name)
        
        if not self.client.is_available():
            logger.warning(f"Ollama model {model_name} not available, falling back to default")
            self.client = OllamaClient()
        
        logger.info(f"TranslatorAgent initialized with model: {model_name}")
    
    def translate(
        self,
        text: str,
        target_language: str = "Hindi",
        preserve_formatting: bool = True
    ) -> Dict:
        """
        Translate the provided text.
        
        Args:
            text: Text to translate
            target_language: Target language for translation
            preserve_formatting: Whether to preserve original formatting
        
        Returns:
            Translation result dictionary
        """
        if not text or len(text.strip()) < 2:
            return {
                "translation": text,
                "source_text": text,
                "target_language": target_language,
                "status": "TOO_SHORT",
                "error": "Text too short to translate"
            }
        
        with Timer("Translation"):
            try:
                prompt = TRANSLATION_PROMPT.format(
                    target_language=target_language,
                    source_text=text
                )
                
                translation = self.client.generate(
                    prompt,
                    temperature=0.3,  # Some creativity for natural phrasing
                    max_tokens=Config.ollama_multi_model.HEAVY_MAX_TOKENS
                )
                
                translation = translation.strip()
                
                result = {
                    "translation": translation,
                    "source_text": text,
                    "target_language": target_language,
                    "source_word_count": len(text.split()),
                    "translation_word_count": len(translation.split()),
                    "status": "SUCCESS",
                    "model_used": self.model_name
                }
                
            except Exception as e:
                logger.error(f"Translation failed: {e}")
                result = {
                    "translation": f"Translation failed: {str(e)}",
                    "source_text": text,
                    "target_language": target_language,
                    "status": "ERROR",
                    "error": str(e)
                }
        
        logger.info(f"Translation to {target_language} complete")
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_translator(model_name: str = None) -> TranslatorAgent:
    """Get a configured TranslatorAgent instance"""
    return TranslatorAgent(model_name=model_name)


def translate_text(text: str, target_language: str = "Hindi") -> Dict:
    """Convenience function to translate text"""
    translator = get_translator()
    return translator.translate(text, target_language)
