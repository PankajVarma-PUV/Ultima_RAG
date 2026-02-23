import re
import json
from ..core.utils import logger
from ..core.models import Intent
from langchain_ollama import OllamaLLM as Ollama

# Regex pattern for @mention detection: handles @filename.ext or @filename (stops at space/punctuation)
# SOTA Fix: Ensure dots follow by chars are captured (extensions), but trailing dots (end of sentence) are not.
MENTION_PATTERN = re.compile(r'@([\w\- ]+(?:\.[\w\-]+)*)(?=\s|$|\?|!|[,;:])')


def parse_mentions(query: str) -> list:
    """Extract @filename mentions from a query string.
    Returns a list of unique file names (case-preserved)."""
    matches = MENTION_PATTERN.findall(query)
    # Filter out common false positives (like bare @ followed by a space)
    return list(dict.fromkeys([m.strip() for m in matches if m.strip()]))


def strip_mentions(query: str) -> str:
    """Remove @filename mentions from query, leaving the clean question."""
    return MENTION_PATTERN.sub('', query).strip()


class IntentClassifier:
    """
    SOTA Wise Intent Router for Ultima_RAG Metacognitive Brain.
    Classifies user intent into 4 buckets:
    - GENERAL: Chat, Translation, Knowledge
    - RAG: Document search
    - PERCEPTION: Multimodal analysis
    - MULTI_TASK: Chained agents
    """
    
    def __init__(self):
        from ..core.config import Config
        self.llm = Ollama(
            model=Config.ollama.MODEL_NAME,
            base_url=Config.ollama.BASE_URL,
            timeout=Config.ollama.TIMEOUT
        )
        logger.info("Wise Intent Router (Ultima_RAG Edition) initialized")
    
    async def classify(self, query: str, mentioned_files: list = None) -> tuple:
        """
        Classify user intent and detect target language.
        Returns Tuple[Intent, Optional[str]]
        """
        # 1. Target Language Detection (Regex based for common patterns)
        target_language = None
        lang_patterns = [
            (r"(?i)in\s+(hindi|spanish|french|german|japanese|chinese|russian|marathi|gujarati|bengali|telugu)", 1),
            (r"(?i)to\s+(hindi|spanish|french|german|japanese|chinese|russian|marathi|gujarati|bengali|telugu)", 1),
            (r"(?i)(hindi|spanish|french|german|japanese|chinese|russian|marathi|gujarati|bengali|telugu)\s+language", 0)
        ]
        
        for pattern, group_idx in lang_patterns:
            match = re.search(pattern, query)
            if match:
                target_language = match.group(group_idx).capitalize()
                break

        # 2. Fast-path: HISTORY detection
        history_patterns = [
            r"(?i)past\s*conversations?", 
            r"(?i)history", 
            r"(?i)what\s*have\s*we\s*talked\s*about",
            r"(?i)summarize\s*our\s*chat",
            r"(?i)tell\s*me\s*everything\s*about"
        ]
        if any(re.search(p, query) for p in history_patterns):
            logger.info(f"Intent: Fast-path HISTORY detected")
            return Intent.HISTORY, target_language

        if mentioned_files:
            logger.info(f"Intent: Forced RAG due to @mentions: {mentioned_files}")
            return Intent.RAG, target_language


        prompt = f"""
<role>
You are the Ultima_RAG Wise Intent Router. Classify inquiries into the most efficient processing pipeline with high-fidelity calibration.
</role>

<taxonomy>
- GENERAL: Greetings, general conversation, or non-grounded knowledge.
- RAG: Questions requiring precision document retrieval and facts.
- PERCEPTION: Inquiries about visual/audio assets (images, videos).
- MULTI_TASK: Complex requests involving multiple distinct actions (e.g., "Summarize this PDF and translate the first page").
- HISTORY: Requests for conversational recall or timeline summaries.
</taxonomy>

<user_input>
"{query}"
</user_input>

<mandate>
1. XML THINKING: First, explain your classification logic in a <thinking> block.
2. JSON OUTPUT: Provide a JSON object with "intent" and "confidence_score" (0.0-1.0).
</mandate>

INTENT JSON:"""
        
        try:
            # SOTA: Using ainvoke for parallel agentic performance
            response_raw = await self.llm.ainvoke(prompt)
            result_text = response_raw.strip()
            
            # Clean JSON
            if "```" in result_text:
                result_text = re.sub(r'```(?:json)?\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)
            
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result_data = json.loads(json_match.group())
            else:
                result_data = {"intent": "GENERAL", "confidence_score": 0.5}

            intent_str = result_data.get("intent", "GENERAL").upper()
            confidence = float(result_data.get("confidence_score", 0.5))
            
            intent = Intent.GENERAL
            if "PERCEPTION" in intent_str: intent = Intent.PERCEPTION
            elif "RAG" in intent_str: intent = Intent.RAG
            elif "MULTI" in intent_str: intent = Intent.MULTI_TASK
            elif "HISTORY" in intent_str: intent = Intent.HISTORY
            
            logger.info(f"Wise Intent Routing: {intent} (Confidence: {confidence:.2f})")
            return intent, target_language
            
        except Exception as e:
            logger.error(f"Wise Intent Routing Error: {e}")
            return Intent.GENERAL, target_language # Safe default

    def detect_context_rejection(self, query: str) -> bool:
        """
        Detect if the user explicitly requests to ignore context/scrapped knowledge.
        Matches common patterns like 'ignore context', 'do not use documents', etc.
        """
        patterns = [
            r"(?i)do\s*not\s*use\s*(context|scrapped|knowledge|documents|files|scrapped content)",
            r"(?i)ignore\s*(context|scrapped|knowledge|documents|files|scrapped content)",
            r"(?i)without\s*(using|referring\s*to)\s*(context|documents|files)",
            r"(?i)don'?t\s*use\s*(context|documents|files)"
        ]
        
        for p in patterns:
            if re.search(p, query):
                logger.info(f"Intent: Explicit context rejection detected in query: '{query}'")
                return True
        return False

