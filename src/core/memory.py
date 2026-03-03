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

from typing import List, Dict, Optional
from datetime import datetime
from .database import UltimaRAGDatabase
from .config import Config
from .utils import logger

# ---------------------------------------------------------------------------
# MemGPT Token Budget Constants (Dynamic Configuration)
# ---------------------------------------------------------------------------
# The memory system auto-scales based on the MAX_INPUT_TOKENS set in .env.
# It reserves a percentage (MEMGPT_OVERFLOW_RATIO) for active history.
# The remainder is reserved for system prompts, RAG chunks, and multimodal data.
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    UltimaRAG Memory Manager (MemGPT implementation).
    Controls the 'Virtual Context' by paging messages in and out of the LLM window.
    Tuned for Gemma3:4b with a 2048-token total context window.
    """

    def __init__(self, db: UltimaRAGDatabase):
        self.db = db
        # MemGPT Overflow Threshold — derived from Config.memgpt so it auto-scales
        # with MAX_INPUT_TOKENS when you upgrade hardware via .env
        # Default: 80% of 2048 = 1638 tokens for history
        self.threshold = Config.memgpt.get_threshold()

    def get_prompt_context(self, conversation_id: str) -> List[Dict]:
        """
        Retrieves the active context for an LLM prompt.
        If the active context is empty or too short, it may 'recall' recent paged data.
        """
        active_msgs = self.db.get_active_messages(conversation_id)
        
        # If conversation is new or empty, return as is
        if not active_msgs:
            return []
            
        # SOTA: Sanitize context to prevent internal field leakage (vectors, state, etc.)
        sanitized = []
        for msg in active_msgs:
            sanitized.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })
        return sanitized

    def get_all_context(self, conversation_id: str) -> List[Dict]:
        """
        Retrieves the entire conversation history for specialized reasoning.
        """
        all_msgs = self.db.get_full_history(conversation_id)
        
        sanitized = []
        for msg in all_msgs:
            sanitized.append({
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("created_at")
            })
        return sanitized

    def get_semantic_history(self, conversation_id: str, query_vector: List[float], limit: int = 5) -> List[Dict]:
        """
        Performs semantic search within a specific conversation's history.
        """
        results = self.db.search_messages(query_vector, limit=limit, conversation_id=conversation_id)
        
        sanitized = []
        for msg in results:
            sanitized.append({
                "role": msg.get("role"),
                "content": msg.get("content"),
                "timestamp": msg.get("created_at")
            })
        return sanitized

    def count_tokens(self, messages: List[Dict]) -> int:
        """
        Estimates the total token count for a list of messages.
        Uses a conservative approximation: 1 token ≈ 4 characters.
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            # 4 chars per token is a safe approximation (slightly over-estimates)
            total += len(content) // 4 + 4  # +4 for role/overhead per message
        return total

    async def manage_overflow(self, conversation_id: str):
        """
        MemGPT Resource Guard: Called before every LLM request.
        Checks if the active context window exceeds the threshold for Gemma3:4b.
        If so, summarizes and pages out the oldest turn (User + Assistant pair).
        This is the core of the MemGPT paging mechanism.
        """
        active_msgs = self.db.get_active_messages(conversation_id)
        
        if not active_msgs:
            return
        
        # Count tokens in the active context
        current_tokens = self.count_tokens(active_msgs)
        
        if current_tokens < self.threshold:
            logger.debug(f"MemGPT: Context OK ({current_tokens}/{self.threshold} tokens) for {conversation_id}")
            return

        logger.info(f"🧠 MemGPT Overflow Guard: {current_tokens} tokens > {self.threshold} threshold. Paging out oldest turn for {conversation_id}.")

        # Page out in 'Turns' (User + Assistant pair = 2 messages)
        if len(active_msgs) < 2:
            return

        turn_to_page = active_msgs[:2]
        msgs_to_page = [m['id'] for m in turn_to_page]

        # Summarize before discarding (for recall/archival quality)
        summary = await self._summarize_turn(turn_to_page)

        if msgs_to_page:
            self.db.page_out_messages(msgs_to_page, conversation_id, summary, token_count=current_tokens)
            logger.info(f"✅ MemGPT: Paged out 1 oldest turn. Summary: '{summary[:80]}...'")

    async def _summarize_turn(self, turn: List[Dict]) -> str:
        """
        Summarizes a conversation turn for archival.
        Token cap and content truncation length are read from Config.memgpt (env-driven).
        """
        import asyncio
        from .ollama_client import OllamaClient
        client = OllamaClient(model_name=Config.ollama.MODEL_NAME)

        # Truncate each turn to safe length before building prompt (env-driven)
        truncate = Config.memgpt.CONTENT_TRUNCATE_CHARS
        text_parts = []
        for m in turn:
            role = m.get('role', 'unknown')
            content = m.get('content', '')[:truncate]
            text_parts.append(f"{role}: {content}")
        text = "\n".join(text_parts)

        prompt = f"""Summarize the following conversation turn in 1-2 sentences for long-term archival.
Focus only on key facts and outcomes. Be brief.

Turn:
{text}

Summary:"""

        try:
            # Native async generation using httpx
            result = await client.generate(
                prompt,
                temperature=0.0,
                max_tokens=Config.memgpt.SUMMARY_MAX_TOKENS
            )
            response_text = result.get("response", "") if isinstance(result, dict) else result
            return response_text.strip() if response_text else "Turn archived."
        except Exception as e:
            logger.error(f"MemGPT Summarization Error: {e}")
            return "Turn archived."

    def recall_context(self, query_vector: List[float]) -> List[Dict]:
        """
        The 'Recall' action: Searches the entire paged history for relevant context.
        Used when the AI needs to remember something 'long-term'.
        """
        results = self.db.search_messages(query_vector, limit=5)
        
        # Format for inclusion in prompt
        recalled_context = []
        for r in results:
            if r.get('state') == 'PAGED': # Only count things NOT in current window
                recalled_context.append({
                    "role": r.get('role', 'system'),
                    "content": r.get('content', ''),
                    "recalled": True
                })
        
        return recalled_context

    async def extract_facts(self, conversation_id: str, new_messages: List[Dict]):
        """
        Tier 2 Semantic Distillation (Fact Extraction Sub-Agent).
        Reads new User/Assistant messages and extracts immutable facts about the user or project
        to be stored in the 'knowledge_distillation' table for permanent recall.
        Runs asynchronously in the background.
        """
        if not new_messages:
            return

        from .ollama_client import OllamaClient
        import json
        import uuid
        
        # Use the fast, distilled model for fact extraction if configured
        client = OllamaClient(model_name=Config.ollama.MODEL_NAME)
        
        # Combine new messages
        conversation_block = ""
        for msg in new_messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            conversation_block += f"{role}: {content}\n"
            
        system_prompt = """
You are a Background Memory Extraction Agent.
Your goal is to extract permanent, immutable facts about the User, their Preferences, or the Project context from the provided conversation snippet.
Examples of facts:
- "User is building a React Native app"
- "User prefers Python for backend scripting"
- "The main server IP is 192.168.1.50"

Output a strictly valid JSON array of facts.
Format:
[
  {"fact": "Extracted string", "domain": "user_preference|project_context", "confidence": 0.95}
]
If no concrete facts are found, output an empty array: []
"""
        user_prompt = f"Analyze this conversation:\n\n{conversation_block}\n\nExtract facts as JSON."
        
        try:
            # We use the new async generator with JSON format constraint
            result = await client.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.0,
                format="json"
            )
            
            response_text = result.get("response", "[]") if isinstance(result, dict) else result
            
            try:
                facts = json.loads(response_text)
                if not isinstance(facts, list):
                    facts = [facts]
            except json.JSONDecodeError:
                logger.warning(f"Fact Extraction Sub-Agent failed to parse JSON: {response_text[:100]}")
                return
                
            if facts:
                # Store extracted facts in the database
                for fact in facts:
                    if not isinstance(fact, dict) or "fact" not in fact:
                        continue
                        
                    metric_fact = fact.get("fact", "")
                    domain = fact.get("domain", "general")
                    confidence = float(fact.get("confidence", 0.5))
                    
                    if len(metric_fact) > 5 and confidence > 0.6:
                        self.db.conn.open_table("knowledge_distillation").add([{
                            "id": str(uuid.uuid4()),
                            "conversation_id": conversation_id,
                            "extracted_fact": metric_fact,
                            "domain": domain,
                            "confidence": confidence,
                            "created_at": datetime.utcnow().isoformat()
                        }])
                logger.info(f"✅ Extracted {len(facts)} permanent facts into Knowledge Distillation layer.")
                
        except Exception as e:
            logger.error(f"Fact Extraction Sub-Agent Error: {e}")

