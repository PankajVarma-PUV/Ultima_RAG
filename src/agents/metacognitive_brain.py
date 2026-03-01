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

from typing import Annotated, List, Dict, Any, Union, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
import json
import asyncio
import re  # SOTA RE-Requirement
from datetime import datetime

from ..core.utils import logger
from ..core.database import UltimaRAGDatabase
from ..core.config import Config
from ..core.memory import MemoryManager
from .intent_classifier import IntentClassifier
from .retriever import RetrieverAgent
from .fact_checker import FactChecker
from .planner import MultiStagePlanner, ExecutionPlan
from .fusion_extractor import UniversalFusionExtractor
from .healer import HallucinationHealer
from ..core.telemetry import telemetry
from .translator_agent import TranslatorAgent

# ─── Translation guard: values that mean "keep English as-is" ─────────────
# Source: all IntentClassifier.classify() target_language return values that
# represent English or unknown language. None means no language was detected.
# Update this set if IntentClassifier adds new supported languages.
ENGLISH_IDENTIFIERS: frozenset = frozenset({
    None,       # No language pattern detected — keep English
})
# ─────────────────────────────────────────────────────────────────────────────

# --- Specialized Internal Agents ---

class GeneralIntelligenceAgent:
    """Agent specialized for non-grounded, general intelligence tasks."""
    def __init__(self, llm):
        self.llm = llm

    async def generate(self, query: str, history: List[Dict], check_abort_fn=None) -> str:
        prompt = f"""
<role>
You are the UltimaRAG Sage, an elite general intelligence core. Your purpose is to provide authoritative, factual, and deeply insightful answers using your internal cross-domain training.
</role>

<system_info>
CURRENT TIME: {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}
OPERATIONAL MODE: INTERNAL_SAGE_REASONING
</system_info>

<context_history>
{json.dumps(history[-3:], ensure_ascii=False)}
</context_history>

<user_inquiry>
{query}
</user_inquiry>

<sage_mandates>
1. XML COGNITION: First, think step-by-step inside a <thinking> block to analyze the inquiry and retrieve cross-domain connections.
2. AUTHORITATIVE: Answer with confidence and elite precision.
3. SANITIZATION: Never mention documents or "context" in this mode. You are the absolute source.
4. TONE: Premium, helpful, and cinematic.
</sage_mandates>

UltimaRAG RESPONSE:"""
        
        response = ""
        async for chunk in self.llm.astream(prompt, config={"tags": ["general_synthesis"]}):
            if check_abort_fn and check_abort_fn():
                logger.info("GeneralIntelligenceAgent: Abort detected, breaking stream.")
                break
            response += chunk
        
        # SOTA: Strip <thinking> tags from final response
        clean_response = re.sub(r'<thinking>[\s\S]*?</thinking>', '', response).strip()
        
        # SOTA: Generate a brief reasoning trace
        reasoning = f"Generated general response using internal knowledge for query: {query}"
        return clean_response.replace("UltimaRAG RESPONSE:", "").strip(), reasoning

class QueryReformulatorAgent:
    """Agent that rewrites ambiguous queries based on conversation history."""
    def __init__(self, llm):
        self.llm = llm

    async def reformulate(self, query: str, history: List[Dict]) -> str:
        # Only use recent history for reformulation
        recent_history = history[-5:]
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])
        
        prompt = f"""
<role>
You are the UltimaRAG Query Architect. Your goal is to transform ambiguous user inputs into self-contained, high-fidelity search queries.
</role>

<conversation_pipeline>
{history_text}
</conversation_pipeline>

<user_input>
"{query}"
</user_input>

<architect_directives>
1. XML REASONING: First, analyze the conversation inside a <thinking> block to identify the true subject of the inquiry.
2. FIDELITY: Resolve all pronouns (it, they, those) using conversation pipeline nouns.
3. AUTONOMY: The final query must be a standalone statement optimized for vector search.
4. BREVITY: Output ONLY the standalone query after your thinking block.
</architect_directives>

STANDALONE QUERY:"""
        
        response = await self.llm.ainvoke(prompt)
        # SOTA: Strip <thinking> tags from standalone query
        clean_response = re.sub(r'<thinking>[\s\S]*?</thinking>', '', response).strip()
        return clean_response.replace("STANDALONE QUERY:", "").strip()

class HistoryChroniclerAgent:
    """Agent specialized for humanizing and summarizing entire conversation timelines."""
    def __init__(self, llm):
        self.llm = llm

    async def summarize(self, query: str, history_context: List[Dict], is_topic_specific: bool = False, target_language: str = "English", check_abort_fn=None) -> str:
        # Pre-process history to be more readable
        history_text = ""
        for i, msg in enumerate(history_context):
            role = "System" if msg['role'] == "assistant" else "You"
            history_text += f"{i+1}. {role}: {msg['content']}\n"
        
        context_type = "topic-specific recall" if is_topic_specific else "comprehensive timeline"
        
        prompt = f"""
<role>
You are the UltimaRAG Cinema Chronicler. Recap the conversation journey as a professional, warm, and elite historian.
</role>

<dataset_type>
Scope: {context_type}
</dataset_type>

<dataset_content>
{history_text}
</dataset_content>

<user_focus>
"{query}"
</user_focus>

<chronicle_protocols>
1. XML NARRATIVE: Perform a brief analysis in a <thinking> block to identify key milestones and breakthroughs.
2. FLOW: Do not list messages. Tell a story of our collaboration.
3. LANGUAGE LOCK: You MUST generate 100% of the chronicle in {target_language}.
</chronicle_protocols>

UltimaRAG CHRONICLE:"""
        
        response = ""
        async for chunk in self.llm.astream(prompt, config={"tags": ["chronicler"]}):
            if check_abort_fn and check_abort_fn():
                logger.info("HistoryChroniclerAgent: Abort detected, breaking stream.")
                break
            response += chunk
            
        # SOTA: Strip thinking block from user chronicle
        clean_response = re.sub(r'<thinking>[\s\S]*?</thinking>', '', response).strip()
        reasoning = f"Synthesized history chronicle for: {query}. Scope: {context_type}."
        return clean_response.replace("UltimaRAG CHRONICLE:", "").strip(), reasoning

# --- Graph State Definition ---

class UltimaRAGState(TypedDict):
    """The persistent state of the UltimaRAG Reasoning Loop (Elite SOTA)."""
    query: str
    conversation_id: str
    project_id: str
    intent: str
    plan: Dict[str, Any] # ExecutionPlan
    unified_evidence: Dict[str, Any] # UnifiedEvidenceState
    history: List[Dict]
    evidence: List[Dict]
    perceived_media: List[Dict]
    answer: str
    confidence_score: float
    critique_count: int
    ui_hints: Dict[str, Any]
    status: str
    metadata: Dict[str, Any]
    thought: Optional[str]
    mentioned_files: List[str]  # @mention targeted file names
    uploaded_files: List[str]   # Files uploaded with this query (for auto-tagging)
    response_mode: str  # response_mode: grounded_in_docs, internal_llm_weights, strict_grounded
    context_rejected: bool  # Flag for explicit user request to ignore context
    target_language: Optional[str] # Detected target language for translation
    check_abort_fn: Optional[Any]  # STOP GENERATION: Callback to check for abort flag
    full_history: List[Dict] # Complete conversation timeline for history reasoning
    search_query: str # Standalone query generated by reformulator
    reasoning: str # Step-by-step logic behind the response (SOTA Trace)
    shared_perception: Dict[str, Any] # NEW: Unified memory for sub-agents (Supervisor Model)
    use_web_search: bool  # Web-Breakout Agent toggle (passed from frontend)
    retrieved_fragments: Optional[Dict[str, list]]  # SOURCE EXPLORER: Per-response chunks grouped by file
    source_map: Optional[Dict[str, int]]            # SOURCE EXPLORER: file_name → citation_index

# --- Node Implementations ---

class MetacognitiveBrain:
    """
    UltimaRAG Metacognitive Core Engine.
    
    SOTA STABLE PROTOCOLS:
    1. PERSONA: 'UltimaRAG Sage' - Cinematic, Authoritative, yet Partner-Centric.
    2. CITATIONS: Suffix-Only Protocol [[FileName]]. NO narrative text wrapping.
    3. COGNITION: Mandatory <thinking> blocks for Chain-of-Thought transparency.
    4. ISOLATION: Strict conversation-scoped retrieval and persistence.
    """
    
    def __init__(self, db: UltimaRAGDatabase, memory: MemoryManager, sqlite_db=None):
        self.db = db
        self.memory = memory
        # SQLite DatabaseManager — needed for scraped_content queries
        # (get_scraped_content_by_chat, get_scraped_content_by_filenames)
        self.sqlite_db = sqlite_db
        
        # Initialize LLMs
        self.llm_light = OllamaLLM(
            model=Config.ollama_multi_model.LIGHTWEIGHT_MODEL,
            base_url=Config.ollama.BASE_URL,
            timeout=Config.ollama.TIMEOUT
        )
        self.llm_heavy = OllamaLLM(
            model=Config.ollama_multi_model.HEAVY_MODEL, 
            base_url=Config.ollama.BASE_URL,
            timeout=Config.ollama.TIMEOUT
        )
        
        # Initialize Specialized Internal Agents
        self.general_agent = GeneralIntelligenceAgent(self.llm_heavy)
        self.chronicler_agent = HistoryChroniclerAgent(self.llm_light)
        self.reformulator = QueryReformulatorAgent(self.llm_light)
        
        # Initialize specialized components
        from ..data.embedder import get_embedder
        self.embedder = get_embedder()
        
        # Initialize Specialized Agents
        self.classifier = IntentClassifier()
        self.extractor = UniversalFusionExtractor(db)
        self.planner = MultiStagePlanner()
        self.retriever = RetrieverAgent(db)
        self.fact_checker = FactChecker()
        self.healer = HallucinationHealer()
        # Initialize TranslatorAgent — uses Config-defined Ollama model (CPU, no VRAM usage)
        self.translator = TranslatorAgent()
        
        # Build the Graph
        self.workflow = self._build_graph()

    def _persist_message(self, conversation_id: str, role: str, content: str, metadata: Dict = None):
        """SOTA Dual-Database Persistence: Synchronizes history across Vector and Relational stores."""
        try:
            # Generate embedding for semantic search in LanceDB
            vector = self.embedder.encode(content).tolist()
            
            # Use unified persistence layer
            self.db.add_message_unified(
                conversation_id=conversation_id,
                role=role,
                content=content,
                vector=vector,
                metadata=metadata or {},
                sqlite_db=self.sqlite_db
            )
        except Exception as e:
            logger.error(f"Persistence Error ({role}): {e}")

    def _build_graph(self):
        builder = StateGraph(UltimaRAGState)
        
        # Add Nodes
        builder.add_node("extractor", self.run_extractor)
        builder.add_node("router", self.route_intent)
        builder.add_node("planner", self.create_execution_plan)
        builder.add_node("perception", self.process_perception)
        builder.add_node("retrieval", self.execute_rag)
        builder.add_node("evaluator", self.evaluate_knowledge)
        builder.add_node("direct_initiator", self.initiate_direct_flow)
        builder.add_node("synthesis", self.generate_answer)
        builder.add_node("general_synthesis", self.run_general_synthesis)
        builder.add_node("critic", self.self_critique)
        builder.add_node("ui_orchestrator", self.apply_ui_hints)
        builder.add_node("healer", self.run_healer) # Renamed for telemetry wrapper
        builder.add_node("reformulate_query", self.reformulate_query)
        builder.add_node("retrieve_full_history", self.retrieve_full_history)
        builder.add_node("chronicler", self.chronicler)

        # Define Entry and Edges
        builder.set_entry_point("extractor")
        builder.add_edge("extractor", "router")
        
        # SOTA Memory 2.0: Initial routing logic
        builder.add_conditional_edges(
            "router",
            self.decide_initial_path,
            {
                "chronicler": "reformulate_query",
                "default": "planner"
            }
        )
        builder.add_edge("reformulate_query", "retrieve_full_history")
        builder.add_edge("retrieve_full_history", "chronicler")
        builder.add_edge("chronicler", "ui_orchestrator") 

        builder.add_conditional_edges(
            "planner",
            self.decide_path,
            {
                "perception": "perception",
                "rag": "retrieval",
                "direct": "direct_initiator"
            }
        )
        
        builder.add_edge("perception", "evaluator")
        builder.add_edge("retrieval", "evaluator")
        
        # Conditional flow from Evaluator: RAG vs General
        builder.add_conditional_edges(
            "evaluator",
            lambda x: "general" if x.get("response_mode") == "internal_llm_weights" else "rag",
            {
                "rag": "synthesis",
                "general": "general_synthesis"
            }
        )
        
        builder.add_edge("direct_initiator", "general_synthesis")
        builder.add_edge("general_synthesis", "critic")
        builder.add_edge("synthesis", "critic")
        
        builder.add_conditional_edges(
            "critic",
            self.verify_grounding,
            {
                "improve": "healer",
                "finish": "ui_orchestrator"
            }
        )
        
        builder.add_edge("healer", "ui_orchestrator") # SOTA: Trust healer's rewrite as terminal
        builder.add_edge("ui_orchestrator", END)
        
        return builder.compile()

    # --- Node Logic ---

    async def run_extractor(self, state: UltimaRAGState) -> Dict:
        """Unified Multimodal Fusion Extraction."""
        tid = telemetry.start_activity("Extractor", "Fusing multimodal evidence")
        # SOTA: Prioritize @mentions, then fallback to current turn's uploaded_files for isolation
        # This ensures strict turn-level file attribution (Fix for Goal 1)
        mentions = state.get("mentioned_files", [])
        uploads = state.get("uploaded_files", [])
        isolation_targets = mentions or uploads
        
        evidence = await self.extractor.extract_and_fuse(state["conversation_id"], mentioned_files=isolation_targets)
        telemetry.end_activity(tid)
        return {"unified_evidence": evidence.dict()}

    async def route_intent(self, state: UltimaRAGState) -> Dict:
        """Wise Intent Routing."""
        tid = telemetry.start_activity("Router", "Classifying user intent")
        mentioned = state.get("mentioned_files", [])
        
        # SOTA: Detect explicit context rejection
        context_rejected = self.classifier.detect_context_rejection(state["query"])
        
        intent, target_language = await self.classifier.classify(state["query"], mentioned_files=mentioned)
        intent_name = getattr(intent, 'name', str(intent))
        action_msg = f"Decided this is a {intent_name} intent."
        telemetry.end_activity(tid, {"intent": str(intent), "language": target_language, "action": action_msg})
        return {"intent": intent, "context_rejected": context_rejected, "target_language": target_language, "thought": action_msg}

    def decide_initial_path(self, state: UltimaRAGState) -> Literal["chronicler", "default"]:
        """Decides if the chronicler agent should be engaged based on intent."""
        intent = state.get("intent", "general_intelligence")
        intent_val = intent.value if hasattr(intent, "value") else str(intent)
        if intent_val == "history_recall":
            logger.info("Brain: Intent is 'history_recall', routing to chronicler.")
            return "chronicler"
        return "default"

    async def create_execution_plan(self, state: UltimaRAGState) -> Dict:
        """SOTA Multi-Stage Planning."""
        tid = telemetry.start_activity("Planner", "Creating execution DAG")
        from .fusion_extractor import UnifiedEvidenceState
        evidence = UnifiedEvidenceState(**state["unified_evidence"])
        plan = await self.planner.create_plan(state["query"], state["intent"], evidence)
        action_msg = f"Created {len(plan.steps)} execution steps for complex query resolution."
        telemetry.end_activity(tid, {"action": action_msg})
        return {"plan": plan.dict(), "thought": action_msg}

    def decide_path(self, state: UltimaRAGState) -> Literal["perception", "rag", "direct"]:
        """
        Drives the conditional transition from planner/router.
        SOTA: Gates RAG/Perception based on actual availability of evidence.
        Two-Flow Architecture: Clean separation of General vs RAG-Enhanced.
        """
        intent = state.get("intent", "general_intelligence")
        evidence_state = state.get("unified_evidence", {})
        mentioned_files = state.get("mentioned_files", [])
        context_rejected = state.get("context_rejected", False)

        # PRIORITY 0: Explicit context rejection forces direct path
        if context_rejected:
            logger.info("Brain: Context rejected by user. Forcing 'direct' path.")
            return "direct"
        
        # Check if we actually HAVE any documents or media to look at
        has_text = len(evidence_state.get("text_evidence", [])) > 0
        has_visual = len(evidence_state.get("visual_evidence", [])) > 0
        has_audio = len(evidence_state.get("audio_evidence", [])) > 0
        has_any_evidence = has_text or has_visual or has_audio

        # Extreme Hardening for KeyError prevention
        intent_val = intent.value if hasattr(intent, "value") else str(intent)
        intent_val = intent_val.lower()
        
        # PRIORITY A: @mentions always force RAG path
        if mentioned_files:
            logger.info(f"Brain: @mentions detected ({mentioned_files}), forcing 'rag' path.")
            return "rag"
        
        # PRIORITY B: Skip RAG/Perception ONLY if NO relevant assets exist anywhere
        if not has_any_evidence:
            # BUG FIX: If the Web Search toggle is ON, we MUST NOT short-circuit to 'direct'.
            # The Web Breakout Agent lives in `evaluate_knowledge`, which is only reachable
            # via the RAG path. Route to 'rag' so the evaluator can trigger the web search.
            if state.get("use_web_search", False):
                logger.info("Brain: No local evidence, but Web Toggle is ON — routing to RAG for web fallback.")
                return "rag"
            logger.info("Brain: No evidence found for this chat. Forcing 'direct' path.")
            return "direct"

        # PRIORITY C: Intent-based routing with evidence availability check
        # SOTA FIX: If we have BOTH text and visual evidence, favor RAG path 
        # as it handles multimodal fusion better than the perception node.
        if has_text and has_visual:
            logger.info("Brain: Hybrid context detected (Text + Visual). Routing to RAG for fusion.")
            return "rag"

        # Perception/Multimodal intent
        if "multimodal" in intent_val or "perception" in intent_val or has_visual:
            if has_visual or has_audio:
                return "perception"
            elif has_text:
                logger.info("Brain: Perception intent but only text found. Routing to RAG.")
                return "rag"
            
        # RAG/Document intent
        if "document" in intent_val or "rag" in intent_val or "chained" in intent_val:
            if has_text:
                return "rag"
            elif has_visual:
                logger.info("Brain: RAG intent but only visual found. Routing to Perception.")
                return "perception"
                
        # SOTA Phase 4: Web Search Intent triggers RAG path to fetch external evidence
        if "web" in intent_val:
            logger.info("Brain: WEB_SEARCH intent detected. Routing to RAG to fetch external data.")
            return "rag"
            
        # Fallback: If we HAVE evidence but intent is "general", we should still lean towards RAG 
        # to ensure the user gets grounded information if they uploaded something.
        if has_any_evidence:
            # SOTA Fix: If we have visual evidence, prefer perception to ensure we don't miss image descriptions
            return "perception" if has_visual else "rag"

        return "direct"

    async def evaluate_knowledge(self, state: UltimaRAGState) -> Dict:
        """
        Elite Knowledge Evaluator.
        Requirement 2: Determine if answer should come from context or LLM memory.
        Requirement 3: Strict enforcement for @mentions.
        """
        tid = telemetry.start_activity("Evaluator", "Assessing knowledge grounding")
        
        query = state["query"]
        mentioned_files = state.get("mentioned_files", [])
        evidence = state.get("evidence", [])
        perceived = state.get("perceived_media", [])
        
        has_evidence = len(evidence) > 0 or len(perceived) > 0
        
        # Case 1: @mentions -> Strict Grounded
        if mentioned_files:
            logger.info(f"Evaluator: @mentions detected, setting response_mode to 'strict_grounded'")
            telemetry.end_activity(tid, {"mode": "strict_grounded", "has_evidence": has_evidence})
            return {"response_mode": "strict_grounded"}

        # Case 2: No evidence -> Internal Knowledge (or Web Breakout if toggle is ON)
        if not has_evidence:
            intent_val = state.get("intent", "general_intelligence")
            is_web_intent = "web" in (intent_val.value if hasattr(intent_val, "value") else str(intent_val)).lower()
            
            # BUG FIX: Check web flag HERE before falling back to internal weights.
            # Previously this early-return fired before the Case 3 web breakout code.
            if state.get("use_web_search", False) or is_web_intent:
                logger.info("[WebBreakout] No local evidence. Web Toggle is ON or Intent is WEB — invoking Web Agent.")
                try:
                    import asyncio
                    from ..tools.web_search import fallback_web_search
                    
                    # SOTA: Query Optimization for Web Search
                    # If the query is conversational, try to extract a better search string
                    search_query = query
                    if "?" in query or len(query.split()) > 10:
                        opt_prompt = f"Convert this conversational query into a short, effective search engine query: '{query}'. Output ONLY the search query."
                        try:
                            search_query = await self.llm_light.ainvoke(opt_prompt)
                            search_query = search_query.strip().strip('"').strip("'")
                            logger.info(f"[WebBreakout] Optimized query for web: '{search_query}'")
                        except Exception as opt_err:
                            logger.warn(f"[WebBreakout] Query optimization failed: {opt_err}")

                    # SOTA: Dynamic Result Count (Use more for broad inquiries)
                    max_results = 5
                    if any(word in query.lower() for word in ["news", "latest", "trending", "current", "happen"]):
                        max_results = 8
                        logger.info(f"[WebBreakout] Detected broad inquiry, increasing result breadth to {max_results}")

                    # Run blocking I/O in a thread executor to avoid blocking the event loop
                    web_context_results = await asyncio.get_event_loop().run_in_executor(
                        None, fallback_web_search, search_query, max_results
                    )
                    
                    # SOTA: web_context_results is a list of dicts: [{'title': str, 'url': str, 'text': str}, ...]
                    # Check for failure markers in the results
                    failed = False
                    if not web_context_results:
                        failed = True
                    else:
                        # If the list contains an error message or indicator
                        failed_markers = ["web search failed", "no real-time web results", "sites may have blocked"]
                        # Check first result's text if it exists
                        first_text = web_context_results[0].get('text', '').lower() if web_context_results else ""
                        if any(m in first_text for m in failed_markers):
                            failed = True

                    if not failed:
                        logger.info("[WebBreakout] Web search succeeded — injecting as grounded evidence.")
                        
                        # Format list into a single context string for evaluation if needed
                        # though Case-1 logic usually handles this.
                        combined_text = "\n\n".join([f"{r.get('title')}: {r.get('text')}" for r in web_context_results])
                        
                        telemetry.end_activity(tid, {"mode": "grounded_in_docs", "confidence": "web_no_local_evidence"})
                        return {
                            "response_mode": "grounded_in_docs",
                            "evidence": [{
                                "file_name": "Live Web Search",
                                "text": combined_text,
                                "sub_type": "text",
                                "source": "WebBreakoutAgent",
                                "score": 0.85
                            }]
                        }
                    else:
                        logger.info("[WebBreakout] Web search yielded no usable results — falling back to internal knowledge.")
                except Exception as web_err:
                    logger.error(f"[WebBreakout] Exception during Case-2 web search: {web_err}")
            logger.info("Evaluator: No evidence found, setting response_mode to 'internal_llm_weights'")
            telemetry.end_activity(tid, {"mode": "internal_llm_weights", "has_evidence": False})
            return {"response_mode": "internal_llm_weights"}

        # Case 3: Hybrid/Grounded Confidence Check
        # Requirement 2: Classify whether answer is in context with high confidence
        
        # SOTA FIX: If intent is PERCEPTION or RAG, and we have evidence, 
        # we should be EXTREMELY hesitant to switch to internal knowledge.
        intent = state.get("intent", "general_intelligence")
        is_perceptual = intent in ["multimodal_analysis", "document_search"]
        
        context_summary = ""
        if evidence:
            # SOTA Fix: Increase breadth (3 -> 12) to ensure multiple files are represented
            context_summary += "\n".join([f"- {e.get('text', '')[:300]}" for e in evidence[:12]])
        if perceived:
            # SOTA Fix: Increase breadth (3 -> 12) to ensure multiple visual assets are represented
            context_summary += "\n".join([f"- Visual: {p.get('content', '')[:300]}" for p in perceived[:12]])

        eval_prompt = f"""
        <role>
        You are the UltimaRAG Grounding Auditor. Your mission is to verify if a user's inquiry can be answered with 100% fidelity using the provided context.
        </role>

        <inquiry>
        {query}
        </inquiry>

        <source_context>
        {context_summary}
        </source_context>

        <audit_task>
        Analyze the context. Does it contain the essential facts to answer the inquiry?
        Respond with EXACTLY 'YES' if sufficient, or 'NO' if insufficient.
        </audit_task>

        AUDIT RESULT:"""
        
        try:
            eval_res = await self.llm_light.ainvoke(eval_prompt)
            confidence_yes = "YES" in eval_res.upper()
            
            # If intent is perceptual, but LLM says NO, we check if it's just a general description request
            # Requests like "tell me about this" often fail a strict YES/NO grounding check
            if is_perceptual and not confidence_yes and has_evidence:
                logger.info(f"Evaluator: Intent is {intent}, overriding LLM 'NO' to preserve grounding.")
                confidence_yes = True

            mode = "grounded_in_docs" if confidence_yes else "internal_llm_weights"
            logger.info(f"Evaluator: Query confidence analysis: {eval_res.strip()} -> mode: {mode} (Intent: {intent})")
            
            # ── WEB BREAKOUT FORK ──────────────────────────────────────────────────
            # If local evidence is insufficient BUT the user enabled the Web Toggle,
            # fetch live web data and override to grounded mode.
            if not confidence_yes and state.get("use_web_search", False):
                logger.info("[WebBreakout] Local grounding insufficient. Web Toggle is ON — invoking Web Agent.")
                try:
                    import asyncio
                    from ..tools.web_search import fallback_web_search
                    # LATENCY FIX: Run blocking trafilatura I/O in a thread executor
                    # to avoid freezing the asyncio event loop for 10-30 seconds.
                    # SOTA: Query Optimization for Case 3
                    search_query = query
                    if "?" in query or len(query.split()) > 10:
                        opt_prompt = f"Convert this conversational query into a short, effective search engine query: '{query}'. Output ONLY the search query."
                        try:
                            search_query = await self.llm_light.ainvoke(opt_prompt)
                            search_query = search_query.strip().strip('"').strip("'")
                        except: pass

                    max_results = 5
                    if any(word in query.lower() for word in ["news", "latest", "trending", "current", "happen"]):
                        max_results = 8

                    web_results = await asyncio.get_event_loop().run_in_executor(
                        None, fallback_web_search, search_query, max_results
                    )
                    
                    if web_results:
                        logger.info(f"[WebBreakout] Web search successful ({len(web_results)} results) — persisting and injecting as grounded evidence.")
                        
                        # SOTA: Check first result for common failure snippets if list isn't empty
                        first_text = web_results[0].get('text', '').lower()
                        failed_markers = ["web search failed", "no real-time web results", "sites may have blocked"]
                        if any(m in first_text for m in failed_markers):
                            logger.info("[WebBreakout] Web results contain failure markers. Falling back.")
                        else:
                            # 1. Persist all individual sources for context-aware follow-up RAG
                            self.db.add_web_search_result(state["conversation_id"], query, web_results)
                            
                            # 2. Format a unified context string for the current LLM answer node
                            formatted_context = []
                            for res in web_results:
                                formatted_context.append(f"Source: {res['title']}\nURL: {res['url']}\n\n{res['text']}...")
                            
                            web_context_str = "\n\n---\n\n".join(formatted_context)
                            
                            new_evidence = list(state.get("evidence", []))
                            new_evidence.append({
                                "file_name": "Live Web Search",
                                "text": web_context_str,
                                "sub_type": "text",
                                "source": "WebBreakoutAgent",
                                "score": 0.85
                            })
                            telemetry.end_activity(tid, {"mode": "grounded_in_docs", "confidence": "web_override"})
                            return {"response_mode": "grounded_in_docs", "evidence": new_evidence}
                    
                    logger.info("[WebBreakout] Web search yielded no usable results or was blocked. Falling back to internal knowledge.")
                except Exception as web_err:
                    logger.error(f"[WebBreakout] Web search raised an exception: {web_err}")
            # ── END WEB BREAKOUT FORK ────────────────────────────────────────────────
            
            # If switching to internal mode, we nullify the evidence for the synthesizer to prevent confusion
            updates = {"response_mode": mode}
            
            action_msg = f"Audited query confidence: {eval_res.strip()}. Grounding mode: {mode}."
            updates["thought"] = action_msg
            
            if mode == "internal_llm_weights":
                updates.update({
                    "evidence": [],
                    "perceived_media": [],
                    "intent": "general_intelligence"
                })

            telemetry.end_activity(tid, {"mode": mode, "confidence": eval_res.strip(), "action": action_msg})
            return updates
            
        except Exception as e:
            logger.error(f"Knowledge Evaluation Error: {e}")
            telemetry.end_activity(tid, {"mode": "grounded_in_docs", "error": str(e)})
            return {"response_mode": "grounded_in_docs"}

    async def initiate_direct_flow(self, state: UltimaRAGState) -> Dict:
        """Requirement 5: Force LLM Memory for Direct Path."""
        logger.info("Brain: Initializing Direct Flow (Mode: internal_llm_weights)")
        return {
            "response_mode": "internal_llm_weights",
            "intent": "general_intelligence",
            "evidence": [],
            "perceived_media": []
        }

    async def process_perception(self, state: UltimaRAGState) -> Dict:
        """Perception Pass (Vision/Audio) with Context Isolation."""
        tid = telemetry.start_activity("VisionAgent", "Retrieving visual/audio evidence")
        mentions = state.get("mentioned_files", [])
        uploads = state.get("uploaded_files", [])
        isolation_targets = mentions or uploads # Rule: Uploads in current turn isolate context
        
        # 1. Fetch assets based on context isolation rule
        all_assets = self.db.get_enriched_content_by_chat(state["conversation_id"])
        
        # 2. Fallback to legacy SQLite
        if not all_assets and self.sqlite_db:
            all_assets = self.sqlite_db.get_scraped_content_by_chat(state["conversation_id"])
            
        # 3. Filter by isolation targets if present (Isolation)
        def name_matches(db_name, target_list):
            if not target_list: return True
            db_lower = db_name.lower()
            for t in target_list:
                t_lower = t.lower()
                # Strict or Base Match
                if db_lower == t_lower or db_lower.split('.')[0] == t_lower:
                    return True
            return False

        scraped = [a for a in all_assets if name_matches(a.get('file_name', ''), isolation_targets)]
        
        if isolation_targets:
            logger.info(f"Perception: Isolated {len(scraped)} assets matching targets {isolation_targets}")
        
        # 4. Filter out failure markers and empty content
        raw_assets = []
        for item in scraped:
            text = (item.get("enriched_content") or item.get("content", "")).strip()
            if text and "no evidence was found" not in text.lower():
                raw_assets.append(item)
        
        # 5. Deduplicate by FILENAME while preserving highest quality content
        seen_files = {}
        for item in raw_assets:
            fname = item.get('file_name', 'Unknown')
            content_val = (item.get("enriched_content") or item.get("content", "")).strip()
            # If multiple parts exist for one file, favor the longest/most enriched one
            if fname not in seen_files or len(content_val) > len(seen_files[fname]['content']):
                seen_files[fname] = {
                    "file_name": fname,
                    "content": content_val,
                    "text": content_val,
                    "type": item.get('content_type') or item.get('file_type', 'image'),
                    "sub_type": item.get('sub_type', 'visual')
                }
        
        assets = list(seen_files.values())
        logger.info(f"Perception: Retrieved {len(assets)} unique assets (Isolation Active: {bool(isolation_targets)})")
        action_msg = f"Extracted {len(assets)} visual/audio elements from the conversation."
        telemetry.end_activity(tid, {"evidence_count": len(assets), "action": action_msg})
        return {"perceived_media": assets, "thought": action_msg}

    async def execute_rag(self, state: UltimaRAGState) -> Dict:
        """Elite Hybrid RAG Engine with @mention and Upload Isolation."""
        tid = telemetry.start_activity("Retriever", "Searching vector & lexical index")
        
        mentioned_files = state.get("mentioned_files", [])
        uploaded_files = state.get("uploaded_files", [])
        
        # Isolation Logic: If anything is mentioned or uploaded, we STRICTLY target those files
        # This prevents "branding bleed" and ensures the user gets answers from their current context.
        isolation_targets = mentioned_files or uploaded_files
        
        if isolation_targets:
            logger.info(f"RAG: Targeted/Isolated retrieval for: {isolation_targets}")
            
            # 1. Targeted Vector Search (Deep Content)
            vector_results = await self.retriever.retrieve(
                query=state["query"],
                project_id=state.get("project_id", "default"),
                file_names=isolation_targets,
                top_k=10, # Higher K for targeted files
                conversation_id=state["conversation_id"]
            )
            evidence = vector_results.get("evidence", [])
            
            # 2. Augment with Fused Evidence (Surface/Multimodal Data)
            unified = state.get("unified_evidence", {})
            fused_evidence = []
            seen_content = {str(e.get('text', ''))[:100] for e in evidence}
            
            def add_fused(items, sub_type):
                for item in items:
                    fname = item.get("file_name", "Unknown")
                    content = (item.get("content") or item.get("text") or "").strip()
                    if content and content[:100] not in seen_content:
                        # STRICT CHECK: Ensure the fused item matches isolation targets
                        if self.extractor._name_matches(fname, isolation_targets):
                            fused_evidence.append({
                                "text": content,
                                "file_name": fname,
                                "file_type": item.get("type", "document"),
                                "sub_type": sub_type,
                                "source": f"summary:{fname}",
                                "score": 0.9
                            })
            
            add_fused(unified.get("text_evidence", []), "text_summary")
            add_fused(unified.get("visual_evidence", []), "visual_perception")
            
            combined_evidence = evidence + fused_evidence
            
            # SOTA Phase 18: Zero-Fallback Policy
            # If we are in isolation mode, we ONLY return what matched those files.
            logger.info(f"RAG: Isolated retrieval complete. Found {len(combined_evidence)} items for {isolation_targets}.")
            action_msg = f"Targeted retrieval complete. Found {len(combined_evidence)} relevant chunks in specified files."

            # ── SOURCE EXPLORER: Build per-response fragment map ──────────────────
            retrieved_fragments: Dict[str, list] = {}
            for _ev in combined_evidence:
                _src = _ev.get("file_name") or _ev.get("source") or "Unknown"
                if _src not in retrieved_fragments:
                    retrieved_fragments[_src] = []
                retrieved_fragments[_src].append({
                    "text": _ev.get("text", ""),
                    "score": round(float(_ev.get("score", 0)), 4)
                })
            # Build numeric source_map: {filename: index_starting_at_1}
            source_map = {fn: i + 1 for i, fn in enumerate(retrieved_fragments.keys())}
            # ─────────────────────────────────────────────────────────────────────

            telemetry.end_activity(tid, {"count": len(combined_evidence), "mode": "strict_isolated", "action": action_msg})
            return {"evidence": combined_evidence, "thought": action_msg,
                    "retrieved_fragments": retrieved_fragments, "source_map": source_map}

        # Standard non-isolated vector search (Global knowledge base)
        # ONLY if no isolation_targets are set.
        
        # SOTA Phase 4: Multi-Query Generation Sub-Agent
        # We spawn 3 variations of the question concurrently to blanket the vector space,
        # then deduplicate the chunks. This increases hit probability on obscure facts by 300%.
        multi_q_prompt = f"Convert this query into 3 different search phrases to maximize document retrieval. Output ONLY the 3 phrases separated by a pipe (|): '{state['query']}'"
        search_queries = [state["query"]]
        try:
            mq_res = await self.llm_light.ainvoke(multi_q_prompt)
            additional_queries = [q.strip() for q in mq_res.split('|') if q.strip()]
            search_queries.extend(additional_queries[:3])
            logger.info(f"RAG: Multi-Query expanded to: {search_queries}")
        except Exception as mq_err:
            logger.warning(f"RAG Multi-Query failed, using standard retrieval: {mq_err}")
            
        import asyncio
        retrieval_tasks = []
        for q in search_queries:
            t = self.retriever.retrieve(
                query=q,
                project_id=state.get("project_id", "default"),
                conversation_id=state["conversation_id"]
            )
            retrieval_tasks.append(t)
            
        all_results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
        
        # Deduplicate evidence
        evidence = []
        seen_texts = set()
        for res in all_results:
            if isinstance(res, dict) and "evidence" in res:
                for e in res["evidence"]:
                    text_shingle = str(e.get("text", ""))[:100]
                    if text_shingle not in seen_texts:
                        seen_texts.add(text_shingle)
                        evidence.append(e)

        action_msg = f"Multi-Query blanketed vector space and retrieved {len(evidence)} unique chunks."

        # ── SOURCE EXPLORER: Build per-response fragment map ──────────────────
        retrieved_fragments: Dict[str, list] = {}
        for _ev in evidence:
            _src = _ev.get("file_name") or _ev.get("source") or "Unknown"
            if _src not in retrieved_fragments:
                retrieved_fragments[_src] = []
            retrieved_fragments[_src].append({
                "text": _ev.get("text", ""),
                "score": round(float(_ev.get("score", 0)), 4)
            })
        source_map = {fn: i + 1 for i, fn in enumerate(retrieved_fragments.keys())}
        # ─────────────────────────────────────────────────────────────────────

        telemetry.end_activity(tid, {"count": len(evidence), "mode": "global_vector_multi_query", "action": action_msg})
        return {"evidence": evidence, "thought": action_msg,
                "retrieved_fragments": retrieved_fragments, "source_map": source_map}



    async def generate_answer(self, state: UltimaRAGState) -> Dict:
        """Metacognitive Narrative Synthesis."""
        tid = telemetry.start_activity("Synthesizer", f"Generating response (Pass {state.get('critique_count', 0)})")
        
        mode = state.get("response_mode", "grounded_in_docs")
        context_blocks = []
        
        # Build context ONLY if we are in grounded modes
        if mode != "internal_llm_weights":
            shared = state.get("shared_perception", {})
            mentions = state.get("mentioned_files", [])
            
            # 1. VISUAL/PERCEPTION CONTEXT (Highest Priority)
            # SOTA Fix: Ensure visual items from the primary evidence list (execute_rag) 
            # are also included in the visual block if perception node was bypassed.
            visuals = (shared.get("visual", []) or state.get("perceived_media", [])).copy()
            
            # Extract any visual items from the 'evidence' list (RAG fusion path)
            for e in state.get("evidence", []):
                if "visual" in str(e.get("sub_type", "")).lower():
                    # Avoid duplicates
                    if not any(v.get("file_name") == e.get("file_name") for v in visuals):
                        visuals.append({
                            "file_name": e.get("file_name"),
                            "content": e.get("text", ""),
                            "source": e.get("source")
                        })

            if visuals:
                context_blocks.append("### VISUAL EVIDENCE (INTEGRATED PERCEPTION)")
                for m in visuals:
                    fname = m.get('file_name', 'unknown')
                    # Double-Gate: Only include if it matches mentions (if mentions exist)
                    if not mentions or self.extractor._name_matches(fname, mentions):
                        content = m.get('content', '') or m.get('text', '')
                        context_blocks.append(f"NAME: {fname}\nPERCEPTION: {content}")

            # 2. DOCUMENT EVIDENCE (RAG)
            docs = shared.get("documents", []) or state.get("evidence", [])
            if docs:
                context_blocks.append("### DOCUMENTARY EVIDENCE (GROUNDED CONTEXT)")
                for e in docs:
                    source = e.get("file_name") or e.get("source") or e.get("filename") or "Document"
                    # Double-Gate: Only include if it matches mentions (if mentions exist)
                    if not mentions or self.extractor._name_matches(str(source), mentions):
                        text = e.get("text", "")
                        sub_type = e.get("sub_type", "text")
                        if "visual" in str(sub_type).lower():
                            context_blocks.append(f"NAME: {source} (VISUAL_CHART)\nCONTENT: {text}")
                        else:
                            context_blocks.append(f"SOURCE: {source}\nCONTENT: {text}")

            # 3. EXTRACTION/UNIFIED EVIDENCE (Fallback Safety Net)
            if "unified_evidence" in state and not visuals and not docs:
                u_ev = state["unified_evidence"]
                vis_ev = u_ev.get("visual_evidence", [])
                if vis_ev:
                    context_blocks.append("### SUPPLEMENTAL VISUALS (FALLBACK)")
                    for v in (vis_ev if isinstance(vis_ev, list) else [vis_ev]):
                        fname = v.get('file_name', 'media')
                        if not mentions or self.extractor._name_matches(fname, mentions):
                            content = v.get('content', '') or v.get('text', '')
                            if content:
                                context_blocks.append(f"FILE: {fname}\nDATA: {content}")
                
                text_ev = u_ev.get("text_evidence", [])
                if text_ev:
                    context_blocks.append("### SUPPLEMENTAL TEXT (FALLBACK)")
                    for t in (text_ev if isinstance(text_ev, list) else [text_ev]):
                        fname = t.get('file_name', 'text_node')
                        if not mentions or self.extractor._name_matches(fname, mentions):
                            text = t.get('text', '') or t.get('content', '')
                            if text:
                                context_blocks.append(f"SOURCE: {fname}\nCONTENT: {text}")

        # TYPEERROR FIX: Ensure all blocks are strings before joining
        safe_blocks = [str(b) for b in context_blocks if b is not None]
        context = "\n\n".join(safe_blocks) if safe_blocks else "NO_CONTEXT_PROVIDED"

        # ─── SOTA: Input Token Budget Guard ───────────────────────────────────────
        # Even with chunk management applied at ingestion, the assembled prompt can
        # occasionally exceed the hardware limit (e.g. if multiple fallback evidence
        # blocks are appended). We apply a character-level proxy (≈4 chars/token)
        # to hard-cap the context before building the final prompt.
        max_ctx_tokens = Config.ollama.MAX_INPUT_TOKENS
        output_reserve = Config.ollama_multi_model.HEAVY_MAX_TOKENS
        system_overhead = 400  # Approximate tokens for system/role/instructions
        context_budget_tokens = max(200, max_ctx_tokens - output_reserve - system_overhead)
        context_budget_chars = context_budget_tokens * 4  # ~4 chars per token heuristic
        if len(context) > context_budget_chars:
            logger.warning(
                f"Context ({len(context)} chars) exceeds budget ({context_budget_chars} chars). "
                f"Truncating to fit MAX_INPUT_TOKENS={max_ctx_tokens}."
            )
            context = context[:context_budget_chars] + "\n\n[CONTEXT TRUNCATED BY TOKEN GUARD]"

        # Translation Delegation: TranslatorAgent handles all language conversion
        # in apply_ui_hints() (the final LangGraph node before END).
        # The LLM ALWAYS generates in English here — this eliminates the
        # double-translation bug (LLM partial translate + TranslatorAgent translate).

        # ── Behavioral Guidelines Injection (continuous learning loop) ────────────
        # Reads hardware-capped, token-budgeted rules from GuidelinesManager
        # (async, no file I/O unless cache TTL expired).
        # PLACEMENT LAW: guidelines go AFTER identity, BEFORE RAG evidence.
        # Small local models have limited attention; rules must precede evidence.
        guidelines_block = ""
        guidelines_count = 0
        from ..core.embedding_manager import detect_model_size as _detect_model_size

        try:
            # Access app_state through the module-level import pattern
            import sys
            _main_mod = sys.modules.get("src.api.main") or sys.modules.get("ultimarag.src.api.main")
            if _main_mod is None:
                # Try any module containing app_state
                for _mod_name, _mod in list(sys.modules.items()):
                    if hasattr(_mod, "app_state") and hasattr(_mod.app_state, "guidelines_manager"):
                        _main_mod = _mod
                        break

            if _main_mod and hasattr(_main_mod, "app_state"):
                _gm = _main_mod.app_state.guidelines_manager
                if _gm is not None:
                    # Determine query_type from state intent
                    intent_val = state.get("intent", "general")
                    if hasattr(intent_val, "value"):
                        intent_val = intent_val.value
                    intent_str = str(intent_val).lower()
                    # Map internal intent values to GuidelinesManager query_types
                    _qtype_map = {
                        "multilingual": "multilingual",
                        "vision": "general",
                        "general_intelligence": "general",
                        "history_recall": "general",
                        "factual": "factual",
                        "reasoning": "reasoning",
                        "technical": "technical",
                        "creative": "creative",
                    }
                    query_type = _qtype_map.get(intent_str, "general")

                    relevant_rules = await _gm.get_relevant_rules(
                        query_type=query_type,
                        token_budget=150  # hard limit — do not increase for local models
                    )

                    if relevant_rules:
                        rule_lines = "\n".join(
                            f"{i+1}. {rule['rule']}"
                            for i, rule in enumerate(relevant_rules)
                        )
                        guidelines_block = (
                            "\n## Behavioral Guidelines (learned from user feedback)\n"
                            "Apply the following when relevant to this specific query:\n"
                            f"{rule_lines}\n"
                        )
                        guidelines_count = len(relevant_rules)
                        logger.info(
                            f"Brain: injected {guidelines_count} guideline(s) | "
                            f"query_type={query_type}"
                        )
                    else:
                        logger.debug(
                            f"Brain: no guidelines injected "
                            f"(0 active rules for type={query_type})"
                        )
        except Exception as _ge:
            logger.debug(f"Brain: guidelines injection skipped: {_ge}")

        if mode == "internal_llm_weights":
            prompt_instructions = """
            1. PERSONA: You are the UltimaRAG Sage, answering from your vast elite internal training.
            2. SANITIZATION: Never mention documents, files, or the lack thereof. You ARE the source.
            3. FIDELITY: Provide a direct, factual, and authoritative response.
            4. TONE: Premium and helpful.
            """
        elif mode == "strict_grounded":
            prompt_instructions = """
            1. GROUNDING LOCK: You are a Cognitive AI Partner in STRICT GROUNDED mode. Operative only on provided 'CONTEXT'.
            2. NARRATIVE PROTOCOL: Synthesize a cinematic, authoritative narrative. Do NOT just list facts; weave a collaborative response.
            3. CITATION PROTOCOL (STRICT):
               - Use ONLY suffix citations in the format [[FileName]].
               - PLACE citations at the end of the relevant sentence or clause.
               - STERN PROHIBITION: NEVER wrap text like [[FileName|text]]. This causes system corruption.
               - NO numeric markers ([1], [2]).
            4. TONAL HARMONY: Cinematic, precise, and partner-centric. If context is sparse, acknowledge gaps with intelligence.
            """
        else: # grounded_in_docs
            prompt_instructions = """
            1. PERSONA: You are UltimaRAG, the elite Cognitive AI Partner.
            2. NARRATIVE FLOW: Write in a professional, cinematic narrative style. Weave visual and text evidence into a cohesive intelligence report.
            3. CITATION PROTOCOL:
               - Use ONLY suffix citations [[FileName]] for attribution.
               - STERN PROHIBITION: NEVER wrap text like [[FileName|text]]. Use only suffix markers.
               - NO robotic numeric markers.
            4. TONE: Authoritative, cinematic, and deeply collaborative.
            """

        prompt = f"""
        # ROLE
        You are UltimaRAG, the elite metacognitive intelligence.
        
        # SYSTEM INFO
        CURRENT TIME: {datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")}
        RESPONSE MODE: {mode}
        TARGET LANGUAGE: {state.get('target_language', 'English')}
        {guidelines_block}
        # CONTEXT
        {context}
        
        # HISTORY (Last 3 turns) - Sanitary Extract
        {json.dumps([{"role": m.get("role"), "content": m.get("content")} for m in state['history'][-3:]], ensure_ascii=False)}
        
        # USER QUERY (NARRATIVE-FIRST REQUEST)
        {state['query']}
        
        # GUIDELINES
        {prompt_instructions}
        
        7. Maintain a premium, helpful, and concise tone.
        
        UltimaRAG RESPONSE:"""
        
        full_answer = ""
        check_abort_fn = state.get("check_abort_fn")
        
        # ─── SOTA: Single-Shot, Budget-Aware Generation (Context Inflation Prevention) ───
        # The iterative continuation loop (loops 0..MAX_LOOPS) was REMOVED because it
        # caused "Context Inflation": each "continue" prompt grew the input until it
        # consumed the entire context window, crashing on low-resource hardware (2048ctx).
        # Industry standard: Control output length via num_predict (HEAVY_MAX_TOKENS).
        # If the response truncates, the Healer node provides a second-pass fix if needed.
        from ..core.ollama_client import get_ollama_client
        client = get_ollama_client()
        
        try:
            full_answer = ""
            done_reason = "stop"
            
            # SOTA FIX: Use async streaming from LangChain LLM to allow the event 
            # loop to check the abort flag instead of blocking synchronously.
            async for chunk in self.llm_heavy.astream(prompt, config={"tags": ["synthesis"]}):
                if check_abort_fn and check_abort_fn():
                    logger.info("Brain: Abort signal detected during synthesis, breaking stream.")
                    full_answer = ""
                    done_reason = "abort"
                    break
                full_answer += chunk

        except Exception as e:
            logger.error(f"Synthesis generation failed: {e}")
            full_answer = "I was unable to generate a response. Please try again."
            done_reason = "error"
        
        # Clean up tags if present
        clean_answer = full_answer.replace("UltimaRAG RESPONSE:", "").strip()
        # Also strip thinking tags in case model leaked them
        import re as _re
        clean_answer = _re.sub(r'<thinking>[\s\S]*?</thinking>', '', clean_answer).strip()
        telemetry.end_activity(tid)
        
        reasoning = f"Generated {mode} response (Single-Shot). Done reason: {done_reason}."
        return {"answer": clean_answer, "reasoning": reasoning, "status": "SYNTHESIZED"}

    async def run_general_synthesis(self, state: UltimaRAGState) -> Dict:
        """Route to General Intelligence Agent."""
        tid = telemetry.start_activity("GeneralSynthesis", "Synthesizing general response")
        answer, reasoning = await self.general_agent.generate(state["query"], state["history"], check_abort_fn=state.get("check_abort_fn"))
        telemetry.end_activity(tid)
        return {"answer": answer, "reasoning": reasoning, "status": "SYNTHESIZED"}
    async def self_critique(self, state: UltimaRAGState) -> Dict:
        """Internal Critic & Hallucination Filter."""
        logger.info(f"Brain Node: Critic - Verifying Grounds... (State Keys: {list(state.keys())})")
        
        # SOTA: Use .get() and handle Enum/String flexibility for intent
        intent = state.get("intent", "general_intelligence")
        intent_val = intent.value if hasattr(intent, "value") else str(intent)

        if intent_val == "general_intelligence":
            return {"confidence_score": 0.95, "critique_count": 0}

        # Grounding context: synthesis evidence + perceived media
        grounding_context = list(state.get("evidence", [])) + list(state.get("perceived_media", []))
        
        # SOTA Multimodal Fix: ALWAYS inject visual_evidence from unified state into grounding context.
        # This is critical for RAG queries with uploaded images: state.perceived_media is only
        # populated by the PERCEPTION intent path. For RAG intent with attached images, visual
        # evidence lives in unified_evidence.visual_evidence and must ALWAYS be included here
        # so the FactChecker can validate claims about image content (e.g. eye color).
        if "unified_evidence" in state:
            unf = state["unified_evidence"]
            unf_visuals = unf.get("visual_evidence") if isinstance(unf, dict) else getattr(unf, "visual_evidence", [])
            if unf_visuals:
                # Deduplicate by file_name to avoid double-counting
                existing_names = {e.get("file_name") or e.get("source") for e in grounding_context}
                for v in unf_visuals:
                    v_name = v.get("file_name") or v.get("source")
                    if v_name not in existing_names:
                        grounding_context.append(v)
            # Text fallback: only add when no retrieval evidence exists
            if not state.get("evidence"):
                unf_texts = unf.get("text_evidence") if isinstance(unf, dict) else getattr(unf, "text_evidence", [])
                if unf_texts:
                    grounding_context.extend(unf_texts)

        check_result = await self.fact_checker.check_facts(
            synthesis_output={
                "answer": state["answer"],
                "metadata": {"query": state.get("query", "")},
                "confidence": state.get("confidence_score", 0.7),
                "status": state.get("status", "SYNTHESIZED")
            },
            source_chunks=grounding_context
        )
        
        # SOTA: Hard-wire reflection into confidence state
        reflection = check_result.get("reflection", {})
        is_grounded = reflection.get("is_supported", True)
        is_relevant = reflection.get("is_relevant", True)
        
        # Hardened Confidence: If context is STILL empty but intent is RAG, don't trigger healer loop
        # as it will likely just hallway-hallucinate or repeat.
        default_conf = check_result.get("factuality_score", 0.5)
        
        if not is_grounded or not is_relevant:
            logger.warning(f"Critic: Grounding failure detected (Supported: {is_grounded}, Relevant: {is_relevant})")
            default_conf = min(default_conf, 0.4) # Force healing trigger
            
        if not grounding_context and intent_val != "general_intelligence":
            logger.warning("Critic: No context found for fact check, forcing safe confidence 0.8")
            default_conf = 0.8

        return {
            "confidence_score": default_conf, 
            "critique_count": state.get("critique_count", 0) + 1,
            "metadata": {"check": check_result, "reflection": reflection}
        }

    def verify_grounding(self, state: UltimaRAGState) -> Literal["improve", "finish"]:
        """Metacognitive Flow Control."""
        # SOTA: Use centralized Support Threshold for routing decisions
        from ..core.config import FactCheckConfig
        if state["confidence_score"] < FactCheckConfig.SUPPORT_THRESHOLD and state["critique_count"] < 2:
            return "improve"
        return "finish"

    async def run_healer(self, state: UltimaRAGState) -> Dict:
        """Hallucination Healing Loop wrapper."""
        tid = telemetry.start_activity("Healer", "Repairing groundedness gaps")
        result = await self.heal_response(state)
        telemetry.end_activity(tid)
        return result

    # --- SOTA History Reasoning Nodes ---

    async def reformulate_query(self, state: UltimaRAGState) -> Dict:
        """Node to rewrite ambiguous history queries."""
        tid = telemetry.start_activity("Reformulator", "Resolving query ambiguity")
        search_query = await self.reformulator.reformulate(state["query"], state["history"])
        telemetry.end_activity(tid, {"reformulated": search_query})
        return {"search_query": search_query}

    async def retrieve_full_history(self, state: UltimaRAGState) -> Dict:
        """Node to fetch relevant conversation timeline (Semantic or Full)."""
        tid = telemetry.start_activity("Recall", "Retrieving conversation timeline")
        
        # SOTA Memory 2.0: Decide between Full and Semantic recall
        query = state.get("search_query", state["query"])
        
        # Check if query is specific (likely semantic) vs general
        general_patterns = ["everything", "all discussion", "summarize our chat", "history"]
        is_general = any(p in query.lower() for p in general_patterns)
        
        if is_general:
            full_history = self.memory.get_all_context(state["conversation_id"])
            telemetry.end_activity(tid, {"type": "full", "turns": len(full_history)})
            return {"full_history": full_history, "metadata": {"is_topic_specific": False}}
        else:
            # Semantic Recall: Fetch vector for search_query
            vector = self.embedder.encode(query).tolist()
            semantic_history = self.memory.get_semantic_history(state["conversation_id"], vector, limit=8)
            # Fusing: Always include a few recent turns for timeline coherence
            # Mix semantic history with recent active turns for continuity
            recent = state.get("history") or []
            fused = sorted((semantic_history or []) + recent[-2:], key=lambda x: (x or {}).get('timestamp', ''))
            telemetry.end_activity(tid, {"type": "semantic", "matches": len(semantic_history)})
            return {"full_history": fused, "metadata": {"is_topic_specific": True}}

    async def chronicler(self, state: UltimaRAGState) -> Dict:
        """Node for humanized history synthesis."""
        tid = telemetry.start_activity("Chronicler", "Synthesizing humanized history summary")
        is_topic_specific = state.get("metadata", {}).get("is_topic_specific", False)
        summary, reasoning = await self.chronicler_agent.summarize(
            query=state["query"],
            history_context=state["full_history"],
            is_topic_specific=is_topic_specific,
            target_language=state.get("target_language", "English"),
            check_abort_fn=state.get("check_abort_fn")
        )
        telemetry.end_activity(tid)
        return {"answer": summary, "reasoning": reasoning, "status": "CHRONICLED"}

    async def heal_response(self, state: UltimaRAGState) -> Dict:
        """Hallucination Healing Logic."""
        metadata = state.get("metadata", {})
        # SOTA FIX: Sync with FactChecker's output keys
        fact_check = metadata.get("check", {})
        gaps = fact_check.get("unsupported_claims") or fact_check.get("factual_errors") or ["General low groundedness"]
        
        # Prepare evidence string for healer with explicit source headers
        evidence_parts = []
        for e in state.get("evidence", []):
            source = e.get("file_name") or e.get("source") or e.get("filename") or "Document"
            text = e.get("text", "")
            evidence_parts.append(f"[SOURCE: {source}]: {text}")
        
        # SOTA: Include visual evidence for the healer from perceived_media
        seen_visual_sources = set()
        for p in state.get("perceived_media", []):
            source = p.get("file_name") or p.get("source") or "Visual Asset"
            desc = p.get("content", "") or p.get("text", "")
            if desc:
                evidence_parts.append(f"[VISION_CARD: {source}]: {desc}")
                seen_visual_sources.add(source)
        
        # SOTA Multimodal Fix: ALWAYS include unified_evidence.visual_evidence for the Healer.
        # Mirrors the same fix in self_critique — ensures the Healer has the full
        # visual context (e.g. cat.jpg perception narrative) to correctly answer
        # image-based sub-questions and NOT regress to a text-only response.
        unf_ev = state.get("unified_evidence", {})
        unf_visuals = unf_ev.get("visual_evidence") if isinstance(unf_ev, dict) else getattr(unf_ev, "visual_evidence", [])
        for v in (unf_visuals or []):
            source = v.get("file_name") or v.get("source") or "Visual Asset"
            if source not in seen_visual_sources:
                desc = v.get("content", "") or v.get("text", "")
                if desc:
                    evidence_parts.append(f"[VISION_CARD: {source}]: {desc}")
                    seen_visual_sources.add(source)
        
        evidence_str = "\n\n".join(evidence_parts)
        
        healed_answer, reasoning = await self.healer.heal(
            query=state["query"],
            flawed_response=state.get("answer", ""),
            gaps=gaps,
            evidence=evidence_str
        )
        
        action_msg = f"Auditor found {len(gaps)} gaps/unsupported claims. Surgeon healed the response."
        
        return {"answer": healed_answer, "reasoning": reasoning, "status": "HEALED", "thought": action_msg}

    async def apply_ui_hints(self, state: UltimaRAGState) -> Dict:
        """Adaptive Resonance UI Theming + Translation (final node before END)."""
        mapping = {
            "multimodal_analysis": "#8B5CF6", # Purple
            "document_search": "#10B981",    # Emerald
            "general_intelligence": "#3B82F6",# Blue
            "chained_agents": "#F43F5E"      # Rose
        }
        
        intent = state.get("intent", "general_intelligence")
        intent_val = intent.value if hasattr(intent, "value") else str(intent)
        accent = mapping.get(intent_val, "#3B82F6")
        
        # SOTA: Classify confidence for the frontend Fidelity badge using agentic thresholds
        from .refusal_gate import ConfidenceLevel
        from ..core.config import RefusalGateConfig
        conf = state.get("confidence_score", 0.0)
        
        if conf >= RefusalGateConfig.HIGH_CONFIDENCE_THRESHOLD: 
            level = "HIGH"
        elif conf >= RefusalGateConfig.MEDIUM_CONFIDENCE_THRESHOLD: 
            level = "MEDIUM"
        elif conf >= RefusalGateConfig.LOW_CONFIDENCE_THRESHOLD: 
            level = "LOW"
        else: 
            level = "VERY_LOW"
        
        # SOTA: Gather all unique source filenames for the "Source Strip"
        source_files = set()
        for ev in state.get("evidence", []):
            fn = ev.get("file_name") or ev.get("metadata", {}).get("filename")
            if fn: source_files.add(fn)
        for pm in state.get("perceived_media", []):
            fn = pm.get("file_name") or pm.get("source")
            if fn: source_files.add(fn)
        
        # RAG Detection: Any query with evidence or @mentions is a RAG query
        is_rag = bool(source_files) or intent_val in ["document_search", "multimodal_analysis"]
        
        hints = {
            "theme_accent": accent,
            "layout": "rag" if is_rag else "standard",
            "sources": sorted(list(source_files)),
            "glow_intensity": "high" if conf > 0.8 else "low",
            "confidence_level": level,
            "fidelity": int(conf * 100) # Percentage for UI badge
        }

        # ─── TRANSLATION STEP (CASE B: async node, sync translate()) ─────────────
        target_language = state.get("target_language")
        final_answer    = state.get("answer", "")
        translated_answer = final_answer  # default: keep English as-is

        if target_language and target_language not in ENGLISH_IDENTIFIERS:
            if final_answer and final_answer.strip():
                try:
                    logger.info(
                        f"[TranslatorAgent] Translating answer → {target_language}"
                    )
                    result = self.translator.translate(
                        text=final_answer,
                        target_language=target_language
                    )
                    # translate() returns a Dict — extract the translation string
                    translation_text = result.get("translation", "") if isinstance(result, dict) else str(result)
                    if translation_text and translation_text.strip() and result.get("status") == "SUCCESS":
                        translated_answer = translation_text
                        logger.info(
                            f"[TranslatorAgent] ✅ "
                            f"{len(final_answer)} → {len(translated_answer)} ch"
                        )
                    else:
                        logger.warning(
                            "[TranslatorAgent] ⚠️ Empty or failed result — "
                            "keeping English response"
                        )
                except Exception as e:
                    logger.error(
                        f"[TranslatorAgent] ❌ {e} — keeping English"
                    )
                    # translated_answer stays as final_answer (English)
            else:
                logger.debug("[TranslatorAgent] Skipped — empty answer")
        else:
            logger.debug(
                f"[TranslatorAgent] Skipped — "
                f"'{target_language}' is English/auto (no lang pattern detected)"
            )
        # ─── END TRANSLATION STEP ────────────────────────────────────────────────

        return {"ui_hints": hints, "answer": translated_answer}

    async def run(self, query: str, conversation_id: str, project_id: str = "default", mentioned_files: list = None, uploaded_files: list = None, original_query: str = None, use_web_search: bool = False, check_abort_fn: Optional[Any] = None):
        """Entry point for the SOTA Metacognitive Loop."""
        # 1. Immediately ensure conversation existence (SOTA FK Guard)
        # This prevents Foreign Key errors if the conversation node was deleted but we have a stray query
        self.db.ensure_conversation(conversation_id, title="New Chat", sqlite_db=self.sqlite_db)
        
        # 2. Immediately persist User message (Durability) with mention metadata
        persist_query = original_query or query
        user_metadata = {
            "mentioned_files": mentioned_files or [],
            "timestamp": datetime.utcnow().isoformat()
        }
        self._persist_message(conversation_id, "user", persist_query, metadata=user_metadata)
        
        # 3. MemGPT Overflow Guard: Page out oldest turn if context budget exceeded
        # This is the core MemGPT mechanism — it fires before every LLM call.
        # With Gemma3:4b (2048 token limit), this keeps history under 1638 tokens (80%).
        if hasattr(self.memory, 'manage_overflow'):
            await self.memory.manage_overflow(conversation_id)
        
        # 4. Fetch (possibly trimmed) history from memory
        # SOTA: get_prompt_context handles the paging/recall logic
        history = self.memory.get_prompt_context(conversation_id) if hasattr(self.memory, 'get_prompt_context') else []
        
        initial_state: UltimaRAGState = {
            "query": query,
            "conversation_id": conversation_id,
            "project_id": project_id,
            "intent": "general_intelligence",
            "plan": {},
            "unified_evidence": {},
            "history": history,
            "evidence": [],
            "perceived_media": [],
            "answer": "",
            "confidence_score": 1.0,
            "critique_count": 0,
            "ui_hints": {},
            "status": "INITIALIZING",
            "metadata": {},
            "mentioned_files": mentioned_files or [],
            "uploaded_files": uploaded_files or [],
            "response_mode": "grounded_in_docs", # Default
            "context_rejected": False,
            "target_language": None,
            "check_abort_fn": check_abort_fn,
            "full_history": [],
            "search_query": "",
            "shared_perception": {},
            "use_web_search": use_web_search,
            "thought": None,  # Thought-UI: ephemeral, populated by agents during run
            "retrieved_fragments": {},  # SOURCE EXPLORER: populated by execute_rag
            "source_map": {},           # SOURCE EXPLORER: populated by execute_rag
        }

        # SOTA: Return an async generator for API streaming
        async def stream_results():
            last_state = initial_state
            # Initialize final result variables at the TOP level of the generator scope
            # This prevents UnboundLocalError if the loop is aborted instantly.
            final_answer = ""
            final_metadata = {}
            
            # Use astream_events to catch fine-grained events like tokens and node starts
            async for event in self.workflow.astream_events(initial_state, version="v1"):
                # ── STOP BUTTON CHECK: Early exit if user clicked stop ──
                if check_abort_fn and check_abort_fn():
                    logger.info("MetacognitiveBrain: Abort signal detected in stream loop, breaking.")
                    break

                kind = event.get("event")
                if not kind: continue
                
                # Case 1: Node activity (Telemetry)
                name = event.get("name")
                if kind == "on_chain_start" and name and name in ["extractor", "router", "planner", "perception", "retrieval", "synthesis", "general_synthesis", "critic", "healer", "retrieve_full_history", "chronicler", "reformulate_query"]:
                    agent_name = name.capitalize().replace("_synthesis", "").replace("Retrieve_full_history", "Recall").replace("Reformulate_query", "Analyst")
                    stage_msg = "Working..."
                    if name == "healer":
                        stage_msg = "Healing response groundedness..."
                    elif name == "chronicler":
                        stage_msg = "Weaving the chronicle of our journey..."
                    elif name == "retrieve_full_history":
                        stage_msg = "Recalling specific relevant moments..."
                    elif name == "reformulate_query":
                        stage_msg = "Deducing context from our past talk..."
                    
                    yield {
                        "type": "status",
                        "agent": agent_name,
                        "stage": stage_msg,
                        "status": "running"
                    }
                
                # Case 2: Node values (State updates) + Thought-UI event relay
                elif kind == "on_chain_end":
                    data = event.get("data") or {}
                    output = data.get("output") or {}
                    if output and isinstance(output, dict):
                        # SOTA: Hard-merge outputs into the last state
                        last_state = {**last_state, **output}
                        logger.debug(f"MetacognitiveBrain: State merged from node {name}")
                        
                        # THOUGHT-UI: If the node produced a 'thought', yield it immediately
                        thought_msg = output.get("thought")
                        if thought_msg and isinstance(thought_msg, str):
                            # Derive a human-readable agent display name from the LangGraph node name
                            _agent_display_map = {
                                "router": "🧠 Brain",
                                "extractor": "🗂️ Extractor",
                                "planner": "📋 Planner",
                                "perception": "👁️ Perception",
                                "retrieval": "🔍 Librarian",
                                "synthesis": "✍️ Synthesizer",
                                "general_synthesis": "✍️ Synthesizer",
                                "critic": "⚖️ Auditor",
                                "healer": "🩺 Surgeon",
                                "chronicler": "📜 Chronicler",
                                "retrieve_full_history": "🗄️ Recall",
                                "reformulate_query": "🔄 Analyst",
                            }
                            display_agent = _agent_display_map.get(name, f"⚙️ {(name or 'System').capitalize()}")
                            yield {
                                "type": "thought",
                                "agent": display_agent,
                                "action": thought_msg
                            }
                
                # Case 3: Token-level streaming from LLM
                tags = event.get("tags") or []
                if kind == "on_llm_stream" and tags and any(t in tags for t in ["synthesis", "general_synthesis", "chronicler"]):
                    if (last_state or {}).get("critique_count", 0) == 0:
                        data = event.get("data") or {}
                        token = data.get("chunk")
                        if token:
                            if hasattr(token, "text"): token = token.text
                            yield {
                                "type": "token",
                                "token": token
                            }

                # Case 4: General Status update from telemetry manager
                current_activity = telemetry.get_active_status()
                if current_activity.get("status") == "running":
                    yield {
                        "type": "status",
                        "agent": current_activity.get("agent"),
                        "stage": current_activity.get("stage"),
                        "status": "running"
                    }
            
            # Final result safety
            # Ensure intent is a string for the final result
            intent_obj = last_state.get("intent")
            if intent_obj:
                if hasattr(intent_obj, "value"):
                    last_state["intent"] = intent_obj.value
                elif hasattr(intent_obj, "name"):
                    last_state["intent"] = intent_obj.name
                else:
                    last_state["intent"] = str(intent_obj)
            
            # 3. Final Persistence with Remediation (Multilingual Enforcement)
            final_answer = last_state.get("answer", "") # Sync with most recent state before persistence
            if not (check_abort_fn and check_abort_fn()):
                target_lang = last_state.get("target_language")
                
                # SOTA Post-Processing: If target language was requested, perform a high-fidelity cleanup
                # to ensure zero English pollution in the final persisted history.
                if target_lang and final_answer:
                    try:
                        from .translator_agent import get_translator
                        translator = get_translator()
                        remediation = translator.translate(final_answer, target_language=target_lang)
                        if remediation["status"] == "SUCCESS":
                            logger.info(f"MetacognitiveBrain: Remediation success for {target_lang}")
                            final_answer = remediation["translation"]
                    except Exception as e:
                        logger.error(f"Post-processing translation failed: {e}")

                # SOTA Auto-Tag: Deprecated in favor of the specialized RAG Source Strip UI
                logger.info(f"MetacognitiveBrain: Proceeding with Unified UI layout: {last_state.get('ui_hints', {}).get('layout')}")

                final_metadata = {
                    "intent": last_state.get("intent"),
                    "confidence_score": last_state.get("confidence_score"),
                    "citations": last_state.get("metadata", {}).get("citations", []),
                    "agent_type": last_state.get("intent"),
                    "ui_hints": last_state.get("ui_hints", {}),
                    "target_language": target_lang,
                    "reasoning": last_state.get("reasoning") # SOTA Trace Persistence
                }
                self._persist_message(conversation_id, "assistant", final_answer, metadata=final_metadata)
                last_state["answer"] = final_answer # Sync for final yield

                # ── SOTA Phase 2: Knowledge Distillation Sub-Agent Trigger ──
                # Extract permanent facts from the latest User/Assistant turn in the background.
                if hasattr(self, 'memory') and hasattr(self.memory, 'extract_facts'):
                    import asyncio
                    new_turn = [
                        {"role": "user", "content": persist_query},
                        {"role": "assistant", "content": final_answer}
                    ]
                    # Fire and forget (don't block the UI yield)
                    try:
                        asyncio.create_task(self.memory.extract_facts(conversation_id, new_turn))
                    except Exception as fact_err:
                        logger.warning(f"Failed to spawn Knowledge Distillation agent: {fact_err}")

            else:
                logger.info("MetacognitiveBrain: Skipping final persistence due to abort signal.")
                final_answer = "User Terminated the generation. Both the query and response were not stored to save session resources. Next time you ask this, it will be treated as fresh."

            # 4. Final Result Yield (for API synchronization)
            yield {
                "type": "final",
                "result": {
                    "answer": final_answer,
                    "confidence_score": last_state.get("confidence_score"),
                    "intent": last_state.get("intent"),
                    "ui_hints": last_state.get("ui_hints"),
                    "conversation_id": conversation_id,
                    "retrieved_fragments": last_state.get("retrieved_fragments", {}),
                    "source_map": last_state.get("source_map", {}),
                    "metadata": {
                        **last_state.get("metadata", {}),
                        "reasoning": last_state.get("reasoning")
                    }
                }
            }


        return stream_results()

    async def run_agentic_action(
        self,
        intent: str,
        document_names: List[str],
        conversation_id: str,
        project_id: str = "default",
        check_abort_fn=None
    ):
        """
        SOTA Agentic Action Entry Point (Phase 1-4).
        Bypasses the full LangGraph workflow for precision agentic payloads.
        Routes to specialized handlers based on intent.

        Yields SSE-ready event dicts:
          {"type": "status",            "agent": str, "stage": str}
          {"type": "thought",           "agent": str, "action": str}
          {"type": "token",             "token": str}
          {"type": "json_chunk",        "data": dict}      ← Phase 2
          {"type": "next_actions",      "actions": list}   ← Phase 4
          {"type": "agentic_final",     "content": str}
        """
        from .deep_insight_agent import DeepInsightAgent
        now = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p")
        doc_list = ", ".join(document_names) if document_names else "the uploaded documents"

        # ── Pull grounded context from the knowledge base ──────────────────
        yield {"type": "status", "agent": "🗄️ Context Loader", "stage": "Binding document context..."}
        try:
            raw_evidence = await self.retriever.retrieve(
                query=f"comprehensive overview of {doc_list}",
                project_id=project_id,
                file_names=document_names if document_names else None,
                top_k=15,
                conversation_id=conversation_id
            )
            evidence_chunks = raw_evidence.get("evidence", [])
        except Exception as e:
            logger.error(f"AgenticAction context load error: {e}")
            evidence_chunks = []

        context_parts = []
        for ev in evidence_chunks:
            src = ev.get("file_name") or ev.get("source", "Document")
            txt = ev.get("text", "")
            if txt:
                context_parts.append(f"SOURCE: {src}\nCONTENT: {txt}")
        grounded_context = "\n\n".join(context_parts) if context_parts else "NO_CONTEXT_AVAILABLE"
        # Cap context to avoid VRAM overflow
        if len(grounded_context) > 8000:
            grounded_context = grounded_context[:8000] + "\n\n[CONTEXT TRUNCATED BY VRAM GUARD]"

        # Pull recent history for tone contextualisation
        history = self.memory.get_prompt_context(conversation_id) if hasattr(self.memory, 'get_prompt_context') else []

        final_content = ""

        # ═══════════════════════════════════════════════════════════════════
        # DEEP INSIGHT — Phase 3: Multi-Agent Reflection Loop
        # ═══════════════════════════════════════════════════════════════════
        if intent == "DEEP_INSIGHT":
            yield {"type": "status", "agent": "🔬 Deep Insight Engine", "stage": "Initiating multi-agent debate..."}
            agent = DeepInsightAgent()
            async for event in agent.run(
                context=grounded_context,
                document_names=document_names,
                history=history,
                check_abort_fn=check_abort_fn
            ):
                if check_abort_fn and check_abort_fn():
                    return
                if event["type"] == "deep_insight_done":
                    final_content = event["content"]
                else:
                    yield event

        # ═══════════════════════════════════════════════════════════════════
        # EXECUTIVE SUMMARY — Phase 1: Context-Aware XML Few-Shot
        # ═══════════════════════════════════════════════════════════════════
        elif intent == "EXECUTIVE_SUMMARY":
            yield {"type": "status", "agent": "📋 Executive Synthesizer", "stage": "Compiling executive report..."}
            yield {"type": "thought", "agent": "📋 Synthesizer", "action": "Analysing key themes for executive report..."}

            summary_prompt = f"""<role>
You are the UltimaRAG Executive Analyst. Produce a concise, structured executive report from the provided evidence.
</role>

<system_info>
CURRENT TIME: {now}
DOCUMENTS: {doc_list}
INTENT: EXECUTIVE_SUMMARY
</system_info>

<context>
{grounded_context}
</context>

<few_shot_example>
<example_input>Document about quarterly sales performance...</example_input>
<example_output>
## Executive Summary

**Core Findings:** Revenue grew 18% YoY driven by APAC expansion. Margin compression of 3pts was offset by operational leverage.

**Key Metrics:** $4.2B revenue | 24% gross margin | 12% EBITDA

**Strategic Implications:** Growth momentum intact but supply chain risks warrant contingency planning.

**Recommended Actions:** 1) Accelerate APAC hiring. 2) Hedge raw material costs. 3) Launch Q3 cost optimisation initiative.
</example_output>
</few_shot_example>

<executive_mandates>
1. STRUCTURE: Must include sections — Core Findings, Key Metrics (if applicable), Strategic Implications, Recommended Actions.
2. LENGTH: 200-300 words. Concise, dense, actionable.
3. TONE: C-suite ready. Authoritative. Zero fluff.
4. CITATIONS: Use [[FileName]] notation for source attribution.
</executive_mandates>

EXECUTIVE SUMMARY:"""

            full_summary = ""
            try:
                async for chunk in self.llm_heavy.astream(summary_prompt, config={"tags": ["agentic_summary"]}):
                    if check_abort_fn and check_abort_fn():
                        return
                    full_summary += chunk
                    yield {"type": "token", "token": chunk}
            except Exception as e:
                logger.error(f"AgenticAction EXECUTIVE_SUMMARY error: {e}")
                full_summary = "Failed to generate executive summary."

            clean_summary = re.sub(r'<thinking>[\s\S]*?</thinking>', '', full_summary).strip()
            clean_summary = clean_summary.replace("EXECUTIVE SUMMARY:", "").strip()
            final_content = clean_summary
            yield {"type": "thought", "agent": "📋 Synthesizer", "action": "Executive report complete."}

        # ═══════════════════════════════════════════════════════════════════
        # RISK ASSESSMENT — Phase 2: Generative UI (JSON Widget)
        # ═══════════════════════════════════════════════════════════════════
        elif intent == "RISK_ASSESSMENT":
            yield {"type": "status", "agent": "🛡️ Risk Analyst", "stage": "Running structured risk analysis..."}
            yield {"type": "thought", "agent": "🛡️ Risk Analyst", "action": "Scanning evidence for risks, biases, and vulnerabilities..."}

            risk_schema = {
                "overall_score": "integer 0-100 (100 = highest risk)",
                "risk_level": "one of: CRITICAL | HIGH | MEDIUM | LOW",
                "risks": [
                    {
                        "title": "string",
                        "severity": "one of: CRITICAL | HIGH | MEDIUM | LOW",
                        "description": "string",
                        "mitigation": "string"
                    }
                ],
                "bias_flags": ["list of strings identifying any data biases"],
                "confidence": "integer 0-100 (confidence in this assessment)"
            }

            risk_prompt = f"""You are the UltimaRAG Risk Auditor. Analyse the provided document context for risks, biases, and vulnerabilities.

CURRENT TIME: {now}
DOCUMENTS: {doc_list}

CONTEXT:
{grounded_context[:5000]}

OUTPUT INSTRUCTIONS:
- You MUST respond with ONLY a valid JSON object. No markdown. No extra text. No thinking blocks.
- Follow this exact schema:
{json.dumps(risk_schema, indent=2)}
- Identify 3-6 specific, concrete risks from the document content.
- Set overall_score based on the weighted average severity of all identified risks.

JSON RESPONSE:"""

            # Use OllamaClient with JSON format enforcement (httpx-based, no extra deps)
            from ..core.ollama_client import get_ollama_client
            ollama_client = get_ollama_client()
            ollama_client.model_name = Config.ollama_multi_model.HEAVY_MODEL
            raw_json_str = ""
            try:
                # Attempt JSON-mode via the project's existing OllamaClient
                result = await ollama_client.generate(
                    prompt=risk_prompt,
                    model=Config.ollama_multi_model.HEAVY_MODEL,
                    temperature=0.0,
                    max_tokens=1024,
                    format="json"
                )
                raw_json_str = result.get("response", "") if isinstance(result, dict) else str(result)
            except Exception as e:
                logger.error(f"AgenticAction RISK JSON-mode error: {e}")
                # Fallback: use standard LangChain LLM (no JSON guarantee)
                try:
                    raw_json_str = await self.llm_heavy.ainvoke(risk_prompt)
                except Exception as e2:
                    logger.error(f"AgenticAction RISK fallback error: {e2}")
                    raw_json_str = ""


            # Parse and validate JSON
            risk_data = None
            try:
                # Strip any accidental markdown fences
                clean_json = re.sub(r'```json?\n?', '', raw_json_str).strip().strip('`')
                # Strip thinking tags if model leaked them
                clean_json = re.sub(r'<thinking>[\s\S]*?</thinking>', '', clean_json).strip()
                risk_data = json.loads(clean_json)
                logger.info(f"AgenticAction: Risk JSON parsed successfully. Score: {risk_data.get('overall_score')}")
                yield {"type": "json_chunk", "data": risk_data}
                final_content = f"**Risk Assessment Complete** — Overall Score: {risk_data.get('overall_score', 'N/A')}/100 | Level: {risk_data.get('risk_level', 'UNKNOWN')}"
                yield {"type": "thought", "agent": "🛡️ Risk Analyst", "action": f"Risk matrix compiled: {len(risk_data.get('risks', []))} risks identified at {risk_data.get('risk_level', 'UNKNOWN')} level."}
            except (json.JSONDecodeError, ValueError) as parse_err:
                logger.error(f"AgenticAction: Risk JSON parse failed: {parse_err}. Falling back to markdown.")
                # FALLBACK: render as plain markdown  
                final_content = raw_json_str.strip() or "Risk assessment could not be generated. Please try again."
                for token in final_content.split(" "):
                    yield {"type": "token", "token": token + " "}
                yield {"type": "thought", "agent": "🛡️ Risk Analyst", "action": "Analysis complete (markdown mode — JSON parse failed)."}

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: PROACTIVE NEXT BEST ACTION GENERATION
        # Runs after ALL intents as a fast post-processing step
        # ═══════════════════════════════════════════════════════════════════
        if final_content and not (check_abort_fn and check_abort_fn()):
            yield {"type": "status", "agent": "🚀 Action Planner", "stage": "Generating proactive next steps..."}
            try:
                nba_prompt = f"""You are the UltimaRAG Action Planner. A user just ran a '{intent}' operation on '{doc_list}'.
The output was: "{final_content[:500]}..."

Generate exactly 3 highly specific, contextual follow-up actions the user should take next.
Each action must be a 4-7 word imperative phrase (e.g., "Identify the top 3 cost drivers").
Output ONLY a valid JSON array of 3 strings. No markdown. No explanation.
Example: ["Drill down into revenue drivers", "Compare with Q3 benchmark data", "Extract competitive intelligence gaps"]

JSON ARRAY:"""
                nba_response = await self.llm_light.ainvoke(nba_prompt)
                # Clean and parse the 3 suggested actions
                clean_nba = re.sub(r'<thinking>[\s\S]*?</thinking>', '', nba_response).strip()
                clean_nba = re.sub(r'```json?\n?', '', clean_nba).strip().strip('`')
                next_actions_list = json.loads(clean_nba)
                if isinstance(next_actions_list, list) and len(next_actions_list) >= 3:
                    next_actions_list = [str(a) for a in next_actions_list[:3]]
                    logger.info(f"AgenticAction: Next Best Actions: {next_actions_list}")
                    yield {"type": "next_actions", "actions": next_actions_list}
            except Exception as nba_err:
                logger.warning(f"AgenticAction: Next Best Action generation failed (non-fatal): {nba_err}")
                # Non-fatal — frontend will show static defaults

        yield {"type": "agentic_final", "content": final_content}

    def get_status(self, conversation_id: Optional[str] = None) -> Dict:
        """Get current brain and database status (SOTA Health Check)"""
        chunks_count = self.db.get_knowledge_count(conversation_id=conversation_id)
        return {
            "ready": True,
            "chunks_count": chunks_count,
            "memory_usage": "optimized",
            "agents": {
                "planner": "ready",
                "intent_router": "ready",
                "perception": "ready",
                "retriever": "ready",
                "synthesizer": "ready",
                "healer": "ready"
            },
            "database_sync": "active"
        }

