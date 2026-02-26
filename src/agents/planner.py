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

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from ..core.utils import logger
from ..core.models import Intent, UnifiedEvidenceState
from langchain_ollama import OllamaLLM as Ollama

class TaskStep(BaseModel):
    step_id: int
    agent: str # perception, rag, general, transformation
    description: str
    dependencies: List[int] = []
    status: str = "pending"

class ExecutionPlan(BaseModel):
    query: str
    steps: List[TaskStep]
    unified_evidence: Optional[UnifiedEvidenceState] = None

class MultiStagePlanner:
    """
    SOTA Multi-Stage Execution Planner.
    Decomposes compound queries into a logical DAG of tasks.
    """
    
    def __init__(self):
        from ..core.config import Config
        self.llm = Ollama(
            model=Config.ollama_multi_model.AGENT_MODELS.get("planner", Config.ollama_multi_model.HEAVY_MODEL),
            base_url=Config.ollama.BASE_URL,
            timeout=Config.ollama.TIMEOUT
        )
        logger.info("Multi-Stage Execution Planner (UltimaRAG Edition) initialized")

    async def create_plan(self, query: str, intent: Intent, evidence: UnifiedEvidenceState) -> ExecutionPlan:
        """
        Creates a sequential execution plan based on query and intent.
        """
        logger.info(f"Creating execution plan for: {query} (Intent: {intent})")
        
        # Simple logic for now: 
        # If Intent is PERCEPTION -> Step 1: Perception
        # If Intent is RAG -> Step 1: Retrieval
        # If Intent is MULTI_TASK -> Decompose with LLM
        
        if intent != Intent.MULTI_TASK:
            step = TaskStep(
                step_id=1,
                agent="perception" if intent == Intent.PERCEPTION else ("rag" if intent == Intent.RAG else "general"),
                description=f"Direct processing for {intent}"
            )
            return ExecutionPlan(query=query, steps=[step], unified_evidence=evidence)
            
        # For MULTI_TASK, we use Gemma-3-12B to decompose
        prompt = f"""
<role>
You are the UltimaRAG Strategic Architect. Decompose complex user requests into a high-fidelity execution DAG with integrated quality gates.
</role>

<user_input>
"{query}"
</user_input>

<agent_arsenal>
- PERCEPTION: Vision/Audio analysis (OCR, scene description, transcriptions).
- RAG: Precision document retrieval and grounding.
- TRANSFORMATION: Language translation, summarization, or synthesis.
- VERIFICATION: Fact-checking and grounding audit for complex answers.
- GENERAL: Baseline intelligence for non-grounded tasks.
</agent_arsenal>

<planning_directives>
1. STRATEGY: Sequential DAG where dependencies are clear.
2. QUALITY GATE: For complex multi-stage tasks, ALWAYS include a "VERIFICATION" step at the end.
3. ATOMICITY: Each step focuses on a single operation.
</planning_directives>

<output_requirement>
JSON list of steps: step_id (int), agent (string), description (string), dependencies (list of ints).
</output_requirement>

STRATEGIC PLAN:"""
        
        try:
            # SOTA: Dynamic LLM-driven planning
            response_raw = await self.llm.ainvoke(prompt)
            result_text = response_raw.strip()
            
            # Clean JSON
            import re
            if "```" in result_text:
                result_text = re.sub(r'```(?:json)?\n?', '', result_text)
                result_text = re.sub(r'\n?```$', '', result_text)
            
            json_match = re.search(r'\[[\s\S]*\]', result_text)
            if json_match:
                import json
                steps_data = json.loads(json_match.group())
                steps = [TaskStep(**s) for s in steps_data]
            else:
                # Fallback to simulated decomposition
                if "describe" in query.lower() and "translate" in query.lower():
                    steps = [
                        TaskStep(step_id=1, agent="perception", description="Describe the visual content"),
                        TaskStep(step_id=2, agent="transformation", description="Translate description to Hindi", dependencies=[1])
                    ]
                else:
                    steps = [TaskStep(step_id=1, agent="general", description="Process complex query")]
                
            return ExecutionPlan(query=query, steps=steps, unified_evidence=evidence)
        except Exception as e:
            logger.error(f"Planning Error: {e}")
            return ExecutionPlan(query=query, steps=[TaskStep(step_id=1, agent="general", description="Fallback step")], unified_evidence=evidence)

