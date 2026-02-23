"""
RETIRED MODULE — AgentOrchestrator (Legacy v1 Pipeline)
========================================================
STATUS: ❌ DEPRECATED — DO NOT USE

This module was the original v1 sequential RAG pipeline.
It has been completely superseded by the LangGraph-based MetacognitiveBrain
located at: src/agents/metacognitive_brain.py

The MetacognitiveBrain provides:
  - LangGraph StateGraph for parallel/conditional node execution
  - Adaptive routing (grounded_in_docs / internal_llm_weights)
  - Web-Breakout Agent fallback (DuckDuckGo + Trafilatura)
  - Hallucination Healer node
  - Multi-modal evidence fusion
  - Streaming token output

This file is kept for git history preservation only.
Any accidental import will raise a clear error at runtime.
"""

import warnings


class AgentOrchestrator:
    """
    RETIRED: Legacy v1 sequential pipeline.
    Use MetacognitiveBrain from src.agents.metacognitive_brain instead.
    """

    def __init__(self, *args, **kwargs):
        raise DeprecationWarning(
            "[Ultima_RAG] AgentOrchestrator is retired and must not be used. "
            "Use MetacognitiveBrain from src.agents.metacognitive_brain instead."
        )

    async def process_query(self, *args, **kwargs):
        raise DeprecationWarning("AgentOrchestrator is retired.")


# Emit a warning if this module is ever imported at the top level
warnings.warn(
    "[Ultima_RAG] orchestrator.py is a retired legacy module. "
    "The active pipeline is MetacognitiveBrain (metacognitive_brain.py).",
    DeprecationWarning,
    stacklevel=2
)

