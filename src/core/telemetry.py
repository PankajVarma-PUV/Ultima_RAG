"""
Ultima_RAG Telemetry Utility
Captures agentic activity and performance metrics for real-time UI updates.
"""
from typing import Dict, Any, Optional
import time
import json
from .utils import logger

class AgentTelemetry:
    """Captured state of an agentic step"""
    def __init__(self, agent_name: str, stage: str):
        self.agent_name = agent_name
        self.stage = stage
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

    def finish(self, metadata: Optional[Dict[str, Any]] = None):
        self.end_time = time.time()
        if metadata:
            self.metadata.update(metadata)

    def to_dict(self) -> Dict[str, Any]:
        duration = (self.end_time - self.start_time) if self.end_time else (time.time() - self.start_time)
        return {
            "agent": self.agent_name,
            "stage": self.stage,
            "duration": round(duration, 3),
            "status": "completed" if self.end_time else "running",
            "metadata": self.metadata
        }

class TelemetryManager:
    """Manages recording and broadcasting of agent activities"""
    def __init__(self):
        self.activities: Dict[str, AgentTelemetry] = {}

    def start_activity(self, agent_name: str, stage: str) -> str:
        # Auto-prune old running activities to prevent HUD ghosting
        dead_threshold = time.time() - 60 # 60 seconds max longevity
        self.activities = {k: v for k, v in self.activities.items() if v.end_time or (time.time() - v.start_time) < 60}
        
        activity_id = f"{agent_name}_{int(time.time() * 1000)}"
        self.activities[activity_id] = AgentTelemetry(agent_name, stage)
        logger.info(f"📊 Telemetry Start: {agent_name} -> {stage}")
        return activity_id

    def clear_all(self):
        """Emergency reset for telemetry state"""
        self.activities = {}
        logger.info("📊 Telemetry: State cleared")

    def end_activity(self, activity_id: str, metadata: Optional[Dict[str, Any]] = None):
        if activity_id in self.activities:
            self.activities[activity_id].finish(metadata)
            logger.info(f"📊 Telemetry End: {self.activities[activity_id].agent_name}")

    def get_active_status(self) -> Dict[str, Any]:
        """Returns the most recent active activity"""
        running = [a.to_dict() for a in self.activities.values() if a.end_time is None]
        return running[-1] if running else {"status": "idle"}

# Global instance
telemetry = TelemetryManager()

