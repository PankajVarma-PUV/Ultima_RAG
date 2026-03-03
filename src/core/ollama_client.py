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
Ollama Client for UltimaRAG
Asynchronous HTTP client for local Ollama API with automatic token management
and strict Pydantic Output Guards.
"""

import json
import time
import httpx
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, AsyncGenerator, Callable

from .config import OllamaConfig, Config
from .utils import logger

# Singleton instance
_client_instance = None


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """
    Asynchronous HTTP client for Ollama local API.
    
    Features:
    - High-performance Async concurrency using httpx
    - Streaming and non-streaming generation
    - Pydantic Output Guards for strict JSON mode
    """
    
    def __init__(
        self,
        base_url: str = None,
        model_name: str = None,
        max_input_tokens: int = None,
        timeout: int = None
    ):
        """
        Initialize Ollama client.
        """
        self.base_url = base_url or OllamaConfig.BASE_URL
        self.model_name = model_name or OllamaConfig.MODEL_NAME
        # Use a generous timeout for massive 32k context reasoning
        self.timeout = httpx.Timeout(timeout or OllamaConfig.TIMEOUT, connect=5.0)
        
    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "").split(":")[0] for m in models]
                    return self.model_name.split(":")[0] in model_names or len(models) > 0
                return False
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False
            
    async def ensure_connection(self):
        """Warm up method called during app lifespan"""
        global _client_instance
        if _client_instance is None:
            if not await self.is_available():
                logger.warning(f"Ollama not available at {self.base_url}")
            else:
                logger.info(f"OllamaClient initialized: {self.model_name} @ {self.base_url}")
                _client_instance = self
    
    async def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stream: bool = False,
        repeat_penalty: float = 1.1,
        model: Optional[str] = None,
        response_format: Optional[str] = None,
        system: Optional[str] = None,
        format: Optional[str] = None,
        check_abort_fn: Optional[Callable] = None
    ) -> Union[str, AsyncGenerator[Dict[str, Any], None]]:
        """
        Generate text asynchronously using Ollama.
        """
        if check_abort_fn and check_abort_fn():
            logger.info("Abort signaled before Ollama generation. Skipping.")
            if stream:
                async def empty_gen(): yield {"done": True, "response": "Aborted"}
                return empty_gen()
            return {"response": "Aborted", "done": True}

        
        # Build request
        payload = {
            "model": model or self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "repeat_penalty": repeat_penalty
            }
        }
        
        if system is not None:
            payload["system"] = system
            
        final_format = format or response_format
        if final_format is not None:
            payload["format"] = final_format
            
        if images is not None:
            payload["images"] = images
        
        # SOTA Phase 1 Remediation: We DO NOT force `format="json"` natively via Ollama API.
        # DeepSeek R1 and similar reasoning models require the freedom to emit <think> tags.
        # Natively forcing JSON breaks their grammar constraint engine.
        # JSON parsing is handled defensively at the agent layer.
        
        try:
            if stream:
                return self._generate_stream(payload, check_abort_fn=check_abort_fn)
            else:
                return await self._generate_sync(payload, fallback_model=Config.ollama_multi_model.LIGHTWEIGHT_MODEL, check_abort_fn=check_abort_fn)
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            # SOTA Phase 4: Dynamic Model Routing & Fallback
            if not stream:
                logger.warning(f"🚨 SOTA Self-Healing: Primary model '{model or self.model_name}' failed ({e}). Re-routing to lightweight fallback...")
                try:
                    fallback = Config.ollama_multi_model.LIGHTWEIGHT_MODEL
                    payload["model"] = fallback
                    result = await self._generate_sync(payload)
                    logger.info(f"✅ SOTA Self-Healing: Fallback to '{fallback}' succeeded.")
                    return result
                except Exception as fallback_error:
                    logger.error(f"❌ SOTA Self-Healing: Fallback model also failed: {fallback_error}")
                    raise RuntimeError(f"Ollama generation and fallback failed: {e}")
            else:
                logger.error(f"Ollama stream generation failed (no fallback possible): {e}")
                raise RuntimeError(f"Ollama generation failed: {e}")
    
    async def _generate_sync(self, payload: Dict, fallback_model: str = None, check_abort_fn: Optional[callable] = None) -> Dict:
        """Asynchronous generation (non-streaming) - returns {'response': str, 'done_reason': str}"""
        if check_abort_fn and check_abort_fn(): return {"response": "Aborted", "done": True}
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
            
            result = response.json()
            return {
                "response": result.get("response", ""),
                "done_reason": result.get("done_reason", "stop")
            }
    
    async def _generate_stream(self, payload: Dict, check_abort_fn: Optional[callable] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Asynchronous streaming generation - yields JSON dicts"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise RuntimeError(f"Ollama API error {response.status_code}: {error_text.decode('utf-8')}")
                
                async for line in response.aiter_lines():
                    # ── STRICT ABORT CHECK: Terminate stream immediately ──
                    if check_abort_fn and check_abort_fn():
                        logger.info("Abort signaled during Ollama stream. Terminating.")
                        yield {"done": True, "response": "Aborted"}
                        return

                    if line:
                        try:
                            # Ollama streams JSON line-by-line
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
    
    async def chat(
        self,
        messages: list,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict:
        """
        Chat completion format.
        """
        temperature = temperature if temperature is not None else OllamaConfig.TEMPERATURE
        max_tokens = max_tokens or OllamaConfig.MAX_OUTPUT_TOKENS
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload
                )
                
                if response.status_code != 200:
                    raise RuntimeError(f"Ollama chat API error: {response.status_code}")
                
                result = response.json()
                return {
                    "message": result.get("message", {}).get("content", ""),
                    "done_reason": result.get("done_reason", "stop")
                }
                
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running."
            )
        except Exception as e:
            # SOTA Phase 4: Dynamic Model Routing & Fallback for Chat
            logger.warning(f"🚨 SOTA Self-Healing: Primary chat model '{self.model_name}' failed ({e}). Re-routing to fallback...")
            try:
                fallback = Config.ollama_multi_model.LIGHTWEIGHT_MODEL
                payload["model"] = fallback
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/api/chat",
                        json=payload
                    )
                    if response.status_code != 200:
                        raise RuntimeError(f"Ollama fallback chat API error: {response.status_code}")
                    result = response.json()
                    logger.info(f"✅ SOTA Self-Healing: Chat Fallback to '{fallback}' succeeded.")
                    return {
                        "message": result.get("message", {}).get("content", ""),
                        "done_reason": result.get("done_reason", "stop")
                    }
            except Exception as fallback_error:
                logger.error(f"❌ SOTA Self-Healing: Chat Fallback model also failed: {fallback_error}")
                raise RuntimeError(f"Ollama chat generation and fallback failed: {e}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_ollama_client() -> OllamaClient:
    """Get or create a singleton OllamaClient instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = OllamaClient()
    return _client_instance

# We keep this as synchronous for legacy code compatibility, 
# but warn that agents should use the actual await client.generate() instead.
def generate_with_ollama(
    prompt: str,
    temperature: float = None,
    max_tokens: int = None
) -> str:
    """
    WARNING: Legacy synchronous wrapper. 
    New agent code MUST use: await get_ollama_client().generate(...)
    """
    import asyncio
    client = get_ollama_client()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            logger.warning("Synchronous generate_with_ollama called from running event loop. Use await client.generate() instead.")
            return loop.create_task(client.generate(prompt, temperature=temperature, max_tokens=max_tokens))
    except RuntimeError:
        pass
    
    return asyncio.run(client.generate(prompt, temperature=temperature, max_tokens=max_tokens))
