"""
Ollama Client for Ultima_RAG
HTTP client for local Ollama API with automatic token management.
"""

import requests
import json
import time
from typing import Dict, List, Optional, Any, Union, Generator

from .config import OllamaConfig
from .utils import logger

# Singleton instance
_client_instance = None


# =============================================================================
# OLLAMA CLIENT
# =============================================================================

class OllamaClient:
    """
    HTTP client for Ollama local API.
    
    Features:
    - No token compression (Gemma3:12b handles large inputs)
    - Streaming and non-streaming generation
    - Connection health checks
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
        self.timeout = timeout or OllamaConfig.TIMEOUT
        
        # Verify connection - Only log if this is the first initialization
        global _client_instance
        if _client_instance is None:
            if not self.is_available():
                logger.warning(f"Ollama not available at {self.base_url}")
            else:
                logger.info(f"OllamaClient initialized: {self.model_name} @ {self.base_url}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                return self.model_name.split(":")[0] in model_names or len(models) > 0
            return False
        except Exception as e:
            logger.debug(f"Ollama availability check failed: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        stream: bool = False,
        repeat_penalty: float = 1.1,
        model: Optional[str] = None
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text using Ollama.
        """
        
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
        
        try:
            if stream:
                return self._generate_stream(payload)
            else:
                return self._generate_sync(payload)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running: `ollama serve`"
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def _generate_sync(self, payload: Dict) -> Dict:
        """Synchronous generation - returns {'response': str, 'done_reason': str}"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return {
            "response": result.get("response", ""),
            "done_reason": result.get("done_reason", "stop")
        }
    
    def _generate_stream(self, payload: Dict) -> Dict:
        """Streaming generation - returns {'response': str, 'done_reason': str}"""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.timeout
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code}")
        
        full_response = ""
        done_reason = "stop"
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    full_response += chunk
                    if data.get("done", False):
                        done_reason = data.get("done_reason", "stop")
                        break
                except json.JSONDecodeError:
                    continue
        
        return {
            "response": full_response,
            "done_reason": done_reason
        }
    
    def chat(
        self,
        messages: list,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Chat completion format (for future use).
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            temperature: Sampling temperature
            max_tokens: Max output tokens
        
        Returns:
            Assistant response text
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
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama chat API error: {response.status_code}")
            
            result = response.json()
            return {
                "message": result.get("message", {}).get("content", ""),
                "done_reason": result.get("done_reason", "stop")
            }
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please ensure Ollama is running."
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def get_ollama_client() -> OllamaClient:
    """Get or create a singleton OllamaClient instance"""
    global _client_instance
    if _client_instance is None:
        _client_instance = OllamaClient()
    return _client_instance


def generate_with_ollama(
    prompt: str,
    temperature: float = None,
    max_tokens: int = None
) -> str:
    """Convenience function to generate with Ollama (Text-only)"""
    client = get_ollama_client()
    return client.generate(prompt, temperature=temperature, max_tokens=max_tokens)

