"""
Vision Perception Agent for Ultima_RAG.
Handles Qwen2-VL-2B inference via HuggingFace Transformers.
Optimized for 6GB VRAM using 4-bit quantization and CPU offloading.
"""

import os
import torch
from typing import List, Dict, Optional
from PIL import Image
from ..core.utils import logger

class QwenVisionAgent:
    """Agent 3: Vision Perception Agent (Qwen2-VL-2B)"""
    
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _check_gpu_health(self):
        """Forensic GPU diagnostic for Ultima_RAG Elite."""
        logger.info("🕵️ GPU Diagnostic: Running health check...")
        try:
            torch_ver = torch.__version__
            cuda_avail = torch.cuda.is_available()
            logger.info(f"  - PyTorch Version: {torch_ver}")
            logger.info(f"  - CUDA Available: {cuda_avail}")
            
            if cuda_avail:
                logger.info(f"  - Device Count: {torch.cuda.device_count()}")
                logger.info(f"  - Device Name: {torch.cuda.get_device_name(0)}")
                logger.info(f"  - CUDA Version: {torch.version.cuda}")
                # Log VRAM status
                vram_free, vram_total = torch.cuda.mem_get_info()
                logger.info(f"  - VRAM: {vram_free/1024**3:.1f}GB free / {vram_total/1024**3:.1f}GB total")
            else:
                logger.warning("  - [CAUTION] CUDA not detected by PyTorch.")
                # Look for common issues: environment variables
                vis_dev = os.environ.get("CUDA_VISIBLE_DEVICES")
                if vis_dev:
                    logger.warning(f"  - CUDA_VISIBLE_DEVICES is set to: {vis_dev}")
        except Exception as e:
            logger.error(f"  - GPU Diagnostic failed: {e}")

    def _lazy_load(self):
        """Load model with adaptive configuration (4-bit for GPU, standard for CPU)."""
        if self.model is not None:
            return

        self._check_gpu_health()
        
        logger.info(f"Loading Qwen2-VL-2B from HuggingFace ({self.model_id})...")
        try:
            import transformers
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            import qwen_vl_utils
            import accelerate
            
            logger.info("Found all vision dependencies: transformers, qwen_vl_utils, accelerate.")

            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            # Logic: Use quantization ONLY if GPU is available
            kwargs = {
                "pretrained_model_name_or_path": self.model_id,
                "low_cpu_mem_usage": True,
            }

            # Check if user wants to force GPU (can be useful if basic check fails but path exists)
            force_gpu = os.getenv("Ultima_FORCE_GPU", "false").lower() == "true"
            use_cuda = torch.cuda.is_available() or force_gpu

            if use_cuda:
                from transformers import BitsAndBytesConfig
                # 4-bit quantization config for 6GB VRAM
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["quantization_config"] = bnb_config
                kwargs["device_map"] = "auto"
                # Dynamic max memory based on actual VRAM
                try:
                    vram_free, _ = torch.cuda.mem_get_info()
                    # Leave 1.5GB for system/overhead, cap model at remaining
                    safe_vram = f"{max(2.0, (vram_free/1024**3) - 1.5):.1f}GiB"
                    kwargs["max_memory"] = {0: safe_vram, "cpu": "24GiB"}
                except:
                    kwargs["max_memory"] = {0: "4GiB", "cpu": "24GiB"}
                
                logger.info(f"GPU Mode Enabled: Using 4-bit quantization (Cap: {kwargs.get('max_memory', {}).get(0, '4GiB')}).")
                self.device = "cuda"
            else:
                # CPU fallback: No quantization (bitsandbytes doesn't support CPU)
                kwargs["device_map"] = "cpu"
                kwargs["torch_dtype"] = torch.float32 
                logger.info("CPU Mode Enabled: Falling back to Float32 (No GPU).")
                self.device = "cpu"

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(**kwargs)
            
            # Final device verification
            actual_device = next(self.model.parameters()).device
            logger.info(f"Qwen2-VL-2B loaded successfully on {actual_device}.")
            self.device = str(actual_device)
        except ImportError as e:
            logger.error(f"Missing dependency for Qwen2-VL: {e}. Please run 'pip install qwen-vl-utils accelerate bitsandbytes'")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model ({self.model_id}): {e}")
            if "accelerator device" in str(e).lower():
                logger.error("HINT: This usually happens when bitsandbytes quantization is attempted on a CPU-only machine.")
            raise
        except ImportError as e:
            logger.error(f"Missing dependency for Qwen2-VL: {e}. Please run 'pip install qwen-vl-utils accelerate bitsandbytes'")
            raise
        except Exception as e:
            logger.error(f"Failed to load Qwen2-VL model ({self.model_id}): {e}")
            if "accelerator device" in str(e).lower():
                logger.error("HINT: This usually happens when bitsandbytes quantization is attempted on a CPU-only machine.")
            raise

    async def describe_image(self, pil_image: Image.Image, prompt: str = None) -> str:
        """Generate a description for an image."""
        self._lazy_load()
        
        if prompt is None:
            prompt = (
                "Describe this image in great detail. Focus on the main subject, "
                "colors, textures, and any visible text."
            )

        import tempfile
        import os
        from qwen_vl_utils import process_vision_info
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.convert("RGB").save(tmp_path, "JPEG", quality=95)

        try:
            # Clear cache before heavy work
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": f"file://{tmp_path}",
                            "min_pixels": 256 * 28 * 28,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            
            # Memory safe processor call
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Robust device placement
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            # Inference with memory-conscious settings
            with torch.no_grad():
                logger.info(f"Qwen2-VL: Running inference on {device}...")
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512, # Sufficient for description
                    do_sample=False,
                    use_cache=True
                )
                logger.info(f"Qwen2-VL: Generation complete.")
                
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Qwen2-VL: Final output: {output_text[:100]}...")
            
            return output_text.strip()
            
        except Exception as e:
            logger.error(f"Qwen2-VL inference failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Error during vision perception: {str(e)}"
        finally:
            if os.path.exists(tmp_path):
                try: os.remove(tmp_path)
                except: pass

# Singleton instance for the pipeline
_vision_agent = None

def get_vision_agent() -> QwenVisionAgent:
    global _vision_agent
    if _vision_agent is None:
        _vision_agent = QwenVisionAgent()
    return _vision_agent

