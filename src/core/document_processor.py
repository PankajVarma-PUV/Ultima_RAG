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

import fitz
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from PIL import Image
from .utils import logger
from .file_manager import ensure_chat_dir
from ..vision.qwen_agent import get_vision_agent

class DocumentProcessor:
    """SOTA Document Processor for UltimaRAG. Extracts text and images from PDFs."""
    
    @staticmethod
    async def extract_from_pdf(conversation_id: str, file_path: str, file_name: str, check_abort_fn: Optional[Callable] = None) -> Tuple[str, List[Dict]]:
        """
        Extract native text and perform OCR on embedded images from a PDF asynchronously.
        SOTA: Returns image metadata to enable pipeline reuse and eliminate redundant VLM hits.
        
        Returns:
            Tuple of (full_text, list_of_image_dicts[{path, ocr_text}])
        """
        logger.info(f"📄 Extracting content from PDF: {file_name}")
        full_text = ""
        image_metadata = [] 
        
        try:
            doc = await asyncio.to_thread(fitz.open, file_path)
            # SOTA: Enable Image Extraction for Multimodal Enrichment
            temp_img_dir = ensure_chat_dir(conversation_id) / "extracted_images"
            temp_img_dir.mkdir(exist_ok=True)
            
            processed_xrefs = set()
            vision_agent = get_vision_agent()
            
            for page_num in range(len(doc)):
                # ── STRICT ABORT CHECK: check between pages ──
                if check_abort_fn and check_abort_fn():
                    logger.info(f"Abort signaled during PDF extraction of {file_name} at page {page_num+1}. Terminating.")
                    break
                    
                page = doc[page_num]
                
                # 1. Extract Native Text
                page_text = page.get_text()
                full_text += f"\n\n[Page {page_num + 1}]\n"
                
                # Check if page has significant text (vs a scanned page image)
                if len(page_text.strip()) < 50:
                    # Scenario A: Mostly empty text -> likely a scanned page.
                    # Render the entire page to image and OCR it using Qwen2-VL.
                    try:
                        logger.info(f"Page {page_num + 1} looks like a scanned page. Performing full-page SOTA Vision extraction...")
                        zoom_matrix = fitz.Matrix(2, 2)
                        pix = page.get_pixmap(matrix=zoom_matrix)
                        
                        # Convert to PIL Image for Vision Agent
                        mode = "RGBA" if pix.alpha else "RGB"
                        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                        if mode == "RGBA":
                            img = img.convert("RGB")
                            
                        # SOTA: Use a combined prompt to get text and context in ONE hit
                        combined_prompt = "Extract all text exactly. Also, provide a 1-sentence description of the visual layout."
                        ocr_res = await vision_agent.describe_image(img, prompt=combined_prompt)
                        
                        if ocr_res and not ocr_res.startswith("Error"):
                            full_text += f"\n[OCR Extracted Text from Scanned Page]:\n{ocr_res}\n"
                            
                        # SOTA: Save this scanned page as an image for enrichment
                        if len(image_metadata) < 10: 
                            img_filename = f"{file_name}_page_{page_num+1}.jpg"
                            img_path = temp_img_dir / img_filename
                            img.save(img_path, "JPEG", quality=85)
                            image_metadata.append({
                                "path": str(img_path),
                                "ocr_text": ocr_res
                            })
                            
                    except Exception as ocr_e:
                        logger.warning(f"Full-page Vision extraction failed on page {page_num + 1}: {ocr_e}")
                else:
                    full_text += page_text
                    
                    # Scenario B: Extract native text + handle embedded images
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        # ── STRICT ABORT CHECK: check between images ──
                        if check_abort_fn and check_abort_fn(): break

                        if len(image_metadata) >= 10: break # Guard
                        
                        xref = img_info[0]
                        if xref in processed_xrefs:
                            continue
                            
                        try:
                            # Use PyMuPDF's Pixmap
                            pix = fitz.Pixmap(doc, xref)
                            width, height = pix.width, pix.height
                            
                            # Guard: Filter out tiny images
                            if width < 150 or height < 150:
                                processed_xrefs.add(xref)
                                pix = None
                                continue
                                
                            # Format conversion
                            if pix.n - pix.alpha >= 4:
                                rgb_pix = fitz.Pixmap(fitz.csRGB, pix)
                                mode = "RGBA" if rgb_pix.alpha else "RGB"
                                img = Image.frombytes(mode, [rgb_pix.width, rgb_pix.height], rgb_pix.samples)
                                rgb_pix = None
                            else:
                                mode = "RGBA" if pix.alpha else "RGB"
                                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                                
                            if mode == "RGBA":
                                img = img.convert("RGB")
                            
                            pix = None 
                            
                            # Perform Combined OCR/Vision hit
                            logger.info(f"Page {page_num + 1}: Extracting perception from embedded image...")
                            combined_prompt = "Extract all text exactly. Also, provide a 1-sentence description of the visual scene."
                            ocr_res = await vision_agent.describe_image(img, prompt=combined_prompt)
                            
                            # Save image for multimodal processing
                            img_filename = f"{file_name}_img_{xref}.jpg"
                            img_path = temp_img_dir / img_filename
                            img.save(img_path, "JPEG", quality=85)
                            
                            image_metadata.append({
                                "path": str(img_path),
                                "ocr_text": ocr_res if ocr_res and not ocr_res.startswith("Error") else "No text found."
                            })
                            
                            if ocr_res and not ocr_res.startswith("Error"):
                                full_text += f"\n[Image Perception]: {ocr_res}\n"
                        except Exception as ocr_e:
                            logger.warning(f"Vision extraction failed for an image on page {page_num + 1}: {ocr_e}")
                            
                        processed_xrefs.add(xref)
                    
            doc.close()
            logger.info(f"✅ Extracted {len(full_text)} chars and {len(image_metadata)} images from {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_name}: {e}")
            
        return full_text, image_metadata

