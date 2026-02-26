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
from typing import Dict, List, Tuple
from PIL import Image
from .utils import logger
from .file_manager import ensure_chat_dir
from ..vision.qwen_agent import get_vision_agent

class DocumentProcessor:
    """SOTA Document Processor for UltimaRAG. Extracts text and images from PDFs."""
    
    @staticmethod
    async def extract_from_pdf(conversation_id: str, file_path: str, file_name: str) -> Tuple[str, List[str]]:
        """
        Extract native text and perform OCR on embedded images from a PDF asynchronously.
        Images are NOT saved to disk. Uses Qwen2-VL to guarantee 100% extraction irrespective of language.
        
        Returns:
            Tuple of (full_text, list_of_image_paths_which_is_empty)
        """
        logger.info(f"📄 Extracting content from PDF: {file_name}")
        full_text = ""
        image_paths = [] # Kept empty as per requirements
        
        try:
            doc = await asyncio.to_thread(fitz.open, file_path)
            processed_xrefs = set()
            vision_agent = get_vision_agent()
            
            for page_num in range(len(doc)):
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
                            
                        ocr_text = await vision_agent.describe_image(img, prompt="Extract all text from this image exactly as written. Return ONLY the text, regardless of the language.")
                        if ocr_text and not ocr_text.startswith("Error"):
                            full_text += f"\n[OCR Extracted Text from Scanned Page]:\n{ocr_text}\n"
                    except Exception as ocr_e:
                        logger.warning(f"Full-page Vision extraction failed on page {page_num + 1}: {ocr_e}")
                else:
                    full_text += page_text
                    
                    # Scenario B: Extract native text + handle embedded images
                    image_list = page.get_images(full=True)
                    for img_index, img_info in enumerate(image_list):
                        xref = img_info[0]
                        if xref in processed_xrefs:
                            continue
                            
                        try:
                            # Use PyMuPDF's Pixmap to handle various encodings safely (JBIG2, JPX, etc.)
                            pix = fitz.Pixmap(doc, xref)
                            width, height = pix.width, pix.height
                            
                            # Guard: Filter out tiny images (icons, spacers, bullets)
                            if width < 100 or height < 100:
                                processed_xrefs.add(xref)
                                pix = None
                                continue
                                
                            # Convert CMYK or other formats to RGB
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
                                
                            pix = None # free resources
                            
                            logger.info(f"Page {page_num + 1}: Extracting text from embedded image...")
                            ocr_text = await vision_agent.describe_image(img, prompt="Extract all text from this image exactly as written. Return ONLY the text, regardless of the language.")
                            if ocr_text and not ocr_text.startswith("Error"):
                                full_text += f"\n[OCR Extracted Text from Embedded Image]: {ocr_text}\n"
                        except Exception as ocr_e:
                            logger.warning(f"Vision extraction failed for an image on page {page_num + 1}: {ocr_e}")
                            
                        processed_xrefs.add(xref)
                    
            doc.close()
            logger.info(f"✅ Extracted {len(full_text)} chars of text from {file_name} (Includes SOTA Vision Extraction)")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_name}: {e}")
            
        return full_text, image_paths

