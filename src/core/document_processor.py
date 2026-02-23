import fitz
import os
from pathlib import Path
from typing import Dict, List, Tuple
from .utils import logger
from .file_manager import ensure_chat_dir

class DocumentProcessor:
    """SOTA Document Processor for Ultima_RAG. Extracts text and images from PDFs."""
    
    @staticmethod
    def extract_from_pdf(conversation_id: str, file_path: str, file_name: str) -> Tuple[str, List[str]]:
        """
        Extract text and images from a PDF.
        Naming Convention: Document_Name_Image_N
        
        Returns:
            Tuple of (full_text, list_of_image_paths)
        """
        logger.info(f"📄 Extracting content from PDF: {file_name}")
        full_text = ""
        image_paths = []
        
        chat_dir = ensure_chat_dir(conversation_id)
        images_dir = chat_dir / "images"
        
        doc_base_name = os.path.splitext(file_name)[0]
        
        try:
            doc = fitz.open(file_path)
            image_counter = 1
            processed_xrefs = set()
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 1. Extract Text
                full_text += f"\n\n[Page {page_num + 1}]\n"
                full_text += page.get_text()
                
                # 2. Extract Images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    # SOTA: Deduplicate images by xref to avoid redundant processing of logos/icons
                    if xref in processed_xrefs:
                        continue
                        
                    base_image = doc.extract_image(xref)
                    
                    # SOTA Guard: Filter out tiny images (icons, spacers, bullets)
                    width = base_image.get("width", 0)
                    height = base_image.get("height", 0)
                    if width < 100 or height < 100:
                        logger.debug(f"Skipping tiny image {xref} ({width}x{height})")
                        processed_xrefs.add(xref)
                        continue
                        
                    image_bytes = base_image["image"]
                    ext = base_image["extension"]
                    
                    # Naming Logic: Document_Name_Image_N
                    img_name = f"{doc_base_name}_Image_{image_counter}.{ext}"
                    img_path = images_dir / img_name
                    
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    image_paths.append(str(img_path))
                    processed_xrefs.add(xref)
                    image_counter += 1
                    
                    # SOTA Cap: Limit extraction to 20 images per document for safety
                    if image_counter > 20:
                        logger.warning(f"Maximum image extraction limit reached for {file_name}")
                        break
                
                if image_counter > 20:
                    break
            doc.close()
            logger.info(f"✅ Extracted {image_counter-1} images and {len(full_text)} chars of text from {file_name}")
            
        except Exception as e:
            logger.error(f"Failed to process PDF {file_name}: {e}")
            
        return full_text, image_paths

