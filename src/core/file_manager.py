"""
File Manager for Ultima_RAG Multimodal uploads.
Handles partitioning of uploads by conversation_id and file type.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from .utils import logger

# Base data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data"
UPLOADS_DIR = DATA_DIR / "uploads"

def ensure_chat_dir(conversation_id: str) -> Path:
    """Ensure directory for a specific conversation exists."""
    chat_path = UPLOADS_DIR / conversation_id
    subfolders = ["documents", "images", "audio", "video"]
    
    for sub in subfolders:
        (chat_path / sub).mkdir(parents=True, exist_ok=True)
        
    return chat_path

def get_upload_path(conversation_id: str, file_type: str, file_name: str) -> Path:
    """Get the target path for an uploaded file."""
    # Map raw extension or type to subfolder
    subfolder = "documents"
    type_map = {
        'pdf': 'documents', 'txt': 'documents', 'word': 'documents', 'docx': 'documents',
        'csv': 'documents', 'xls': 'documents', 'xlsx': 'documents',
        'png': 'images', 'jpg': 'images', 'jpeg': 'images', 'image': 'images',
        'mp3': 'audio', 'wav': 'audio', 'audio': 'audio',
        'mp4': 'video', 'mov': 'video', 'webm': 'video', 'video': 'video'
    }
    
    subfolder = type_map.get(file_type.lower(), "documents")
    ensure_chat_dir(conversation_id)
    return UPLOADS_DIR / conversation_id / subfolder / file_name

def save_upload(conversation_id: str, file_type: str, file_name: str, file_content: bytes) -> str:
    """Save an uploaded file and return the local path."""
    # SOTA Security: Sanitize filename to prevent path traversal
    safe_name = os.path.basename(file_name)
    target_path = get_upload_path(conversation_id, file_type, safe_name)
    
    try:
        with open(target_path, "wb") as f:
            f.write(file_content)
        logger.info(f"Saved {file_name} to {target_path}")
        return str(target_path)
    except Exception as e:
        logger.error(f"Failed to save upload {file_name}: {e}")
        raise


def list_uploads(conversation_id: str) -> list:
    """List all uploaded files for a conversation.
    
    Returns a list of dicts with file info:
    - name: file name
    - path: full path
    - type: file category (documents, images, audio, video)
    - size: file size in bytes
    """
    chat_path = UPLOADS_DIR / conversation_id
    files = []
    
    if not chat_path.exists():
        return files
    
    for subfolder in ["documents", "images", "audio", "video"]:
        folder_path = chat_path / subfolder
        if folder_path.exists():
            for file_path in folder_path.iterdir():
                if file_path.is_file():
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "type": subfolder,
                        "size": file_path.stat().st_size
                    })
    
    return files


def get_file_path(conversation_id: str, file_name: str) -> Optional[Path]:
    """Get the full path for a specific file in a conversation's uploads."""
    chat_path = UPLOADS_DIR / conversation_id
    
    if not chat_path.exists():
        return None
    
    for subfolder in ["documents", "images", "audio", "video"]:
        file_path = chat_path / subfolder / file_name
        if file_path.exists():
            return file_path
    
    return None

def delete_chat_dir(conversation_id: str) -> bool:
    """Delete all uploads for a specific conversation."""
    chat_path = UPLOADS_DIR / conversation_id
    if chat_path.exists():
        try:
            shutil.rmtree(chat_path)
            logger.info(f"Deleted uploads for conversation: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chat uploads: {e}")
            return False
    return True

def nuke_uploads() -> bool:
    """Wipe the entire uploads directory (Nuclear Option)."""
    if UPLOADS_DIR.exists():
        try:
            for item in UPLOADS_DIR.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
            logger.info("Nuclear Factory Reset: Uploads directory wiped.")
            return True
        except Exception as e:
            logger.error(f"Failed to nuke uploads directory: {e}")
            return False
    return True

