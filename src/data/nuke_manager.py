"""
Nuke Manager for Ultima_RAG
Coordinates full-system factory resets across all data layers.
"""

import os
from ..core.utils import logger
from ..core.database import get_database as get_vector_db
from .database import get_database as get_relational_db

class NukeManager:
    """Centralized authority for wiping system data."""
    
    @staticmethod
    def nuke_all_data(password: str) -> tuple[bool, str]:
        """
        Wipes all conversations, messages, and vector chunks.
        Requires password 'ADMIN'.
        Returns: (success, message)
        """
        if password != "ADMIN":
            logger.warning("FAILED NUKE ATTEMPT: Incorrect Administrative Password")
            return False, "Administrative privilege escalation failed. Access Denied."
            
        logger.critical("PERFORMING FULL SYSTEM NUKE...")
        
        try:
            # 1. Reset Relational Database (SQLite/PostgreSQL)
            relational_db = get_relational_db()
            relational_db.reset_database()
            logger.info("Relational database wiped.")
            
            # 2. Reset Vector Database (LanceDB)
            vector_db = get_vector_db()
            vector_db.reset_database()
            logger.info("Vector database wiped.")
            
            # 3. Reset Physical Uploads
            from ..core.file_manager import nuke_uploads
            nuke_uploads()
            logger.info("Physical uploads wiped.")
            
            # 4. Future-proofing: reset other caches/history if needed
            
            logger.critical("SYSTEM NUKE SUCCESSFUL. APPLICATION RESET TO NEW.")
            return True, "System nuked successfully. Application reset to fresh state."
            
        except Exception as e:
            logger.error(f"CRITICAL ERROR DURING SYSTEM NUKE: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)

