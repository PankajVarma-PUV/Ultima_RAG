"""
Ultima_RAG Conversation-to-PDF Exporter
Generates structured PDF documents from conversation history with full Unicode support.
"""

import io
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from fpdf import FPDF

from .utils import logger


class ConversationPDF(FPDF):
    """SOTA Custom PDF generator with Ultima_RAG branding and Unicode support."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)
        
        # SOTA: Register Nirmala UI for robust Hindi/Devanagari support
        font_paths = [
            ("Nirmala", r"C:\Windows\Fonts\nirmala.ttf"),
            ("Nirmala-Bold", r"C:\Windows\Fonts\nirmalab.ttf")
        ]
        
        self._has_unicode_font = False
        for name, path in font_paths:
            if os.path.exists(path):
                style = "B" if "Bold" in name else ""
                self.add_font("NirmalaUI", style=style, fname=path)
                self._has_unicode_font = True
            else:
                logger.warning(f"Font not found: {path}. Unicode support may be limited.")

    def header(self):
        # Professional Minimalist Header
        self.set_xy(10, 10)
        self.set_font('NirmalaUI' if self._has_unicode_font else 'Helvetica', 'B', 14)
        self.set_text_color(6, 182, 212) # Ultima Cyan
        self.cell(0, 10, 'Ultima RAG | INTELLIGENCE EXPORT', align='L')
        
        self.set_font('NirmalaUI' if self._has_unicode_font else 'Helvetica', '', 8)
        self.set_text_color(148, 163, 184) # Slate 400
        self.cell(0, 10, f'GEN-ID: {datetime.now().strftime("%Y%m%d%H%M%S")}', align='R')
        
        # Aesthetic Divider
        self.set_draw_color(6, 182, 212)
        self.set_line_width(0.2)
        self.line(10, 20, self.w - 10, 20)
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('NirmalaUI' if self._has_unicode_font else 'Helvetica', 'I', 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f'Authenticated Ultima_RAG Document - Page {self.page_no()}/{{nb}}', align='C')

    def render_chat_bubble(self, role: str, content: str, metadata: Dict = None):
        """Render a premium chat bubble style message."""
        is_user = role.upper() == 'USER'
        
        # Colors & Styles
        bg_color = (248, 250, 252) if is_user else (241, 245, 249) # Light gray variations
        border_color = (203, 213, 225) if is_user else (6, 182, 212) # User: Slate, AI: Cyan
        text_color = (15, 23, 42)
        label_color = (100, 116, 139)
        
        font_main = 'NirmalaUI' if self._has_unicode_font else 'Helvetica'
        
        # 1. Header (Label)
        self.set_font(font_main, 'B', 9)
        self.set_text_color(*label_color)
        icon = "👤 USER" if is_user else "🤖 Ultima AI"
        self.cell(0, 8, icon, ln=1)
        
        # 2. Bubble Body logic using MultiCell for wrapping
        self.set_font(font_main, '', 11)
        self.set_text_color(*text_color)
        
        # Calculate height needed
        content_width = self.w - 30
        prev_y = self.get_y()
        
        # Draw background and border
        # We perform a "Pre-flight" to get height if needed, but FPDF's multi_cell handles it.
        # To make it look like a bubble, we use a rect behind it.
        # We'll use a simple border for now for high-fidelity printing.
        self.set_draw_color(*border_color)
        self.set_line_width(0.1)
        
        # Render the content
        self.set_x(15)
        self.multi_cell(content_width, 7, content, border=0, align='L', fill=False)
        
        # 3. Metadata Footer (AI Only)
        if not is_user and metadata:
            self.ln(1)
            self.set_x(15)
            self.set_font(font_main, 'I', 8)
            self.set_text_color(6, 182, 212)
            
            intent = metadata.get('intent', 'General')
            conf = metadata.get('confidence_score')
            sources = metadata.get('sources', [])
            
            meta_parts = [f"Direct Logic: {intent}"]
            if conf: meta_parts.append(f"Fidelity: {round(float(conf)*100)}%")
            if sources: meta_parts.append(f"Grounded: {len(sources)} sources")
            
            self.cell(0, 5, " | ".join(meta_parts), ln=1)
            
        self.ln(10)

def generate_conversation_pdf(
    conversation: Dict,
    messages: List[Dict],
    scope: str = "full"
) -> bytes:
    """
    Generate a SOTA PDF document with full Hindi support and high-fidelity design.
    """
    pdf = ConversationPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    font_main = 'NirmalaUI' if pdf._has_unicode_font else 'Helvetica'

    # --- Cover Section ---
    title = conversation.get('title') or 'Ultima_RAG Intelligence Log'
    pdf.set_font(font_main, 'B', 24)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 15, title.upper(), align='L')
    
    pdf.set_font(font_main, '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, f"CONVERSATION IDENTITY: {conversation.get('conversation_id', 'N/A')}", ln=1)
    pdf.cell(0, 5, f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    
    if scope == "summary":
        pdf.ln(10)
        pdf.set_font(font_main, 'B', 12)
        pdf.set_text_color(6, 182, 212)
        pdf.cell(0, 10, "EXECUTIVE SUMMARY", ln=1)
        pdf.set_font(font_main, '', 11)
        pdf.set_text_color(15, 23, 42)
        summary = next((m.get('content') for m in reversed(messages) if m.get('role') == 'assistant' and 'summary' in m.get('content', '').lower()), "Brief conversation overview generated per user query.")
        pdf.multi_cell(0, 7, summary)
        pdf.ln(10)
    
    pdf.ln(10)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
    pdf.ln(15)

    # --- Message Flow ---
    # Filter messages based on scope
    filtered_messages = messages
    if scope == "latest":
        filtered_messages = messages[-2:] if len(messages) >= 2 else messages

    for msg in filtered_messages:
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
        
        # Metadata parsing
        meta_raw = msg.get('metadata_json') or msg.get('metadata', '{}')
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        except:
            meta = {}
            
        pdf.render_chat_bubble(role, content, meta)

    return pdf.output()


def generate_evidence_report(
    filename: str,
    asset: Optional[Dict],
    chunks: List[Dict]
) -> bytes:
    """
    SOTA: Evidence Report Generator for Source Explorer.
    Produces a professional dossier on a specific file's content and grounding.
    """
    pdf = ConversationPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    font_main = 'NirmalaUI' if pdf._has_unicode_font else 'Helvetica'
    
    # --- Title Section ---
    pdf.set_font(font_main, 'B', 20)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 15, f"EVIDENCE REPORT: {filename.upper()}", ln=1)
    
    # --- Meta Data Section ---
    pdf.set_font(font_main, 'B', 10)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(40, 7, "ASSET METADATA", ln=1)
    
    pdf.set_font(font_main, '', 9)
    pdf.set_text_color(71, 85, 105)
    
    if asset:
        pdf.cell(60, 5, f"ID: {asset.get('id', 'N/A')}")
        pdf.cell(60, 5, f"Type: {asset.get('file_type', 'N/A')}")
        pdf.cell(60, 5, f"Pages/Duration: {asset.get('total_pages') or asset.get('duration_sec') or 'N/A'}", ln=1)
    else:
        pdf.cell(0, 5, "Metadata not available in registry.", ln=1)
    
    pdf.ln(6)
    pdf.cell(0, 5, f"CHUNKS ANALYZED: {len(chunks)}", ln=1)
    pdf.cell(0, 5, f"REPORT GENERATED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    
    pdf.ln(10)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
    pdf.ln(10)
    
    # --- Content Section ---
    pdf.set_font(font_main, 'B', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 10, "GROUNDED CONTEXT BLOCKS", ln=1)
    pdf.ln(5)
    
    for i, chunk in enumerate(chunks, 1):
        # Header for chunk
        pdf.set_font(font_main, 'B', 9)
        pdf.set_text_color(100, 116, 139)
        
        meta = chunk.get('metadata')
        if isinstance(meta, str):
            try: meta = json.loads(meta)
            except: meta = {}
            
        page_info = f" | Page: {meta.get('page', meta.get('page_number', 'N/A'))}" if meta else ""
        pdf.cell(0, 8, f"BLOCK {i:02d}{page_info}", ln=1)
        
        # Chunk Text
        pdf.set_font(font_main, '', 10)
        pdf.set_text_color(30, 41, 59)
        text = chunk.get('text', '')
        pdf.multi_cell(0, 6, text)
        
        pdf.ln(5)
        # Separator between chunks
        pdf.set_draw_color(241, 245, 249)
        pdf.line(20, pdf.get_y(), pdf.w - 20, pdf.get_y())
        pdf.ln(5)
        
        # Check for page break
        if pdf.get_y() > 250:
            pdf.add_page()
            
    return pdf.output()


def generate_query_pdf(
    query: str,
    response: str,
    conversation_id: str,
    mentioned_files: Optional[List[str]] = None,
    conversation_title: str = "Ultima_RAG Intelligence Export"
) -> bytes:
    """
    Generate a branded, Unicode-capable PDF for a specific query and its AI response.
    Supports Hindi/English and any language that Nirmala UI covers.
    """
    pdf = ConversationPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    font_main = 'NirmalaUI' if pdf._has_unicode_font else 'Helvetica'
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Cover ──────────────────────────────────────────────────────────────
    pdf.set_font(font_main, 'B', 22)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 14, conversation_title.upper(), align='L')

    pdf.set_font(font_main, '', 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, f"QUERY-BASED EXPORT  |  {now_str}", ln=1)
    pdf.cell(0, 5, f"CONVERSATION: {conversation_id}", ln=1)

    if mentioned_files:
        pdf.cell(0, 5, f"GROUNDED SOURCES: {', '.join(mentioned_files)}", ln=1)

    pdf.ln(6)
    pdf.set_draw_color(6, 182, 212)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
    pdf.ln(10)

    # ── User Query Block ─────────────────────────────────────────────────────
    pdf.set_font(font_main, 'B', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 7, "📝  USER QUERY", ln=1)

    pdf.set_font(font_main, '', 12)
    pdf.set_text_color(15, 23, 42)
    pdf.set_x(15)
    pdf.multi_cell(pdf.w - 30, 7, query, border=0, align='L')
    pdf.ln(8)

    # ── AI Response Block ────────────────────────────────────────────────────
    pdf.set_font(font_main, 'B', 10)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(0, 7, "🤖  Ultima_RAG AI RESPONSE", ln=1)

    pdf.set_draw_color(6, 182, 212)
    pdf.set_line_width(0.1)
    pdf.line(10, pdf.get_y(), 14, pdf.get_y())   # left accent bar

    pdf.ln(4)
    pdf.set_font(font_main, '', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.set_x(15)
    pdf.multi_cell(pdf.w - 30, 7, response, border=0, align='L')
    pdf.ln(12)

    # ── Footer Certification ──────────────────────────────────────────────
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.2)
    pdf.line(10, pdf.get_y(), pdf.w - 10, pdf.get_y())
    pdf.ln(5)

    pdf.set_font(font_main, 'I', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, f"Certified Ultima_RAG Intelligence Document  |  Generated: {now_str}", ln=1, align='C')

    return pdf.output()

