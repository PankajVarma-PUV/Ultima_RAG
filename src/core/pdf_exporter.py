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
UltimaRAG Conversation-to-PDF Exporter
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
    """SOTA Custom PDF generator with UltimaRAG branding and Unicode support."""

    def __init__(self):
        # SOTA: Explicit A4 Format (210 x 297mm)
        super().__init__(orientation='P', unit='mm', format='A4')
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(15, 15, 15) # Standard A4 Margins
        
        # GRID SYSTEM
        self.GRID_WIDTH = self.w - 30 # 180mm effective content width
        self.LEFT_MARGIN = 15
        
        # SOTA: Register Fonts for robust Hindi/Devanagari and Emoji support
        # We try both Windows standard paths and potential local copies
        self._has_unicode_font = False
        self._has_emoji_font = False
        
        # 1. Register Main Unicode Font (Hindi support)
        # We discovered .ttf fonts in the system that support Devanagari
        hindi_fonts = [
            ("Baloo", r"C:\Windows\Fonts\Baloo-Regular.ttf"),
            ("Teko", r"C:\Windows\Fonts\Teko-SemiBold.ttf"),
            ("Yatra", r"C:\Windows\Fonts\YatraOne-Regular.ttf"),
            ("NirmalaUI", r"C:\Windows\Fonts\nirmala.ttc")
        ]
        
        self._has_unicode_font = False
        for name, path in hindi_fonts:
            try:
                if os.path.exists(path):
                    # 1. Register Regular Style (Normal)
                    self.add_font(name, style="", fname=path)
                    
                    if name == "NirmalaUI" and path.lower().endswith(".ttc"):
                        # 2. Register Bold Style (Index 1 in Nirmala.ttc)
                        try:
                            self.add_font(name, style="B", fname=path)
                            logger.info(f"PDF: Registered Bold for {name}")
                        except Exception as be:
                            logger.warning(f"Nirmala Bold reg failed: {be}")

                        # 3. Handle Missing Italic Styles (Aliasing Strategy)
                        # Nirmala UI lacks native Italic files. We register Regular as 'I' 
                        # and Bold as 'BI' to prevent FPDF crash on 'nirmalauiI'/'nirmalauiBI'.
                        try:
                            self.add_font(name, style="I", fname=path) # Alias I -> Regular
                            self.add_font(name, style="BI", fname=path) # Alias BI -> Bold (effectively)
                            logger.info(f"PDF: Registered style aliases (I, BI) for {name}")
                        except Exception as ae:
                            logger.warning(f"Nirmala Alias reg failed: {ae}")

                    self._has_unicode_font = True
                    logger.info(f"PDF: Registered {name} for Hindi support.")
            except Exception as e:
                logger.error(f"Error registering {name}: {e}")

        # 2. Register Emoji Font (Segoe UI Emoji)
        emoji_path = r"C:\Windows\Fonts\seguiemj.ttf"
        try:
            if os.path.exists(emoji_path):
                self.add_font("SegoeUIEmoji", style="", fname=emoji_path)
                self._has_emoji_font = True
                logger.info("PDF: Registered Segoe UI Emoji for emoji support.")
        except Exception as e:
            logger.error(f"Error registering Segoe UI Emoji: {e}")

        # 3. Set Fallback Logic
        fallbacks = []
        # Add all discovered Hindi fonts to fallbacks
        for name, path in hindi_fonts:
            if os.path.exists(path):
                fallbacks.append(name)
        
        if self._has_emoji_font:
            fallbacks.append("SegoeUIEmoji")
        
        if fallbacks:
            self.set_fallback_fonts(fallbacks)
            logger.info(f"PDF: Set fallback fonts: {fallbacks}")

    def header(self):
        # Premium Minimalist Header (Content pages only)
        if self.page_no() == 1:
            return

        self.set_fill_color(6, 182, 212) # Heritage Cyan
        self.rect(0, 0, self.w, 1.5, 'F')
        
        font_main = 'NirmalaUI' if self._has_unicode_font else 'Helvetica'
        self.set_xy(self.LEFT_MARGIN, 8)
        self.set_font(font_main, 'B', 9)
        self.set_text_color(100, 116, 139) # Muted text
        self.cell(self.GRID_WIDTH / 2, 8, 'ULTIMARAG INTELLIGENCE DOSSIER', align='L')
        
        self.set_font(font_main, '', 8)
        self.cell(self.GRID_WIDTH / 2, 8, f'PAGE {self.page_no()}', align='R', ln=1)

        self.set_draw_color(241, 245, 249)
        self.line(self.LEFT_MARGIN, 16, self.w - self.LEFT_MARGIN, 16)
        self.set_y(22)

    def render_cover_page(self, title: str, conv_id: str, metadata: Dict = None):
        """Executive Technical Report Cover (One-pass, no page add)."""
        font_main = 'NirmalaUI' if self._has_unicode_font else 'Helvetica'
        
        # 1. Background Fill
        self.set_fill_color(255, 255, 255)
        self.rect(0, 0, self.w, self.h, 'F')
        
        # 2. Side Decorative Bar (Large)
        self.set_fill_color(15, 23, 42) # Slate Deep
        self.rect(0, 0, 8, self.h, 'F')
        self.set_fill_color(6, 182, 212) # Cyan thin
        self.rect(8, 0, 1, self.h, 'F')

        # 3. Content Block
        content_x = 25
        self.set_xy(content_x, 60)
        
        # Accent Tag
        self.set_fill_color(6, 182, 212)
        self.rect(content_x, 60, 10, 1.5, 'F')
        self.ln(8)
        
        # Main Title (SOTA Wrapping)
        self.set_font(font_main, 'B', 38)
        self.set_text_color(15, 23, 42)
        # Use content_x to calculate width left
        self.multi_cell(self.w - content_x - 15, 14, title.upper(), align='L')
        
        self.ln(5)
        self.set_font(font_main, '', 16)
        self.set_text_color(100, 116, 139)
        self.multi_cell(0, 10, "OFFICIAL AI INTELLIGENCE EXPORT", align='L')
        
        # 4. Registry Identity (Fixed Clipping)
        self.set_y(-90)
        self.set_x(content_x)
        self.set_font(font_main, 'B', 11)
        self.set_text_color(6, 182, 212)
        self.multi_cell(self.w - content_x - 15, 10, "REGISTRY IDENTITY", align='L')
        
        self.set_font(font_main, '', 10)
        self.set_text_color(51, 65, 85)
        self.set_x(content_x)
        # CONVERSATION ID WRAPPING
        self.multi_cell(self.w - content_x - 15, 6, f"IDENTITY: {conv_id}", align='L')
        self.set_x(content_x)
        self.multi_cell(0, 6, f"GENERATED: {datetime.now().strftime('%B %d, %Y | %H:%M:%S UTC')}", align='L')
        
        if metadata:
            for k, v in metadata.items():
                self.set_x(content_x)
                self.multi_cell(self.w - content_x - 15, 6, f"{k.upper()}: {v}", align='L')

        # 5. Footer Disclaimer
        self.set_y(-25)
        self.set_font(font_main, 'B', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 5, "AUTHENTicated ULTIMARAG SOTA INTEL", align='C', ln=1)
        self.cell(0, 5, "CONFIDENTIAL / INTERNAL USE ONLY", align='C', ln=1)

    def footer(self):
        self.set_y(-15)
        self.set_font('NirmalaUI' if self._has_unicode_font else 'Helvetica', 'I', 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 10, f'Authenticated UltimaRAG Document - Page {self.page_no()}/{{nb}}', align='C')

    def render_chat_bubble(self, role: str, content: str, metadata: Dict = None):
        """High-Contrast Investigation Section (Stable V3)."""
        is_user = role.upper() == 'USER'
        font_main = 'NirmalaUI' if self._has_unicode_font else 'Helvetica'
        
        # GRID LAYOUT
        content_w = self.GRID_WIDTH - 5
        
        # 1. Heading
        self.set_x(self.LEFT_MARGIN)
        self.set_font(font_main, 'B', 9)
        self.set_text_color(6, 182, 212) if not is_user else self.set_text_color(100, 116, 139)
        label = "INVESTIGATIVE PROMPT" if is_user else "ULTIMARAG INTELLIGENCE"
        self.multi_cell(self.GRID_WIDTH, 8, label, align='L')
        
        # 2. Content Block
        self.set_x(self.LEFT_MARGIN)
        self.set_font(font_main, '', 10.5)
        self.set_text_color(30, 41, 59)
        
        # MultiCell with fill for premium look
        # AI response gets a very subtle off-white background
        fill = not is_user
        if fill: self.set_fill_color(252, 252, 254)
        
        # Start coordinate for side accent
        start_y = self.get_y()
        self.multi_cell(self.GRID_WIDTH, 6.5, content, border=0, align='L', fill=fill)
        end_y = self.get_y()
        
        # 3. Side Accent Bar
        accent_color = (6, 182, 212) if not is_user else (148, 163, 184)
        self.set_fill_color(*accent_color)
        self.rect(self.LEFT_MARGIN - 4, start_y, 1.5, end_y - start_y, 'F')
        
        # 4. Intelligence Metadata (AI Only)
        if not is_user and metadata:
            self.set_x(self.LEFT_MARGIN)
            self.set_font(font_main, 'I', 8)
            self.set_text_color(100, 116, 139)
            
            trace = metadata.get('intent', 'Reasoning')
            srcs = len(metadata.get('sources', []))
            meta_str = f"Trace: {trace} | Resolution: High-Fidelity"
            if srcs > 0: meta_str += f" | Evidence Base: {srcs} Nodes"
            
            self.multi_cell(self.GRID_WIDTH, 6, meta_str, align='L')
            
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

    # SOTA V3: Primary Orchestration
    # 1. First Page: Cover
    pdf.render_cover_page(
        title=conversation.get('title', 'Intelligence Dossier'),
        conv_id=conversation.get('conversation_id', 'N/A')
    )
    
    # 2. Add Content Page
    pdf.add_page()
    
    font_main = 'NirmalaUI' if pdf._has_unicode_font else 'Helvetica'
    
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
    
    pdf.ln(5)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
    pdf.ln(10)

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
    conversation_title: str = "UltimaRAG Intelligence Export"
) -> bytes:
    """
    Generate a branded, Unicode-capable PDF for a specific query and its AI response.
    Supports Hindi/English and any language that Nirmala UI covers.
    """
    pdf = ConversationPDF()
    pdf.alias_nb_pages()
    # SOTA V3: Targeted Export
    # 1. Cover
    pdf.render_cover_page(
        title=conversation_title,
        conv_id=conversation_id,
        metadata={"Report Type": "Investigation Segment", "Analysis": "Deep Grounding"}
    )
    
    # 2. Content
    pdf.add_page()

    font_main = 'NirmalaUI' if pdf._has_unicode_font else 'Helvetica'
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if mentioned_files:
        pdf.cell(0, 5, f"GROUNDED SOURCES: {', '.join(mentioned_files)}", ln=1)

    pdf.ln(5)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
    pdf.ln(10)

    # ── User Query Card ─────────────────────────────────────────────────────
    pdf.render_chat_bubble("USER", query)

    # ── AI Response Card ────────────────────────────────────────────────────
    meta = {
        "intent": "Targeted Export",
        "confidence_score": 1.0,
        "sources": mentioned_files if mentioned_files else []
    }
    pdf.render_chat_bubble("ASSISTANT", response, metadata=meta)

    # ── Footer Certification ──────────────────────────────────────────────
    pdf.set_y(pdf.get_y() + 10)
    pdf.set_draw_color(226, 232, 240)
    pdf.set_line_width(0.2)
    pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
    pdf.ln(5)

    pdf.set_font(font_main, 'I', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, f"Certified UltimaRAG Intelligence Document  |  Generated: {now_str}", ln=1, align='C')

    return pdf.output()

