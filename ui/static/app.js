/*
 * UltimaRAG — Multi-Agent RAG System
 * Copyright (C) 2026 Pankaj Varma
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/**
 * UltimaRAG SOTA Frontend Controller
 * Logic for Workspace Explorer, Generative UI, and Adaptive Resonance
 */

class UltimaApp {
    constructor() {
        this.state = {
            activeProject: 'default',
            activeConversation: null,
            projects: [],
            conversations: [],
            recentConversations: [],
            tree: { projects: [] },
            isProcessing: false,
            pendingFiles: [],
            mentionedFiles: [],    // @mention targeted file names
            mentionSearch: '',     // current @mention search prefix
            abortController: null,  // STOP GENERATION: AbortController for active fetch
            workspaceFileCount: 0,  // tracks whether current convo has indexed files (for action button RAG gate)
            webSearchEnabled: false // Web-Breakout Agent toggle (OFF by default)
        };

        // UI Elements
        this.treeContainer = document.getElementById('treeContainer');
        this.conversationList = document.getElementById('conversationList');
        this.chatLog = document.getElementById('chat-log');
        this.userInput = document.getElementById('userInput');
        this.sendBtn = document.getElementById('sendBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.actionButtonContainer = document.getElementById('actionButtonContainer');
        this.fileInput = document.getElementById('fileInput');
        this.convTitle = document.getElementById('activeConversationTitle');
        this.searchChats = document.getElementById('searchChats');
        this.statusDot = document.getElementById('statusDot');
        this.statusText = document.getElementById('statusText');
        this.dbDot = document.getElementById('dbDot');
        this.dbText = document.getElementById('dbText');

        // SOTA Dashboard elements
        this.telemetryAgent = document.getElementById('telemetryAgent');
        this.telemetryStage = document.getElementById('telemetryStage');
        this.telemetryHud = document.getElementById('telemetryHud');
        this.unifiedOverlay = document.getElementById('unifiedOverlay');
        this.filePreview = document.getElementById('filePreview');

        // @Mention elements
        this.mentionBtn = document.getElementById('mentionBtn');
        this.mentionDropdown = document.getElementById('mentionDropdown');
        this.mentionSuggestions = document.getElementById('mentionSuggestions');
        this.mentionChips = document.getElementById('mentionChips');

        // Workspace File Viewer elements
        this.workspaceFilesContainer = document.getElementById('workspaceFiles');
        this.workspaceCount = document.getElementById('workspaceCount');
        this.fileActionModal = document.getElementById('fileActionModal');
        this.fileActionName = document.getElementById('fileActionName');
        this.fileActionIcon = document.getElementById('fileActionIcon');
        this.fileActionView = document.getElementById('fileActionView');
        this.fileActionClose = document.getElementById('fileActionClose');

        this.init();
    }

    async init() {
        if (window.marked) {
            // SOTA Phase 13: Custom Marked Renderer for Elite UI
            const renderer = new marked.Renderer();

            renderer.code = (code, language) => {
                const lang = language || 'text';
                const escapedCode = code.replace(/'/g, "\\'").replace(/"/g, '&quot;');
                return `
                    <div class="code-block-sota">
                        <div class="code-header">
                            <span class="code-lang-badge">${lang}</span>
                            <button class="code-copy-btn" onclick="window.app.copyCode(this, \`${escapedCode}\`)">
                                <i class="fa-regular fa-copy"></i>
                                <span>Copy Code</span>
                            </button>
                        </div>
                        <pre><code class="language-${lang}">${code}</code></pre>
                    </div>
                `;
            };

            renderer.table = (header, body) => {
                return `<table><thead>${header}</thead><tbody>${body}</tbody></table>`;
            };

            marked.setOptions({ renderer, headerIds: false, mangle: false, breaks: true });
        }

        this.setupEventListeners();
        await this.loadWorkspace();
        this.updateSystemStatus();
        this.connectTelemetry();

        // [SOTA] Disable continuous UI polling for /health to avoid console spam.
        // We now fetch status strictly on-demand (e.g. init, file upload, chat switch).
        // setInterval(() => this.updateSystemStatus(), 30000);
        console.log("UltimaRAG Metacognitive Core Interface Active.");
    }

    setupEventListeners() {
        if (this.sendBtn) this.sendBtn.addEventListener('click', () => this.handleQuery());

        // STOP GENERATION button
        if (this.stopBtn) {
            this.stopBtn.addEventListener('click', () => this.stopGeneration());
        }

        if (this.userInput) {
            this.userInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.handleQuery();
                }
            });
            this.userInput.oninput = () => {
                this.userInput.style.height = 'auto';
                this.userInput.style.height = (this.userInput.scrollHeight) + 'px';
            };

            // Clipboard paste: capture screenshots/images from clipboard
            this.userInput.addEventListener('paste', (e) => {
                const items = e.clipboardData?.items;
                if (!items) return;

                for (const item of items) {
                    if (item.type.startsWith('image/')) {
                        e.preventDefault();
                        const blob = item.getAsFile();
                        if (!blob) continue;

                        // Create a File with a descriptive name
                        const ext = blob.type.split('/')[1] || 'png';
                        const fileName = `screenshot_${Date.now()}.${ext}`;
                        const file = new File([blob], fileName, { type: blob.type });

                        this.state.pendingFiles.push(file);
                        this.updateFileIndicator();
                        this.renderFileChips();
                        this.applyAdaptiveResonance({ theme_accent: '#06b6d4', status: 'Screenshot Captured' });
                        break; // Only handle first image
                    }
                }
            });
        }

        if (this.fileInput) {
            this.fileInput.addEventListener('change', (e) => this.handleFileSelection(e));
        }

        if (this.searchChats) {
            this.searchChats.oninput = () => this.renderRecentActivity();
        }

        const profileBtn = document.getElementById('profileBtn');
        const contactModal = document.getElementById('contactModal');
        if (profileBtn && contactModal) {
            profileBtn.onclick = (e) => {
                e.stopPropagation();
                contactModal.classList.toggle('hidden');
            };
            contactModal.onclick = (e) => e.stopPropagation(); // BUGFIX: Prevent modal from closing when clicking inside it
            document.addEventListener('click', () => contactModal.classList.add('hidden'));
        }

        // @Mention button
        if (this.mentionBtn) {
            this.mentionBtn.addEventListener('click', () => {
                if (!this.state.activeConversation) return;
                // Insert @ into textarea and trigger autocomplete
                this.userInput.value += '@';
                this.userInput.focus();
                this.fetchMentionSuggestions('');
            });
        }

        // @Mention detection in textarea
        if (this.userInput) {
            this.userInput.addEventListener('input', (e) => {
                this.detectMentionTrigger();
            });
        }

        // Close mention dropdown on outside click
        document.addEventListener('click', (e) => {
            if (this.mentionDropdown && !this.mentionDropdown.contains(e.target) && e.target !== this.mentionBtn) {
                this.hideMentionDropdown();
            }
        });

        // Escape key: close modals/dropdowns
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideMentionDropdown();
                this.hideFileActionModal();
                closeFileViewer();
            }
        });

        // File Action Modal: Close button
        if (this.fileActionClose) {
            this.fileActionClose.addEventListener('click', () => this.hideFileActionModal());
        }
        // File Action Modal: click outside to close
        if (this.fileActionModal) {
            this.fileActionModal.addEventListener('click', (e) => {
                if (e.target === this.fileActionModal) this.hideFileActionModal();
            });
        }
    }

    handleFileSelection(e) {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;

        this.state.pendingFiles = [...this.state.pendingFiles, ...files];
        this.updateFileIndicator();
        this.renderFileChips();
        this.applyAdaptiveResonance({ theme_accent: '#06b6d4', status: 'Assets Pending Extraction' });

        // Reset input value to allow re-uploading the same file
        e.target.value = '';
    }

    renderFileChips() {
        if (!this.filePreview) return;
        if (this.state.pendingFiles.length === 0) {
            this.filePreview.classList.add('hidden');
            return;
        }

        this.filePreview.classList.remove('hidden');
        this.filePreview.innerHTML = this.state.pendingFiles.map((file, idx) => `
            <div class="glass-panel px-3 py-1.5 rounded-full flex items-center gap-2 border-white/10 text-xs animate-slide-up">
                <i class="fa-solid ${this.getFileIcon(file.name)} opacity-50"></i>
                <span class="truncate max-w-[120px]">${file.name}</span>
                <button onclick="window.app.removeFile(${idx})" class="hover:text-red-400 transition-colors">
                    <i class="fa-solid fa-xmark"></i>
                </button>
            </div>
        `).join('');
    }

    getFileIcon(name) {
        const ext = name.split('.').pop().toLowerCase();
        if (['png', 'jpg', 'jpeg'].includes(ext)) return 'fa-file-image';
        if (['pdf'].includes(ext)) return 'fa-file-pdf';
        if (['mp3', 'wav'].includes(ext)) return 'fa-file-audio';
        if (['mp4', 'mov', 'webm'].includes(ext)) return 'fa-file-video';
        return 'fa-file-lines';
    }

    removeFile(idx) {
        this.state.pendingFiles.splice(idx, 1);
        this.updateFileIndicator();
        this.renderFileChips();
    }

    updateFileIndicator() {
        const count = this.state.pendingFiles.length;
        const uploadBtn = document.querySelector('label[for="fileInput"]');
        if (!uploadBtn) return;

        if (count > 0) {
            uploadBtn.innerHTML = `<i class="fa-solid fa-paperclip text-cyan-400"></i><span class="absolute -top-1 -right-1 bg-cyan-500 text-[8px] w-3 h-3 flex items-center justify-center rounded-full text-white">${count}</span>`;
        } else {
            uploadBtn.innerHTML = `<i class="fa-solid fa-paperclip opacity-50"></i>`;
        }
    }

    // --- Core Actions ---

    async startNewChat() {
        this.state.activeConversation = null;
        this.state.pendingFiles = []; // Clear pending files on new session
        this.updateFileIndicator();
        this.renderFileChips();
        this.renderWorkspaceFiles([]); // Clear workspace files

        this.chatLog.innerHTML = `
            <div class="max-w-3xl mx-auto bg-white/5 p-6 rounded-3xl border border-white/5 shadow-xl">
                <div class="text-sm leading-relaxed">
                    Welcome to <strong class="text-cyan-400">UltimaRAG</strong>. I am your metacognitive assistant.
                    Upload documents or ask questions to begin reasoning with grounded evidence.
                </div>
                <div class="text-[10px] font-bold uppercase tracking-widest text-cyan-500/50 mt-4 flex items-center gap-2">
                    <span class="w-1.5 h-1.5 rounded-full bg-cyan-500"></span> Core System Active
                </div>
            </div>
        `;
        if (this.convTitle) this.convTitle.innerText = "New Inquiry";

        try {
            const resp = await fetch('/conversations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ project_id: this.state.activeProject })
            });
            const data = await resp.json();
            if (data.success) {
                this.state.activeConversation = data.conversation_id;
                await this.loadWorkspace();
            }
        } catch (e) { console.error("New Chat Error:", e); }
    }

    async selectConversation(cid, title) {
        this.state.activeConversation = cid;
        if (this.convTitle) this.convTitle.innerText = title || "Current Inquiry";

        try {
            const msgResp = await fetch(`/workspace/conversations/${cid}`);
            const data = await msgResp.json();

            this.chatLog.innerHTML = "";

            if (data.messages && data.messages.length > 0) {
                data.messages.forEach(msg => {
                    // SOTA: metadata is now returned as a direct object from the API
                    const meta = typeof msg.metadata === 'string' ? JSON.parse(msg.metadata || '{}') : (msg.metadata || {});
                    this.appendMessage(msg.role, msg.content, meta);
                });
            } else {
                this.appendMessage('ai', "Inquiry history initialized.");
            }
            this.renderWorkspace();
            await this.loadWorkspaceFiles(); // Load files for this conversation
        } catch (err) {
            console.error("Fetch Error:", err);
            this.appendMessage('ai', "Failed to load conversation history.");
        }
    }

    // --- Workspace & Sidebar ---

    async loadWorkspace() {
        try {
            const [treeResp, recentResp] = await Promise.all([
                fetch('/workspace/tree'),
                fetch('/workspace/recent')
            ]);

            const treeData = await treeResp.json();
            const recentData = await recentResp.json();

            if (treeData.success) this.state.tree = treeData.tree;
            if (recentData.success) this.state.recentConversations = recentData.conversations;

            this.renderWorkspace();
            this.renderRecentActivity();
        } catch (err) { console.error("Workspace Load Error:", err); }
    }

    renderRecentActivity() {
        if (!this.conversationList) return;
        const searchTerm = this.searchChats ? this.searchChats.value.toLowerCase() : '';

        let html = '';
        const filtered = this.state.recentConversations.filter(c =>
            !searchTerm || (c.name || c.title || '').toLowerCase().includes(searchTerm)
        );

        if (filtered.length === 0) {
            html = '<div class="text-[10px] opacity-40 px-2 py-2">No conversations found</div>';
        } else {
            filtered.forEach(c => {
                const isActive = this.state.activeConversation === c.id;
                const name = c.name || c.title || (c.created_at ? new Date(c.created_at).toLocaleDateString() : null) || `Chat ${c.id.substring(0, 4)}`;
                html += `
                <div class="tree-item conv-item ${isActive ? 'active' : ''} group" 
                     onclick="window.app.selectConversation('${c.id}', '${name}')">
                    <div class="flex items-center gap-2 flex-1 min-w-0">
                        <i class="fa-regular fa-message opacity-60"></i>
                        <span class="truncate">${name}</span>
                    </div>
                    <div class="flex items-center gap-1.5 opacity-0 group-hover:opacity-100 transition-opacity pr-1">
                        <button onclick="event.stopPropagation(); window.app.renameConversation('${c.id}', '${name.replace(/'/g, "\\'")}')" 
                                class="w-6 h-6 rounded-md hover:bg-white/10 flex items-center justify-center transition-all text-[10px]" title="Rename">
                            <i class="fa-solid fa-pencil opacity-40 hover:opacity-100"></i>
                        </button>
                        <button onclick="event.stopPropagation(); window.app.deleteConversation('${c.id}')" 
                                class="w-6 h-6 rounded-md hover:bg-white/10 flex items-center justify-center transition-all text-[10px]" title="Delete">
                            <i class="fa-solid fa-trash-can opacity-40 hover:opacity-100 hover:text-red-400"></i>
                        </button>
                    </div>
                </div>
            `;
            });
        }
        this.conversationList.innerHTML = html;
    }

    renderWorkspace() {
        // Legacy: project tree rendering (kept for backward compat)
    }

    selectProject(pid) {
        this.state.activeProject = pid;
    }

    // --- Workspace File Viewer ---

    async loadWorkspaceFiles() {
        const cid = this.state.activeConversation;
        if (!cid) {
            this.renderWorkspaceFiles([]);
            return;
        }
        try {
            const resp = await fetch(`/workspace/files/${cid}`);
            const data = await resp.json();
            if (data.success) {
                this.renderWorkspaceFiles(data.files || []);
            } else {
                this.renderWorkspaceFiles([]);
            }
        } catch (e) {
            console.error('Workspace files error:', e);
            this.renderWorkspaceFiles([]);
        }
    }

    renderWorkspaceFiles(files) {
        // Track how many files are in current convo (used for action button RAG gate)
        this.state.workspaceFileCount = files.length;

        // Update count badge
        if (this.workspaceCount) {
            if (files.length > 0) {
                this.workspaceCount.textContent = files.length;
                this.workspaceCount.classList.remove('hidden');
            } else {
                this.workspaceCount.classList.add('hidden');
            }
        }

        if (!this.workspaceFilesContainer) return;

        if (files.length === 0) {
            this.workspaceFilesContainer.innerHTML = '<div class="text-[10px] opacity-30 px-2 py-2">No files uploaded</div>';
            return;
        }

        const typeIconMap = {
            'documents': { icon: 'fa-file-pdf', cls: 'doc' },
            'images': { icon: 'fa-file-image', cls: 'img' },
            'video': { icon: 'fa-file-video', cls: 'vid' },
            'audio': { icon: 'fa-file-audio', cls: 'aud' }
        };

        let html = '';
        files.forEach(f => {
            const info = typeIconMap[f.type] || { icon: 'fa-file', cls: 'doc' };
            const sizeKB = f.size ? (f.size / 1024).toFixed(1) + ' KB' : '';
            html += `
                <div class="workspace-file-item" onclick="window.app.showFileAction('${f.name.replace(/'/g, "\\'")}')"
                     title="${f.name} (${sizeKB})">
                    <div class="file-icon ${info.cls}"><i class="fa-solid ${info.icon}"></i></div>
                    <span class="file-name">${f.name}</span>
                    <span class="file-type-badge">${f.type}</span>
                </div>
            `;
        });
        this.workspaceFilesContainer.innerHTML = html;
    }

    showFileAction(fileName) {
        if (!this.fileActionModal) return;
        const cid = this.state.activeConversation;
        if (!cid) return;

        // Update modal content
        if (this.fileActionName) this.fileActionName.textContent = fileName;
        if (this.fileActionIcon) {
            const ext = fileName.split('.').pop().toLowerCase();
            let iconClass = 'fa-file';
            if (['pdf', 'txt', 'md', 'docx'].includes(ext)) iconClass = 'fa-file-pdf';
            else if (['png', 'jpg', 'jpeg'].includes(ext)) iconClass = 'fa-file-image';
            else if (['mp4', 'mov', 'webm'].includes(ext)) iconClass = 'fa-file-video';
            else if (['mp3', 'wav'].includes(ext)) iconClass = 'fa-file-audio';
            this.fileActionIcon.className = `fa-solid ${iconClass} text-cyan-400`;
        }

        // Wire View button
        if (this.fileActionView) {
            this.fileActionView.onclick = () => {
                const url = `/workspace/files/${cid}/view/${encodeURIComponent(fileName)}`;
                window.open(url, '_blank');
                this.hideFileActionModal();
            };
        }



        this.fileActionModal.classList.remove('hidden');
    }

    hideFileActionModal() {
        if (this.fileActionModal) this.fileActionModal.classList.add('hidden');
    }

    toggleWebSearch() {
        this.state.webSearchEnabled = !this.state.webSearchEnabled;
        const btn = document.getElementById('webSearchToggle');
        if (!btn) return;
        if (this.state.webSearchEnabled) {
            btn.classList.add('web-toggle-active');
            btn.title = 'Web Search: ON — will search the web if local docs have no answer';
        } else {
            btn.classList.remove('web-toggle-active');
            btn.title = 'Web Search: OFF — enable to search the web as a fallback';
        }
        console.log(`[UltimaRAG] Web Search toggle: ${this.state.webSearchEnabled ? 'ON' : 'OFF'}`);
    }

    // --- Message Rendering ---

    renderSummaryCard(summary) {
        if (!this.chatLog) return;
        const wrapper = document.createElement('div');
        wrapper.className = 'max-w-3xl mx-auto w-full flex flex-col items-start mb-6';

        let htmlContent = summary.content;
        if (typeof marked !== 'undefined') {
            htmlContent = marked.parse(htmlContent);
            if (typeof DOMPurify !== 'undefined') htmlContent = DOMPurify.sanitize(htmlContent);
        }

        wrapper.innerHTML = `
            <div class="resonance-card !bg-indigo-500/10 !border-indigo-500/30">
                <div class="intent-pill !bg-indigo-500/20 !text-indigo-300"><i class="fa-solid fa-file-invoice mr-1"></i>DOCUMENT SUMMARY</div>
                <div class="text-[10px] font-bold opacity-60 ml-2 truncate max-w-[200px] hover:max-w-none transition-all">${summary.file_name}</div>
            </div>
            <div class="bubble ai-bubble flex-1 w-full !bg-indigo-900/10 !border-indigo-500/20 backdrop-blur-sm shadow-lg inset-shadow-sm">
                <div class="prose prose-invert max-w-none text-[13px] leading-relaxed">
                    ${htmlContent}
                </div>
            </div>
        `;

        this.chatLog.appendChild(wrapper);
        this.chatLog.scrollTop = this.chatLog.scrollHeight;
    }

    appendMessage(role, content, metadata = null) {
        if (!this.chatLog) return;

        const wrapper = document.createElement('div');
        wrapper.className = `max-w-3xl mx-auto w-full flex flex-col ${role === 'user' ? 'items-end' : 'items-start'}`;

        if (role === 'user') {
            // SOTA: Reconstruct @ mention chips for historical messages
            let mentionChipsHtml = '';
            if (metadata && metadata.mentioned_files && metadata.mentioned_files.length > 0) {
                const chips = metadata.mentioned_files.map(fileName =>
                    `<span class="mention-chip-history">${fileName}</span>`
                ).join('');
                mentionChipsHtml = `<div class="mention-chips-history">${chips}</div>`;
            }

            wrapper.innerHTML = `
                ${mentionChipsHtml}
                <div class="bubble user-bubble">${content}</div>
            `;
        } else {
            let metaHtml = '';
            if (metadata && metadata.confidence_score !== undefined) {
                const confPercent = Math.round(metadata.confidence_score * 100);
                metaHtml = `
                <div class="resonance-card">
                    <div class="intent-pill">${metadata.intent || metadata.agent_type || 'Ultima'}</div>
                    <div class="text-[10px] font-bold opacity-40">Fidelity: ${confPercent}%</div>
                </div>
            `;
            }

            const layout = (metadata && metadata.ui_hints) ? metadata.ui_hints.layout : 'standard';
            const mainContentHtml = (layout === 'rag')
                ? this.renderGroundedUI(content, metadata)
                : this.renderStructuredResponse(content);

            wrapper.innerHTML = `
                ${metaHtml}
                <div class="bubble ai-bubble flex-1 w-full">
                    ${mainContentHtml}
                </div>
            `;
        }

        this.chatLog.appendChild(wrapper);
        this.chatLog.scrollTop = this.chatLog.scrollHeight;
    }

    renderStructuredResponse(text, forcedSources = []) {
        // SOTA Phase 15: Clean unnecessary whitespace and excessive newlines
        let processedText = text.replace(/\n{3,}/g, '\n\n').trim();
        const usedSources = new Set(forcedSources);

        // 3. Mark Body for Tags (Placeholder system to survive Markdown parsing)
        const shadowPlaceholders = [];
        const shadowRegex = /\[\[([^|\]]+)\|([^\]]+)\]\]/gi;
        processedText = processedText.replace(shadowRegex, (match, fileName, content) => {
            const id = `SOTASHADOWTOKEN${shadowPlaceholders.length}TKN`;
            shadowPlaceholders.push({ id, fileName: fileName.trim(), content: content.trim() });
            return id;
        });

        const badgePlaceholders = [];
        const badgeRegex = /\[\[(?:Source:\s*)?([^|\]]+)\]\]|\[Source:\s*([^\]]+)\]/gi;
        processedText = processedText.replace(badgeRegex, (match, dblFile, sglFile) => {
            const fileNamePart = (dblFile || sglFile).trim();
            if (/^\d+$/.test(fileNamePart)) return match; // Ignore numeric hallucinations
            const id = `SOTACITATIONTOKEN${badgePlaceholders.length}TKN`;
            badgePlaceholders.push({ id, fileName: fileNamePart });
            return id;
        });

        // 4. Adaptive Content Cards (Insight Detection)
        const insightRegex = /^(Insight|Key Fact|Note|Warning):(.*)$/gim;
        processedText = processedText.replace(insightRegex, (match, label, content) => {
            return `
                <div class="insight-card animate-slide-up">
                    <span class="insight-label">${label}</span>
                    <div class="insight-content">${content.trim()}</div>
                </div>
            `;
        });

        // 5. Streaming Guard: Don't render half-opened tags
        if (processedText.includes('[[')) {
            const lastIdx = processedText.lastIndexOf('[[');
            if (!processedText.includes(']]', lastIdx)) {
                processedText = processedText.substring(0, lastIdx);
            }
        }

        // 6. Final Body Parsing (Markdown first)
        const tasks = [];
        const taskRegex = /\[TASK:([^\]]+)\]([\s\S]*?)\[\/TASK\]/gi;
        let match;
        while ((match = taskRegex.exec(processedText)) !== null) {
            tasks.push({ type: match[1], content: match[2].trim() });
        }

        let bodyContent = processedText.replace(/\[TASK:[^\]]+\][\s\S]*?\[\/TASK\]/gi, '').trim();

        let html = `${window.marked ? marked.parse(bodyContent) : bodyContent}`;

        // 7. Inject Placeholders back as SOTA Interactive Elements
        shadowPlaceholders.forEach(p => {
            usedSources.add(p.fileName);
            const tooltipContent = window.marked ? marked.parse(p.content) : p.content;

            // SOTA Phase 16: Use spans for valid HTML nesting
            const replacement = `
                <span class="lens-highlight" data-file="${p.fileName}" onclick="window.app.openArtifact('${p.fileName.replace(/'/g, "\\'")}')">
                    ${p.content}
                    <span class="perception-tooltip">
                        <span class="perception-tooltip-header">
                            <span class="flex items-center gap-2">
                                <i class="fa-solid fa-eye text-[8px] animate-pulse"></i>
                                <span>Ultima PERCEPTION: ${p.fileName}</span>
                            </span>
                        </span>
                        <span class="perception-tooltip-body">${tooltipContent}</span>
                    </span>
                </span>`.trim();

            // SOTA Phase 21: Global Replacement Fix
            html = html.split(p.id).join(replacement);
        });

        badgePlaceholders.forEach(p => {
            usedSources.add(p.fileName);
            const index = Array.from(usedSources).indexOf(p.fileName) + 1;
            const safeFile = p.fileName.replace(/'/g, "\\'")
            const replacement = `<sup class="citation-badge" data-source-file="${p.fileName}" onclick="window.app.openArtifact('${safeFile}', this)" title="${p.fileName}">[${index}]</sup>`;

            // SOTA Phase 21: Global Replacement Fix
            html = html.split(p.id).join(replacement);
        });

        // 6. Source Footer Generation
        if (usedSources.size > 0) {
            // SOTA: Cleanse sources for display (unique base names)
            const uniqueSources = new Map();
            Array.from(usedSources).forEach(s => {
                const base = s.includes('.') ? s.split('.').slice(0, -1).join('.') : s;
                if (!uniqueSources.has(base.toLowerCase())) {
                    uniqueSources.set(base.toLowerCase(), s);
                }
            });

            const footerItems = Array.from(uniqueSources.values()).map((s, i) => `
                <div class="source-item" data-source-file="${s}" onclick="window.app.openArtifact('${s.replace(/'/g, "\\'")}'.toString(), this)">
                    <span class="source-index">${i + 1}</span>
                    <span class="source-name" title="${s}">${s}</span>
                </div>
            `).join('');

            html += `
                <div class="sources-footer">
                    <div class="sources-footer-header">
                        <i class="fa-solid fa-book-open"></i>
                        <span>GROUNDED SOURCES</span>
                    </div>
                    <div class="sources-grid">${footerItems}</div>
                </div>
            `;
        }

        if (tasks.length > 0) {
            tasks.forEach(t => {
                const typeParts = t.type.split(':');
                const taskType = typeParts[0];
                const iconMap = { 'SUMMARY': 'fa-list-check', 'TRANSLATION': 'fa-language', 'REWRITE': 'fa-pen-nib' };
                const icon = iconMap[taskType] || 'fa-bolt-lightning';

                html += `
                <div class="task-card task-${taskType.toLowerCase()} mt-4">
                    <div class="card-header">
                        <div class="card-title-group">
                            <div class="task-icon"><i class="fa-solid ${icon}"></i></div>
                            <span class="card-title">${taskType}</span>
                        </div>
                        <button class="copy-btn" onclick="window.app.copyText(this)">Copy Result</button>
                    </div>
                    <div class="card-body">${t.content}</div>
                </div>`;
            });
        }

        return html;
    }

    renderGroundedUI(content, metadata) {
        // SOTA: Merge explicit sources from metadata with those auto-detected in content
        const metadataSources = (metadata && metadata.ui_hints && metadata.ui_hints.sources) || [];

        // Pass a proxy or pre-populate usedSources if we want to force them into the footer
        const bodyHtml = this.renderStructuredResponse(content, metadataSources);

        return `
            <div class="grounded-card">
                ${metadataSources.length > 0 ? `
                <div class="source-strip">
                    <div class="source-strip-label">Contextual Assets:</div>
                    ${metadataSources.map(s => `
                        <div class="source-pill" onclick="window.app.openArtifact('${s.replace(/'/g, "\\'")}')">
                            <i class="fa-solid fa-file-lines"></i>
                            <span>${s}</span>
                        </div>
                    `).join('')}
                </div>` : ''}
                <div class="grounded-body">
                    ${bodyHtml}
                </div>
            </div>
        `;
    }

    copyText(btn) {
        const body = btn.closest('.task-card').querySelector('.card-body');
        navigator.clipboard.writeText(body.innerText).then(() => {
            const old = btn.innerText;
            btn.innerText = 'Copied!';
            setTimeout(() => btn.innerText = old, 1500);
        });
    }

    // --- Query & Upload ---

    async handleQuery() {
        const query = this.userInput.value.trim();
        if (!query || this.state.isProcessing) return;

        const hasFiles = this.state.pendingFiles.length > 0;
        this.state.isProcessing = true;
        this.userInput.value = '';
        this.userInput.style.height = 'auto';

        this.appendMessage('user', query);
        this.showStopButton();

        if (hasFiles) {
            // SOTA: Auto-tagging for UI feedback
            const uploadedNames = this.state.pendingFiles.map(f => f.name);
            this.state.mentionedFiles = [...new Set([...this.state.mentionedFiles, ...uploadedNames])];

            // Re-append user message with metadata to show chips
            this.chatLog.lastElementChild.remove();
            this.appendMessage('user', query, { mentioned_files: this.state.mentionedFiles });

            this.showUnifiedOverlay();
            await this.handleUnifiedQuery(query);
            return;
        }

        this.showThinkingIndicator();
        this.showTelemetryHud();

        // STOP GENERATION: Create AbortController for this request
        const controller = new AbortController();
        this.state.abortController = controller;

        try {
            const response = await fetch('/query/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    conversation_id: this.state.activeConversation,
                    project_id: this.state.activeProject,
                    mentioned_files: this.state.mentionedFiles.length > 0 ? this.state.mentionedFiles : null,
                    use_web_search: this.state.webSearchEnabled
                }),
                signal: controller.signal
            });

            // Clear mention state after sending
            this.state.mentionedFiles = [];
            this.renderMentionChips();

            if (!response.ok) {
                let errorMsg = "Metacognitive Core connection failed";
                try {
                    const errorData = await response.json();
                    if (errorData.detail) errorMsg = errorData.detail;
                    else if (errorData.message) errorMsg = errorData.message;
                } catch (e) {
                    errorMsg += ` (${response.status} ${response.statusText})`;
                }
                throw new Error(errorMsg);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiBubble = null;
            let summaryBubbles = {};
            let currentSummaryBubble = null;

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));

                            // Capture conversation_id early (from any stage)
                            if (!this.state.activeConversation && data.conversation_id) {
                                console.log("Capturing early conversation_id:", data.conversation_id);
                                this.state.activeConversation = data.conversation_id;
                            }

                            if (data.stage === 'processing') {
                                this.updateTelemetry(data.agent, data.message);
                            } else if (data.type === 'thought') {
                                if (!aiBubble) aiBubble = this.createAiBubble();
                                aiBubble.appendThought(data.agent, data.action);
                            } else if (data.stage === 'streaming') {
                                this.removeThinkingIndicator();
                                if (!aiBubble) aiBubble = this.createAiBubble();
                                aiBubble.appendToken(data.token);


                            } else if (data.stage === 'terminated') {
                                // STOP GENERATION: Force-terminated by user
                                this.removeThinkingIndicator();
                                this.hideTelemetryHud();
                                if (!aiBubble) {
                                    aiBubble = this.createAiBubble(data.message, {
                                        agent_type: 'TERMINATED', confidence_score: 0
                                    });
                                } else {
                                    aiBubble.finalize(data.message, {
                                        agent_type: 'TERMINATED', confidence_score: 0
                                    });
                                }
                            } else if (data.stage === 'result') {
                                this.removeThinkingIndicator();
                                this.hideTelemetryHud();
                                if (data.ui_hints) this.applyAdaptiveResonance(data.ui_hints);

                                if (!aiBubble) aiBubble = this.createAiBubble(data.final_response, data);
                                else aiBubble.finalize(data.final_response, data);

                                if (!this.state.activeConversation && data.conversation_id) {
                                    this.state.activeConversation = data.conversation_id;
                                }
                                await this.loadWorkspace();
                            } else if (data.stage === 'error') {
                                this.appendMessage('ai', `**Reasoning Error**: ${data.message}`);
                            }
                        } catch (e) { console.error("Parse error in stream:", e, line); }
                    }
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                // STOP GENERATION: Fetch was aborted — the server-side abort handler
                // saves the terminated message; show it on the client side
                this.removeThinkingIndicator();
                this.hideTelemetryHud();
                this.appendMessage('ai', 'User Terminated the generation', {
                    intent: 'TERMINATED', confidence_score: 0
                });
            } else {
                console.error(err);
                this.appendMessage('ai', `**Critical Error**: ${err.message}`);
            }
        } finally {
            this.state.isProcessing = false;
            this.state.abortController = null;
            this.removeThinkingIndicator();
            this.hideTelemetryHud();
            this.hideStopButton();
        }
    }

    async handleUnifiedQuery(query) {
        // STOP GENERATION: Create AbortController for this request
        const controller = new AbortController();
        this.state.abortController = controller;

        try {
            const formData = new FormData();
            formData.append('query', query);
            formData.append('conversation_id', this.state.activeConversation || '');
            formData.append('project_id', this.state.activeProject || 'default');
            formData.append('use_web_search', this.state.webSearchEnabled.toString());

            this.state.pendingFiles.forEach(file => {
                formData.append('files', file);
            });

            const response = await fetch('/query/unified', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });

            if (!response.ok) {
                let errorMsg = "Unified Reasoning Bridge failed";
                try {
                    const errorData = await response.json();
                    if (errorData.detail) errorMsg = errorData.detail;
                    else if (errorData.message) errorMsg = errorData.message;
                } catch (e) {
                    // Fallback if not JSON
                    errorMsg += ` (${response.status} ${response.statusText})`;
                }
                throw new Error(errorMsg);
            }

            // Reset pending files
            this.state.pendingFiles = [];
            this.state.mentionedFiles = []; // SOTA Fix: Clear mentions to prevent bleed into next turn
            this.updateFileIndicator();
            this.renderFileChips();
            this.renderMentionChips();

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let aiBubble = null;
            let currentSummaryBubble = null;
            let summaryBubbles = {};

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6));

                            // Capture conversation_id early (from any stage)
                            if (!this.state.activeConversation && data.conversation_id) {
                                console.log("Capturing early conversation_id (unified):", data.conversation_id);
                                this.state.activeConversation = data.conversation_id;
                            }

                            if (data.stage === 'status') {
                                // SOTA Overlay Update
                                if (data.status === 'enriching') {
                                    const overlayText = document.querySelector('#unifiedOverlay p');
                                    if (overlayText) overlayText.innerText = data.message;
                                } else if (data.status === 'enrichment_complete') {
                                    // CRITICAL: Switch UI State
                                    this.hideUnifiedOverlay();
                                    this.showThinkingIndicator();
                                    this.showTelemetryHud();
                                    this.updateTelemetry('Ultima', 'Starting Reasoning Analysis...');
                                }

                            } else if (data.stage === 'workspace_updated') {
                                // SOTA: Files are ready in DB — update workspace NOW (before AI answers)
                                if (this.state.activeConversation || data.conversation_id) {
                                    await this.loadWorkspace();
                                    await this.loadWorkspaceFiles();
                                }

                            } else if (data.stage === 'processing') {
                                // Brain process telemetry
                                this.updateTelemetry(data.agent, data.message);

                            } else if (data.type === 'thought') {
                                if (!aiBubble) aiBubble = this.createAiBubble();
                                aiBubble.appendThought(data.agent, data.action);

                            } else if (data.stage === 'streaming') {
                                // Token-level streaming
                                this.hideUnifiedOverlay(); // Safety fallback
                                this.removeThinkingIndicator();
                                if (!aiBubble) aiBubble = this.createAiBubble();
                                aiBubble.appendToken(data.token);


                            } else if (data.stage === 'terminated') {
                                // STOP GENERATION: Force-terminated by user
                                this.hideUnifiedOverlay();
                                this.removeThinkingIndicator();
                                this.hideTelemetryHud();
                                if (!aiBubble) {
                                    aiBubble = this.createAiBubble(data.message, {
                                        agent_type: 'TERMINATED', confidence_score: 0
                                    });
                                } else {
                                    aiBubble.finalize(data.message, {
                                        agent_type: 'TERMINATED', confidence_score: 0
                                    });
                                }

                            } else if (data.stage === 'result') {
                                this.hideUnifiedOverlay();
                                this.removeThinkingIndicator();
                                this.hideTelemetryHud();

                                if (data.ui_hints) this.applyAdaptiveResonance(data.ui_hints);

                                if (!aiBubble) aiBubble = this.createAiBubble(data.final_response, data);
                                else aiBubble.finalize(data.final_response, data);

                                if (!this.state.activeConversation && data.conversation_id) {
                                    this.state.activeConversation = data.conversation_id;
                                }
                                await this.loadWorkspace();
                                await this.loadWorkspaceFiles();

                            } else if (data.stage === 'error') {
                                this.hideUnifiedOverlay();
                                this.appendMessage('ai', `**Unified Core Error**: ${data.message}`);
                            }
                        } catch (e) { console.error("Parse error in unified stream:", e, line); }
                    }
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') {
                this.hideUnifiedOverlay();
                this.removeThinkingIndicator();
                this.hideTelemetryHud();
                this.appendMessage('ai', 'User Terminated the generation', {
                    intent: 'TERMINATED', confidence_score: 0
                });
            } else {
                console.error(err);
                this.hideUnifiedOverlay();
                this.appendMessage('ai', `**Critical Error**: ${err.message}`);
            }
        } finally {
            this.state.isProcessing = false;
            this.state.abortController = null;
            this.removeThinkingIndicator();
            this.hideTelemetryHud();
            this.hideStopButton();
        }
    }

    createAiBubble(initialContent = "", metadata = null) {
        const wrapper = document.createElement('div');
        wrapper.className = `max-w-3xl mx-auto w-full flex flex-col items-start gap-1`;

        const thoughtProcessContainer = document.createElement('details');
        thoughtProcessContainer.className = 'thought-process-container w-full text-xs bg-[#1E293B]/60 backdrop-blur-md rounded-lg overflow-hidden border border-[#334155]/60 hidden transition-all duration-300';
        thoughtProcessContainer.innerHTML = `<summary class="p-2 cursor-pointer text-gray-400 hover:text-cyan-400 select-none flex items-center gap-2 font-mono"><i class="fa-solid fa-microchip pulse-icon"></i> <span>Cognitive Trace</span></summary><ul class="thought-list p-2 pt-0 text-gray-300 space-y-2 font-mono ml-4 border-l border-gray-600/50 pl-2"></ul>`;
        const thoughtList = thoughtProcessContainer.querySelector('.thought-list');

        const bubble = document.createElement('div');
        bubble.className = "bubble ai-bubble flex-1 w-full mt-1";

        let metaContainer = document.createElement('div');
        metaContainer.className = "resonance-container mb-2";

        wrapper.appendChild(thoughtProcessContainer);
        wrapper.appendChild(metaContainer);
        wrapper.appendChild(bubble);
        this.chatLog.appendChild(wrapper);

        let streamedText = "";

        const updateMetadata = (data) => {
            if (data) {
                let confPercent = 0;
                if (data.ui_hints && data.ui_hints.fidelity !== undefined) {
                    confPercent = data.ui_hints.fidelity;
                } else if (data.confidence_score !== undefined) {
                    confPercent = Math.round(data.confidence_score * 100);
                }

                const intent = data.intent || data.agent_type || (data.ui_hints ? data.ui_hints.intent : null) || 'Ultima';

                metaContainer.innerHTML = `
                    <div class="resonance-card">
                        <div class="intent-pill">${intent.toUpperCase()}</div>
                        <div class="text-[10px] font-bold opacity-40">Fidelity: ${confPercent}%</div>
                    </div>
                `;
            }
        };

        if (metadata) updateMetadata(metadata);

        const api = {
            appendThought: (agent, action) => {
                thoughtProcessContainer.classList.remove('hidden');
                thoughtProcessContainer.open = true; // Auto open while generating
                const li = document.createElement('li');
                li.className = "opacity-0 translate-y-2 animate-slide-up relative before:content-[''] before:absolute before:-left-[13px] before:top-2 before:w-[6px] before:h-[6px] before:rounded-full before:bg-cyan-500";
                li.innerHTML = `<span class="text-cyan-400 font-semibold mr-1">${agent}:</span> <span class="text-gray-300/90">${action}</span>`;
                thoughtList.appendChild(li);

                // Trigger reflow for animation
                void li.offsetWidth;
                li.classList.remove('opacity-0', 'translate-y-2');
                li.classList.add('opacity-100', 'translate-y-0');

                this.chatLog.scrollTop = this.chatLog.scrollHeight;
            },
            appendToken: (token) => {
                streamedText += token;
                // SOTA Phase 13: Incremental Streaming Parser
                // We render via StructuredResponse to enable real-time markdown/tags
                bubble.innerHTML = this.renderStructuredResponse(streamedText);
                this.chatLog.scrollTop = this.chatLog.scrollHeight;
            },
            finalize: (content, data) => {
                const finalContent = content || streamedText;
                if (finalContent && finalContent.trim()) {
                    bubble.innerHTML = this.renderStructuredResponse(finalContent);
                    // SOTA: Inject RAG Suggested Actions ONLY when:
                    //   – intent is RAG / evidence-based (not 'summarize')
                    //   – user had document context (@mention or uploaded file)
                    const intent = (data && (data.intent || (data.ui_hints && data.ui_hints.intent) || ''));
                    const isSummarize = (data && data.agent_type === 'SUMMARIZE') ||
                        intent.toLowerCase().includes('summarize') ||
                        intent.toLowerCase().includes('summary');
                    const hasDocContext = this.state.mentionedFiles.length > 0 ||
                        (this.state.activeConversation && this.state.workspaceFileCount > 0);
                    if (!isSummarize && hasDocContext) {
                        this.injectActionButtons(bubble);
                    }
                }
                // ── SOURCE EXPLORER: Attach per-response data to wrapper for openArtifact() ──
                if (data && data.retrieved_fragments && Object.keys(data.retrieved_fragments).length > 0) {
                    wrapper.dataset.retrievedFragments = JSON.stringify(data.retrieved_fragments);
                    wrapper.dataset.sourceMap = JSON.stringify(data.source_map || {});
                }
                // ────────────────────────────────────────────────────────────────
                updateMetadata(data);
                if (thoughtProcessContainer.open) {
                    thoughtProcessContainer.open = false; // Collapse upon finalization for a clean UI
                }
                this.chatLog.scrollTop = this.chatLog.scrollHeight;
            },
            // ── Phase 2: Generative UI — Mount the Risk Dashboard Widget ──
            mountRiskWidget: (data) => {
                bubble.innerHTML = this.renderRiskWidget(data);
                this.chatLog.scrollTop = this.chatLog.scrollHeight;
            },
            // ── Phase 3: Close cognitive trace (used after agentic actions) ──
            closeThoughts: () => {
                if (thoughtProcessContainer.open) {
                    thoughtProcessContainer.open = false;
                }
                this.chatLog.scrollTop = this.chatLog.scrollHeight;
            },
            // ── Phase 4: Expose the bubble DOM element for button injection ──
            getBubbleEl: () => bubble
        };


        if (initialContent) api.finalize(initialContent, metadata);
        return api;
    }

    copyCode(btn, code) {
        navigator.clipboard.writeText(code).then(() => {
            const span = btn.querySelector('span');
            const icon = btn.querySelector('i');
            const oldText = span.innerText;

            span.innerText = 'Copied!';
            icon.className = 'fa-solid fa-check text-green-400';

            setTimeout(() => {
                span.innerText = oldText;
                icon.className = 'fa-regular fa-copy';
            }, 2000);
        });
    }

    // ── Phase 1: Context-Aware Action Buttons ──────────────────────────
    // Each button fires a context-bound agentic payload instead of injecting text.

    injectActionButtons(container) {
        const bar = document.createElement('div');
        bar.className = 'action-injector-bar animate-slide-up';
        bar.dataset.agenticBar = 'static'; // Mark as static so Phase 4 can replace it
        bar.innerHTML = `
            <div class="action-bar-label"><i class="fa-solid fa-bolt-lightning text-cyan-500/70"></i> Agentic Actions</div>
            <div class="action-bar-btns flex flex-row flex-wrap gap-3 mt-2">
                <button class="action-btn" onclick="window.app.triggerAgenticAction('DEEP_INSIGHT', this.closest('[data-agentic-bar]'))">
                    <i class="fa-solid fa-microscope text-cyan-400"></i>
                    <span>Deep Insight</span>
                </button>
                <button class="action-btn" onclick="window.app.triggerAgenticAction('EXECUTIVE_SUMMARY', this.closest('[data-agentic-bar]'))">
                    <i class="fa-solid fa-file-invoice text-purple-400"></i>
                    <span>Executive Summary</span>
                </button>
                <button class="action-btn" onclick="window.app.triggerAgenticAction('RISK_ASSESSMENT', this.closest('[data-agentic-bar]'))">
                    <i class="fa-solid fa-shield-halved text-red-400"></i>
                    <span>Risk Assessment</span>
                </button>
            </div>
        `;
        container.appendChild(bar);
    }

    // ── Phase 4: Dynamic AI-Generated Follow-Up Buttons ───────────────
    injectDynamicActionButtons(container, nextActions) {
        // Remove any existing static action bars attached to this bubble
        const existingBar = container.querySelector('[data-agentic-bar]');
        if (existingBar) existingBar.remove();

        const bar = document.createElement('div');
        bar.className = 'action-injector-bar action-bar-dynamic animate-slide-up';
        bar.dataset.agenticBar = 'dynamic';

        const btnsHtml = nextActions.map((label, i) => `
            <button class="action-btn action-btn-dynamic" onclick="window.app.triggerSuggestedAction(${JSON.stringify(label)})">
                <i class="fa-solid fa-arrow-right text-emerald-400"></i>
                <span>${label}</span>
            </button>
        `).join('');

        bar.innerHTML = `
            <div class="action-bar-label action-bar-label-dynamic">
                <i class="fa-solid fa-wand-magic-sparkles text-emerald-400/80"></i> AI Suggested Next Steps
            </div>
            <div class="action-bar-btns flex flex-row flex-wrap gap-3 mt-2">${btnsHtml}</div>
        `;
        container.appendChild(bar);
    }

    triggerSuggestedAction(prompt) {
        this.userInput.value = prompt;
        this.userInput.focus();
        // Trigger input event to auto-resize textarea if applicable
        this.userInput.dispatchEvent(new Event('input', { bubbles: true }));
    }

    // ── Phase 1: Agentic Action Trigger ─────────────────────────────────
    async triggerAgenticAction(intentType, barEl) {
        if (this.state.isProcessing) return;
        if (!this.state.activeConversation) {
            this.appendMessage('ai', '**No active conversation.** Start a chat and upload documents first.');
            return;
        }

        // Collect documents visible in the workspace for this conversation
        const documentIds = this._getWorkspaceFileNames();
        if (documentIds.length === 0) {
            // Graceful fallback: still allow the action with empty doc_ids
            // The backend will attempt to retrieve all chunks from the conversation
            console.warn('[AgenticAction] No document IDs found — backend will use all conversation chunks.');
        }

        this.state.isProcessing = true;
        this.showStopButton();
        this.showThinkingIndicator();
        this.showTelemetryHud();

        const controller = new AbortController();
        this.state.abortController = controller;

        let aiBubble = null;
        let nextActionsReceived = [];
        let riskData = null;

        try {
            const response = await fetch('/query/agentic_action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    intent: intentType,
                    conversation_id: this.state.activeConversation,
                    project_id: this.state.activeProject || 'default',
                    document_ids: documentIds.length > 0 ? documentIds : null
                }),
                signal: controller.signal
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.detail || `HTTP ${response.status}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const data = JSON.parse(line.substring(6));

                        if (data.stage === 'initializing') {
                            this.updateTelemetry('Ultima', `${intentType.replace('_', ' ')} pipeline starting...`);

                        } else if (data.stage === 'agentic_user_msg') {
                            // Phase 1: Show the synthetic user message in chat
                            this.removeThinkingIndicator();
                            this.appendMessage('user', data.message);

                        } else if (data.stage === 'processing') {
                            this.updateTelemetry(data.agent || 'Ultima', data.message || 'Processing...');

                        } else if (data.type === 'thought') {
                            if (!aiBubble) aiBubble = this.createAiBubble();
                            aiBubble.appendThought(data.agent, data.action);

                        } else if (data.stage === 'streaming') {
                            this.removeThinkingIndicator();
                            if (!aiBubble) aiBubble = this.createAiBubble();
                            aiBubble.appendToken(data.token);

                        } else if (data.stage === 'json_chunk') {
                            // Phase 2: Generative UI — Risk Widget
                            this.removeThinkingIndicator();
                            if (!aiBubble) aiBubble = this.createAiBubble();
                            riskData = data.data;
                            aiBubble.mountRiskWidget(data.data);

                        } else if (data.stage === 'next_actions') {
                            // Phase 4: Proactive Next Best Actions received
                            nextActionsReceived = data.actions || [];

                        } else if (data.stage === 'result') {
                            this.removeThinkingIndicator();
                            this.hideTelemetryHud();

                            if (!aiBubble) {
                                aiBubble = this.createAiBubble(data.final_response, data);
                            } else {
                                // For JSON widget (risk), the bubble already has the widget mounted
                                // Only call finalize if we have textual content (not risk widget)
                                if (!riskData) {
                                    aiBubble.finalize(data.final_response, data);
                                } else {
                                    // Close the thought accordion
                                    aiBubble.closeThoughts();
                                }
                            }

                            // Phase 4: Inject dynamic or fallback static buttons
                            if (aiBubble && aiBubble.getBubbleEl) {
                                const bubbleEl = aiBubble.getBubbleEl();
                                if (nextActionsReceived.length >= 3) {
                                    this.injectDynamicActionButtons(bubbleEl, nextActionsReceived);
                                } else {
                                    // Fallback to static action buttons
                                    this.injectActionButtons(bubbleEl);
                                }
                            }

                            await this.loadWorkspace();

                        } else if (data.stage === 'terminated') {
                            this.removeThinkingIndicator();
                            this.hideTelemetryHud();
                            if (aiBubble) aiBubble.closeThoughts();
                            else this.appendMessage('ai', 'Agentic action terminated.');

                        } else if (data.stage === 'error') {
                            this.removeThinkingIndicator();
                            this.hideTelemetryHud();
                            this.appendMessage('ai', `**Agentic Pipeline Error**: ${data.message}`);
                        }

                    } catch (e) { console.error('[AgenticAction] Parse error:', e, line); }
                }
            }
        } catch (err) {
            if (err.name !== 'AbortError') {
                console.error('[AgenticAction] Error:', err);
                this.removeThinkingIndicator();
                this.appendMessage('ai', `**Agentic Action Failed**: ${err.message}`);
            } else {
                this.removeThinkingIndicator();
                this.appendMessage('ai', 'Agentic action cancelled.');
            }
        } finally {
            this.state.isProcessing = false;
            this.state.abortController = null;
            this.removeThinkingIndicator();
            this.hideTelemetryHud();
            this.hideStopButton();
        }
    }

    /** Helper: get file names currently in workspace for the active conversation */
    _getWorkspaceFileNames() {
        if (!this.workspaceFilesContainer) return [];
        const items = this.workspaceFilesContainer.querySelectorAll('.workspace-file-item .file-name');
        return Array.from(items).map(el => el.textContent.trim()).filter(Boolean);
    }

    // ── Phase 2: Generative Risk Widget Renderer ─────────────────────────
    renderRiskWidget(data) {
        const score = data.overall_score ?? 0;
        const level = (data.risk_level || 'LOW').toUpperCase();
        const risks = Array.isArray(data.risks) ? data.risks : [];
        const biases = Array.isArray(data.bias_flags) ? data.bias_flags : [];
        const confidence = data.confidence ?? 80;

        const severityColor = {
            CRITICAL: { bar: '#ef4444', badge: 'risk-severity-critical' },
            HIGH: { bar: '#f97316', badge: 'risk-severity-high' },
            MEDIUM: { bar: '#eab308', badge: 'risk-severity-medium' },
            LOW: { bar: '#22c55e', badge: 'risk-severity-low' }
        };
        const levelCols = severityColor[level] || severityColor.LOW;

        // Dial angle: 0-100 maps to -135deg to +135deg
        const dialAngle = -135 + (score / 100) * 270;

        const risksHtml = risks.map(r => {
            const sev = (r.severity || 'LOW').toUpperCase();
            const style = severityColor[sev] || severityColor.LOW;
            return `
                <div class="risk-item">
                    <div class="risk-item-header">
                        <span class="risk-severity-badge ${style.badge}">${sev}</span>
                        <span class="risk-item-title">${r.title || 'Unknown Risk'}</span>
                    </div>
                    <p class="risk-item-desc">${r.description || ''}</p>
                    ${r.mitigation ? `<div class="risk-item-mitigation"><i class="fa-solid fa-shield-check"></i> ${r.mitigation}</div>` : ''}
                </div>
            `;
        }).join('');

        const biasHtml = biases.length > 0 ? `
            <div class="risk-bias-section">
                <div class="risk-section-title"><i class="fa-solid fa-triangle-exclamation"></i> Detected Biases</div>
                ${biases.map(b => `<div class="risk-bias-item">${b}</div>`).join('')}
            </div>` : '';

        return `
            <div class="risk-widget animate-slide-up">
                <div class="risk-widget-header">
                    <div class="risk-dial-container">
                        <svg class="risk-dial-svg" viewBox="0 0 120 70" xmlns="http://www.w3.org/2000/svg">
                            <path d="M10 65 A50 50 0 0 1 110 65" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="10" stroke-linecap="round"/>
                            <path d="M10 65 A50 50 0 0 1 110 65" fill="none" stroke="${levelCols.bar}" stroke-width="10" stroke-linecap="round"
                                stroke-dasharray="157" stroke-dashoffset="${157 - (157 * score / 100)}"
                                style="filter: drop-shadow(0 0 6px ${levelCols.bar}88)"
                            />
                            <text x="60" y="58" text-anchor="middle" font-size="20" font-weight="700" fill="white">${score}</text>
                            <text x="60" y="68" text-anchor="middle" font-size="7" fill="rgba(255,255,255,0.5)" letter-spacing="1">RISK SCORE</text>
                        </svg>
                    </div>
                    <div class="risk-summary-block">
                        <div class="risk-level-badge risk-severity-badge ${levelCols.badge}" style="font-size:15px;padding:4px 14px">${level}</div>
                        <div class="risk-meta-row"><i class="fa-solid fa-bullseye"></i> <span>${risks.length} risk(s) identified</span></div>
                        <div class="risk-meta-row"><i class="fa-solid fa-circle-check"></i> <span>Confidence: ${confidence}%</span></div>
                    </div>
                </div>
                <div class="risk-section-title"><i class="fa-solid fa-triangle-exclamation text-red-400"></i> Risk Matrix</div>
                <div class="risk-items-container">${risksHtml}</div>
                ${biasHtml}
            </div>
        `;
    }

    // ── STOP GENERATION HELPERS ──────────────────────────────────────────

    async stopGeneration() {
        const convId = this.state.activeConversation;
        if (!convId) return;

        // 1. Signal the server to set the abort flag
        try {
            await fetch('/query/abort', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conversation_id: convId })
            });
        } catch (e) {
            console.error('Abort signal failed:', e);
        }

        // 2. Abort the client-side fetch stream
        if (this.state.abortController) {
            this.state.abortController.abort();
        }
    }

    showStopButton() {
        console.log("Setting UI State: Processing");
        if (this.actionButtonContainer) {
            this.actionButtonContainer.dataset.state = 'processing';
        }
    }

    hideStopButton() {
        console.log("Setting UI State: Idle");
        if (this.actionButtonContainer) {
            this.actionButtonContainer.dataset.state = 'idle';
        }
    }

    showUnifiedOverlay() {
        if (this.unifiedOverlay) this.unifiedOverlay.classList.remove('hidden');
    }

    hideUnifiedOverlay() {
        if (this.unifiedOverlay) this.unifiedOverlay.classList.add('hidden');
    }

    showThinkingIndicator() {
        if (!this.chatLog) return;
        const wrapper = document.createElement('div');
        wrapper.id = 'thinking-indicator';
        wrapper.className = "max-w-3xl mx-auto w-full flex flex-col items-start animate-slide-down";
        wrapper.innerHTML = `
            <div class="thinking-bubble">
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                <span class="text-[10px] font-bold opacity-40 uppercase tracking-widest ml-1">Metacognitive Core Processing...</span>
            </div>
        `;
        this.chatLog.appendChild(wrapper);
        this.chatLog.scrollTop = this.chatLog.scrollHeight;
    }

    removeThinkingIndicator() {
        const indicator = document.getElementById('thinking-indicator');
        if (indicator) indicator.remove();
    }

    showTelemetryHud() {
        if (this.telemetryHud) this.telemetryHud.classList.remove('hidden');
    }

    hideTelemetryHud() {
        if (this.telemetryHud) this.telemetryHud.classList.add('hidden');
    }

    updateTelemetry(agent, stage) {
        if (!this.telemetryHud) return;

        // SOTA Phase 25: Enhanced Telemetry Mapping
        const stageMap = {
            'indexing': '<i class="fa-solid fa-database animate-pulse mr-2"></i>INDEXING',
            'enriching': '<i class="fa-solid fa-microscope animate-bounce mr-2"></i>ENRICHING',
            'reasoning': '<i class="fa-solid fa-brain animate-pulse mr-2"></i>REASONING',
            'retrieving': '<i class="fa-solid fa-magnifying-glass mr-2"></i>RETRIEVING'
        };

        const displayStage = stageMap[stage.toLowerCase()] || stage.toUpperCase();

        if (this.telemetryAgent) this.telemetryAgent.innerHTML = `<span class="opacity-50">AGENT:</span> ${agent.toUpperCase()}`;
        if (this.telemetryStage) {
            this.telemetryStage.innerHTML = displayStage;
            this.telemetryStage.style.opacity = '1';
        }

        this.showTelemetryHud();
    }

    // =========================================================================
    // SOTA Phase 5: WebSocket UI Telemetry Loop
    // =========================================================================

    connectTelemetry() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/telemetry/ws`;

        console.log(`🔌 Initializing Global UI Telemetry connect: `, wsUrl);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log("🔌 Connected to Ultima SOTA Telemetry Socket");
            // Ping loop to keep connection alive
            this.pingInterval = setInterval(() => {
                if (this.ws.readyState === WebSocket.OPEN) this.ws.send("ping");
            }, 15000);
        };

        this.ws.onmessage = (event) => {
            try {
                if (event.data === "pong") return;
                const data = JSON.parse(event.data);

                if (data.type === "telemetry_start") {
                    this.updateTelemetry(data.agent, data.stage || "processing");
                }
                else if (data.type === "telemetry_end") {
                    // Hide the HUD if there are no more active agents running.
                    // Instead of full state tracking, we just hide it on end, 
                    // and allow the streaming HTTP response to show/hide it as well.
                    this.hideTelemetryHud();
                }
            } catch (e) { /* ignore parse errors for ping/pong */ }
        };

        this.ws.onclose = () => {
            console.warn("🔌 Telemetry Socket disconnected. Retrying in 5s.");
            clearInterval(this.pingInterval);
            setTimeout(() => this.connectTelemetry(), 5000);
        };

        this.ws.onerror = (err) => {
            console.error("🔌 Telemetry socket error", err);
        };
    }

    async updateSystemStatus() {
        try {
            const convId = this.state.activeConversation || '';
            const resp = await fetch(`/health?conversation_id=${convId}`);
            const data = await resp.json();
            if (this.statusDot) {
                this.statusDot.className = `w-2 h-2 rounded-full bg-${data.ready ? 'green' : 'yellow'}-500`;
                this.statusText.innerText = data.ready ? `System: Ready (${data.chunks_count} chunks)` : "System: No Knowledge Indexed";
            }
            if (this.dbDot) {
                this.dbDot.className = `w-2 h-2 rounded-full bg-${data.db_connected ? 'green' : 'red'}-500`;
                this.dbText.innerText = data.db_connected ? "Database: Connected" : "Database: Offline";
            }
        } catch (e) { console.error("Status Sync Error:", e); }
    }

    applyAdaptiveResonance(hints) {
        if (!hints) return;
        if (hints.theme_accent) document.documentElement.style.setProperty('--accent', hints.theme_accent);
        if (hints.status) {
            this.statusText.innerText = hints.status;
            this.statusDot.style.background = hints.theme_accent || 'var(--accent)';
        }
    }

    async renameConversation(cid, currentName) {
        const newName = prompt("Enter new title for this conversation:", currentName);
        if (!newName || newName === currentName) return;

        try {
            const resp = await fetch(`/conversations/${cid}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ title: newName })
            });
            const data = await resp.json();
            if (data.success) {
                if (this.state.activeConversation === cid) {
                    if (this.convTitle) this.convTitle.innerText = data.title;
                }
                await this.loadWorkspace();
            } else {
                alert("Failed to rename conversation: " + (data.detail || "Unknown error"));
            }
        } catch (e) { console.error("Rename Error:", e); }
    }

    async deleteConversation(cid) {
        if (!confirm("Are you sure you want to delete this conversation? All associated messages and cognitive assets will be purged.")) return;

        try {
            const resp = await fetch(`/conversations/${cid}`, {
                method: 'DELETE'
            });
            const data = await resp.json();
            if (data.success) {
                if (this.state.activeConversation === cid) {
                    this.state.activeConversation = null;
                    this.chatLog.innerHTML = "";
                    if (this.convTitle) this.convTitle.innerText = "New Inquiry";
                    // Optionally start a new chat or leave it empty
                }
                await this.loadWorkspace();
            } else {
                alert("Failed to delete conversation: " + (data.detail || "Unknown error"));
            }
        } catch (e) { console.error("Delete Error:", e); }
    }

    // ── Source Explorer ───────────────────────────────────────────────────────────────────

    async openArtifact(sourceName, triggerEl = null) {
        const explorer = document.getElementById('sourceExplorer');
        const content = document.getElementById('sourceContent');
        if (!explorer || !content) return;

        explorer.classList.remove('translate-x-full');
        content.innerHTML = `
            <div class="flex flex-col items-center justify-center h-64 gap-4">
                <div class="w-12 h-12 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin"></div>
                <div class="text-sm font-bold tracking-widest text-cyan-400 uppercase">Retrieving Artifact: ${sourceName}</div>
            </div>
        `;

        try {
            const convId = this.state.activeConversation || '';
            const ext = sourceName.split('.').pop().toLowerCase();
            const isImage = ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp'].includes(ext);
            const isVideo = ['mp4', 'mov', 'webm', 'avi', 'mkv'].includes(ext);
            const isAudio = ['mp3', 'wav', 'ogg', 'm4a', 'flac', 'aac'].includes(ext);
            const isPdf = ext === 'pdf';
            const isMedia = isImage || isVideo || isAudio;
            const isVisual = isImage || isVideo;

            const fileUrl = `/workspace/files/${convId}/view/${encodeURIComponent(sourceName)}`;
            const downloadUrl = `${fileUrl}?download=true`;

            // ── SOURCE EXPLORER: Attempt to resolve per-response fragments ──────
            // Walk up the DOM from triggerEl to find the message wrapper that
            // has data-retrieved-fragments attached by createAiBubble.finalize()
            let perResponseFragments = null;
            let isPerResponse = false;
            if (triggerEl) {
                const wrapper = triggerEl.closest
                    ? triggerEl.closest('[data-retrieved-fragments]')
                    : null;
                if (wrapper && wrapper.dataset.retrievedFragments) {
                    try {
                        const allFragments = JSON.parse(wrapper.dataset.retrievedFragments);
                        // Find the entry matching sourceName (case-insensitive fallback)
                        const matchKey = Object.keys(allFragments).find(k =>
                            k === sourceName || k.toLowerCase() === sourceName.toLowerCase()
                        );
                        if (matchKey) {
                            perResponseFragments = allFragments[matchKey];
                            isPerResponse = true;
                        }
                    } catch (_) { /* ignore JSON parse errors */ }
                }
            }
            // ────────────────────────────────────────────────────────────────────

            // Fetch full metadata only if we need it (non-media types or no per-response data)
            let data = { success: true, chunks: [], total_chunks: 0, type: 'unknown', metadata: null };
            if (!isPerResponse || !isMedia) {
                try {
                    const resp = await fetch(`/api/workspace/files/${encodeURIComponent(sourceName)}/details?conversation_id=${convId}`);
                    const fetched = await resp.json();
                    if (fetched.success) data = fetched;
                } catch (_) { /* non-fatal — media preview will still work */ }
            }

            // ── Media Preview Block ──
            let mediaHtml = '';
            if (isImage) {
                mediaHtml = `
                    <div class="flex items-center justify-center p-2 bg-black/60 rounded-xl min-h-[200px]">
                        <img src="${fileUrl}" alt="${sourceName}"
                             class="max-w-full max-h-[360px] rounded-lg object-contain cursor-zoom-in"
                             onclick="window.open(this.src, '_blank')" />
                    </div>`;
            } else if (isVideo) {
                mediaHtml = `
                    <div class="bg-black rounded-xl overflow-hidden">
                        <video controls class="w-full max-h-[300px] rounded-xl" preload="metadata">
                            <source src="${fileUrl}" type="video/${ext === 'mov' ? 'quicktime' : ext}">
                            <p class="text-white/40 p-4">Your browser doesn't support inline video. <a href="${downloadUrl}" class="text-cyan-400">Download it</a>.</p>
                        </video>
                    </div>`;
            } else if (isAudio) {
                mediaHtml = `
                    <div class="p-4 bg-white/5 rounded-xl">
                        <div class="flex items-center gap-3 mb-3">
                            <i class="fa-solid fa-waveform-lines text-cyan-400 text-2xl"></i>
                            <div class="text-sm font-bold text-white/80">${sourceName}</div>
                        </div>
                        <audio controls class="w-full" preload="metadata">
                            <source src="${fileUrl}" type="audio/${ext === 'm4a' ? 'mp4' : ext}">
                            <p class="text-white/40 text-xs">Your browser doesn't support inline audio. <a href="${downloadUrl}" class="text-cyan-400">Download it</a>.</p>
                        </audio>
                    </div>`;
            } else if (isPdf) {
                mediaHtml = `
                    <div class="rounded-xl overflow-hidden border border-white/10" style="height:480px">
                        <embed src="${fileUrl}" type="application/pdf" width="100%" height="100%" />
                    </div>
                    <a href="${fileUrl}" target="_blank" class="block text-center text-[11px] text-cyan-400 mt-2 hover:underline">
                        <i class="fa-solid fa-arrow-up-right-from-square mr-1"></i>Open PDF in new tab
                    </a>`;
            } else {
                // Generic doc — show vector identity info
                const fragCount = isPerResponse
                    ? (perResponseFragments ? perResponseFragments.length : 0)
                    : (data.total_chunks || 0);
                const fragLabel = isPerResponse
                    ? `${fragCount} retrieved fragment${fragCount !== 1 ? 's' : ''} used in response`
                    : `${fragCount} indexed fragment${fragCount !== 1 ? 's' : ''}`;
                mediaHtml = `
                    <div class="p-4 text-xs text-cyan-400/60 font-mono bg-cyan-950/20 border border-white/5 rounded-xl">
                        <i class="fa-solid fa-microscope mr-2"></i>Vector Identity: ${fragLabel}
                    </div>`;
            }

            // ── Fragment Display ──
            const fragments = isPerResponse ? (perResponseFragments || []) : (data.chunks || []);
            const fragmentLabel = isPerResponse ? 'Response Fragments' : 'Neural Content Fragments';
            const SHOW_MORE_THRESHOLD = 600; // chars

            let chunksHtml = '';
            if (fragments.length > 0) {
                chunksHtml = fragments.map((chunk, idx) => {
                    const text = chunk.text || chunk.content || '';
                    const score = chunk.score != null ? `<span class="ml-2 text-cyan-400/60">${(chunk.score * 100).toFixed(1)}% match</span>` : '';
                    const isLong = text.length > SHOW_MORE_THRESHOLD;
                    const truncated = isLong ? text.slice(0, SHOW_MORE_THRESHOLD) + '…' : text;
                    const rendered = window.marked ? marked.parse(truncated) : truncated;
                    const fullRendered = isLong ? (window.marked ? marked.parse(text) : text) : null;
                    const showMoreId = `frag-${idx}-${Date.now()}`;
                    return `
                    <div class="mb-4 p-4 bg-white/5 rounded-xl border border-white/5 hover:border-cyan-500/20 transition-all text-xs leading-relaxed">
                        <div class="flex items-center gap-2 mb-2 opacity-40 font-bold uppercase tracking-widest text-[9px]">
                            <span class="w-1.5 h-1.5 rounded-full bg-cyan-500"></span> Fragment ${idx + 1}${score}
                        </div>
                        <div id="${showMoreId}-short">${rendered}</div>
                        ${isLong ? `
                        <div id="${showMoreId}-full" style="display:none">${fullRendered}</div>
                        <button class="mt-2 text-[10px] text-cyan-400/70 hover:text-cyan-300 transition-colors"
                            onclick="(function(b){const s=document.getElementById('${showMoreId}-short');const f=document.getElementById('${showMoreId}-full');const show=s.style.display!=='none';s.style.display=show?'none':'';f.style.display=show?'':'none';b.textContent=show?'Show less':'Show more';})(this)">
                            Show more
                        </button>` : ''}
                    </div>`;
                }).join('');
            } else {
                chunksHtml = `<div class="opacity-30 italic text-xs py-4 text-center">No textual fragments extracted for this asset.</div>`;
            }

            content.innerHTML = `
                <div class="artifact-stage animate-slide-down">
                    <div class="mb-6 pb-4 border-b border-white/10 flex justify-between items-center">
                        <div>
                            <h1 class="text-2xl font-black text-white tracking-tight break-all">${sourceName}</h1>
                            <div class="text-[10px] uppercase tracking-widest text-white/40 font-bold mt-1">
                                ${isVisual ? '<i class="fa-solid fa-eye-low-vision text-cyan-400"></i> Visual Artifact' :
                    isAudio ? '<i class="fa-solid fa-waveform-lines text-purple-400"></i> Audio Artifact' :
                        isPdf ? '<i class="fa-solid fa-file-pdf text-red-400"></i> Document' :
                            '<i class="fa-solid fa-file-shield text-emerald-400"></i> Grounded Context'}
                            </div>
                        </div>
                        <div class="flex items-center gap-2">
                            <a href="${downloadUrl}" title="Download" class="p-2 hover:bg-white/10 rounded-full transition-colors text-white/60 hover:text-white">
                                <i class="fa-solid fa-download text-sm"></i>
                            </a>
                            <button onclick="document.getElementById('sourceExplorer').classList.add('translate-x-full')" class="p-2 hover:bg-white/10 rounded-full transition-colors">
                                <i class="fa-solid fa-xmark"></i>
                            </button>
                        </div>
                    </div>

                    <!-- Media / Preview Block -->
                    <div class="artifact-preview mb-6">
                        ${mediaHtml}
                    </div>

                    <!-- Neural Content Fragments -->
                    <div class="artifact-fragments">
                        <div class="text-[10px] uppercase font-bold tracking-widest text-white/30 mb-4 flex items-center gap-2">
                            <i class="fa-solid fa-brain text-[8px]"></i> ${fragmentLabel}
                            ${isPerResponse ? '<span class="ml-2 px-1.5 py-0.5 bg-cyan-500/20 text-cyan-300 rounded text-[9px] normal-case font-normal">This response</span>' : ''}
                        </div>
                        ${chunksHtml}
                    </div>

                    <div class="mt-8 pt-6 border-t border-white/5">
                        <div class="text-[10px] uppercase font-bold tracking-widest text-white/30 mb-4">Internal Meta-Data</div>
                        <div class="space-y-2">
                            <div class="flex justify-between text-[11px] font-mono"><span class="opacity-40">Status</span><span class="text-green-400">Grounded</span></div>
                            <div class="flex justify-between text-[11px] font-mono"><span class="opacity-40">Persistence</span><span class="text-cyan-400">Converged</span></div>
                            <div class="flex justify-between text-[11px] font-mono"><span class="opacity-40">Type</span><span class="text-white">${data.type || ext.toUpperCase() || 'Unknown'}</span></div>
                            ${isPerResponse ? `<div class="flex justify-between text-[11px] font-mono"><span class="opacity-40">Fragment Scope</span><span class="text-cyan-400">Per-Response</span></div>` : ''}
                        </div>
                    </div>
                </div>
            `;
        } catch (e) {
            console.error("Artifact Error:", e);
            content.innerHTML = `<div class="text-red-400 p-8 border border-red-500/20 bg-red-500/5 rounded-xl">Error staging artifact: ${e.message}</div>`;
        }
    }


    // Deprecated: kept in case referenced from elsewhere
    async triggerPivotSearch(sourceName) {
        if (this.state.isProcessing) return;

        // Visual UX: slide sidebar out and populate prompt
        document.getElementById('sourceExplorer').classList.add('translate-x-full');
        this.userInput.value = `@${sourceName} (Pivot Search: Identify semantically similar patterns across this document's vector space)`;

        // Auto-tag if not already tagged
        if (!this.state.mentionedFiles.includes(sourceName)) {
            this.state.mentionedFiles.push(sourceName);
            this.renderMentionChips();
        }

        this.handleQuery();
    }

    async exportFileContext(sourceName) {
        this.updateStatus(`Generating Evidence Report for ${sourceName}...`, 'process');

        try {
            const resp = await fetch(`/api/workspace/files/${encodeURIComponent(sourceName)}/export`);
            if (!resp.ok) throw new Error("Export generation failed");

            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `Ultima_Evidence_${sourceName.replace(/\.[^/.]+$/, "")}.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);

            this.updateStatus('Report Exported', 'idle');
        } catch (e) {
            console.error("Export Error:", e);
            this.appendMessage('ai', `**Export Error**: Could not generate evidence report for ${sourceName}`);
            this.updateStatus('Export Failed', 'idle');
        }
    }

    updateStatus(message, state = 'idle') {
        if (this.statusText) this.statusText.innerText = `System: ${message}`;
        if (state === 'process') {
            this.showTelemetryHud();
        } else {
            this.hideTelemetryHud();
        }
    }

    // =========================================================================
    // @MENTION SYSTEM
    // =========================================================================

    detectMentionTrigger() {
        const text = this.userInput.value;
        const cursorPos = this.userInput.selectionStart;

        // Look backwards from cursor for an @ symbol
        const beforeCursor = text.substring(0, cursorPos);
        const lastAt = beforeCursor.lastIndexOf('@');

        if (lastAt === -1) {
            this.hideMentionDropdown();
            return;
        }

        // Check that the @ is at start or preceded by whitespace
        if (lastAt > 0 && !/\s/.test(beforeCursor[lastAt - 1])) {
            this.hideMentionDropdown();
            return;
        }

        const prefix = beforeCursor.substring(lastAt + 1);
        // If there's a space after the filename-like text, stop autocomplete
        if (prefix.includes(' ') && prefix.indexOf('.') < prefix.lastIndexOf(' ')) {
            this.hideMentionDropdown();
            return;
        }

        this.state.mentionSearch = prefix;
        this.fetchMentionSuggestions(prefix);
    }

    async fetchMentionSuggestions(prefix) {
        if (!this.state.activeConversation) {
            this.hideMentionDropdown();
            return;
        }

        try {
            const resp = await fetch(
                `/api/workspace/files/autocomplete?conversation_id=${this.state.activeConversation}&prefix=${encodeURIComponent(prefix)}`
            );
            const data = await resp.json();

            if (data.success && data.suggestions.length > 0) {
                this.renderMentionSuggestions(data.suggestions);
            } else {
                this.hideMentionDropdown();
            }
        } catch (e) {
            console.error('Mention autocomplete error:', e);
            this.hideMentionDropdown();
        }
    }

    renderMentionSuggestions(suggestions) {
        if (!this.mentionSuggestions || !this.mentionDropdown) return;

        const typeIcons = {
            image: 'fa-image text-cyan-400',
            video: 'fa-film text-purple-400',
            audio: 'fa-headphones text-emerald-400',
            pdf: 'fa-file-pdf text-red-400',
            document: 'fa-file-alt text-blue-400',
            file: 'fa-file text-slate-400'
        };

        let html = '';
        suggestions.forEach((s, idx) => {
            const ext = s.name.split('.').pop().toLowerCase();
            let type = s.type || 'file';
            if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext)) type = 'image';
            else if (['mp4', 'mov', 'webm'].includes(ext)) type = 'video';
            else if (['mp3', 'wav', 'ogg'].includes(ext)) type = 'audio';
            else if (ext === 'pdf') type = 'pdf';

            const icon = typeIcons[type] || typeIcons.file;

            html += `
                <div class="mention-suggestion px-4 py-2.5 flex items-center gap-3 cursor-pointer hover:bg-white/10 transition-colors ${idx === 0 ? 'bg-white/5' : ''}"
                     onclick="window.app.selectMention('${s.name}')">
                    <i class="fa-solid ${icon} text-sm w-5 text-center"></i>
                    <span class="text-sm font-medium text-white/90 truncate">${s.name}</span>
                </div>
            `;
        });

        this.mentionSuggestions.innerHTML = html;
        this.mentionDropdown.classList.remove('hidden');
    }

    hideMentionDropdown() {
        if (this.mentionDropdown) {
            this.mentionDropdown.classList.add('hidden');
        }
    }

    selectMention(fileName) {
        // Replace the @prefix in textarea with @filename
        const text = this.userInput.value;
        const cursorPos = this.userInput.selectionStart;
        const beforeCursor = text.substring(0, cursorPos);
        const lastAt = beforeCursor.lastIndexOf('@');
        const afterCursor = text.substring(cursorPos);

        if (lastAt >= 0) {
            this.userInput.value = beforeCursor.substring(0, lastAt) + '@' + fileName + ' ' + afterCursor;
        }

        // Add to mentioned files (dedupe)
        if (!this.state.mentionedFiles.includes(fileName)) {
            this.state.mentionedFiles.push(fileName);
        }

        this.hideMentionDropdown();
        this.renderMentionChips();
        this.userInput.focus();
    }

    removeMention(fileName) {
        this.state.mentionedFiles = this.state.mentionedFiles.filter(f => f !== fileName);
        // Also remove from textarea
        this.userInput.value = this.userInput.value.replace(`@${fileName}`, '').replace(/  +/g, ' ').trim();
        this.renderMentionChips();
    }

    renderMentionChips() {
        if (!this.mentionChips) return;

        if (this.state.mentionedFiles.length === 0) {
            this.mentionChips.classList.add('hidden');
            this.mentionChips.innerHTML = '';
            return;
        }

        const chipColors = [
            'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
            'bg-purple-500/20 text-purple-300 border-purple-500/30',
            'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
            'bg-amber-500/20 text-amber-300 border-amber-500/30',
            'bg-rose-500/20 text-rose-300 border-rose-500/30'
        ];

        let html = '<span class="text-[10px] uppercase tracking-widest font-bold opacity-40 mr-1">Targeting:</span>';
        this.state.mentionedFiles.forEach((f, i) => {
            const color = chipColors[i % chipColors.length];
            html += `
                <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${color}">
                    <i class="fa-solid fa-at text-[9px]"></i>
                    ${f}
                    <button onclick="window.app.removeMention('${f}')"
                            class="ml-1 opacity-60 hover:opacity-100 transition-opacity">
                        <i class="fa-solid fa-xmark text-[9px]"></i>
                    </button>
                </span>
            `;
        });

        this.mentionChips.innerHTML = html;
        this.mentionChips.classList.remove('hidden');
    }

    // =========================================================================
    // PDF EXPORT  (2-mode: Full Conversation | Query-Based)
    // =========================================================================

    togglePdfMenu() {
        // Show the new 2-mode PDF modal instead of the old dropdown
        this.openPdfExportModal();
    }

    openPdfExportModal() {
        // Remove any existing modal
        const old = document.getElementById('pdfExportOverlay');
        if (old) old.remove();

        if (!this.state.activeConversation) {
            this.appendMessage('ai', '**Export Error**: Please open a conversation first.');
            return;
        }

        const overlay = document.createElement('div');
        overlay.id = 'pdfExportOverlay';
        overlay.style.cssText = `
            position:fixed; inset:0; z-index:9999;
            background:rgba(0,0,0,0.7); backdrop-filter:blur(6px);
            display:flex; align-items:center; justify-content:center;
        `;

        overlay.innerHTML = `
            <div id="pdfExportModal" style="
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border: 1px solid rgba(6,182,212,0.25);
                border-radius: 20px;
                padding: 32px;
                width: 520px;
                max-width: 95vw;
                max-height: 85vh;
                overflow-y: auto;
                box-shadow: 0 25px 60px rgba(0,0,0,0.8);
                font-family: inherit;
                color: #e2e8f0;
            ">
                <!-- Header -->
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:24px;">
                    <div>
                        <div style="font-size:10px; letter-spacing:0.15em; color:#06b6d4; font-weight:700; text-transform:uppercase; margin-bottom:4px;">
                            <i class="fa-solid fa-file-pdf"></i> Intelligence Export
                        </div>
                        <div style="font-size:18px; font-weight:800; color:#f8fafc;">Select Export Mode</div>
                    </div>
                    <button id="pdfModalClose" style="
                        width:36px; height:36px; border-radius:50%; background:rgba(255,255,255,0.05);
                        border:1px solid rgba(255,255,255,0.1); color:#94a3b8; cursor:pointer;
                        display:flex; align-items:center; justify-content:center; font-size:14px;
                        transition:all 0.2s;
                    " onmouseenter="this.style.background='rgba(255,255,255,0.1)'"
                       onmouseleave="this.style.background='rgba(255,255,255,0.05)'">
                        <i class="fa-solid fa-xmark"></i>
                    </button>
                </div>

                <!-- Mode Selection Tabs -->
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:28px;" id="pdfModeCards">
                    <button id="pdfModeFullBtn" onclick="window.app._selectPdfMode('full')" style="
                        padding:20px 16px; border-radius:14px;
                        background:rgba(6,182,212,0.15); border:1.5px solid rgba(6,182,212,0.5);
                        color:#e2e8f0; cursor:pointer; text-align:left; transition:all 0.2s;
                    ">
                        <i class="fa-solid fa-layer-group" style="color:#06b6d4; font-size:18px; margin-bottom:8px; display:block;"></i>
                        <div style="font-weight:700; font-size:13px; margin-bottom:4px;">Full Conversation</div>
                        <div style="font-size:10px; color:#94a3b8; line-height:1.4;">Export the entire conversation history as a structured PDF.</div>
                    </button>
                    <button id="pdfModeQueryBtn" onclick="window.app._selectPdfMode('query')" style="
                        padding:20px 16px; border-radius:14px;
                        background:rgba(255,255,255,0.03); border:1.5px solid rgba(255,255,255,0.1);
                        color:#e2e8f0; cursor:pointer; text-align:left; transition:all 0.2s;
                    ">
                        <i class="fa-solid fa-magnifying-glass-chart" style="color:#a855f7; font-size:18px; margin-bottom:8px; display:block;"></i>
                        <div style="font-weight:700; font-size:13px; margin-bottom:4px;">Query-Based</div>
                        <div style="font-size:10px; color:#94a3b8; line-height:1.4;">Ask a question — AI generates a response using your files. Preview before downloading.</div>
                    </button>
                </div>

                <!-- FULL CONVERSATION PANEL -->
                <div id="pdfPanelFull" style="display:block;">
                    <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:16px; margin-bottom:20px; font-size:12px; color:#94a3b8; line-height:1.6;">
                        <i class="fa-solid fa-circle-info text-cyan-400 mr-2"></i>
                        Exports <strong style="color:#e2e8f0;">all messages</strong> in this conversation, including user queries and AI responses, with source metadata.
                        Supports <strong style="color:#e2e8f0;">Hindi, English</strong> and all Devanagari scripts via Unicode font.
                    </div>
                    <button onclick="window.app._downloadFullConversationPdf()" style="
                        width:100%; padding:14px; border-radius:12px;
                        background:linear-gradient(135deg, #06b6d4, #0284c7);
                        border:none; color:white; font-weight:700; font-size:13px;
                        cursor:pointer; letter-spacing:0.05em; transition:opacity 0.2s;
                    " onmouseenter="this.style.opacity='0.85'" onmouseleave="this.style.opacity='1'">
                        <i class="fa-solid fa-file-arrow-down mr-2"></i> Download Full Conversation PDF
                    </button>
                </div>

                <!-- QUERY-BASED PANEL -->
                <div id="pdfPanelQuery" style="display:none;">
                    <div style="margin-bottom:16px;">
                        <label style="font-size:10px; font-weight:700; letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase; display:block; margin-bottom:8px;">
                            Your Query <span style="color:#06b6d4;">(any language)</span>
                        </label>
                        <textarea id="pdfQueryInput" rows="4" placeholder="e.g. Summarize the key findings from cats.txt in Hindi..." style="
                            width:100%; padding:12px 14px; border-radius:10px;
                            background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.12);
                            color:#f1f5f9; font-size:13px; line-height:1.6; resize:vertical;
                            font-family:inherit; outline:none; box-sizing:border-box;
                        " onfocus="this.style.borderColor='rgba(6,182,212,0.5)'" onblur="this.style.borderColor='rgba(255,255,255,0.12)'"></textarea>
                    </div>

                    <!-- File mentions -->
                    <div style="margin-bottom:16px; font-size:11px; color:#64748b;">
                        <i class="fa-solid fa-link mr-1"></i>
                        Tagged files: <span id="pdfMentionedFiles" style="color:#06b6d4;">
                            ${this.state.mentionedFiles.length > 0
                ? this.state.mentionedFiles.join(', ')
                : (this.state.workspaceFileCount > 0 ? 'All workspace files' : 'None — upload a file first')}
                        </span>
                    </div>

                    <button id="pdfGenerateBtn" onclick="window.app._generateQueryPdf()" style="
                        width:100%; padding:14px; border-radius:12px;
                        background:linear-gradient(135deg, #a855f7, #7c3aed);
                        border:none; color:white; font-weight:700; font-size:13px;
                        cursor:pointer; letter-spacing:0.05em; transition:opacity 0.2s;
                    " onmouseenter="this.style.opacity='0.85'" onmouseleave="this.style.opacity='1'">
                        <i class="fa-solid fa-wand-magic-sparkles mr-2"></i> Generate AI Response
                    </button>

                    <!-- Preview Area (hidden until response arrives) -->
                    <div id="pdfPreviewArea" style="display:none; margin-top:20px;">
                        <div style="font-size:10px; font-weight:700; letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase; margin-bottom:10px;">
                            <i class="fa-solid fa-eye mr-1"></i> AI Response Preview
                        </div>
                        <div id="pdfPreviewText" style="
                            background:rgba(255,255,255,0.04); border:1px solid rgba(6,182,212,0.2);
                            border-radius:12px; padding:16px; font-size:12px; line-height:1.8;
                            color:#e2e8f0; max-height:260px; overflow-y:auto;
                            white-space:pre-wrap; word-break:break-word;
                        "></div>
                        <div style="margin-top:12px; text-align:right;">
                            <button id="pdfDownloadQueryBtn" onclick="window.app._downloadQueryPdf()" style="
                                padding:12px 24px; border-radius:10px;
                                background:linear-gradient(135deg, #10b981, #059669);
                                border:none; color:white; font-weight:700; font-size:13px;
                                cursor:pointer; transition:opacity 0.2s;
                            " onmouseenter="this.style.opacity='0.85'" onmouseleave="this.style.opacity='1'">
                                <i class="fa-solid fa-file-arrow-down mr-2"></i> Download as PDF
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Status Line -->
                <div id="pdfStatusLine" style="margin-top:16px; font-size:11px; color:#64748b; min-height:16px; text-align:center;"></div>
            </div>
        `;

        document.body.appendChild(overlay);

        // Close on backdrop click
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.remove();
        });
        document.getElementById('pdfModalClose').addEventListener('click', () => overlay.remove());

        // Default mode: full
        this._selectPdfMode('full');

        // Store session token for query-based download
        this._pdfSessionToken = null;
    }

    _selectPdfMode(mode) {
        const fullPanel = document.getElementById('pdfPanelFull');
        const queryPanel = document.getElementById('pdfPanelQuery');
        const fullBtn = document.getElementById('pdfModeFullBtn');
        const queryBtn = document.getElementById('pdfModeQueryBtn');
        if (!fullPanel) return;

        if (mode === 'full') {
            fullPanel.style.display = 'block';
            queryPanel.style.display = 'none';
            fullBtn.style.background = 'rgba(6,182,212,0.15)';
            fullBtn.style.border = '1.5px solid rgba(6,182,212,0.5)';
            queryBtn.style.background = 'rgba(255,255,255,0.03)';
            queryBtn.style.border = '1.5px solid rgba(255,255,255,0.1)';
        } else {
            fullPanel.style.display = 'none';
            queryPanel.style.display = 'block';
            queryBtn.style.background = 'rgba(168,85,247,0.15)';
            queryBtn.style.border = '1.5px solid rgba(168,85,247,0.5)';
            fullBtn.style.background = 'rgba(255,255,255,0.03)';
            fullBtn.style.border = '1.5px solid rgba(255,255,255,0.1)';
        }
    }

    async _downloadFullConversationPdf() {
        const status = document.getElementById('pdfStatusLine');
        if (status) status.textContent = 'Generating Full Conversation PDF…';

        try {
            const resp = await fetch(`/api/conversations/${this.state.activeConversation}/export/pdf?scope=full`, {
                method: 'POST'
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || 'Export failed');
            }
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const cd = resp.headers.get('Content-Disposition');
            const match = cd ? cd.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/) : null;
            a.download = match ? match[1].replace(/['"]/g, '') : 'UltimaRAG_Conversation.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            if (status) status.textContent = '✅ Download started!';
            setTimeout(() => { const o = document.getElementById('pdfExportOverlay'); if (o) o.remove(); }, 1500);
        } catch (e) {
            if (status) status.textContent = `❌ Error: ${e.message}`;
            console.error('Full PDF export error:', e);
        }
    }

    async _generateQueryPdf() {
        const queryInput = document.getElementById('pdfQueryInput');
        const status = document.getElementById('pdfStatusLine');
        const previewArea = document.getElementById('pdfPreviewArea');
        const previewText = document.getElementById('pdfPreviewText');
        const generateBtn = document.getElementById('pdfGenerateBtn');

        const query = queryInput ? queryInput.value.trim() : '';
        if (!query) {
            if (status) status.textContent = '⚠️ Please enter a query first.';
            return;
        }

        if (generateBtn) generateBtn.disabled = true;
        if (status) status.textContent = '⏳ Generating AI response via RAG pipeline…';
        if (previewArea) previewArea.style.display = 'none';

        try {
            const formData = new FormData();
            formData.append('query', query);
            formData.append('conversation_id', this.state.activeConversation);
            // Pass tagged files if any
            const files = this.state.mentionedFiles.length > 0
                ? JSON.stringify(this.state.mentionedFiles)
                : '[]';
            formData.append('mentioned_files', files);

            const resp = await fetch('/api/pdf/query-generate', {
                method: 'POST',
                body: formData
            });

            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || 'Generation failed');
            }

            const data = await resp.json();
            this._pdfSessionToken = data.session_token;

            // Show preview
            if (previewText) previewText.textContent = data.response_text;
            if (previewArea) previewArea.style.display = 'block';
            if (status) status.textContent = '✅ Response generated. Read, review, then download.';
        } catch (e) {
            if (status) status.textContent = `❌ ${e.message}`;
            console.error('PDF query generate error:', e);
        } finally {
            if (generateBtn) generateBtn.disabled = false;
        }
    }

    async _downloadQueryPdf() {
        if (!this._pdfSessionToken) {
            const status = document.getElementById('pdfStatusLine');
            if (status) status.textContent = '⚠️ No generated response found. Please generate first.';
            return;
        }

        const status = document.getElementById('pdfStatusLine');
        const btn = document.getElementById('pdfDownloadQueryBtn');
        if (status) status.textContent = '⏳ Building PDF…';
        if (btn) btn.disabled = true;

        try {
            const resp = await fetch(`/api/pdf/download/${this._pdfSessionToken}`);
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                throw new Error(err.detail || 'Download failed');
            }

            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            const cd = resp.headers.get('Content-Disposition');
            const match = cd ? cd.match(/filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/) : null;
            a.download = match ? match[1].replace(/['"]/g, '') : 'UltimaRAG_QueryExport.pdf';
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
            if (status) status.textContent = '✅ PDF downloaded!';
        } catch (e) {
            if (status) status.textContent = `❌ ${e.message}`;
            console.error('PDF download error:', e);
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    async exportConversationPdf(scope = 'full') {
        // Legacy compatibility shim — maps old scope calls to new modal
        if (scope === 'full' || scope === 'latest') {
            await this._downloadFullConversationPdf();
        } else {
            this.openPdfExportModal();
        }
    }

    async nukeConversations() {
        const password = prompt("ADMINISTRATIVE ACTION REQUIRED\nEnter the master reset password to purge all data:");
        if (!password) return;

        if (!confirm("CRITICAL WARNING: This will permanently delete ALL conversations, ALL uploaded files, and ALL vector databases. This action cannot be undone. Proceed?")) {
            return;
        }

        // --- SOTA Phase 5: System Purge Visuals (Inline Styles to bypass Cache) ---
        // Ensure old style block is removed if it exists
        const oldStyle = document.getElementById('nuke-dynamic-styles');
        if (oldStyle) oldStyle.remove();

        const style = document.createElement('style');
        style.id = 'nuke-dynamic-styles';
        style.innerHTML = `
            #nuke-overlay {
                position: fixed; inset: 0; background: rgba(5, 5, 10, 0.95); backdrop-filter: blur(24px); 
                z-index: 99999; display: flex; flex-direction: column; align-items: center; justify-content: center;
                color: #ef4444; font-family: 'Inter', monospace; opacity: 0; transition: opacity 0.5s;
            }
            #nuke-overlay.active { opacity: 1; }
            .nuke-radar-container {
                position: relative; width: 220px; height: 220px; border-radius: 50%;
                border: 2px solid rgba(239, 68, 68, 0.2); box-shadow: 0 0 50px rgba(239, 68, 68, 0.1);
                display: flex; align-items: center; justify-content: center; margin-bottom: 40px;
            }
            .nuke-radar-sweep {
                position: absolute; inset: 0; border-radius: 50%;
                background: conic-gradient(rgba(239, 68, 68, 0.6) 0deg, transparent 70deg);
                animation: radar-spin 1.5s linear infinite;
            }
            .nuke-radar-circle {
                position: absolute; width: 100%; height: 100%; border-radius: 50%; 
                border: 2px dashed rgba(239, 68, 68, 0.5);
                animation: pulse-ring 2s cubic-bezier(0.1, 0.8, 0.3, 1) infinite;
            }
            .nuke-icon-center {
                font-size: 4rem; z-index: 2; text-shadow: 0 0 30px #ef4444;
                animation: shake-icon 0.5s infinite;
            }
            .nuke-text-glitch {
                font-size: 2.2rem; font-weight: 900; letter-spacing: 6px; text-transform: uppercase;
                text-shadow: 0 0 15px rgba(239, 68, 68, 0.8); margin-bottom: 15px; text-align: center;
            }
            .nuke-subtext {
                color: #fca5a5; font-size: 1rem; letter-spacing: 3px; font-weight: 600;
                animation: blink-text 1s infinite alternate; text-align: center;
            }
            .nuke-data-stream {
                position: absolute; right: 40px; top: 40px; font-size: 0.75rem; color: #ef4444; opacity: 0.6;
                text-align: right; line-height: 1.8; font-family: monospace; letter-spacing: 1px;
            }
            @keyframes radar-spin { 100% { transform: rotate(360deg); } }
            @keyframes pulse-ring { 0% { transform: scale(0.6); opacity: 1; } 100% { transform: scale(1.6); opacity: 0; } }
            @keyframes shake-icon { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
            @keyframes blink-text { from { opacity: 0.3; } to { opacity: 1; } }
            
            /* Success State */
            #nuke-overlay.success-state { color: #10b981; }
            #nuke-overlay.success-state .nuke-radar-sweep { display: none; }
            #nuke-overlay.success-state .nuke-radar-container { border-color: #10b981; box-shadow: 0 0 50px rgba(16, 185, 129, 0.2); }
            #nuke-overlay.success-state .nuke-radar-circle { border: 2px solid rgba(16, 185, 129, 0.5); animation: none; transform: scale(1.1); }
            #nuke-overlay.success-state .nuke-icon-center { color: #10b981; text-shadow: 0 0 30px #10b981; animation: none; }
            #nuke-overlay.success-state .nuke-text-glitch { color: #10b981; text-shadow: 0 0 15px rgba(16, 185, 129, 0.8); }
            #nuke-overlay.success-state .nuke-subtext { color: #6ee7b7; animation: none; font-weight: bold; }
            #nuke-overlay.success-state .nuke-data-stream { color: #10b981; }
        `;
        document.head.appendChild(style);

        const overlay = document.createElement('div');
        overlay.id = 'nuke-overlay';
        overlay.innerHTML = `
            <div class="nuke-data-stream" id="nukeDataStream">
                INITIATING OVERRIDE PROTOCOL<br>
                BYPASSING SAFETY INTERLOCKS<br>
                ERASING VECTOR EMBEDDINGS<br>
                DROPPING SCHEMA TABLES
            </div>
            <div class="nuke-radar-container">
                <div class="nuke-radar-circle"></div>
                <div class="nuke-radar-sweep"></div>
                <i class="fa-solid fa-radiation nuke-icon-center" id="nukeIcon"></i>
            </div>
            <div class="nuke-text-glitch" id="nukeTitle">SYSTEM PURGE ACTIVE</div>
            <div class="nuke-subtext" id="nukeSub">WIPING ALL DATABASES AND MEMORY... DO NOT CLOSE WINDOW.</div>
        `;
        document.body.appendChild(overlay);

        // Randomize data stream text to make it look cool/animated
        let streamInterval = setInterval(() => {
            const streamEl = document.getElementById('nukeDataStream');
            if (streamEl) {
                const hex1 = Math.floor(Math.random() * 0xFFFFF).toString(16).toUpperCase();
                const hex2 = Math.floor(Math.random() * 0xFFFFF).toString(16).toUpperCase();
                streamEl.innerHTML = `
                    DELETING SECTOR 0x${hex1}...<br>
                    OVERWRITING MEMORY...<br>
                    SHREDDING 0x${hex2}...<br>
                    PURGING VECTORS...
                `;
            }
        }, 300);

        // Force reflow
        void overlay.offsetWidth;
        overlay.classList.add('active');

        try {
            const resp = await fetch('/api/admin/nuke', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ password })
            });

            const data = await resp.json();

            clearInterval(streamInterval);

            if (data.success) {
                overlay.classList.add('success-state');
                document.getElementById('nukeIcon').className = 'fa-solid fa-check-circle nuke-icon-center';
                document.getElementById('nukeTitle').innerText = 'PURGE SUCCESSFUL';
                document.getElementById('nukeSub').innerText = 'ALL CONTEXT HAS BEEN ERASED. RELOADING...';
                document.getElementById('nukeDataStream').innerHTML = 'SYSTEM RESET COMPLETE.';

                setTimeout(() => {
                    window.location.reload();
                }, 2000);
            } else {
                overlay.classList.remove('active');
                setTimeout(() => { overlay.remove(); style.remove(); }, 500);
                alert("PURGE FAILED: " + (data.detail || "Administrative privilege escalation failed."));
            }
        } catch (e) {
            clearInterval(streamInterval);
            console.error("Nuke Error:", e);
            overlay.classList.remove('active');
            setTimeout(() => { overlay.remove(); style.remove(); }, 500);
            alert("A critical error occurred during the system purge.");
        }
    }

}
// =========================================================================
// FILE VIEWER MODAL (Global functions for onclick handlers)
// =========================================================================

function openFileViewer(fileUrl, fileType, fileName) {
    const modal = document.getElementById('fileViewerModal');
    const title = document.getElementById('fileViewerTitle');
    const body = document.getElementById('fileViewerBody');
    if (!modal || !body) return;

    title.querySelector('span:last-child').textContent = fileName || 'File Viewer';

    const ext = (fileName || '').split('.').pop().toLowerCase();
    let content = '';

    if (['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext)) {
        content = `<img src="${fileUrl}" alt="${fileName}" class="max-w-full max-h-full object-contain rounded-xl" />`;
    } else if (['mp4', 'mov', 'webm'].includes(ext)) {
        content = `<video controls class="max-w-full max-h-full rounded-xl"><source src="${fileUrl}" /></video>`;
    } else if (['mp3', 'wav', 'ogg'].includes(ext)) {
        content = `<div class="text-center"><p class="text-white/60 mb-4">${fileName}</p><audio controls src="${fileUrl}" class="w-full max-w-md"></audio></div>`;
    } else if (ext === 'pdf') {
        content = `<iframe src="${fileUrl}" class="w-full h-full rounded-xl" frameborder="0"></iframe>`;
    } else {
        content = `<div class="text-center"><p class="text-white/60 mb-4">${fileName}</p><a href="${fileUrl}" download class="px-4 py-2 bg-cyan-500/20 text-cyan-300 rounded-lg hover:bg-cyan-500/30 transition">Download File</a></div>`;
    }

    body.innerHTML = content;
    modal.classList.remove('hidden');
    modal.classList.add('flex');
}

function closeFileViewer() {
    const modal = document.getElementById('fileViewerModal');
    if (modal) {
        modal.classList.add('hidden');
        modal.classList.remove('flex');
        const body = document.getElementById('fileViewerBody');
        if (body) body.innerHTML = '';
    }
}

// SOTA Phase 4: Purge Level 10 (Cache Busting)
const Ultima_VERSION = "4.0.0-REV-ULTRA";

window.addEventListener('load', () => {
    console.log(`%c Ultima RAG CORE: ${Ultima_VERSION} ACTIVE %c`, "background:#00d9ff; color:#000; font-weight:bold; padding:4px;", "");
    window.app = new UltimaApp();

    // Auto-purge old state if version mismatch detected (Simulated)
    const lastVer = localStorage.getItem('Ultima_version');
    if (lastVer !== Ultima_VERSION) {
        console.warn("Ultima: Version shift detected. Purging legacy cognitive cache.");
        localStorage.setItem('Ultima_version', Ultima_VERSION);
    }
});

