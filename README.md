<h1 align="center">🛡️ SpandaOS: The Metacognitive RAG Operating System</h1>

<p align="center">
  <em>An advanced, autonomous Retrieval-Augmented Generation (RAG) system powered by a 30-Agent LangGraph Architecture, Continuous Learning, and Multimodal Senses.</em>
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/Status-SOTA-blueviolet?style=for-the-badge">
  <img alt="License" src="https://img.shields.io/badge/License-AGPL--3.0-green?style=for-the-badge">
  <img alt="Tech Stack" src="https://img.shields.io/badge/Framework-LangGraph%20%7C%20FastAPI-blue?style=for-the-badge">
  <img alt="Database" src="https://img.shields.io/badge/DB-LanceDB%20%7C%20SQLite-darkred?style=for-the-badge">
</p>

<hr>

## 🌌 What is SpandaOS?

SpandaOS (Spanda Operating Soul) goes far beyond traditional "retrieve and generate" pipelines. It is a **Metacognitive Brain** built on a LangGraph state machine orchestrating **30 specialized AI agents**. It features seamless multimodal ingestion (vision, audio, video, documents), inline cross-encoder reranking, strict NLI-based fact-checking, surgical hallucination healing, and a closed-loop continuous learning system that permanently adapts to user feedback.

With a rich, cinematic streaming UI, SpandaOS surfaces the "inner monologue" of the agents to the user in real-time, providing total transparency into its reasoning process and evaluation metrics.

## ✨ Key Features

- **🧠 The Metacognitive Brain (LangGraph)**: The absolute supervisor. It dynamically routes queries (RAG, Perception, Multi-Task, History Recall), generates execution DAGs via a Multi-Stage Planner, and fuses multimodal state.
- **👁️ Multimodal Senses (Ingestion Pipeline)**: Background extractor agents process formats natively. Supported by `Qwen-VL-2B` vision descriptions, `EasyOCR` tiled scanning, `faster-whisper` audio transcription, and intelligent video keyframe sampling.
- **🛡️ NLI Fact-Checking & Self-Healing**: Synthesized responses are rigorously audited by a CrossEncoder Fact Checker. Unsupported or loose claims are surgically rewritten by a DeepSeek-powered Hallucination Healer before reaching the user.
- **🔄 Continuous Learning Loop**: A background `ReflectionAgent` processes negative feedback, deduces system mistakes, and atomically writes new behavioral rules to a system guidelines database using semantic deduplication, dynamically altering future responses.
- **⚡ Specialized Action Agents**: Lightning-fast, direct vector API pathways triggered by UI buttons for **Executive Summaries**, multi-agent **Deep Insights** (Analyst ↔ Skeptic ↔ Synthesizer debate), and rigorous **Risk Assessments**.
- **📚 SOTA Hybrid Retrieval**: Combines LanceDB dense vector search (all-MiniLM-L6-v2) with BM25 keyword matching, refined inline by a Neural Reranker.
- **🔒 API Middleware & Security**: A `PromptFirewall` scans for jailbreaks and prompt injections before queries can access the Metacognitive Brain.

## 🤖 The 30-Agent Ecosystem

SpandaOS operates a complex society of specialized agents and autonomous subsystems. Highlights include:
1. **Phase 1: Security Gateway** - `IdentityAgent` (Zero-latency branding) & `PromptFirewall`.
2. **Phase 2: Multimodal Ingestion** - `Vision Perception`, `Multilingual OCR`, `Video/Audio Processors`, and `NarrativeAgent`.
3. **Phase 3: The Active Mind** - `Intent Classifier`, `Multi-Stage Planner`, `RetrieverAgent`, `MemoryManager`.
4. **Phase 4: Synthesis & Auditing** - `Cognitive Synthesis`, `NLI Fact Checker`, `Hallucination Healer`, `Quality Indicator`, and `TranslatorAgent`.
5. **Phase 5: Autonomous Infrastructure** - CPU-bound `EmbeddingManager`, `IngestionWatchdog` fail-safes, and the `ReflectorAgent`.

## 🛠️ Technical Model Stack

SpandaOS natively orchestrates multiple local models based on task complexity, maintaining graceful fallbacks and strict hardware limits:
- **Routing & Planning (Lightweight)**: `qwen3:4b`
- **Synthesis & Analysis (Heavyweight)**: `qwen3:8b`
- **Hallucination Healing**: `deepseek-r1:8b`
- **Content Enrichment & History**: `gemma3:4b`
- **Vision & OCR**: `Qwen2-VL-2B-Instruct` & `EasyOCR`
- **Audio Transcription**: `faster-whisper`
- **Embeddings & NLI**: `all-MiniLM-L6-v2` & `ms-marco-MiniLM-L-6-v2` (CrossEncoder)

## 📁 Core Architecture / Project Structure

```text
SpandaOS/
├── src/
│   ├── api/        # FastAPI layer, routing, and middleware (main.py)
│   ├── agents/     # LangGraph nodes and specialized agents (retriever.py, healer.py)
│   ├── core/       # Infrastructure (database.py, memory.py, config.py)
│   ├── data/       # LanceDB operations, chunking, embedder
│   ├── tools/      # External integrations (web_search.py)
│   └── vision/     # Multimodal processors (video, image, audio, qwen_agent)
├── ui/             # Vanilla JavaScript & Modern ES6+ / Rich Glassmorphism frontend
├── data/           # Configs, system_guidelines.json, local SQLite DB
└── README.md
```

## 🚀 Getting Started (Local Execution)

SpandaOS is designed to run locally, ensuring 100% data privacy and control. It includes a **Smart setup script** that detects your hardware (NVIDIA GPU vs CPU) and dynamically installs the correct PyTorch and ML software stack.

### Prerequisites
- **Python 3.10+**
- **[Ollama](https://ollama.com/)** running locally (or equivalent API credentials).
- *Minimum Hardware*: Recommended 8GB+ VRAM (NVIDIA) for optimal local execution, with graceful CPU fallbacks for embeddings and lightweight agents.

### Installation & Setup

#### Method 1: Streamlined Setup (Recommended)
1. Download and run the `clone_SpandaOS.bat` script to automatically pull the repository.
2. Enter the newly created `SpandaOS` directory.
3. Run **`Run_SpandaOS.bat`**. This script handles virtual environment creation, module installation, and component initialization automatically.

#### Method 2: Manual Installation
```bash
# 1. Clone the repository
git clone https://github.com/PankajVarma-PUV/SpandaOS.git
cd SpandaOS

# 2. Start the application
Run_SpandaOS.bat
```

The API server will launch at `http://localhost:8000`. Navigate to the provided local URL in your web browser to access the cinematic UI.

## 🤝 Contributing & Auditing

The system's architecture, including LangGraph agent interactions and continuous learning lifecycles, has been rigorously documented and audited. For detailed architectural insights, refer to the documentation in the `About Apllication/` directory:
- `AI_Agnets.md` -> Source of truth for all 30 sub-agents
- `SpandaOS_Agents_Overview.md` -> High-level agent connectivity and logic graphs
- `SpandaOS_Modules.md` -> Modular `.py` structural documentation

Contributions are welcome! Please review `CONTRIBUTING.md` for our standardization guidelines.

## 📄 License & Attribution

SpandaOS is proudly open-source and licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**. See the [LICENSE](LICENSE) file for more information.

---
*Built with ❤️ by Pankaj Varma and the SpandaOS Team*
