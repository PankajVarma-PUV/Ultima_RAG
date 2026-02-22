# 🛡️ SentinelRAG: SOTA Metacognitive RAG Pipeline

SentinelRAG is an advanced, autonomous Retrieval-Augmented Generation (RAG) system built with a **Metacognitive Brain** architecture. It leverages multi-agent orchestration, state-of-the-art vector storage, and a cinematic streaming UI to deliver high-precision, grounded, and multimodal AI responses.

![SentinelRAG Banner](https://img.shields.io/badge/Status-SOTA-blueviolet?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Tech](https://img.shields.io/badge/Framework-LangGraph%20%7C%20FastAPI-blue?style=for-the-badge)

## 🧠 The Metacognitive Architecture

SentinelRAG doesn't just "retrieve and generate." It **reasons** through its own process using a LangGraph-powered graph:

1.  **Intent Classification**: Dynamically identifies the nature of the query (e.g., Multi-task, Summarization, Retrieval).
2.  **Streaming Thought-UI**: Surfaces the internal "inner monologue" of the agents to the user in real-time (o1-style).
3.  **Self-Healing Loop**: An autonomous auditor checks for hallucinations and triggers a "Healer" agent to reground responses if gaps are found.
4.  **Hybrid Search Alpha**: Combines semantic vector search with keyword-based full-text search for maximum recall.

## ✨ Key Features

-   **🚀 Streaming Thought Process**: Real-time visibility into agent reasoning via SSE-powered UI.
-   **📚 SOTA Vector Storage**: High-performance retrieval using LanceDB.
-   **🖼️ Multimodal Capabilities**: Process images, videos, audios, PDFs, and text files seamlessly.
-   **🛠️ Agentic Healing**: Automatic verification and correction of AI-generated content.
-   **🌐 Web Breakout**: Autonomous web search capabilities for trending news and external verification.

## 🛠️ Technical Stack

-   **Backend**: Python, FastAPI, LangGraph, LangChain.
-   **Database**: LanceDB (with Tantivy for full-text index).
-   **Frontend**: Vanilla JavaScript (Modern ES6+), CSS (Rich Glassmorphism).
-   **Agents**: Ollama / HuggingFace hosted models.

## 🚀 Getting Started (Local Execution)

SentinelRAG is designed to be run **locally** to ensure complete data privacy and control.

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) (running locally) or equivalent API credentials.

### Installation & Setup

SentinelRAG features a **Smart setup script** that automatically detects your hardware (NVIDIA GPU vs CPU) and installs the correct ML stack for you.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/SentinelRAG.git
    cd SentinelRAG
    ```
2.  **Start the Application**:
    Simply run the automated startup script. It will create a virtual environment, install the correct version of Torch for your hardware, and launch the server:
    ```bash
    run_sentinelrag.bat
    ```
The API server will start at `http://localhost:8000`, and the UI will be accessible in your web browser.

## 📁 Project Structure

-   `src/agents/`: Specialized AI agents (Planner, Healer, Retriever, etc.).
-   `src/core/`: Central orchestration logic and metacognitive graph.
-   `ui/`: Frontend assets and HTML.
-   `data/`: Local storage for vector DB and processed embeddings.

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Built with ❤️ by the SentinelRAG Team*
