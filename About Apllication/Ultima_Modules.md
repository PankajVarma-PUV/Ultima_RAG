# UltimaRAG Module Documentation
This document serves as the single source of truth for the UltimaRAG AI system architecture and its corresponding `.py` modular files. It breaks down the system into three main areas: Database, FastAPI (Routing/State), and Application Core (Agents, Core Utility, Vision, and Tools).

---

## Section 1: Database Related Files

### database.py (LanceDB)
**Location:** `src/core/database.py`
**Role:** Primary interface for the LanceDB vector database. Manages tables, indexing, schema enforcement, vector searches, and data migrations for document chunks.
**Listens To:** `EmbeddingManager` (for vectors), Application routes, `MetacognitiveBrain` (during retrieval).
**Reports To:** `RetrieverAgent`, `MetacognitiveBrain`.
**User Defined Functions:**
- `__init__(db_path)`: Configures and initializes the LanceDB connection at the given path.
- `connect()`: Establishes synchronous connection to LanceDB and auto-creates required tables.
- `_create_tables_if_needed()`: Applies schemas to create the `knowledge_base` table (for chunks) and `system_guidelines` table.
- `add_chunks(conversation_id, file_name, file_id, chunks)`: Inserts document text chunks and their embeddings into LanceDB.
- `search_knowledge(query_vector, top_k, conversation_id)`: Performs vector similarity search over the knowledge base.
- `migrate_v1_to_v2()`: Handles schema migration for continuous learning objects.
- `delete_conversation_knowledge(...)`: Deletes vector data for a given conversation.
- `nuke_knowledge_base()`: Factory reset; deletes all vector tables.

### database.py (SQL/Metadata)
**Location:** `src/data/database.py`
**Role:** Abstract base class and implementations (SQLite, PostgreSQL) for relational metadata such as conversations, chat histories, agent rules, and system telemetry.
**Listens To:** Database Connection Pool, Environment Variables (`DATABASE_URL`).
**Reports To:** Application API endpoints, Telemetry dashboard, Memory Managers.
**User Defined Functions:**
- `__init__()`: Abstract initializer for the database interface.
- `connect()`: Opens the database session or pool connection.
- `disconnect()`: Closes the database connection.
- `is_connected()`: Validates if the SQL database connection is active.
- `reset_database()`: Drops all tables and re-initializes the schema (fresh start).
- `initialize_schema()`: Creates tables if they don't exist.
- `create_conversation(...)`: Creates a new conversation record.
- `list_conversations(...)`: Retrieves a list of conversations.
- `get_conversation(conversation_id)`: Fetches a single conversation by ID.
- `update_conversation(...)`: Updates a conversation's title or archive status.
- `delete_conversation(conversation_id)`: Deletes a conversation and its messages.
- `add_message(...)`: Appends a new message to a conversation.
- `get_messages(conversation_id)`: Retrieves all messages for a conversation.
- `get_active_messages(conversation_id)`: Retrieves non-deleted messages for a conversation.
- `find_duplicate_query(...)`: Finds an identical query in the conversation for caching.
- `get_message_count(conversation_id)`: Gets the total number of messages in a conversation.
- `ensure_conversation(...)`: Ensures a conversation exists, creating it if necessary.
- `setup()`: Creates relational tables for messages, conversations, ingestion status, and analytics.
- `get_cursor()`: Context manager that yields a cursor or session for executing SQL transactions.
- `get_database()`: Singleton constructor that returns PostgreSQL or SQLite connection.

### chunking.py
**Location:** `src/data/chunking.py`
**Role:** Implements text chunking strategies to split raw document text into semantically cohesive, overlapping segments prior to vectorization.
**Listens To:** `DocumentProcessor` (ingested text), System settings for chunk size.
**Reports To:** `database.py` (LanceDB layer for storage).
**User Defined Functions:**
- `__init(...)`: Initializes the chunker with a specified strategy (semantic, fixed, or paragraph).
- `chunk_document(text, doc_id, metadata)`: Chunks a single document and enriches it with metadata.
- `chunk_documents(documents)`: Processes multiple documents into chunks.
- `chunk_text(text, chunk_size, overlap)`: Primary chunker. It attempts semantic paragraph splitting first, then falls back to fixed size splits.
- `fixed_chunk(text, chunk_size, overlap)`: Fixed-size chunking with token overlap.
- `_fixed_chunk_words(text, chunk_size, overlap)`: Fallback method for word-based fixed chunking.
- `semantic_chunk(text, max_chunk_size, min_chunk_size)`: Semantic chunking respecting sentence boundaries.
- `paragraph_chunk(text, max_chunk_size)`: Paragraph-based chunking that splits on double newlines.
- `enrich_chunk(...)`: Adds metadata to a chunk for retrieval and attribution.
- `_split_by_paragraphs(text)`: Helper function to naturally split documents using double-newline sequences.
- `_chunk_fixed_size(text, chunk_size, overlap)`: Fallback method that rigorously enforces size constraints for very long continuous text blocks.
- `get_token_count(text)`: Gets the number of tokens in the text.

### embedder.py
**Location:** `src/data/embedder.py`
**Role:** Generates embeddings for text chunks using a deterministic, caching pipeline. Ensures reproducibility by leveraging deterministic hashes.
**Listens To:** HuggingFace `SentenceTransformers`, input text from `chunking.py`.
**Reports To:** `database.py` (LanceDB layer).
**User Defined Functions:**
- `__init__(model_name)`: Loads the target sentence transformer on the CPU to prevent GPU VRAM contention.
- `encode(text)`: Generates dense vectors for a single string or a list of strings, checking memory cache first.
- `_encode_single(...)`: Encodes a single text with a cache lookup.
- `_encode_batch(...)`: Encodes a batch of texts with partial cache lookups.
- `_get_cache_key(text)`: Creates a deterministic hash from the input text to leverage the embedding cache.
- `_load_cache()`: Loads the embedding cache from disk.
- `_save_cache()`: Saves the embedding cache to disk.
- `save_cache()`: Manually triggers saving the cache to disk.
- `clear_cache()`: Clears the in-memory and on-disk embedding cache.
- `embedding_dimension()`: Returns the embedding dimension size.
- `get_embedder(...)`: Retrieves a configured embedder instance.
- `embed_chunks(...)`: Adds embeddings to a list of chunk dictionaries.
- `warmup()`: Performs a dummy encoding upon initialization to load model weights into RAM.

### nuke_manager.py
**Location:** `src/data/nuke_manager.py`
**Role:** Administrative utility that performs a complete factory reset of the application, wiping all SQL data, vector tables, and uploaded multimedias.
**Listens To:** FastAPI trigger endpoints.
**Reports To:** System logger, HTTP response payload.
**User Defined Functions:**
- `__init__(admin_password)`: Initializes the object with a security check.
- `nuke_all_data(password)`: Coordinates a sequenced teardown: Drops vector tables, drops SQL tables, clears upload folders, and resets caches.
- `execute_nuke()`: Alias/wrapper for nuke operations.
- `verify_auth(provided_token)`: Ensures that the command originated from a trusted administrator.

---

## Section 2: FastAPI Related Files

### main.py
**Location:** `src/api/main.py`
**Role:** The core FastAPI application definition. Handles server lifespan events, configures middleware, sets up application state, and maps core API endpoints.
**Listens To:** HTTP requests from the frontend client.
**Reports To:** The User (HTTP Responses), internal event loops.
**User Defined Functions:**
- `lifespan(app)`: Context manager that bootstraps the LLM wrapper, database connections, and Agent components when the server starts.
- `_run_startup_diagnostics(state)`: Prints a structured summary banner after all singletons are initialized.
- `ping()`: A simple health-check route.
- `ask_query(request)`: Main RAG interaction endpoint. It captures the query and passes it to `MetacognitiveBrain`.
- `ingest_document(file)`: Endpoint for uploading multimodal files, routing them to the `DocumentProcessor`.
- `get_system_status()`: Administrative endpoint to read hardware statistics (VRAM, CPU) and telemetry.
- `create_conversation(request)`: Endpoint to create a new conversation thread.
- `get_conversations()`: Endpoint to retrieve a list of all conversations.
- `get_conversation(conversation_id)`: Endpoint to retrieve history for a specific conversation.
- `update_conversation(conversation_id, request)`: Endpoint to rename or archive a conversation.
- `delete_conversation(conversation_id)`: Endpoint to delete a conversation from both SQL and vector storage.
- `factory_reset(request)`: Administrative endpoint triggering a full system wipe via `NukeManager`.

### routes.py
**Location:** `src/api/routes.py`
**Role:** Houses supplementary API router endpoints like batch queries, user preference updates, and guidelines retrieval.
**Listens To:** HTTP requests.
**Reports To:** `main.py` API Router.
**User Defined Functions:**
- `analyze_query(request)`: Analyzes a query without running the full pipeline (useful for debugging).
- `search_only(request)`: Performs a knowledge search without synthesis, returning raw retrieval results.
- `batch_query(queries)`: Takes an array of questions, sequentially passes them to the AI, and returns a compiled response.
- `get_config()`: Retrieves current non-sensitive system configuration thresholds.
- `get_metrics()`: Retrieves placeholder system metrics.
- `update_persona(persona_config)`: Updates the in-memory or database properties for `UserPersona`.
- `download_guidelines()`: Exposes `system_guidelines.json` for frontend review and auditing.

### utils.py (API Layer)
**Location:** `src/api/utils.py`
**Role:** Contains helper classes, exception handlers, and streaming simulators used exclusively by the FastAPI layer.
**Listens To:** API Endpoint functions.
**Reports To:** FastAPI application.
**User Defined Functions:**
- `format_error_response(message, code)`: Wraps exceptions in a consistent JSON dictionary schema.
- `simulate_streaming(text_chunks)`: An asynchronous generator that artificially delays chunk delivery.
- `validate_api_key(token)`: Middleware hook to authenticate endpoints via basic token matching against `.env`.

---

## Section 3: All the Application Related Core Files

### Metacognitive Brain
**Location:** `src/agents/metacognitive_brain.py`
**Role:** The central orchestrator (LangGraph equivalent). It manages application state across agents, generates execution DAGs, routes intents, and constructs the final compiled `UnifiedResponse`.
**Listens To:** API `ask_query`, All AI Agents.
**Reports To:** FastAPI application.
**User Defined Functions:**
- `__init__()`: Instantiates all AI sub-agents (FactChecker, Retriever, Synthesizer, etc.).
- `generate_answer(query, conversation_id)`: The main entry point. Constructs a plan, initiates sub-agents, gathers context, synthesizes, and logs.
- `_route_intent(query)`: Uses the `IntentClassifier` to branch the execution logic.
- `_evaluate_knowledge()`: Reflects on whether the retrieved evidence is sufficient, or if a fallback is needed.
- `_construct_response()`: Aggregates agent output strings into a structured Pydantic model (`UnifiedResponse`).
- `_persist_message(...)`: SOTA Dual-Database Persistence: Synchronizes history across Vector and Relational stores.
- `_build_graph()`: Constructs the LangGraph-style execution state machine.
- `run_extractor(state)`: Unified Multimodal Fusion Extraction step for state graph.
- `route_intent(state)`: Wise Intent Routing step for state graph.
- `decide_initial_path(state)`: Decides if chronicler agent should be engaged.
- `create_execution_plan(state)`: Invokes the `MultiStagePlanner` to build a DAG.
- `decide_path(state)`: Conditional transition from planner/router (gates RAG based on evidence).
- `evaluate_knowledge(state)`: Elite Knowledge Evaluator (determines LLM memory vs context).
- `initiate_direct_flow(state)`: Forces LLM Memory for Direct Path.

### Intent Classifier & Firewall
**Location:** `src/agents/intent_classifier.py`
**Role:** Analyzes queries to route them dynamically to the correct AI path. Its firewall component actively blocks known Prompt Injection attacks.
**Listens To:** `MetacognitiveBrain` (user query input).
**Reports To:** `MetacognitiveBrain` (routing instructions).
**User Defined Functions:**
- `__init__()`: Initializes the classifier.
- `classify(query)`: Uses an LLM prompt to classify intent (e.g., MULTI_HOP, GENERAL) and target language.
- `detect_context_rejection(query)`: Detects if user explicitly requests to ignore context.
- `_run_firewall(query)`: Heuristically and semantically checks the query for malicious attempts.
- `detect_injection(query)`: Fast local heuristic check for Prompt Injections (Firewall).
- `_log_security_event(...)`: Logs security threats.
- `parse_mentions(query)`: Extracts @filename mentions.
- `strip_mentions(query)`: Removes @filename mentions from query.

### Universal Fusion Extractor
**Location:** `src/agents/fusion_extractor.py`
**Role:** Aggregates and merges multi-source multimodal evidence (text, visual analysis, OCR, audio) into a single cohesive `UnifiedEvidenceState` object.
**Listens To:** `VisionProcessor`, `AudioProcessor`, `DocumentProcessor`.
**Reports To:** `MultiStagePlanner`, `SynthesizerAgent`.
**User Defined Functions:**
- `__init__(db)`: Initializes with database connection.
- `extract_and_fuse(...)`: Gathers evidence from all assets and bucketizes them into Text, Visual, and Audio (supports strict filtering).
- `fuse_evidence(source_list)`: Joins heterogeneous data chunks, prioritizing visually enriched narrations over raw text.
- `_format_for_context()`: Flattens the unified state into a token-efficient string.
- `_name_matches(...)`: Helper to check if a file name matches target mentions.

### MultiStage Planner
**Location:** `src/agents/planner.py`
**Role:** Decomposes complex, multi-hop user queries into a Directed Acyclic Graph (DAG) of sequential or parallel tasks.
**Listens To:** `MetacognitiveBrain`.
**Reports To:** `MetacognitiveBrain`.
**User Defined Functions:**
- `__init__()`: Initializes with Ollama configuration.
- `create_plan(query, intent, evidence)`: Asks an LLM to generate an `ExecutionPlan`.
- `_parse_dag(llm_json)`: Converts raw LLM outputs into structured `TaskStep` models.

### Query Analyzer
**Location:** `src/agents/query_analyzer.py`
**Role:** Analyzes queries to detect sub-questions, extract named entities, and rephrase queries for better database semantic similarity retrieval.
**Listens To:** `MetacognitiveBrain`.
**Reports To:** `RetrieverAgent`.
**User Defined Functions:**
- `__init__()`: Initializes Query Analyzer with Ollama client.
- `analyze(query)`: Evaluates user intent and decomposes complex questions into targeted sub-queries.
- `_analyze_with_llm(query)`: Uses Ollama to analyze the query.
- `_analyze_fallback(query)`: Fallback rule-based analysis when LLM fails.
- `_validate_and_complete(...)`: Ensures all required fields exist with valid values.
- `extract_entities(text)`: Pulls core topics, dates, and names from the prompt to apply rigid database filters.

### Retriever Agent
**Location:** `src/agents/retriever.py`
**Role:** Interacts directly with LanceDB. Executes hybrid searches (Dense Vector + BM25 keyword overlap).
**Listens To:** `MetacognitiveBrain`, `QueryAnalyzer`.
**Reports To:** `RerankerModule` or `SynthesizerAgent`.
**User Defined Functions:**
- `__init__(db, embedder)`: Initializes retriever with database and embedder.
- `retrieve(query, top_k, ...)`: Executes both dense and sparse searches across the database.
- `retrieve_multimodal(conversation_id, query)`: Fetches multimodal evidence for ALL files in a conversation.
- `_deduplicate(results)`: Removes identical document chunks coming from different search strategies.

### Reranker Module
**Location:** `src/agents/reranker.py`
**Role:** Uses a Cross-Encoder to re-score and finely tune raw retrieval results, prioritizing chunks with high contextual relevance over simple vector similarity.
**Listens To:** `RetrieverAgent`.
**Reports To:** `SynthesizerAgent`.
**User Defined Functions:**
- `__init__(...)`: Initializes Reranker with cross-encoder model.
- `rerank(query, documents, threshold, top_k)`: Applies Cross-Encoder logic to re-order the provided chunks.
- `score_single(query, text)`: Scores a single query-text pair.
- `filter_threshold(ranked_docs, min_score)`: Discards chunks that fall below the configured certainty threshold.

### Synthesizer Agent
**Location:** `src/agents/synthesizer.py`
**Role:** Constructs the final, human-readable text answer from retrieved chunks, prioritizing direct quotes and enforcing in-line citations.
**Listens To:** `MetacognitiveBrain`, `RerankerModule`.
**Reports To:** `MetacognitiveBrain`, `FactChecker`.
**User Defined Functions:**
- `__init__()`: Initializes Synthesis Agent with Ollama client.
- `synthesize(query, context_chunks)`: Uses the primary LLM to draft a response using strict guidelines.
- `_parse_response(response, chunks)`: Parses the synthesis response and extracts structure.
- `_extract_citations(answer, chunks)`: Extracts `[Source: docname]` citations from the answer.
- `_calculate_confidence(...)`: Calculates confidence score based on answer quality indicators.
- `_enforce_citations(text, sources)`: Post-processing step to ensure output uses defined inline brackets referencing documents.

### Fact Checker
**Location:** `src/agents/fact_checker.py`
**Role:** Security and accuracy layer. Extracts individual claims from outputs and validates them against source evidence utilizing Natural Language Inference.
**Listens To:** `SynthesizerAgent`.
**Reports To:** `MetacognitiveBrain`, `HallucinationHealer`.
**User Defined Functions:**
- `__init(...)`: Initialize Fact Checker with NLI model.
- `check_facts(...)`: Extraces atomic claims, runs NLI overlap, and assigns a factuality score based on chunks.
- `_evaluate_with_llm(...)`: Use Gemma3:4b to evaluate response accuracy against context.
- `evaluate_relevance(...)`: Evaluate if the user's question is relevant to document content.
- `_extract_claims(text)`: Uses LLM to break the narrative into discrete, testable bullet points.
- `_extract_claims_with_llm(...)`: Explicitly uses Ollama to extract atomic claims.
- `_extract_claims_fallback(...)`: Fallback logic to split sentences into claims.
- `_verify_claim(...)`: Verify a claim against evidence using NLI.

### Hallucination Healer
**Location:** `src/agents/healer.py`
**Role:** Rewrites the response to fix hallucinations flagged by `FactChecker`, omitting ungrounded statements and patching missing facts.
**Listens To:** `FactChecker`, `MetacognitiveBrain`.
**Reports To:** `MetacognitiveBrain`.
**User Defined Functions:**
- `__init__()`: Initializes hallucinational healer with Ollama client.
- `heal(query, flawed_response, gaps, evidence)`: Engages a targeted LLM to actively re-write and correct identified issues.

### Humanizer Agent
**Location:** `src/agents/humanizer_agent.py`
**Role:** Re-writes text into a natural, conversational tone without altering factual content or citations.
**Listens To:** `MetacognitiveBrain`.
**Reports To:** `MetacognitiveBrain` (Final Output).
**User Defined Functions:**
- `__init__()`: Initialize Humanizer Agent with Ollama client.
- `humanize(query, raw_answer, citations)`: Modifies tone to match the user's `UserPersona`.
- `_fix_source_names(citations)`: Re-formats citation source strings for UI clarity.

### Content Enricher
**Location:** `src/agents/content_enricher.py`
**Role:** Transforms raw multimodal scraping data (like OCR strings) into descriptive, high-fidelity narratives.
**Listens To:** Multimodal Document ingestion pipeline.
**Reports To:** `FusionExtractor`.
**User Defined Functions:**
- `__init__(model_name)`: Initialize content enricher.
- `enrich_content(...)`: Takes raw extracts and converts them into semantic prose based on specific asset type.

### Translator Agent
**Location:** `src/agents/translator_agent.py`
**Role:** Specializes purely in cross-lingual translation, maintaining tone and facts.
**Listens To:** `IntentClassifier`.
**Reports To:** `MetacognitiveBrain`.
**User Defined Functions:**
- `__init__(model_name)`: Initializes Translator Agent.
- `translate(text, target_language, preserve_formatting)`: Translates input string explicitly to the designated language.

### Deep Insight Agent
**Location:** `src/agents/deep_insight_agent.py`
**Role:** Explores secondary layers of data inference. Provides multi-stage reflective analysis (Analyst + Skeptic) for highly complex documents.
**Listens To:** User UI (Deep Insight Action Button).
**Reports To:** `MetacognitiveBrain`.
**User Defined Functions:**
- `__init__()`: Initialize the loop.
- `analyze_deeply(query, context)`: Initiates a multi-turn conversation between LLM roles.
- `run(...)`: Execute the 3-stage deep insight debate and yield SSE-ready events.
- `_strip_thinking(text)`: Strip `<thinking>` blocks emitted by reasoning models.
- `_run_stage(...)`: Run a single LLM stage and collect response.
- `_skeptic_review(insight)`: Generates adversarial counter-points to improve robust analysis.

### Reflector Agent (Continuous Learning)
**Location:** `src/agents/reflector.py`
**Role:** Processes negative feedback, deduces system mistakes, generating and writing new behavioral rules to `system_guidelines.json`.
**Listens To:** Feedback API endpoints.
**Reports To:** `GuidelinesManager` (Disk writes).
**User Defined Functions:**
- `__init__(app_state)`: Initializes the agent with app_state.
- `schedule_reflection(feedback_data)`: Creates a background task for reflection learning (fire-and-forget).
- `_task_done_callback(task)`: Removes completed task from registry.
- `_run_with_semaphore(feedback_data)`: Ensures only one reflection write runs at a time.
- `_process(feedback_data)`: Full reflection pipeline.
- `reflect_on_feedback(context, bad_response, feedback)`: Evaluates errors and extracts generalized rules.
- `_generate_rule(query, response)`: Calls Ollama to generate structured rule via Pydantic output.
- `_find_duplicate(...)`: Finds duplicate rules via embedding dedup or keyword fallback.
- `_keyword_dedup(...)`: Jaccard overlap deduplication.
- `_reinforce_rule(...)`: Increments trigger_count and boosts confidence for an existing rule.
- `_deduplate_rules(new_rule, existing_rules)`: Uses cosine similarity to prevent duplicate instructions.
- `_create_rule_entry(generated)`: Builds the full rule dict from a `GeneratedRule`.
- `_run_lifecycle(rules)`: Triggers retirement checks and rule cap limits.
- `_atomic_write(...)`: Write-to-temp then `os.replace` for atomic disk changes.
- `await_pending_tasks()`: Waits for in-progress reflection tasks during shutdown.

### Quality Indicator / Refusal Gate
**Location:** `src/agents/refusal_gate.py`
**Role:** Analyzes response quality components and calculates a final aggregate CONFIDENCE SCORE.
**Listens To:** `FactChecker`, `MetacognitiveBrain`.
**Reports To:** User UI Headers.
**User Defined Functions:**
- `__init__(...)`: Initializes Quality Indicator with classification thresholds.
- `evaluate_quality(response, metrics)`: Derives a mathematical score for the synthesized output's security and accuracy.
- `decide(...)`: Analyzes quality and ALWAYS returns the answer with confidence indicators.
- `_calculate_fact_score(synthesis, fact_check)`: Calculates overall QualityScore.
- `_classify_confidence(fact_score)`: Classifies the confidence level based on fact score.
- `_collect_warnings(synthesis, fact_check)`: Collects relevant quality warnings.
- `_generate_response(...)`: Generates the response with quality indicators.
- `_suggest_alternatives(...)`: Generates alternative query suggestions.

### Orchestrator (DEPRECATED)
**Location:** `src/agents/orchestrator.py`
**Role:** Legacy sequential pipeline orchestrator. Unused. Superseded by `MetacognitiveBrain`.
**Listens To:** None.
**Reports To:** None.
- `__init__(*args, **kwargs)`: Raises DeprecationWarning.
- `process_query(*args, **kwargs)`: Raises DeprecationWarning.

### Core Config (config.py)
**Location:** `src/core/config.py`
**Role:** Central Hub for environment variables and threshold properties. Exposes Pydantic configurations.
**Listens To:** System `.env` files.
**Reports To:** All configuration-dependent modules.
**User Defined Functions:** Contains Configuration Classes (`OllamaConfig`, `MemGPTConfig`) matching ENV keys.

### Document Processor
**Location:** `src/core/document_processor.py`
**Role:** Extracts native strings and seamlessly identifies embedded images or scanned pages in PDFs, shipping visual fragments to `QwenAgent`.
**Listens To:** `main.py` ingestion routes.
**Reports To:** Database indexing agents.
**User Defined Functions:** 
- `extract_from_pdf(conversation_id, file_path, file_name)`: Scans PDF page by page using native scraping and Qwen2-VL OCR.

### Embedding Manager
**Location:** `src/core/embedding_manager.py`
**Role:** Manages the CPU-bound HuggingFace `SentenceTransformer` process to perform inference and cache embeddings locally.
**Listens To:** Data embedders, Retrieval vectors.
**Reports To:** Database query functions.
**User Defined Functions:**
- `__init__()`: Initializes the EmbeddingManager.
- `initialize()`: Loads the embedding model; called ONCE at startup.
- `encode(text)`: Returns float array representing normalized text.
- `cosine_similarity(vecA, vecB)`: Fast numpy distance calculator.
- `detect_model_size(model_name)`: Single source of truth for dynamic memory limits.
- `is_ready()`: Property indicating if the model loaded successfully.

### File Manager
**Location:** `src/core/file_manager.py`
**Role:** Disk IO operations. Manages the directory partitions for multimodal uploads.
**Listens To:** Ingestion application routes.
**Reports To:** Processors (giving paths to files).
**User Defined Functions:**
- `ensure_chat_dir(conversation_id)`: Ensures directory for a specific conversation exists.
- `get_upload_path(...)`: Gets the target path for an uploaded file.
- `save_upload(uid, content)`: Saves binary files to correct chat UUID folder.
- `list_uploads(conversation_id)`: Lists all uploaded files for a conversation.
- `get_file_path(uuid)`: Fetches file for reading.
- `delete_chat_dir(conversation_id)`: Deletes all uploads for a specific conversation.
- `nuke_uploads()`: Wipes the entire uploads directory (Nuclear Option).

### Guidelines Manager
**Location:** `src/core/guidelines_manager.py`
**Role:** Reads, caches, and enforces filtering on `system_guidelines.json`. Ensures token limits aren't breached.
**Listens To:** `ReflectorAgent` (disk writes), Initial Startup.
**Reports To:** `MetacognitiveBrain`.
**User Defined Functions:**
- `__init__(...)`: Initializes GuidelinesManager.
- `_load()`: Synchronous file load.
- `_async_reload()`: Reloads file from disk without blocking the event loop.
- `get_relevant_rules(intent)`: Fetches active, scored user rules.
- `force_reload()`: Bypasses TTL. Reloads immediately from disk.
- `run_schema_migration()`: Atomically converts V1 guidelines arrays to V2 metric objects.
- `get_stats()`: Returns summary statistics.

### Ingestion Watchdog
**Location:** `src/core/ingestion_watchdog.py`
**Role:** Background async task. Scans database for stalled UI ingestion progress flags, marking them FAILED for clean retries.
**Listens To:** Time intervals, `ingestion_status` table.
**Reports To:** Database state updater.
**User Defined Functions:**
- `__init__(check_interval_seconds, stale_timeout_minutes)`: Initializes the watchdog.
- `start()` / `stop()`: Life-cycle controls.
- `_watchdog_loop()`: Loop mechanism for async operation.
- `_check_stalled_jobs()`: Interval execution checking threshold diffs and marking STALLED status to FAILED.

### Memory Manager (MemGPT)
**Location:** `src/core/memory.py`
**Role:** Pages active dialogue constraints to fit memory limits context. Archives old histories.
**Listens To:** Interaction states, LLM requests.
**Reports To:** Database (Storage), Metacognitive Brain (Recall).
**User Defined Functions:**
- `__init__(db)`: Initializes the MemoryManager.
- `get_prompt_context(conversation_id)`: Retrieves the active context for an LLM prompt.
- `get_all_context(conversation_id)`: Retrieves the entire conversation history for specialized reasoning.
- `get_semantic_history(...)`: Performs semantic search within a specific conversation's history.
- `count_tokens(messages)`: Estimates the total token count for a list of messages.
- `manage_overflow(conversation)`: If context > limit, summarizes and DB-archives oldest turns.
- `_summarize_turn(turn)`: Summarizes a conversation turn for archival.
- `recall_context(query_vector)`: Searches the entire paged history for relevant context.
- `extract_facts()`: Distills explicit facts during background operation.

### Models (models.py)
**Location:** `src/core/models.py`
**Role:** Declarative Pydantic validation models matching application states.
**Listens To:** Python typing.
**Reports To:** Data transport architectures.
**User Defined Functions:** None. (Uses Pydantic BaseModels like `UnifiedResponse`).

### Ollama Client
**Location:** `src/core/ollama_client.py`
**Role:** Async HTTP client wrappers managing the local LLM model inferencing.
**Listens To:** Agent components requiring LLM decisions.
**Reports To:** Output to requesting agents.
**User Defined Functions:**
- `__init__(...)`: Initializes the Ollama client.
- `is_available()`: Health checks the Ollama endpoint.
- `ensure_connection()`: Warm up method called during app lifespan.
- `generate()`: High speed async string generation.
- `_generate_sync(...)`: Asynchronous generation (non-streaming).
- `_generate_stream(...)`: Asynchronous streaming generation.
- `chat()`: Primary JSON message generation wrapper.
- `get_ollama_client()`: Gets or creates a singleton OllamaClient instance.
- `generate_with_ollama(...)`: Legacy synchronous wrapper.

### PDF Exporter
**Location:** `src/core/pdf_exporter.py`
**Role:** Generates structured PDF outputs of application dialogues using FPDF with Unicode support.
**Listens To:** Presentation / Export endpoints.
**Reports To:** Disk filesystem / user response.
**User Defined Functions:**
- `__init__()`: Initializes ConversationPDF.
- `header()`: PDF header logic.
- `footer()`: PDF footer logic.
- `render_chat_bubble(...)`: Renders a premium chat bubble style message.
- `generate_conversation_pdf()`: Exports entire dialogue history.
- `generate_evidence_report()`: Generates standalone AI research dossiers.
- `generate_query_pdf(...)`: Generates a branded, Unicode-capable PDF for a specific query and its AI response.

### Prompts
**Location:** `src/core/prompts.py`
**Role:** Stores constant strings for prompts and formatted templates used throughout the application.
**Listens To:** Agent imports.
**Reports To:** All string manipulations containing few-shot contexts.
**User Defined Functions:**
- `format_context_for_synthesis(chunks)`: Parses database rows into clean LLM injection strings.

### Telemetry Manager
**Location:** `src/core/telemetry.py`
**Role:** Tracks AI processing durations for the frontend UI. Broadcasts WebSocket updates.
**Listens To:** Activity event triggers from Agents.
**Reports To:** WebSocket interfaces (Frontend UI HUD).
**User Defined Functions:**
- `__init__()`: Initializes Telemetry elements (AgentTelemetry, TelemetryManager, WebSocketTelemetryManager).
- `finish(...)`: Finishes telemetry recording.
- `to_dict()`: Converts captured state to dict.
- `start_activity()` / `end_activity()`: Tracks chronometers.
- `clear_all()`: Emergency reset for telemetry state.
- `get_active_status()`: Gets the current running activity for telemetry.
- `connect(...)` / `disconnect(...)`: WebSocket connection controls.
- `WebSocketTelemetryManager.broadcast()`: Fires payloads to port.

### Utils (utils.py)
**Location:** `src/core/utils.py`
**Role:** Central home for text processing, normalization, timing contexts, and similarity calculations.
**Listens To:** Application-wide imports.
**Reports To:** System logger, String generators.
**User Defined Functions:**
- `setup_logging()`: Initializes JSON or console telemetry.
- `set_seed()`: System deterministic hashing initialization.
- `get_deterministic_hash(text)`: Generates a deterministic hash for caching.
- `normalize_text()`: Base string cleaning.
- `tokenize_simple(text)`: Simple whitespace tokenization.
- `calculate_word_overlap()`: N-gram matching logic.
- `truncate_text(...)`: Truncates text to max length with a suffix.
- `cosine_similarity(vec1, vec2)`: Calculates cosine similarity between two vectors.
- `normalize_scores(scores)`: Min-max normalizes scores to [0, 1] range.
- `validate_chunks(chunks)`: Validates chunk structure.
- `validate_query_analysis(analysis)`: Validates query analysis output structure.
- `get_file_category(extension)`: Maps file extensions to database-allowed categories.
- `Timer(context_manager)`: SOTA tracing wrapper.
- `test_determinism()`: Tests that deterministic mode is working correctly.

### Web Search Tool
**Location:** `src/tools/web_search.py`
**Role:** Live web scraping Breakout Tool. Wraps DuckDuckGo and Trafilatura to prevent hallucination when knowledge is absent.
**Listens To:** `MetacognitiveBrain` knowledge gap checks.
**Reports To:** Context array for synthesis generation.
**User Defined Functions:**
- `fallback_web_search(query, max_results)`: Acquires search engine links, extracts body text, handles hard truncation.

### Vision: Audio Processor
**Location:** `src/vision/audio_processor.py`
**Role:** Integrates `faster_whisper` to transcribe audio assets into chunk text data.
**Listens To:** `MultimodalManager`.
**Reports To:** Raw ingestion text pool.
**User Defined Functions:**
- `__init__(model_size)`: Initializes AudioProcessor.
- `_load_model()`: Lazy loads WhisperModel.
- `transcribe(file_path)`: Loads whisper model and executes CPU pipeline.

### Vision: Image Processor
**Location:** `src/vision/image_processor.py`
**Role:** Orchestrates Vision analysis. Splits huge images into 2x2 quadrants to protect VRAM constraints, handles EasyOCR routing.
**Listens To:** `MultimodalManager`.
**Reports To:** Ingestion pipelines.
**User Defined Functions:**
- `__init__()`: Pipeline Orchestrator for Preprocessing + OCR + Vision.
- `warm_up()`: Pre-load vision and OCR models to eliminate cold-start latency.
- `_get_ocr_reader()`: Lazy initializer for easyocr.
- `_process_tiled(img)`: Splits image into 2x2 grid and processes tiles sequentially.
- `process(file_path)`: Multimodal image processing flow.

### Vision: Manager
**Location:** `src/vision/manager.py`
**Role:** Master orchestrator for multimodal ingestion branching. Determines if file goes to Audio, Video, Image, or Document processors.
**Listens To:** `main.py` ingestion routes.
**Reports To:** Respective modality module.
**User Defined Functions:**
- `__init__()`: Initializes the MultimodalManager.
- `get_file_hash(file_path)`: Calculates SHA-256 for duplicate checks.
- `process_file(conversation_id, file_path, file_type, file_name)`: Main switch-board router based on MIME types.
- `_background_enrichment(...)`: Helper to run enrichment in the background and persist to DB.

### Vision: Qwen Agent
**Location:** `src/vision/qwen_agent.py`
**Role:** Exclusively manages the HuggingFace `Qwen2-VL-2B` vision-language model using adaptive 4-bit quantification limits.
**Listens To:** Image wrappers (`VideoProcessor`, `ImageProcessor`, `DocumentProcessor`).
**Reports To:** Raw descriptive extraction outputs.
**User Defined Functions:**
- `__init__(model_id)`: Initializes QwenVisionAgent.
- `_check_gpu_health()`: Diagnostic logging indicating active memory and available CUDA tensors.
- `_lazy_load()`: Loads model with adaptive configuration.
- `describe_image(pil_image, prompt)`: Core vision reasoning generating semantic interpretation off image frames.
- `get_vision_agent()`: Singleton instance getter.

### Vision: Video Processor
**Location:** `src/vision/video_processor.py`
**Role:** Orchestrates frame sampling using SSIM-lite visual difference thresholds to avoid repetitive analysis. Stitches visual narratives with audio.
**Listens To:** `MultimodalManager`.
**Reports To:** Aggregated ingestion text pipelines.
**User Defined Functions:**
- `__init__()`: Initializes VideoProcessor.
- `_get_ocr_reader()`: Lazy initializer for easyocr.
- `_is_frame_significant(prev_frame, curr_frame)`: Runs the rapid OpenCV threshold comparison to determine if semantic change occurred.
- `_is_text_quality_sufficient(text)`: Filters out garbage OCR noise.
- `process(file_path)`: Generates timeline chunks processing visuals selectively and marrying it explicitly to OCR frames.
