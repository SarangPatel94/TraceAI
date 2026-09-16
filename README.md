# 🧭 TraceAI (RepoScout)

An intelligent, context-aware **Hybrid Retrieval-Augmented Generation (RAG)** pipeline designed to map complex backend architectures into plain business language. 

Built specifically for cross-functional alignment, **TraceAI** allows Business Analysts (BAs), Product Owners, and Technical Leads to perform deep semantic audits of localized code repositories using open-source models completely offline.

---

## ⚡ Key Features

*   🎯 **Strict Directory Scoping:** Tailored for modular monolithic frameworks (like Spryker eCommerce), explicitly restricting file discovery to target architecture directories (`src/Pyz` and `vendor/spryker`) while pruning testing framework noise or massive build caches automatically.
*   🧠 **Multi-Language Parsing Syntax:** Utilizes structure-aware AST segment chunking via LangChain to naturally partition specialized multi-extension stacks including `.php`, `.twig`, `.scss`, `.js`, `.py`, `.ts`, `.json`, `.md`, and `.xml`.
*   🚀 **High-Fidelity Hybrid Local Search:** Configures a local **Qdrant DB instance** to fuse **Dense Semantic Vector Fields** (`BAAI/bge-large-en-v1.5`) alongside structural **Lexical Keyword Identifiers** (Local BM25 Tokenizer mappings) using Reciprocal Rank Fusion (RRF).
*   💾 **OOM (Out-of-Memory) Safe Batching Ingestion:** Uses a generator-driven data streaming design to clear text-indexing buffers and forcefully trigger runtime Garbage Collection hooks, establishing a constant memory ceiling over huge codebases.
*   🛡️ **Self-Auditing Confidence Loop Fallback:** Integrates local **Ollama** runtimes to generate deterministic, strictly structured JSON data responses. If the codebase context contains gaps or ambiguous documentation, the pipeline sets confidence to `LOW` and automatically formats a Tech Lead escalation verification sheet.

---
