# Engineering Decisions and Tradeoffs

This document captures key architecture decisions for Memories and the tradeoffs behind each one.

---

## D1: Use ONNX Runtime Instead of PyTorch for Embeddings

- Status: accepted
- Decision: run `all-MiniLM-L6-v2` through ONNX Runtime (`onnx_embedder.py`) rather than `sentence-transformers` + PyTorch runtime.
- Why:
  - significantly smaller runtime/image footprint
  - faster cold start for local container workflows
  - same embedding space/model family retained
- Tradeoff:
  - custom embedder wrapper adds maintenance surface
  - less out-of-the-box flexibility than full PyTorch stack

## D2: Use Memories `IndexFlatIP` for Vector Search

- Status: accepted
- Decision: use exact inner-product search with normalized embeddings.
- Why:
  - deterministic and simple behavior
  - no training/build phase for IVF/HNSW graph parameters
  - easier correctness/debugging for local-first usage
- Tradeoff:
  - linear scan cost grows with index size
  - not optimized for very large multi-tenant datasets

## D3: Use Hybrid Retrieval (Vector + BM25) with RRF

- Status: accepted
- Decision: combine semantic similarity with lexical ranking and fuse via reciprocal-rank fusion.
- Why:
  - improves precision on exact-term queries
  - better robustness across short/long queries and naming-heavy content
  - avoids overfitting to either dense or sparse retrieval alone
- Tradeoff:
  - extra compute and index maintenance (BM25 rebuilds on writes)
  - introduces fusion parameters to tune

## D4: Keep Metadata in JSON Alongside the Vector Index

- Status: accepted
- Decision: persist metadata in `metadata.json` and vectors in `vector_index.bin`.
- Why:
  - transparent, debuggable local files
  - easy backup/restore mechanics
  - no external database dependency
- Tradeoff:
  - full-file rewrites on update-heavy workloads
  - not ideal for high-write concurrent scenarios

## D5: Favor Integrity and Recovery Over Max Write Throughput

- Status: accepted
- Decision: use lock-guarded writes, pre-change backups for destructive operations, and strict index/metadata integrity checks.
- Why:
  - predictable behavior for single-user and small-team workflows
  - safer recovery from interruptions/corruption
  - simpler operational model
- Tradeoff:
  - slower write paths under heavy mutation
  - limited horizontal write scaling

## D6: Keep LLM Extraction Optional and Provider-Agnostic

- Status: accepted
- Decision: extraction lives behind explicit env configuration (`EXTRACT_PROVIDER`), with provider abstraction for Anthropic/OpenAI/ChatGPT Subscription/Ollama.
- Why:
  - base memory features remain fully local and zero-cost
  - users can choose accuracy/cost/privacy profile
  - avoids coupling core service availability to an external LLM
- Tradeoff:
  - more configuration combinations to test
  - behavior differences across providers (e.g., AUDN support)

## D7: Add Post-Extraction Memory Reclamation Controls

- Status: accepted
- Decision: bound extraction payload sizes/concurrency and run post-extract GC + allocator trim (`MEMORY_TRIM_*`).
- Why:
  - extraction bursts can create transient allocation spikes
  - long-running containers should return toward baseline after heavy workloads
  - keeps memory behavior more predictable in shared hosts
- Tradeoff:
  - small CPU overhead after extraction requests
  - allocator trim effectiveness depends on runtime/platform allocator behavior

## D8: Embed OAuth Logic Directly Instead of Gateway Services

- Status: accepted
- Decision: port ChatGPT OAuth2+PKCE token exchange as pure-stdlib Python helpers (`chatgpt_oauth.py`) rather than running separate gateway services.
- Why:
  - zero new pip dependencies (all `urllib.request`, `hashlib`, `secrets`)
  - no runtime services to manage — token exchange happens at provider init
  - same pattern as existing Anthropic OAuth auto-detection
  - CLI tool (`python -m memories auth chatgpt`) handles one-time browser-based setup
- Tradeoff:
  - eager token exchange in `__init__` blocks startup on network call (other providers defer to first `complete()`)
  - refresh token is a long-lived secret stored in `~/.config/memories/env`

## D9: Enable Full AUDN for Ollama via JSON Format Constraint

- Status: accepted
- Decision: set `supports_audn = True` on Ollama and add `"format": "json"` to the Ollama API payload.
- Why:
  - modern local models (Llama 3, Mistral, Qwen) reliably produce structured JSON when constrained
  - `"format": "json"` is natively supported by Ollama and dramatically reduces parse failures
  - eliminates the ADD/NOOP-only limitation for local-model users
- Tradeoff:
  - smaller/older local models may still produce malformed AUDN decisions
  - fallback behavior on parse failure is ADD-all (same as before)

---

## Revisit Triggers

Re-evaluate these decisions when:

- index size or query QPS outgrows exact-search economics
- multi-writer or multi-tenant requirements become first-class
- durability requirements exceed file + backup semantics
- extraction latency/cost dominates end-to-end workflow performance
