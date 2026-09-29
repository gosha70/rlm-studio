
<h1>
  <img
    src="docs/RLM_Studio_Logo.png"
    width="200"
    alt="RLM Studio Logo"
    style="vertical-align: middle; margin-right: 12px; position: relative; top: -2px;" />
  RLM Studio
</h1>

**A workbench for Recursive Language Models** — measure whether the RLM pattern beats direct prompting or RAG on *your* documents, with *your* model (local or cloud), at what cost.

RLM Studio is a working implementation of the MIT [Recursive Language Models paper](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab; official repo: [alexzhang13/rlm](https://github.com/alexzhang13/rlm)), extended in areas the original work does not address: side-by-side comparison of Direct / RAG / RLM across providers, persistent telemetry, step-by-step traces and replay, budget guardrails, and a one-command web UI. Formerly published as *RLMKit*.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()
[![CI](https://github.com/gosha70/rlm-studio/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/gosha70/rlm-studio/actions/workflows/ci.yml)

## What is RLM Studio?

LLMs have fixed context windows. When a document is too large for a single prompt, an RLM gives the model a Python sandbox in which it writes code to navigate the content — reading only what is relevant and calling sub-LLMs over the pieces. That helps a lot on some tasks and not at all on others, and the answer depends on the model, the document, and the budget. RLM Studio exists to answer that question with numbers instead of anecdotes.

It ships as a Python library (`rlmstudio`) plus a web UI, and provides:

- **Three execution modes** — Direct (full context), RAG (retrieval-augmented), and RLM (recursive code generation), selectable per run
- **Compare matrix** — runs any combination of the modes across N providers in parallel, ranking by cost, latency, TTFT, cache hit rate, or LLM-judge score
- **Auto mode** — selects Direct or RLM based on content size (< 8K tokens → Direct, ≥ 8K → RLM)
- **100+ LLM providers** — via LiteLLM (OpenAI, Anthropic, Ollama, LM Studio, vLLM, and more) — local models are first-class
- **Persistent telemetry, traces and replay** — every step the model took, what it cost, and how long it waited, across restarts
- **Budget controls** — cap tokens, cost, steps, and time; failures are classified, not silent
- **Sandboxed execution** — RestrictedPython / subprocess / Docker sandboxes behind one interface
- **[The Studio UI](#rlm-studio-web-ui)** — Chat, Compare, Dashboard, Traces, Learn, Settings, served from a single process

### How it relates to other RLM implementations

| | [`rlms`](https://github.com/alexzhang13/rlm) (the paper authors) | [`dspy.RLM`](https://dspy.ai/api/modules/RLM) | **RLM Studio** |
|---|---|---|---|
| What it is | The reference implementation of the paper | A DSPy module (Deno / Pyodide sandbox), optimizer-compatible | A workbench that runs RLM next to Direct and RAG and measures the difference |
| Sandboxes | local, IPython, Docker, Modal, Prime, Daytona, E2B | Deno / Pyodide | RestrictedPython, subprocess, Docker |
| Providers | OpenAI, Anthropic, OpenRouter, Portkey, vLLM, Gemini, Azure | Whatever DSPy is configured with | 100+ via LiteLLM — Ollama, LM Studio and vLLM first-class |
| Compare modes × providers with ranking | — | — | ✅ Compare matrix (cost, latency, TTFT, cache hit rate, judge) |
| Persistent telemetry, traces, replay, budgets, outcome classification | JSONL logger + visualizer | — | ✅ built in |
| Runs the reference implementation as a baseline | n/a | — | ✅ `rlm_official` (`[interop]` extra) |
| Training / RL harness | ✅ | via DSPy optimizers | — (out of scope) |

Use `rlms` when you want the paper's implementation and its training story; `dspy.RLM` inside DSPy pipelines; RLM Studio when the question is *"does RLM beat Direct or RAG on my documents, with my model, at what cost?"* — it runs `rlms` for you as one of the rows. Landscape as of September 2026.

## Where RLM Studio Shines

RLM Studio solves a specific class of problem. Knowing when it helps (and when it doesn't) saves you from over-engineering.

### Good fit

**Large documents that blow past context windows.** When inputs regularly exceed `50K`–`100K` tokens, full-context prompting gets expensive and answer quality degrades. RLM mode lets the model explore selectively, - reading only the sections it needs at a fraction of the token cost.

**Production workloads that need cost and time guardrails.** RLM Studio caps tokens, cost, steps, and wall-clock time per execution. The outcome classifier flags failures and degraded results automatically. If you're running LLM workloads in production, budget enforcement is built in.

**Tasks where targeted exploration beats brute-force context.** Compliance reviews, contract analysis, multi-section report synthesis, and codebase Q&A all benefit from an LLM that can navigate to relevant sections instead of processing everything at once. RLM's `peek()`, `grep()`, and `chunk()` sandbox gives the model surgical access.

**Provider and mode benchmarking.** The **LLM Tuner** panel runs the same query across N providers × M modes in parallel, ranking results by cost, speed, or token efficiency. Ideal for choosing between providers or tuning RLM vs Direct tradeoffs.

<img height="500" alt="image" src="https://github.com/user-attachments/assets/4adebfaf-0380-4566-894b-a5292bfb674e" />


### Not a fit

**Documents that fit comfortably in context.** If your inputs stay under `8K` tokens, Direct mode works, but a standard LLM client is likely sufficient for your needs.

**Workloads that don't benefit from code generation in the loop.** RLM's core mechanism is an LLM writing `Python` to explore content. For straightforward Q&A, short-text summarization, or classification, the recursive loop adds latency without adding value.

**Massive production retrieval workloads.** RLM Studio includes a RAG mode that's a good fit for embedded knowledge bases and Chat Provider workflows where retrieval beats context-window navigation. For very large retrieval workloads (millions of documents, sharded indexes), a dedicated RAG infrastructure (Qdrant / Weaviate / Pinecone + reranker) is the right tool — RLM Studio's RAG is not built to compete at that scale.

## Deployment model & support boundary

RLM Studio v1.0 targets **single-node self-hosted use** by developers, prompt engineers, and small teams. That is the supported surface.

**Supported in v1.0:**

- One backend process + one frontend process on a single machine (laptop, workstation, or a shared GPU host like DGX Spark).
- Per-execution budgets (tokens, cost, time, steps) and per-provider connection testing.
- Large documents end-to-end through `/api/chat` on a single user's session — the 8MB-class payloads described in [docs/hosts/dgx-spark.md](docs/hosts/dgx-spark.md) work today.
- State persistence across restarts via `~/.rlm-studio/` and the SQLite telemetry DB, with auto-migration on upgrade.

**Explicitly out of scope in v1.0:**

- **Shared multi-tenant service.** There is no per-user request queue, no tenant isolation, and no admission-control layer. Two users uploading 8MB documents simultaneously will both run; the inference backend (vLLM, Ollama, cloud API) is responsible for handling that load, not RLM Studio.
- **Automatic failover between inference backends.** If your configured provider goes offline, the run fails with a classified error; RLM Studio does not silently retry against a different provider.
- **Horizontal scale-out.** There is no clustering, no distributed telemetry, no cross-process queue (Redis/Celery).
- **Enterprise deployment packaging** beyond the Dockerfiles + compose stack shipped here.

If those are requirements, RLM Studio v1.0 is not the right fit today; they are tracked as later milestones. For operational details — state paths, backup/restore, and upgrade behavior — see [docs/OPERATIONS.md](docs/OPERATIONS.md).

## Quick Start

### One-click — RLM Studio (the easiest path)

```bash
pip install "rlm-studio[all]"
rlm-studio studio
```

That installs the package with all extras, starts the API server, mounts the bundled web UI, and opens [http://localhost:8000/studio](http://localhost:8000/studio) in your default browser. Chat, Dashboard, Traces, Compare, Learn, and Settings all run from a single process — no Node, no second terminal.

> **Pre-release (September 2026):** 1.0.0 is not on PyPI yet, so the `pip install` above will not resolve until the release tag lands. Until then, run from a checkout — `git clone https://github.com/gosha70/rlm-studio && cd rlm-studio && uv sync --extra all` — then either the dev stack (`uv run python -m rlmstudio.server --reload` plus `cd frontend && npm run dev`) or build the bundled UI once (see [RELEASING.md](RELEASING.md)) and `uv run rlm-studio studio`. This note is removed in the release PR.

To skip the browser auto-open: `rlm-studio studio --no-browser`. To change the bind address: `rlm-studio studio --host 0.0.0.0 --port 9000`.

### One-click — self-hosted via Docker Compose

```bash
git clone https://github.com/gosha70/rlm-studio.git
cd rlm-studio
cp .env.example .env   # add at least one provider key
docker compose up --build
```

Backend at [http://localhost:8000](http://localhost:8000), frontend at [http://localhost:3000](http://localhost:3000). State persists to the `rlm-studio-data` named volume. See `docs/OPERATIONS.md` for the backup/restore procedure.

### Library use (Python only, no UI)

```bash
# Using uv (recommended)
uv sync --extra all

# Using pip
pip install -e ".[all]"
```

```python
from rlmstudio import interact

result = interact(
    content="Your document text here...",
    query="What is this about?",
    provider="openai",
    model="gpt-4o"
)

print(result.answer)
print(f"Tokens: {result.total_tokens:,}")
print(f"Cost: ${result.total_cost:.4f}")
print(f"Mode used: {result.mode_used}")
```

### Execution Modes

```python
# Direct — send full content in one LLM call (small documents)
result = interact(content, query, mode="direct")

# RLM — LLM writes Python to explore content recursively (large documents)
result = interact(content, query, mode="rlm")

# Auto — let RLM Studio choose based on content size (default)
result = interact(content, query, mode="auto")

# Compare — run both RLM and Direct, return metrics for each
result = interact(content, query, mode="compare")
```

| Mode | Best For | How It Works |
|------|----------|--------------|
| `direct` | < 8K tokens | Full context in single LLM call |
| `rag` | Static corpora, repeated retrieval | Chunk + embed + retrieve top-k, then LLM call with retrieved context |
| `rlm` | Any size, especially >50K tokens | LLM writes code to navigate content via `peek()`, `grep()`, `chunk()` |
| `rlm_official` | Benchmarking Studio's loop against the paper authors' | Runs [alexzhang13/rlm](https://github.com/alexzhang13/rlm) (`rlms`) as an engine under Studio's budgets, traces and ranking. Needs `pip install "rlm-studio[interop]"`; UI and REST API only (not the Python client) |
| `auto` | Mixed / unknown sizes | Selects `direct` (< 8K) or `rlm` (≥ 8K) automatically |
| `compare` | Benchmarking | Runs the other modes concurrently, returns metrics for each side by side |

### Configuration

```python
result = interact(
    content=large_document,
    query="Summarize the key findings",
    mode="auto",
    provider="anthropic",
    model="claude-sonnet-4-6",
    max_steps=16,          # RLM loop budget
    temperature=0.7,
    verbose=True           # Print progress
)
```

### Supported Providers

Set the API key as an environment variable, then pass the provider name:

| Provider | Env Variable | Example Model |
|----------|-------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-4-6` |
| Ollama | (local, no key) | `llama3.2` (must specify `model=`) |
| LM Studio | (local, no key) | Any served model (must specify `model=`) |
| Google | `GOOGLE_API_KEY` | `gemini-pro` |
| 100+ more | via LiteLLM | See [LiteLLM docs](https://docs.litellm.ai/) |

## RLM Studio web UI

The web UI is where you experiment interactively: demo RLM behaviour, tune runtime parameters, compare execution modes side by side, and monitor cost/performance metrics. The one-click `rlm-studio studio` command above runs it; the two-process layout below is the development workflow.

### Starting the UI in development mode

```bash
# Terminal 1: Backend API server (default port 8000)
uv run python -m rlmstudio.server --reload

# Custom port — set RLM_STUDIO_PORT and point the frontend at the same address
RLM_STUDIO_PORT=8080 uv run python -m rlmstudio.server --reload
NEXT_PUBLIC_API_URL=http://localhost:8080 npm run dev  # Terminal 2

# Terminal 2: Frontend (default, connects to port 8000)
cd frontend && npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### What You Can Do

- **Chat** — Upload documents and query them using one or more Chat Providers in parallel. Responses appear in a side-by-side column layout with per-response metrics (tokens, cost, latency).
- **Dashboard** — View aggregated metrics per session: total tokens, cost, average latency, token savings. Charts break down performance by provider and execution mode.
- **Traces** — Inspect every execution step-by-step. See the code the LLM generated, the output it received, token counts, and timing for each step. Visualize as timeline, tree, or raw code. Each row has a **Replay in Learn** action that reconstructs the run as a scrubbable walkthrough on the Learn tab.
- **Learn** — Concepts, Cookbook, and Troubleshooting guides with an interactive replay walkthrough that steps through the RLM loop. Open a bundled demo from the Concepts page, or deep-link from Traces to replay any past execution.
- **Settings** — Configure providers, create Chat Providers, set budgets, manage profiles, customize system prompts, and switch themes.

For a complete walkthrough of all RLM Studio features, see **[docs/rlm-studio-guide.md](docs/rlm-studio-guide.md)**.

## Documentation

In-depth guides live under [`docs/`](docs/). Start with whichever matches what you're trying to do:

| Guide | For |
|-------|-----|
| [docs/rlm-concepts.md](docs/rlm-concepts.md) | Understanding the RLM paradigm — the recursive loop, circuit breakers, the JSON action protocol, and what RLM Studio adds beyond the original paper |
| [docs/rlm-studio-guide.md](docs/rlm-studio-guide.md) | Full walkthrough of the RLM Studio web app — pages, settings, workflows, Learn / Compare / Judge |
| [docs/rlm-prompt-tuning.md](docs/rlm-prompt-tuning.md) | Safe patterns for customizing system prompts in Direct / RLM / RAG modes |
| [docs/hosts/README.md](docs/hosts/README.md) | Connecting an LLM backend — decision tree across providers, deployment topologies, security boundaries. Per-provider guides for OpenAI, Anthropic, Ollama, LM Studio, vLLM, DGX Spark |
| [docs/RLM_Studio_Design_Document.md](docs/RLM_Studio_Design_Document.md) | Design rationale, architecture decisions, and internals for contributors |

## Architecture

```mermaid
graph TB
    subgraph Frontend["Frontend (Next.js 16 + React 19)"]
        Chat[Chat Page]
        Dash[Dashboard]
        Traces[Traces]
        Learn[Learn]
        Settings[Settings]
    end

    subgraph Server["FastAPI Backend"]
        REST[REST API]
        WS[WebSocket]
    end

    subgraph App["Application Layer"]
        Direct[RunDirectUseCase]
        RLM[RunRLMUseCase]
        RAG[RunRAGUseCase]
    end

    subgraph Infra["Infrastructure"]
        LiteLLM[LiteLLMAdapter<br/>100+ providers]
        Sandbox[RestrictedPython<br/>Sandbox]
    end

    subgraph Domain["Domain (zero deps)"]
        Entities[Entities & Value Objects]
        Ports[Ports / Protocols]
    end

    Frontend -->|HTTP + WebSocket| Server
    REST --> App
    WS -->|streaming| App
    App --> Infra
    App --> Domain
    Infra -.->|implements| Ports
```

**Dependency rule:** inner layers never import from outer layers. All external dependencies sit behind port interfaces.

```
src/rlmstudio/
├── domain/            # Entities, ports (zero external deps)
├── application/       # Use cases (run_rlm, run_direct, run_rag)
├── infrastructure/    # Adapters (LiteLLM, sandbox, storage)
├── server/            # FastAPI REST + WebSocket server
└── api.py             # Public interact() / complete() API

frontend/              # Next.js 16 + React 19 + shadcn/ui
├── src/app/           # Pages: chat, dashboard, traces, settings
├── src/components/    # 50+ React components (shadcn/ui + custom)
└── src/lib/           # API client, WebSocket hook, types
```

## Running Tests

```bash
# All tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=rlmstudio --cov-report=term-missing

# Frontend type check
cd frontend && npx tsc --noEmit
```

## License

Copyright (c) 2026 EGOGE. MIT License — see [LICENSE](LICENSE).
