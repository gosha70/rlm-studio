# Contributing to RLM Studio

Thank you for your interest in contributing to RLM Studio! This guide will help you get started.

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for Python dependency management
- Node.js 22+ (for the frontend)
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/gosha70/rlm-studio.git
cd rlm-studio

# Install Python dependencies
uv sync

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys (optional — providers also configurable in the UI)

# Run the backend test suite
uv run pytest

# (Optional) Install and test the frontend
cd frontend && npm install && npm test
```

### Running the App Locally

```bash
# Terminal 1: backend API
uv run python -m rlmstudio.server --reload

# Terminal 2: frontend
cd frontend && npm run dev
```

Open <http://localhost:3000> in your browser.

## Project Conventions

See `CLAUDE.md` for the full set of project conventions. Summary:

### Architecture

- RLM Studio uses Clean Architecture. The dependency rule is non-negotiable: `domain` → `application` → `infrastructure` / `server`. Inner layers never import from outer.
- Ports (`LLMPort`, `SandboxPort`, `StoragePort`, `EmbeddingPort`) are `@runtime_checkable` Protocols. Use structural checks (`hasattr`, `callable`), not `isinstance(x, Protocol)`.
- All LLM calls go through `LLMPort` (primary adapter: `LiteLLMAdapter`). No direct SDK imports outside `infrastructure/llm/`.
- All sandbox executions go through `SandboxPort` — choose from Local, Subprocess, Restricted, Docker.

### Prompts

- All prompts live as versioned YAML files in `src/rlmstudio/prompts/`. Never inline system-prompt or rubric strings in code.

### LLM Calls

- Every LLM call must be wrapped with latency and token logging.
- Embedding model IDs live in config — never hardcode.

### Code Style

- **Type hints** on all public APIs.
- **Docstrings** on all public APIs.
- **Pydantic v2** for request/response models.
- **Ruff** + **Black** for lint and format.
- **No magic strings** for mode / trace-key literals — import from the centralized constants.
- Target **80%+** test coverage on `application/` and `infrastructure/`.

## How to Contribute

### Reporting Bugs

Use the [Bug Report](https://github.com/gosha70/rlm-studio/issues/new?template=bug_report.md) issue template. Include:

- Steps to reproduce
- Expected vs. actual behavior
- Python version, OS, and relevant model/provider info

### Suggesting Features

Use the [Feature Request](https://github.com/gosha70/rlm-studio/issues/new?template=feature_request.md) issue template.

### Submitting Pull Requests

1. **Fork** the repository and create a branch from `master`.
2. **Follow** the coding standards described above.
3. **Write tests** — no live API calls in unit tests. Use mock LLM response fixtures.
4. **Run the test suites** before submitting:
   ```bash
   uv run pytest
   cd frontend && npm test
   ```
5. **Fill out the PR template** completely.
6. **Keep PRs focused** — one logical change per PR.

### Testing Guidelines

- Use `pytest` with fakes for LLM responses; no live API calls in unit tests.
- Use `TestClient` for REST endpoint tests; real WebSocket connections for streaming e2e.
- `vitest` for frontend component and integration tests (`cd frontend && npm test`).
- Use `reset_state()` to isolate the AppState singleton between tests.
- If your change affects retrieval or generation quality, include before/after metrics in your PR description.

## Project Structure Overview

| Directory | Purpose |
|-----------|---------|
| `src/rlmstudio/domain/` | Entities, value objects, ports — zero external deps |
| `src/rlmstudio/application/` | Use cases, services, constants |
| `src/rlmstudio/infrastructure/` | Adapters (LiteLLM, sandbox, storage, embeddings) |
| `src/rlmstudio/server/` | FastAPI app, routes, Pydantic models, AppState |
| `src/rlmstudio/core/` | Original RLM controller loop |
| `src/rlmstudio/envs/` | Execution environments (`DockerExecutor`, `PyReplEnv`) |
| `src/rlmstudio/llm/` | Provider registration and auto-detection |
| `src/rlmstudio/prompts/` | Versioned YAML prompt templates |
| `tests/` | Backend test suite |
| `frontend/` | Next.js + React + shadcn/ui UI |
| `docs/` | User-facing docs (`rlm-concepts`, `rlm-studio-guide`, `hosts/`, `troubleshoot.yaml`) |
| `doc_internal/` | Internal specs, plans, ADRs (gitignored — contributor-local) |
| `docker/` | Docker configurations |

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

### Headers in new files

Start a new first-party source file with the two-line header Apache
[documents](https://www.apache.org/foundation/license-faq.html#apply), using the comment
syntax of the language:

```python
# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0
```

Two rules matter more than the header itself:

- **Never replace someone else's notice.** Code brought in from another project keeps its
  own copyright line and license identifier, and its license text belongs in the
  repository alongside ours. Overwriting that notice is how a license violation happens,
  and it is not reversible by a later commit.
- **Leave existing files alone.** Most files here carry no header, which is fine: the
  `LICENSE` file governs the repository either way. Adding one to a file you are already
  editing is welcome; a sweep across files you are not touching only obscures history.
