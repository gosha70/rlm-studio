# RLM Studio Guide

RLM Studio is a web application for experimenting with, tuning, and monitoring RLM Studio. It lets you compare how different LLM providers and execution modes (Direct, RLM, RAG) handle the same queries — side by side, with full cost and performance metrics.

## Prerequisites

- Python 3.11+ with `uv`
- Node.js 22+
- At least one LLM provider API key (OpenAI, Anthropic) or a local model (Ollama, LM Studio)

## Starting the Application

```bash
# Terminal 1: Backend API
uv run python -m rlmstudio.server --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

The frontend runs on `http://localhost:3000`. The backend runs on `http://localhost:8000`. The frontend proxies API calls to the backend automatically (configured in `next.config.ts`).

## Pages Overview

RLM Studio has six pages, accessible from the sidebar:

| Page | Purpose |
|------|---------|
| **Chat** | Send queries to one or more Chat Providers in parallel |
| **Compare** | Run the same query across a Provider × Mode grid (LLM Tuner) |
| **Dashboard** | View aggregated metrics and charts per session |
| **Traces** | Inspect individual execution traces step by step |
| **Learn** | Concepts, Cookbook, and Troubleshooting guides with a scrubbable RLM loop replay |
| **Settings** | Configure providers, Chat Providers, budgets, profiles, prompts, and theme |

---

## Settings

Settings is where you configure everything before running experiments. It has six tabs.

### Providers

Providers are the raw LLM connections. RLM Studio supports OpenAI, Anthropic, Ollama (local), LM Studio (local), and vLLM (local) out of the box — plus any of the 100+ backends LiteLLM supports via env var overrides. For each provider you can:

- **Select a model** from the provider's model catalog
- **Enter an API key** (or rely on environment variables like `OPENAI_API_KEY`)
- **Set a custom endpoint** (for local models or proxies)
- **Configure runtime settings** — temperature, top_p, max output tokens, timeout
- **Enable/disable** the provider for chat
- **Test the connection** to verify the key and endpoint work

Providers that detect an environment variable show "API key set" automatically. You only need to configure them manually if you want to override the key or change the default model.

For step-by-step setup per backend, see **[docs/hosts/README.md](hosts/README.md)** — it covers the decision tree across backends, deployment topologies (all-local vs laptop + remote GPU vs all-cloud), and security boundaries for keyless local backends.

#### Scheduled connection testing

RLM Studio can re-test every configured provider on a timer so you see offline backends before a chat run fails. Configure it under **Settings → Providers**:

| Setting | Default | Purpose |
|---------|---------|---------|
| **Connection test interval (minutes)** | 0 | How often the background daemon re-tests all providers. `0` disables the daemon. Range: 0–1440 (24 h). |

Each cycle tests up to 5 providers in parallel with a 10-second per-test timeout. A provider flips to **offline** only after 2 consecutive failures (to avoid flap from a transient network blip). A single manual **Test Connection** success flips an offline provider back to connected immediately.

Each provider row shows three audit fields:

- **Last tested at** — UTC timestamp of the most recent test.
- **Last tested by** — `manual` (you clicked Test Connection) or `background` (the daemon).
- **Consecutive failures** — number of back-to-back failures against the current threshold.

### Chat Providers and Profiles

A **Chat Provider** binds a specific LLM (provider + model) to a **Profile**. The Profile controls execution mode, runtime settings, and budget limits. Editing a Profile immediately affects all Chat Providers that reference it — no caching, no duplication.

| | Chat Provider | Profile |
|---|---|---|
| **Purpose** | A runnable configuration you select on the Chat page | A reusable settings template referenced by Chat Providers |
| **Bound to an LLM?** | Yes — specific provider + model (e.g., Anthropic / claude-sonnet-4-6) | No — provider-agnostic |
| **Used in Chat?** | Yes — select one or more, each executes independently | Indirectly — through the Chat Providers that reference it |
| **Controls** | LLM selection, RAG-specific config | Execution mode, runtime settings (temp, top_p, max tokens, timeout), budget limits |
| **Editable fields** | Name, LLM provider, model, profile, RAG config | Strategy, runtime settings, budget, system prompts, description |

**In practice:** Pick or create a Profile (e.g., "RLM deep" — temperature 0.4, 32 steps, 4096 max tokens). Then create Chat Providers that pair that Profile with specific LLMs. Change the Profile's temperature and every Chat Provider referencing it picks up the new value immediately.

### Chat Providers

A **Chat Provider** is a named, runnable configuration that pairs a specific LLM with a Profile. You select Chat Providers on the Chat page to execute queries.

Each Chat Provider specifies:

- **LLM Provider + Model** — e.g., Anthropic / claude-sonnet-4-6
- **Profile** — controls execution mode (Direct / RLM / RAG), runtime settings, and budget limits
- **RAG config** (optional) — chunk size, overlap, top_k, embedding model (only shown when the profile's strategy is RAG)

Create multiple Chat Providers with different LLM + Profile combinations, then select them on the Chat page to compare results side by side.

**Example setup for comparing modes:**

| Chat Provider Name | Provider | Profile | Notes |
|-------------------|----------|---------|-------|
| GPT-4o Direct | OpenAI / gpt-4o | Accurate | Baseline — full context, direct mode |
| GPT-4o RLM | OpenAI / gpt-4o | RLM deep | Same model, recursive exploration |
| Claude Sonnet Direct | Anthropic / claude-sonnet-4-6 | Accurate | Cross-provider comparison |
| Claude Sonnet RLM | Anthropic / claude-sonnet-4-6 | RLM deep | Cross-provider + cross-mode |

**Example setup for tuning RLM parameters:** Create custom profiles with different step budgets, then assign them to Chat Providers using the same LLM.

| Chat Provider Name | Profile | Steps | Temperature | Notes |
|-------------------|---------|-------|-------------|-------|
| RLM Conservative | Custom: RLM-8 | 8 | 0.3 | Fewer steps, lower randomness |
| RLM Balanced | RLM deep (built-in) | 32 | 0.4 | Built-in default |
| RLM Aggressive | Custom: RLM-64 | 64 | 0.7 | More exploration budget |

### Budget

Global budget limits that apply to all executions:

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| Max Steps | 16 | Maximum RLM loop iterations |
| Max Tokens | 50,000 | Total token limit across all steps |
| Max Cost (USD) | $2.00 | Hard cost cap per execution |
| Max Time (seconds) | 30 | Wall-clock timeout |
| Max Recursion Depth | 5 | Nested subcall depth limit |

Use budget controls to prevent runaway costs during experimentation. Start with conservative limits and increase as you understand your workload's requirements.

### Profiles

Profiles are saved presets of global defaults — runtime settings, budget limits, and system prompts bundled into a reusable template. Profiles are **not bound to any LLM provider**. They set the baseline that Chat Providers and the programmatic API inherit.

Each profile includes:

- **Name** and **description**
- **Strategy** (the default execution mode: Direct, RLM, etc.)
- **Runtime settings** — temperature, top_p, max output tokens, timeout
- **Budget limits** — max steps, tokens, cost, time, recursion depth
- **System prompts** — per-mode prompt overrides

RLM Studio ships with four **built-in profiles** (marked with a lock icon, cannot be deleted):

| Profile | Mode | Temp | Max Tokens | Steps | Use Case |
|---------|------|------|------------|-------|----------|
| **Fast & cheap** | Direct | 0.5 | 1,000 | 8 | Quick, low-cost responses with conservative token limits |
| **Accurate** | Direct | 0.2 | 4,096 | 16 | High-quality, precise answers with low temperature |
| **RLM deep** | RLM | 0.4 | 4,096 | 32 | Deep recursive reasoning with high step budget |
| **RAG retrieval** | RAG | 0.3 | 4,096 | 8 | Retrieval-augmented generation with moderate token budget |

**Actions on profiles:**

- **Activate** (play button) — apply the profile's settings as the new global defaults
- **Clone** (copy button) — create an editable copy of any profile (including built-ins)
- **Edit** (pencil button) — modify strategy, runtime settings, and budget inline (custom profiles only)
- **Delete** (trash button) — remove a custom profile. Profiles referenced by Chat Providers cannot be deleted — reassign the Chat Providers first.

### Prompts

Customize the system prompts used for each execution mode:

- **Direct** — the system prompt sent with full-context queries
- **RLM** — the system prompt that instructs the LLM how to use the sandbox tools (`peek()`, `grep()`, `chunk()`, `select()`)
- **RAG** — the system prompt for retrieval-augmented queries

You can also apply **prompt templates** — predefined prompt sets that you can load and customize.

For safe RLM customization patterns and multi-document anti-patterns, see [rlm-prompt-tuning.md](rlm-prompt-tuning.md).

### Appearance

Switch between **Light**, **Dark**, and **System** themes.

---

## Chat

The Chat page is where you run experiments.

### Workflow

1. **Select Chat Providers** — Use the toolbar dropdown to select one or more Chat Providers. Each selected provider will receive the same query in parallel.

2. **Upload a document** (optional) — Click the upload button to attach a PDF, DOCX, TXT, or other supported file. The file is processed server-side and its content becomes available to the LLM.

3. **Ask a question** — Type your query and press Enter (or click Send).

4. **Compare responses** — Each Chat Provider's response appears in its own column. Below each response you see per-execution metrics:
   - **Tokens** — total tokens consumed
   - **Cost** — USD cost for this execution
   - **Latency** — wall-clock time

5. **Continue the conversation** — Conversation history is maintained per Chat Provider. Each follow-up question includes the relevant conversation history for that provider, enabling multi-turn dialogues. The `conversation_memory_enabled` toggle on each Chat Provider controls whether history is delivered at all; see [Conversation memory](#conversation-memory) below for the per-mode delivery differences.

### Conversation memory

History delivery differs by execution mode:

- **Direct / Compare** — prior turns are assembled into the prompt as a "Previous conversation:" prefix, token-budgeted (default 30% of the model's context window). The LLM sees history as native chat messages.
- **RLM / RAG / Auto** — prior turns are bound as a Python variable named `history` inside the sandbox REPL. The variable is a list of `{"turn": int, "user": str, "assistant": str}` dicts, byte-capped so the sandbox can't exhaust host memory. The model only reads history if its generated code references `history` — meaning history costs zero tokens when the model doesn't need it.

Toggle `conversation_memory_enabled = False` on a Chat Provider to disable history entirely — useful for stateless benchmarking where every query should start clean.

### Side-by-Side Comparison

The column layout scales with the number of selected Chat Providers. For example, selecting three Chat Providers produces a three-column grid where you can visually compare:

- Answer quality and completeness
- Token efficiency (RLM often uses fewer tokens for large content)
- Cost differences between providers and modes
- Latency trade-offs (Direct is faster per call; RLM may be faster overall for large content)

### Sessions

Sessions persist across page reloads. The sidebar lists all sessions. You can:

- Click a session to reload its conversation history
- Create a new session (+ button)
- Delete sessions you no longer need

---

## Dashboard

The Dashboard provides aggregated metrics for a selected session.

### Summary Cards

- **Total Tokens** — sum of all tokens used in the session
- **Total Cost** — sum of all execution costs
- **Avg Latency** — mean execution time
- **Token Savings** — percentage saved by RLM compared to Direct mode

All four summary cards exclude non-usable outcomes (see [Outcome classification](#outcome-classification) below) so averages reflect only runs that produced a real answer.

### Outcome classification

Every execution is classified into one of five outcome categories:

| Category | Meaning | Usable? |
|----------|---------|---------|
| `success` | The run completed and produced an answer | Yes |
| `timeout` | The wall-clock timeout fired before the run finished | No |
| `budget_exhausted` | Token / cost / step budget ran out | No (unless ≥50 chars of answer returned) |
| `context_overflow` | The prompt exceeded the model's context window | No |
| `general_error` | Any other failure — adapter error, sandbox violation, network failure, etc. | No |

Non-usable outcomes are excluded from cost, latency, and token aggregations across the Dashboard — the summary cards, charts, and ranking all count only the runs that actually produced an answer. The Recent Executions table still shows every run including failures, with the outcome category in the status column.

For session-level failure metrics, use `GET /api/metrics/failures/{session_id}` — the response includes the failure rate, a breakdown by category, and a breakdown by provider and mode. The Dashboard's failure chart is driven by this endpoint.

### Charts

- **Comparison Chart** — RLM vs Direct mode metrics side by side
- **Cost by Provider** — pie chart showing cost distribution across LLM providers
- **Provider Performance** — bar chart comparing tokens, latency, and cost per provider
- **Chat Provider Performance** — same breakdown but grouped by Chat Provider (so you can distinguish two configurations using the same LLM)
- **Cost by Chat Provider** — pie chart by Chat Provider
- **Performance Trend** — timeline of tokens, cost, and latency across executions

### Recent Executions Table

A table listing every execution in the session with mode, provider, tokens, cost, and latency.

---

## Traces

The Traces page lets you inspect individual executions in detail.

### Execution List

A table of all recent executions showing:

- **Query** — the user's question
- **Chat Provider** — which Chat Provider was used
- **Mode** — Direct, RLM, or RAG
- **Status** — complete, running, or error
- **Tokens** — total tokens consumed
- **Cost** — USD cost

Click any row to load its full trace.

### Trace Detail

When you select an execution, the trace detail shows:

**Summary bar** — query text, mode, Chat Provider name, steps used/limit, tokens, cost, and status.

**Three views** (tabs):

1. **Timeline** — chronological list of every step the LLM took. Each step shows the action type, token count, cost, and duration.

2. **Tree** — hierarchical view showing recursion depth. Subcalls appear nested under their parent steps.

3. **Code** — shows only the code-generation steps. For each step you see:
   - The Python code the LLM wrote
   - The output that code produced
   - Token and cost metrics

**Step Detail** — click any step in Timeline or Tree to see its full details: the raw code, output, model used, duration, and token breakdown.

### Replay in Learn

Every row in the execution list has a **Replay in Learn** action. Clicking it navigates to `/learn/replay/{execution_id}` and renders that specific run as a scrubbable, step-by-step walkthrough — the same widget the Learn tab uses for bundled demos, but driven by your trace. Useful for sharing a link to a specific failure, teaching the RLM loop from a real run, or reviewing a teammate's session.

### Deleting traces

Traces accumulate as you experiment. Two affordances keep the list manageable:

- **Single-row delete** — each row has a trash icon; clicking opens a confirm dialog and removes just that execution.
- **Bulk delete (Gmail-style)** — tick the checkbox in the header to select all visible rows, or click individual checkboxes to build a selection. The toolbar shows a delete action with the count; confirm to remove all selected executions.

Deletions are permanent — once removed, a trace cannot be replayed, graphed, or judged. Deleting a session does not delete its traces; delete the traces first if that's what you want.

---

## Learn

The Learn page is an in-product tutorial surface. It teaches the RLM paradigm through live examples and renders the same `docs/` content you're reading now as interactive guides.

### Sub-pages

| Sub-page | What it shows |
|----------|---------------|
| **Concepts** | The RLM loop explained with an interactive replay walkthrough (6-node SVG diagram, play / pause / step / reset controls at 1×, 1.5×, 2× speeds). Ships with a bundled demo replay. |
| **Cookbook** | The per-host setup guides from `docs/hosts/*.md` rendered in-app. Pick a backend (Anthropic, OpenAI, Ollama, LM Studio, vLLM, DGX Spark) and get the Install → Start → Model → Add to RLM Studio → Test flow without leaving the app. |
| **Troubleshooting** | A searchable, structured view of [`docs/troubleshoot.yaml`](troubleshoot.yaml). Symptoms are grouped by area (connectivity, runtime, sandbox, etc.); each entry has a cause and a fix. |

### Replay walkthrough

The walkthrough widget is a three-pane layout — controls and step list on the left, a 6-node SVG diagram in the centre, and the currently-focused step's detail on the right. As you step forward, the active node highlights, the step list scrolls, and the right pane updates with the code the LLM wrote, the output it received, and token counts.

Two entry points:

1. **Bundled demo** — the Concepts page ships with a pre-recorded replay so you can try the widget without running anything yourself.
2. **From a trace** — any Traces row exposes **Replay in Learn**, which deep-links to `/learn/replay/{execution_id}`. The same widget loads, but backed by your real execution.

A **truncation banner** appears when the underlying trace was longer than the replay cap — it tells you how many steps the full run had so you know the walkthrough is a head/tail slice rather than the complete run.

### Deep-link from Traces

The Traces row CTA is keyboard-safe — Enter or Space on the **Replay in Learn** button navigates without also firing the row's open-trace handler. Clicking the row itself still opens the full trace detail; clicking the button opens the walkthrough.

---

## Compare

The Compare page — also known as the **LLM Tuner** — runs the same query against a grid of (LLM Provider × Execution Mode) cells and ranks the results. It's the fastest way to answer *"which provider + mode should I use for this kind of workload?"* without clicking through Chat N×M times.

### Workflow

1. Pick one or more **LLM providers** (each becomes a column).
2. Pick one or more **execution modes** — Direct, RLM, RAG, and, when the `interop` extra is installed, Official RLM (each becomes a row).
3. Optionally upload a document, then type a query.
4. Pick a **ranking metric** (see below).
5. Click **Run** — every cell executes in parallel against `POST /api/chat/compare-matrix`.

Each cell shows the answer, token count, cost, and latency. The winner for your selected metric is highlighted.

### Ranking metrics

| Metric | Winner is the cell with… |
|--------|---------------------------|
| **Cost** | Lowest total USD cost |
| **Tokens** | Fewest total tokens consumed |
| **Latency** | Lowest wall-clock time |
| **Answer per cost** | Best answer-length-per-dollar ratio |
| **Judge score** | Highest LLM-as-judge `overall_score` (requires a judge Chat Provider set in Settings; see [Judge & scoring](#judge--scoring)) |

### Official RLM engine (`rlm_official`)

The fourth mode runs the paper authors' own implementation — [alexzhang13/rlm](https://github.com/alexzhang13/rlm), PyPI package `rlms` — as an engine inside the matrix, so one run can put Official-RLM, Studio-RLM, Direct and RAG side by side on the same document, provider and budget, with the same traces, telemetry and ranking. Skeptical of whether a number reflects the RLM idea or Studio's implementation? This is the row that answers it.

**Install.** `pip install "rlm-studio[interop]"` (included in `[all]`), then restart the server. `GET /api/engines` reports whether the engine can run; until it can, the option is shown disabled with the reason as its tooltip, and a request naming the mode is rejected with the same reason before anything is recorded.

**Providers.** `openai` and `anthropic` use the engine's native clients with the LLM Provider's key. `ollama`, `lmstudio` and `vllm` go through the engine's OpenAI-compatible client pointed at the provider's endpoint (`/v1` is appended when missing). Same model id, same endpoint, same key as the sibling Studio cells — that is what makes the comparison honest. See [docs/hosts/README.md §8](hosts/README.md#8-official-rlm-engine-rlm_official--provider-mapping) for the matrix.

**Budgets.** The Chat Provider's *Max steps* and *Timeout* apply (`max_iterations` and the engine's own timeout); the recursion depth, token and cost caps from **Settings → Budget** map onto the engine's limits too. The engine checks its timeout *between* iterations, so an execution that never returns is stopped by Studio's own wall-clock guard and classified as a timeout like any other run.

**Recursion depth.** The two projects count depth differently, and Studio translates. Studio counts levels of sub-RLM, so *Max recursion depth* 1 means the root loop may spawn one child that runs its own REPL; the engine instead counts the depth at which a node stops looping and becomes a plain model call, and its root already occupies one level. Studio therefore asks the engine for one more level than you set. Setting 0 is the one case it cannot reproduce faithfully: the engine has no way to refuse sub-calls outright, so they degrade to plain model calls and the run says so in its notes.

**Sandbox.** With the Docker sandbox selected in Settings the engine runs its REPL in a container; with any other sandbox it runs the engine's in-process `local` environment, which is **not isolated** — every such run carries a note saying so in its trace metadata. That environment also redirects this process's output and working directory around every code cell, so Studio runs official slots **one at a time**: a second official cell waits for the first, and reports the engine as busy if it waits past its own timeout. Pick the Docker sandbox to run official cells in parallel.

**What you will notice.**

- Cells carry an **official rlms x.y.z** badge (Compare and Traces).
- No streaming: no TTFT / decode timings, and no token-by-token output in Chat.
- The engine reports token usage per run, not per step, so step rows show 0 tokens while the run totals are exact.
- If the engine reports no price for a model and Studio has none either, the cell shows $0 but is flagged `cost_known=false` and ranks **last** on the cost metrics rather than winning as "free". A model Studio's price table answers $0 for while the run really did consume tokens counts as no price, not as free.
- A run the engine stopped for breaking a cap still reports what it spent up to that point, plus the steps it had taken and any partial answer, instead of looking like a run that cost nothing.
- Conversation memory is not delivered to the engine — it receives only the document and the question.
- The Python client's `compare_matrix()` does not accept the mode in 1.0; use the UI or `POST /api/chat/compare-matrix`.

### Ephemeral Chat Providers

Compare builds Chat Providers on the fly from your picked Provider × Mode combinations. They do not appear under **Settings → Chat Providers** and do not persist after the run — the grid is a disposable experiment surface. If you find a combination worth keeping, recreate it as a permanent Chat Provider in Settings.

### Latency expectation

The matrix endpoint is synchronous — the UI waits until every cell completes (or errors) before rendering. Plan for the slowest cell to dominate the clock: a 3×3 grid with one slow provider will wait for that provider before anything renders. Budget overruns and provider failures surface per-cell without blocking siblings, so you still see partial results even when one combination errors.

---

## Judge & scoring

LLM-as-judge is an optional scoring layer: a dedicated judge LLM rates every answer on a rubric, producing an `overall_score` you can sort and compare by. The Compare page uses it as a ranking metric; the Traces page shows it per execution.

### Picking a judge provider

Under **Settings → Providers**, designate one Chat Provider as the **judge**. The judge config lives at the app level (`judge_chat_provider_id`), so every execution across the app that gets scored uses the same judge. Picking a strong model (e.g. `claude-sonnet-4-6` or `gpt-4o`) gives more reliable scores than a small local model; the judge cost is separate from the scored run's cost.

### Pointwise rubric (v2.0)

The default scoring path is **pointwise** — the judge scores one answer at a time on five dimensions, each on a 1–5 scale:

| Dimension | Anchor |
|-----------|--------|
| **Relevance** | Does the answer address the query? |
| **Correctness** | Is the answer factually / logically correct against the source? |
| **Completeness** | Are all relevant parts of the query addressed? |
| **Coherence** | Is the answer well-structured and readable? |
| **Conciseness** | Is the answer free of padding and repetition? |

`overall_score = mean(dimension_scores)`, rounded to 2 decimals and clamped to `[1.0, 5.0]`.

The rubric prompts live in `src/rlmstudio/prompts/judge_pointwise.yaml`; edit there to tune anchors.

### Pairwise rubric

For head-to-head comparisons, the judge can score two answers against each other on the same five dimensions, plus a winner: `a`, `b`, or `tie`. The prompts live in `src/rlmstudio/prompts/judge_pairwise.yaml`.

### Auto-scoring non-usable outcomes

The judge is an LLM call — it has a cost and a latency. Running it on a failed execution is wasteful, so non-usable outcomes (see [Outcome classification](#outcome-classification)) skip the judge entirely and receive a deterministic auto-score:

| Outcome | `overall_score` | Rationale |
|---------|-----------------|-----------|
| `budget_exhausted` with ≥50 chars of answer | **1.0** | Partial answer; scored at the floor so it doesn't pollute the top of a ranking. |
| All other non-usable outcomes (timeout, context overflow, hard failures, …) | **0.0** | No usable answer; scored below the rubric floor so these sort last when judge_score is the ranking metric. |

This keeps `judge_score` sortable across a mix of successes and failures without spending judge tokens on runs that have nothing to score.

### Where judge scores surface

- **Compare** — pick "Judge score" as the ranking metric; cells are highlighted by `overall_score`.
- **Traces** — the judge score appears on each execution row; unjudged slots sort below judged ones.
- **Dashboard** — average judge score can be tracked across sessions (in charts that include this metric).

---

## Typical Experiment Workflows

### Comparing RLM vs Direct for a Large Document

1. Go to **Settings > Providers** and configure at least one LLM provider (e.g., Anthropic)
2. Go to **Settings > Chat Providers** and create two:
   - "Claude Direct" — Anthropic / claude-sonnet-4-6 / Profile: Accurate
   - "Claude RLM" — Anthropic / claude-sonnet-4-6 / Profile: RLM deep
3. Go to **Chat**, select both Chat Providers
4. Upload a large document
5. Ask a question — both providers respond in parallel
6. Compare: answer quality, tokens used, cost, latency
7. Go to **Dashboard** to see aggregated metrics
8. Go to **Traces** to inspect the RLM execution steps

### Tuning RLM Step Budget

1. Go to **Settings > Profiles** and create three custom profiles with strategy "RLM" and different step budgets: 8, 16, 32
2. Create three Chat Providers with the same LLM, each referencing a different profile
3. Run the same query against all three
4. Check **Traces** for each — does 32 steps find better answers than 8?
5. Check **Dashboard** — what's the cost/quality trade-off?

### Cross-Provider Benchmarking

1. Create Chat Providers for OpenAI/gpt-4o and Anthropic/claude-sonnet-4-6, both using the "Accurate" profile
2. Run identical queries against both
3. Compare response quality, token usage, and cost in the column layout
4. Use **Dashboard > Cost by Chat Provider** to see cumulative cost differences

### Finding the Right Temperature

1. Go to **Settings > Profiles** and create three custom profiles with the same strategy but temperatures 0.2, 0.5, and 0.9
2. Create Chat Providers with the same LLM, each referencing a different profile
3. Run the same factual question against all three
4. Lower temperatures produce more consistent, deterministic answers
5. Higher temperatures produce more creative but potentially less accurate answers

---

## Environment Configuration

### Backend Port

The backend defaults to port 8000. To change it:

```bash
RLM_STUDIO_PORT=8002 uv run python -m rlmstudio.server --reload
```

Then set the frontend to point at the new port. Create or edit `frontend/.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8002
```

Restart the frontend dev server for the change to take effect.

### API Keys

Set provider API keys as environment variables:

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

Or enter them in **Settings > Providers** in the UI (saved to `.env`).

### Docker / Playwright MCP

When accessing RLM Studio from inside a Docker container (e.g., Playwright MCP for UI testing), use `host.docker.internal` instead of `localhost`:

```
http://host.docker.internal:3000
```

The Next.js frontend proxies all API calls through its own server, so the Docker browser never needs direct access to the backend port.
