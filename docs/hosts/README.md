# Connecting an LLM to RLM Studio

RLM Studio treats every LLM as a pluggable backend behind a single adapter (LiteLLM). This page is the landing doc for the `hosts/` subtree: it covers the decision of *which* backend to run, *how* to arrange your network, and *what* RLM Studio needs from you to talk to it. Each per-provider guide ([anthropic](anthropic.md), [openai](openai.md), [ollama](ollama.md), [lmstudio](lmstudio.md), [vllm](vllm.md), [dgx-spark](dgx-spark.md)) keeps the specifics.

## 1. Pick a backend

| If you want… | Use | Keyless? | Guide |
|---|---|---|---|
| Strongest reasoning + long context | Anthropic (Claude) | No (API key) | [anthropic.md](anthropic.md) |
| Broadest model selection, pay-per-use | OpenAI (GPT-4o) | No (API key) | [openai.md](openai.md) |
| Local quantized models, zero setup | Ollama | Yes | [ollama.md](ollama.md) |
| Local GUI-driven inference | LM Studio | Yes | [lmstudio.md](lmstudio.md) |
| High-throughput GPU serving (Linux) | vLLM | Yes (trusted network) | [vllm.md](vllm.md) |
| Self-hosted on Nvidia Grace Blackwell | DGX Spark (Ollama or vLLM) | Yes (trusted network) | [dgx-spark.md](dgx-spark.md) |

**Rule of thumb:** start cloud (OpenAI or Anthropic) unless you have a reason not to — cost, data residency, latency, or hardware you already own. Pick local backends when the network boundary matters more than the model quality.

## 2. Deployment topologies

RLM Studio is two processes — the FastAPI backend (:8000) and the Next.js frontend (:3000). The LLM is a third process, which you can place anywhere you can route to.

### All-local

RLM Studio, frontend, and the LLM (Ollama / LM Studio) all run on one machine. Simplest; bounded by the machine's RAM and GPU.

### Laptop + remote GPU host

RLM Studio runs on your dev laptop; the LLM runs on a remote GPU box (DGX Spark, a workstation, a rented GPU). RLM Studio points at `http://<gpu-host-ip>:<port>`. This is the topology the DGX Spark guide explicitly supports — it keeps your development tools on the laptop and offloads inference to hardware that can handle it.

End-to-end health check for this topology:

```bash
# From the laptop
curl http://<gpu-host-ip>:11434/api/tags          # Ollama reachable?
curl http://localhost:8000/health | python3 -m json.tool   # RLM Studio backend up?
```

### All-cloud

RLM Studio runs locally; the LLM is a cloud API (OpenAI, Anthropic, Google, etc.). No networking to think about beyond the laptop's outbound connection. This is the default path for the two cloud guides.

### Self-hosted behind a VPN / SSH tunnel

Same shape as the remote-GPU-host topology, but the network boundary is a VPN or an SSH tunnel instead of a LAN. Bind the LLM to `127.0.0.1` on the host, forward the port through the tunnel, and RLM Studio points at `http://localhost:<forwarded-port>`. Safer than exposing a local backend over the open network.

## 3. What RLM Studio needs

Every LLM Provider in RLM Studio is four fields. Three are always required; one (API key) depends on the backend.

| Field | Meaning | Required? |
|-------|---------|-----------|
| **Backend** | The adapter id — `anthropic`, `openai`, `ollama`, `lmstudio`, `vllm` | Always |
| **Model** | The model id the backend understands (e.g. `gpt-4o`, `claude-sonnet-4-6`, `llama3.1:8b`) | Always |
| **Base URL** | Override the backend's default endpoint. Leave blank to accept the default. | Optional — cloud backends default correctly |
| **API key** | Cloud backends only. Local backends (`ollama`, `lmstudio`, `vllm`) have no API-key field — secure them with a network boundary instead (see §5). | Cloud only |

**Test Connection** in Settings runs a live probe against the Base URL with your key. On success it records `last_tested_at`, `last_tested_by = "manual"`, and resets `consecutive_failures` to 0. On failure it reports the specific error (401, 429, connection refused, …) so you can diagnose without leaving Settings.

## 4. Configuration surfaces

Two surfaces can configure a provider:

- **Settings → LLM Providers** — runtime, UI-driven. Changes take effect immediately. Stored in the SecretStore (OS keyring when available, or `~/.rlm-studio/api_keys.json` chmod 600).
- **`.env` and environment variables** — startup-only. Read once by pydantic-settings when the server boots.

**Precedence:** real environment variables → SecretStore (from the UI) → legacy `.env`. If you set `OPENAI_API_KEY` in both the UI and the shell, the shell value wins.

See `.env.example` for the full list of overrides (`RLM_STUDIO_OPENAI_DEFAULT_MODEL`, `OLLAMA_BASE_URL`, …).

## 5. Security & network boundaries

- **Secret storage** is handled for you: cloud API keys go to the OS keyring where available, falling back to `~/.rlm-studio/api_keys.json` with `chmod 600`. They are never logged, never returned in API responses, and masked in the UI after first save.
- **Local backends have no API-key field in Settings.** `vllm --api-key` and similar are intentionally not surfaced — for a local backend, the meaningful boundary is the network, not a shared secret. Prefer one of:
  - Bind the server to `127.0.0.1` and route RLM Studio to it through an SSH tunnel.
  - Put the server behind a VPN.
  - Run the server on a LAN you trust.
- **Cloud backends** always need a valid API key with billing on the account. There is no anonymous path.

## 6. Operational hygiene

### Scheduled connection testing

RLM Studio can re-test every configured provider on a timer. Configure the interval under **Settings → Providers** (`connection_test_interval_minutes`, 0–1440; 0 disables the daemon). Each cycle:

- Tests up to 5 providers in parallel.
- Uses a 10-second per-test timeout.
- Marks a provider **offline** only after 2 consecutive failures (to avoid flap).
- A **single manual test success** flips an offline provider back to connected immediately.

Each provider row in Settings shows `last_tested_at`, `last_tested_by` (`manual` vs `background`), and `consecutive_failures` so you can tell at a glance whether a provider is healthy.

### Watching cost and outcome classification

Every execution gets classified into one of `success`, `timeout`, `budget_exhausted`, `context_overflow`, `general_error`. Non-success outcomes are excluded from cost / latency / token aggregations on the Dashboard, so "average cost per query" reflects only the runs that actually produced an answer. See the [rlm-studio-guide Dashboard section](../rlm-studio-guide.md#dashboard) for details.

### When connections go stale

If a provider flips to **offline** and you think it shouldn't have, check:

1. The network path from the RLM Studio backend to the base URL (`curl` the base URL directly).
2. Whether the key is still valid / has billing (`401` vs connection refused).
3. Whether the server process is actually up (`ss -lntp | grep <port>` on the host).

## 7. Go deeper

| Backend | Guide |
|---------|-------|
| Anthropic (Claude) | [anthropic.md](anthropic.md) |
| OpenAI | [openai.md](openai.md) |
| Ollama | [ollama.md](ollama.md) |
| LM Studio | [lmstudio.md](lmstudio.md) |
| vLLM | [vllm.md](vllm.md) |
| DGX Spark | [dgx-spark.md](dgx-spark.md) |

## 8. Official RLM engine (`rlm_official`) — provider mapping

The `rlm_official` execution mode runs the paper authors' implementation ([alexzhang13/rlm](https://github.com/alexzhang13/rlm), PyPI `rlms`; `pip install "rlm-studio[interop]"`) against the **same** LLM Provider as the sibling Studio cells. The engine has its own client layer, so RLM Studio maps each backend onto it:

| Studio backend | Engine client | What is passed | Notes |
|----------------|---------------|----------------|-------|
| `openai` | native `openai` | model id, the provider's API key | Falls back to `OPENAI_API_KEY` if no key is stored |
| `anthropic` | native `anthropic` | model id, the provider's API key | An API key is required — the engine has no env-var fallback here; the option reports "needs an API key" until one is stored |
| `ollama` | `openai` (OpenAI-compatible) | model tag, `<endpoint>/v1`, placeholder key | Default endpoint `http://localhost:11434` gets `/v1` appended |
| `lmstudio` | `openai` (OpenAI-compatible) | model id, endpoint, placeholder key | Default endpoint already ends in `/v1` |
| `vllm` | `openai` (OpenAI-compatible) | model id, endpoint, placeholder key | Same as LM Studio; the engine does not use `hosted_vllm/`-style prefixes |

Sandbox: the engine runs its REPL in a container only when **Settings → Sandbox** is `docker` (Studio's image name is passed through); with any other sandbox it uses the engine's in-process `local` environment, which is not isolated — each such run says so in its trace metadata. That environment also redirects the server process's output and working directory around every code cell, so official runs take it one at a time: pick the Docker sandbox if you want official cells in a Compare grid to run in parallel. The engine does not stream and reports token usage per run, so TTFT / decode timings are absent and step rows show 0 tokens; the run totals are exact. Runtime settings (temperature, top-p) are not forwarded to the engine in 1.0.
