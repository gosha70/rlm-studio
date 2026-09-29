/**
 * Shared constants — single source of truth for mode identifiers,
 * backend keys, and related groupings.  Mirrors the Python-side
 * constants in `src/rlmstudio/application/sandbox_vars.py`.
 */

// ---------------------------------------------------------------------------
// Execution modes
// ---------------------------------------------------------------------------

export const MODE_DIRECT = "direct" as const;
export const MODE_RLM = "rlm" as const;
export const MODE_RAG = "rag" as const;
export const MODE_COMPARE = "compare" as const;
export const MODE_AUTO = "auto" as const;
/** The paper authors' reference implementation (`rlms`) run as an engine. */
export const MODE_RLM_OFFICIAL = "rlm_official" as const;

/** All execution modes the user can select in the UI. */
export const ALL_EXECUTION_MODES = [MODE_DIRECT, MODE_RLM, MODE_RAG] as const;

/** Default mode for new profiles and Chat Providers. */
export const DEFAULT_MODE = MODE_DIRECT;

/** Default enabled modes for ModeConfig. */
export const DEFAULT_ENABLED_MODES = [MODE_DIRECT, MODE_RLM] as const;

// ---------------------------------------------------------------------------
// LLM backend identifiers
// ---------------------------------------------------------------------------

export const BACKEND_OPENAI = "openai" as const;
export const BACKEND_ANTHROPIC = "anthropic" as const;
export const BACKEND_OLLAMA = "ollama" as const;
export const BACKEND_LMSTUDIO = "lmstudio" as const;
export const BACKEND_VLLM = "vllm" as const;

/** Backends that require an API key (cloud providers). */
export const CLOUD_BACKENDS = [BACKEND_OPENAI, BACKEND_ANTHROPIC] as const;

/** Backends that connect to a local endpoint. */
export const LOCAL_BACKENDS = [BACKEND_OLLAMA, BACKEND_LMSTUDIO, BACKEND_VLLM] as const;

/** All available backends for the LLM Provider selector. */
export const ALL_BACKENDS = [...CLOUD_BACKENDS, ...LOCAL_BACKENDS] as const;

/** Human-readable display names for backend keys. */
export const BACKEND_DISPLAY_NAMES: Record<string, string> = {
  [BACKEND_OPENAI]: "OpenAI",
  [BACKEND_ANTHROPIC]: "Anthropic",
  [BACKEND_OLLAMA]: "Ollama",
  [BACKEND_LMSTUDIO]: "LM Studio",
  [BACKEND_VLLM]: "vLLM",
};

/** Model placeholder text per backend. */
export const MODEL_PLACEHOLDER: Record<string, string> = {
  [BACKEND_OPENAI]: "e.g., gpt-4o",
  [BACKEND_ANTHROPIC]: "e.g., claude-sonnet-4-5",
  [BACKEND_VLLM]: "e.g., Qwen/Qwen2.5-7B-Instruct",
  [BACKEND_OLLAMA]: "e.g., llama3.2",
  [BACKEND_LMSTUDIO]: "e.g., llama3.2",
};

/** Endpoint placeholder text per local backend. */
export const ENDPOINT_PLACEHOLDER: Record<string, string> = {
  [BACKEND_OLLAMA]: "http://localhost:11434",
  [BACKEND_VLLM]: "http://<dgx-spark-ip>:8000/v1",
  [BACKEND_LMSTUDIO]: "http://localhost:1234/v1",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Map a raw backend key to a human-readable name. */
export function displayProviderName(key: string): string {
  return BACKEND_DISPLAY_NAMES[key] ?? key;
}
