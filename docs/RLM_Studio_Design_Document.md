  
# **RLM Studio Design Document**

LLM ↔ RLM (REPL) Interaction Architecture

Core Business Logic, Circuit Breakers, and Known Limitations

April 2026  |  Auto-generated from codebase analysis

# **1\. Executive Summary**

RLM Studio implements a Recursive Language Model (RLM) framework that enables LLMs to explore arbitrarily large documents through iterative code execution in a sandboxed REPL. Unlike single-shot RAG or direct prompting, the RLM approach lets the model decide what to read next, re-inspect content, and iteratively build understanding before committing a final answer.

The system uses a deterministic JSON action protocol (v2.0) where the LLM outputs exactly one structured JSON object per turn. The runtime parses that object, dispatches the requested tool, feeds the result back as a conversation message, and repeats until the model emits a terminal action or a circuit breaker fires.

This document maps the exact interaction between the LLM and the RLM runtime, catalogs every circuit breaker and safety valve, documents the tool API the LLM relies on, and identifies architectural assumptions and known weaknesses.

# **2\. Architecture Overview**

## **2.1 Three Execution Modes**

**Direct mode:** A single LLM call with the full document content and query concatenated into one user message. No iteration, no tools, no parsing. System prompt comes from system\_prompt\_templates.json\["direct"\]. *(run\_direct.py:59–69)*

**RAG mode:** Chunks the document, embeds chunks, retrieves top-k similar chunks, and makes a single LLM call with the retrieved context. System prompt from system\_prompt\_templates.json\["rag"\]. *(run\_rag.py:90–128)*

**RLM mode:** An iterative REPL loop. The LLM receives the system prompt (which describes available tools and the JSON protocol) and the user query. On each turn it outputs a JSON action; the runtime executes it in a sandboxed environment and feeds the result back. This continues until the model outputs a "final" action or a circuit breaker terminates the loop. *(run\_rlm.py:45–497)*

## **2.2 Component Map**

| Component | File Path | Responsibility |
| :---- | :---- | :---- |
| RLM REPL Loop | application/use\_cases/run\_rlm.py | Main loop, budget mgmt, fallbacks |
| JSON Parser | core/actions.py | Parse inspect/final/subcall actions |
| Markdown Parser | core/parsing.py | v1.0 fallback: code blocks, FINAL: |
| Sandbox | infrastructure/sandbox/restricted\_sandbox.py | RestrictedPython code execution |
| Content Tools | tools/content.py | peek, grep, chunk, select |
| File Assembly | server/routes/chat.py:99–118 | Multi-file document index format |
| System Prompt | prompts/system\_prompt\_v2\_0.yaml | JSON action protocol definition |
| Prompt Templates | prompts/system\_prompt\_templates.json | Per-profile prompt supplements |
| LLM Port | application/ports/llm\_port.py | LLM provider abstraction |
| LiteLLM Adapter | infrastructure/llm/litellm\_adapter.py | 100+ provider support via LiteLLM |
| Config DTO | application/dto.py:74–106 | RunConfigDTO field definitions |

# **3\. The RLM REPL Loop — Core Business Logic**

The REPL (Read-Eval-Print Loop) is the heart of RLM Studio. It orchestrates a multi-turn conversation between the LLM and a restricted Python sandbox, allowing the model to iteratively explore document content.

## **3.1 Initialization**

The loop begins by building the system prompt from the versioned YAML template (v2.0 by default), appending any profile-specific system\_prompt\_extra, and injecting the document content into the sandbox as variable P.

system\_prompt \= self.\_build\_system\_prompt(len(content))

if config.system\_prompt\_extra:

    system\_prompt \+= "\\n\\n" \+ config.system\_prompt\_extra

*Reference: run\_rlm.py:90–92*

The sandbox binds P and auto-rebinds the four content tools (peek, grep, chunk, select) as partial functions with P pre-filled:

self.\_globals\["peek"\] \= partial(peek, content)

self.\_globals\["grep"\] \= partial(grep, content)

*Reference: restricted\_sandbox.py:186–205*

## **3.2 Step-by-Step Execution Flow**

Each iteration of the loop follows this sequence:

* Budget check: verify steps, tokens, cost, and time are within limits

* Model selection: use root\_model on step 1, recursive\_model thereafter

* Nudge injection: append soft or forced convergence messages if thresholds are reached

* LLM call: send the accumulated messages array to the LLM via self.\_llm.complete(messages)

* Token accounting: accumulate input\_tokens, output\_tokens, and cost from the response

* Response parsing: attempt JSON v2.0 parsing via parse\_action(); fall back to markdown v1.0

* Action dispatch: convert the parsed action into executable Python code

* Sandbox execution: run the code in RestrictedPython and capture stdout

* Result injection: append the execution output as a user message for the next turn

* Repeat detection: hash the output and check for consecutive duplicates

## **3.3 JSON Action Protocol (v2.0)**

The LLM must output exactly one JSON object per turn. Three action types are supported:

### **3.3.1 Inspect Action**

Instructs the runtime to execute a content-navigation tool. The JSON is translated to Python code and run in the sandbox.

{"type": "inspect", "tool": "peek", "args": {"start": 0, "end": 3000}, "note": "Read file 1"}

*Dispatch: run\_rlm.py:1261–1291 (\_inspect\_to\_code method)*

### **3.3.2 Final Action**

Terminates the loop and returns the answer to the user. Execution stops immediately.

{"type": "final", "answer": "The document describes...", "confidence": 0.9}

*Dispatch: run\_rlm.py:1293–1302 (\_extract\_final\_answer method)*

### **3.3.3 Subcall Action**

Creates a recursive RLM execution with a subset of the remaining budget. The child run gets its own loop and sandbox. Recursion depth is bounded by max\_recursion\_depth (default: 1).

{"type": "subcall", "prompt": "\<sub-content\>", "query": "\<sub-question\>"}

*Budget allocation: run\_rlm.py:1146–1191*

## **3.4 Response Parsing Pipeline**

The parser uses a two-tier strategy:

* **Tier 1 — JSON v2.0:** Attempts to extract the first valid JSON object from the response. Strips markdown fences, sanitizes invalid escape sequences, and validates against the action schema. *(core/actions.py:95–223)*

* **Tier 2 — Markdown v1.0 fallback:** If JSON parsing fails, looks for \`\`\`python code blocks and FINAL: prefixed answers. This maintains backward compatibility with v1.0 prompts. *(core/parsing.py:147–184)*

If neither tier produces actionable output (no code, no final answer), the raw text is saved as last\_plain\_answer for the stall-detection circuit breaker to potentially use.

# **4\. Tool API — What the LLM Can Execute**

The LLM has access to exactly four content-navigation tools. These are the only way it can read or search document content. All tools operate on the pre-bound content string P.

## **4.1 peek(start, end, max\_chars)**

Extracts a substring from P by character position. Supports negative indexing (Python slice semantics). Clamps to content bounds.

| Parameter | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| start | int | 0 | Start character position |
| end | int | None | None | End position (exclusive); None \= end of content |
| max\_chars | int | 10000 | Truncation limit for returned text |

*Reference: tools/content.py:9–52*

## **4.2 grep(pattern, context\_lines, max\_matches, ignore\_case, use\_regex)**

Searches P for a literal string or regex pattern. Returns matches with surrounding context lines, formatted with line numbers and separators.

| Parameter | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| pattern | str | (required) | Search pattern (literal or regex) |
| context\_lines | int | 2 | Lines before/after each match |
| max\_matches | int | 100 | Maximum matches returned |
| ignore\_case | bool | False | Case-insensitive search |
| use\_regex | bool | False | Interpret pattern as regex |

*Reference: tools/content.py:55–125*

## **4.3 chunk(size, overlap, by, max\_chunks)**

Splits P into chunks by character count or line count. Useful for systematic scanning of large content.

| Parameter | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| size | int | 1000 | Chunk size (chars or lines) |
| overlap | int | 0 | Overlap between consecutive chunks |
| by | str | "chars" | Unit: "chars" or "lines" |
| max\_chunks | int | 100 | Maximum chunks returned |

*Reference: tools/content.py:128–183*

## **4.4 select(ranges, max\_chars)**

Extracts multiple non-contiguous character ranges from P in a single call. Combines results with separators. Uses peek() internally.

| Parameter | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| ranges | list\[tuple\] | (required) | List of \[start, end\] pairs |
| max\_chars | int | 10000 | Combined truncation limit |

*Reference: tools/content.py:186–226*

## **4.5 read\_result(name, start, end, max\_chars) — and where results live**

Since `specs/rlm-working-state` (PR 1), every v2 inspect action is executed by the controller against the content, not compiled to Python for the sandbox (*application/services/inspect\_dispatcher.py*). The complete return value is registered as `r1, r2, …` in a per-run result registry (*application/services/result\_registry.py*); the newest is also `last`. The model sees a preview of `result_preview_chars` (10,000) characters ending in a marker that names the id and the next offset, and reads any range back with `read_result`. A readback is a step, is answered from the registry without touching the sandbox, and never changes `last`. Results above `spill_result_above_bytes` (16 MiB) are spilled to a scratch directory (kept, not dropped) with character→byte checkpoints so character offsets stay exact; the in-memory total is bounded by `max_registry_bytes` (64 MiB) and spilled results are evicted, oldest first and never `last`, only past `max_spill_bytes` (256 MiB). The scratch directory is removed when the run ends. Only the v1 free-form code path still runs in the sandbox.

| Parameter | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| name | str | "last" | `"last"` or a result id such as `"r3"` |
| start | int | 0 | Start character offset (Python slice semantics) |
| end | int \| None | None | End offset (exclusive); None = end of result |
| max\_chars | int | 10000 | Cap on the returned slice |

# **5\. Circuit Breakers and Safety Valves**

The RLM loop has multiple mechanisms to prevent runaway execution. These are the primary safety guarantees of the system.

## **5.1 Hard Limits (BudgetConfig)**

| Limit | Default | Config Field | Behavior When Hit |
| :---- | :---- | :---- | :---- |
| Max Steps | 16 | max\_steps | Raises BudgetExceededError; triggers synthesis fallback |
| Max Tokens | None | max\_tokens | Raises BudgetExceededError if input+output exceeds limit |
| Max Cost (USD) | None | max\_cost | Raises BudgetExceededError when cumulative cost hits ceiling |
| Max Time (sec) | None | max\_time\_seconds | Raises BudgetExceededError; user sees timeout suggestion |
| Max Recursion | 1 | max\_recursion\_depth | Prevents subcall nesting beyond this depth |

*Enforcement points: run\_rlm.py:127–128 (before step), :196–197 (after LLM call), :419–420 (before synthesis)*

## **5.2 Soft Convergence Nudge**

At nudge\_at\_fraction (default: 0.6) of max\_steps, the system injects a one-time "soft nudge" message telling the model how many steps remain. The message is conversational, not forcing. It fires once and never repeats.

nudge\_step \= max(2, int(max\_steps \* nudge\_at\_fraction))

*Reference: run\_rlm.py:137–152*

## **5.3 Force Final Nudge**

On the very last step (steps \>= max\_steps), the system appends a stronger message explicitly instructing the model to output {"type": "final"} immediately. This is the last chance before synthesis fallback.

*Reference: run\_rlm.py:154–170*

## **5.4 Repeat Detection**

The system maintains a sliding window of the last 5 execution output hashes. If the same output appears consecutively:

* **First repeat:** Appends a "duplicate\_result" message suggesting the model try a different approach.

* **After repeat\_limit (default: 2):** Appends "repeat\_detected" message forcing immediate finalization.

*Reference: run\_rlm.py:293–347*

## **5.5 Stall Detection**

Tracks consecutive steps where the model produces neither executable code nor a final answer. After stall\_limit (default: 3\) consecutive stalls:

* If the model had previously produced a plausible plain-text answer (not JSON), that answer is accepted and returned.

* Otherwise, BudgetExceededError is raised.

*Reference: run\_rlm.py:353–376*

## **5.6 Synthesis Fallback**

When max\_steps is exhausted without a final answer but inspection results exist, the system makes one extra LLM call (outside the step budget) with a simplified synthesis prompt. This call receives the last inspection output (truncated to 3000 chars) and the original query, producing a best-effort answer.

*Reference: run\_rlm.py:411–476, :1305–1319 (\_build\_synthesis\_messages)*

## **5.7 Context Window Overflow**

If the LLM raises a context-window-exceeded exception, the loop breaks to the synthesis fallback path rather than erroring out. The overflow step is recorded and included in the limit warning message.

*Reference: run\_rlm.py:175–182*

# **6\. Sandbox Security Model**

All LLM-generated code runs inside a RestrictedPython sandbox. This is the primary security boundary between the model and the host system.

## **6.1 What Is Allowed**

* The four content tools: peek(), grep(), chunk(), select()

* Standard library modules: json, re, math, datetime, itertools, collections, statistics, string, textwrap, copy, pprint, and others (31 total)

* Python builtins from RestrictedPython’s safe\_builtins (print, len, range, sorted, etc.)

* Variable assignment and function definitions within the sandbox namespace

## **6.2 What Is Blocked**

* File system access: open() is not in safe\_builtins

* Network access: socket, urllib, requests not in allowed modules

* Code execution: exec(), eval(), compile() not available

* Dunder access: \_\_class\_\_, \_\_bases\_\_, \_\_subclasses\_\_ blocked by safer\_getattr

* Arbitrary imports: only the 31 whitelisted modules are importable

## **6.3 Output Limits**

Sandbox stdout is capped at max\_stdout\_chars (default: 10,000 characters). If exceeded, the output is truncated with a "... (truncated)" message. This prevents memory exhaustion from runaway print() calls.

*Reference: restricted\_sandbox.py:90–154*

# **7\. Multi-File Content Assembly**

When multiple files are uploaded, the system concatenates them into a single string (P) with a navigation index.

## **7.1 Document Index Format**

\[DOCUMENT INDEX — 2 files attached\]

Read this index first with peek(0, 1200).

Each file entry includes exact character offsets:

  1\. "Knowledge\_Graphs\_Applied\_v7\_MEAP.pdf" (file\_start=215, content\_start=264, file\_end\_exclusive=451228)

  2\. "AI-Powered\_Search.pdf" (file\_start=451235, content\_start=451268, file\_end\_exclusive=823114)

\[END DOCUMENT INDEX\]

\[File 1: Knowledge\_Graphs\_Applied\_v7\_MEAP.pdf\]

\<content of file 1\>

\---

\[File 2: AI-Powered\_Search.pdf\]

\<content of file 2\>

*Reference: server/routes/chat.py:99–118*

## **7.2 Navigation Pattern**

The expected navigation pattern for multi-file queries is:

* Step 1: peek(0, 500\) to read the document index and beginning of file 1

* Step 2: use the file's content\_start offset from the index to jump directly with peek(start=\<content\_start\>, ...)

* Step 3: use grep(..., context\_lines=...) when you need local context around a file marker or a topic match; grep output also includes char\_offset for precise follow-up peeks

The system prompt v2.0 YAML includes this pattern as an explicit example workflow.

# **8\. System Prompt Architecture**

## **8.1 Prompt Assembly Chain**

The final system prompt sent to the LLM is composed of two layers:

* **Base prompt (v2.0 YAML):** Defines the JSON action protocol, tool signatures, decision protocol, critical rules, quality gates, and example workflows. Contains the {prompt\_length} template variable. *(system\_prompt\_v2\_0.yaml, \~8600 chars)*

* **Profile supplement (system\_prompt\_extra):** Appended as "\\n\\n" \+ extra. Contains per-profile guidance like multi-document rules, answer style preferences. Resolved at runtime via \_resolve\_profile\_prompt(). *(system\_prompt\_templates.json)*

## **8.2 Template Resolution**

Profile prompts are resolved via a two-tier lookup in \_resolve\_profile\_prompt():

* **Priority 1 — Custom per-mode text:** profile.system\_prompts.get(mode)

* **Priority 2 — Named template:** SYSTEM\_PROMPT\_TEMPLATES\[profile.prompt\_template\_name\]\[mode\]

*Reference: server/routes/chat.py:37–57*

## **8.3 Template Caching**

Templates are loaded once at module import time into SYSTEM\_PROMPT\_TEMPLATES (profile\_store.py:262). A reload\_system\_prompt\_templates() function and POST /api/system-prompts/templates/reload endpoint allow hot-reloading without server restart.

# **9\. Known Limitations and Assumptions**

## **9.1 Model Capability Assumptions**

* The v2.0 protocol assumes the LLM can reliably output valid JSON. Smaller models (e.g., Qwen 2.5 7B) frequently produce malformed JSON, hallucinate tool outputs, or ignore multi-step navigation instructions.

* The multi-document navigation contract must stay aligned with the tool API. Prompts should rely on the index's explicit content\_start offsets and grep()'s returned char\_offset values instead of asking the model to infer offsets.

* The system prompt is \~8600 chars before profile supplements. Combined with conversation history, this consumes significant context window on each turn, leaving less room for actual document content.

## **9.2 Peek/Grep Position Mismatch**

Historically, grep() returned only line numbers while peek() used character positions, forcing the model to infer offsets. The current design closes that gap by exposing explicit file offsets in the document index and char\_offset values in grep() output.

## **9.3 Single Content String (P)**

All uploaded files are concatenated into a single string. The tools operate on character positions within this flat string. This means:

* Large multi-file uploads can produce very large P values, causing context window issues on the first peek

* Character positions are global, not per-file; the model must track where each file starts

* File separators ("---") within content could be confused with the inter-file separator

## **9.4 Template Caching**

System prompt templates are cached at import time. Editing system\_prompt\_templates.json requires either a server restart or an explicit POST /api/system-prompts/templates/reload call. The v2.0 YAML prompt also requires a full server restart.

## **9.5 Repeat Detection Hash Window**

The repeat detector hashes only the first 500 characters of execution output. If two different inspections share the same first 500 chars but differ later, the system incorrectly flags them as repeats.

## **9.6 Synthesis Fallback Limitations**

The synthesis fallback truncates the last inspection output to 3000 characters. For large inspection results (e.g., grep across a whole book), this loses most of the retrieved data. The synthesis model also receives no conversation history — only the truncated output and the original query.

## **9.7 Token Counting Heuristic**

If LiteLLM’s model-aware tokenizer is unavailable, the system falls back to a rough heuristic of \~4 characters per token. This can lead to budget miscalculations, especially for non-Latin text or code.

## **9.8 No Per-File Tool Routing**

The tools have no concept of "file" — they operate on the flat concatenated string. A model asking "peek file 2" must first grep for the file marker, find its character position, and then peek at that position. There is no peek\_file(n) shortcut.

## **9.9 Sandbox Stdout Truncation (retired)**

Formerly: sandbox output was capped at 10,000 characters and a grep with many matches lost the rest with no way back. Since `specs/rlm-working-state` PR 1 the complete result is registered on the controller side and the preview's marker names the id and offset to continue from with `read_result` (§4.5). The 10,000-character cap still applies to the v1 free-form Python path, which runs in the sandbox.

## **9.10 No Streaming for Inspect Steps**

The sandbox executes synchronously. Long-running tool calls (e.g., regex grep on very large content) block the event loop. There is no progress feedback during execution.

# **10\. Code Reference Index**

Complete index of files and line ranges referenced in this document, rooted at src/rlmstudio/:

| File | Lines | Contains |
| :---- | :---- | :---- |
| application/use\_cases/run\_rlm.py | 45–497 | REPL loop, budget mgmt, fallbacks |
| application/use\_cases/run\_rlm.py | 1221–1302 | \_parse\_rlm\_response, \_inspect\_to\_code, \_extract\_final\_answer |
| application/use\_cases/run\_rlm.py | 1305–1382 | \_build\_synthesis\_messages, \_format\_limit\_warning |
| application/use\_cases/run\_rlm.py | 1438–1448 | \_build\_system\_prompt |
| application/use\_cases/run\_direct.py | 59–69 | Single-call direct mode |
| application/use\_cases/run\_rag.py | 90–128 | Embed→retrieve→single call RAG |
| application/dto.py | 30–46 | LLMResponseDTO |
| application/dto.py | 74–106 | RunConfigDTO (all budget fields) |
| application/ports/llm\_port.py | 16–93 | LLMPort protocol |
| core/actions.py | 95–288 | JSON parsing, action dispatch |
| core/parsing.py | 36–184 | v1.0 markdown fallback parser |
| tools/content.py | 9–226 | peek, grep, chunk, select |
| infrastructure/sandbox/restricted\_sandbox.py | 78–227 | RestrictedPython sandbox |
| infrastructure/llm/litellm\_adapter.py | 77–429 | LiteLLM adapter (100+ providers) |
| prompts/system\_prompt\_v2\_0.yaml | 1–308 | JSON action protocol v2.0 |
| prompts/system\_prompt\_templates.json | 1–20 | Profile prompt supplements |
| server/routes/chat.py | 37–57 | \_resolve\_profile\_prompt |
| server/routes/chat.py | 99–118 | Multi-file content assembly |
| domain/entities.py | 152–210 | BudgetConfig, BudgetState |
