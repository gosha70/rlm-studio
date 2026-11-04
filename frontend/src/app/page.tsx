"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import useSWR from "swr";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { User, Bot, Sparkles, Loader2, ChevronDown, ChevronRight, Terminal, Upload } from "lucide-react";
import { toast } from "sonner";
import { AppShell } from "@/components/shared/app-shell";
import { ChatProviderSelector } from "@/components/chat/chat-provider-selector";
import { ChatInput } from "@/components/chat/chat-input";
import { FileAttachment } from "@/components/chat/file-attachment";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { ResponseRating } from "@/components/chat/response-rating";
import { PickWinner } from "@/components/chat/pick-winner";
import { JudgeScores } from "@/components/chat/judge-scores";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  getConfig,
  getSession,
  getChatProviders,
  getTrace,
  getExecutions,
  submitChat,
  uploadFile,
  getLLMProviders,
  getSessionEvaluations,
  submitThumbRating,
  removeThumbRating,
  submitBestPick,
  triggerJudge,
  type AppConfig,
  type ChatProviderConfig,
  type LLMProviderConfig,
  type FileUploadResponse,
  type MessageMetrics,
  type JudgeScoreData,
  type TraceStep,
} from "@/lib/api";
import {
  STORAGE_KEY_ACTIVE_SESSION,
  STORAGE_KEY_CHAT_PROVIDER_ORDER,
  STORAGE_KEY_FILES_DRAFT,
  STORAGE_KEY_SELECTED_CHAT_PROVIDERS,
  migrateLegacyStorageKeys,
  storageKeyFilesForSession,
} from "@/lib/branding";

const MAX_FILES = 10;
const BYTES_PER_MB = 1024 * 1024;
const MAX_MESSAGE_MB = 50;
/** Extracted text per message; mirrors backend `_MAX_TOTAL_CONTENT_BYTES`. */
const MAX_BYTES = MAX_MESSAGE_MB * BYTES_PER_MB;
const MAX_UPLOAD_FILE_MB = 200;
/** One upload's raw size; mirrors `_MAX_FILE_SIZE` in `server/routes/files.py`. */
const MAX_UPLOAD_FILE_BYTES = MAX_UPLOAD_FILE_MB * BYTES_PER_MB;
/** Types the server decodes as-is, so their text is no smaller than the file. */
const PLAIN_TEXT_EXTENSIONS = [".txt", ".md", ".py", ".json", ".csv"];

interface ChatTurn {
  id: string;
  userMessage: {
    content: string;
    timestamp: string;
  };
  responses: Record<
    string,
    {
      chatProviderId: string;
      chatProviderName: string;
      profileName?: string | null;
      executionId: string;
      content: string;
      isStreaming: boolean;
      metrics?: MessageMetrics;
      steps?: TraceStep[];
    }
  >;
}

// Map execution_id -> { turnId, chatProviderId } for WS event matching
interface ExecMapping {
  turnId: string;
  chatProviderId: string;
}

export default function ChatPage() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [selectedChatProviderIds, setSelectedChatProviderIds] = useState<string[]>([]);
  const [chatProviderOrder, setChatProviderOrder] = useState<string[]>([]);
  // Prevents the provider-cleanup effect from running before localStorage is read,
  // which would reset the selection to the first available provider.
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from localStorage after mount to avoid SSR/client mismatch
  // Track whether the provider-filter effect has seen the hydrated selection.
  // Without this, React may batch setSelectedChatProviderIds and setHydrated
  // together, causing the filter effect to see hydrated=true but still the
  // initial empty [] for selectedChatProviderIds — then it overwrites the
  // restored selection with the first available provider.
  const selectionHydratedRef = useRef(false);

  useEffect(() => {
    try {
      migrateLegacyStorageKeys();
    } catch {}
    try {
      const sid = localStorage.getItem(STORAGE_KEY_ACTIVE_SESSION);
      if (sid) setSessionId(sid);
    } catch {}
    try {
      const saved = localStorage.getItem(STORAGE_KEY_SELECTED_CHAT_PROVIDERS);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setSelectedChatProviderIds(parsed);
          selectionHydratedRef.current = true;
        }
      }
    } catch {}
    try {
      const saved = localStorage.getItem(STORAGE_KEY_CHAT_PROVIDER_ORDER);
      if (saved) setChatProviderOrder(JSON.parse(saved));
    } catch {}
    setHydrated(true);
  }, []);

  const [uploadedFiles, setUploadedFiles] = useState<FileUploadResponse[]>([]);
  const [uploadProgress, setUploadProgress] = useState<number | "processing" | null>(null);
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  // Track whether uploaded files have been hydrated from localStorage so we
  // don't clobber persisted state with an empty array on mount.
  const filesHydratedRef = useRef(false);
  const [isAnyStreaming, setIsAnyStreaming] = useState(false);
  const [ratings, setRatings] = useState<Record<string, "up" | "down">>({});
  const [bestPicks, setBestPicks] = useState<Record<string, string>>({});
  const [judgeScores, setJudgeScores] = useState<Record<string, JudgeScoreData>>({});
  const [isJudging, setIsJudging] = useState(false);
  const [stepsVisible, setStepsVisible] = useState<Record<string, boolean>>({});
  const skipSessionLoadRef = useRef(false);
  const wsRef = useRef<WebSocket | null>(null);
  const execMapRef = useRef<Record<string, ExecMapping>>({});
  const pendingCountRef = useRef(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsReadyRef = useRef(false);
  const pollTimersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());
  // Monotonic generation counter — incremented on every session change.
  // Callbacks captured under a stale generation bail out immediately.
  const sessionGenRef = useRef(0);

  // Sort selected IDs by chip order for consistent grid column ordering
  const orderedSelectedIds = chatProviderOrder.length > 0
    ? [...selectedChatProviderIds].sort((a, b) => {
        const ai = chatProviderOrder.indexOf(a);
        const bi = chatProviderOrder.indexOf(b);
        return (ai === -1 ? Infinity : ai) - (bi === -1 ? Infinity : bi);
      })
    : selectedChatProviderIds;

  const { data: chatProviders = [] } = useSWR<ChatProviderConfig[]>("chat-providers", getChatProviders);
  const { data: llmProviders = [] } = useSWR<LLMProviderConfig[]>("llm-providers", getLLMProviders);
  const { data: config } = useSWR<AppConfig>("config", getConfig);

  // Persist active session to localStorage
  useEffect(() => {
    if (sessionId) {
      localStorage.setItem(STORAGE_KEY_ACTIVE_SESSION, sessionId);
    } else {
      localStorage.removeItem(STORAGE_KEY_ACTIVE_SESSION);
    }
  }, [sessionId]);

  // Persist selected chat providers to localStorage (only after hydration
  // to avoid overwriting the saved selection with the initial empty array).
  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY_SELECTED_CHAT_PROVIDERS, JSON.stringify(selectedChatProviderIds));
  }, [selectedChatProviderIds, hydrated]);

  // Persist chip order (only after hydration)
  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(STORAGE_KEY_CHAT_PROVIDER_ORDER, JSON.stringify(chatProviderOrder));
  }, [chatProviderOrder, hydrated]);

  // Hydrate uploaded files from localStorage.
  // Uses session-keyed storage when a session exists, or a "draft" key
  // for uploads that happen before the first message creates a session.
  const filesStorageKey = sessionId ? storageKeyFilesForSession(sessionId) : STORAGE_KEY_FILES_DRAFT;

  useEffect(() => {
    if (!hydrated) return;
    filesHydratedRef.current = true;
    try {
      // First try the session-specific key, then fall back to draft key.
      // This handles the case where a user uploaded files before sending
      // (stored under draft key), navigated away, and came back — the
      // main hydration may have restored a previous sessionId, so
      // filesStorageKey would point at the old session's files, not the draft.
      const saved = localStorage.getItem(filesStorageKey);
      if (saved) {
        const parsed = JSON.parse(saved) as FileUploadResponse[];
        if (Array.isArray(parsed) && parsed.length > 0) {
          setUploadedFiles(parsed);
          return;
        }
      }
      // If no files for the current session, check for orphaned drafts
      if (filesStorageKey !== STORAGE_KEY_FILES_DRAFT) {
        const draft = localStorage.getItem(STORAGE_KEY_FILES_DRAFT);
        if (draft) {
          const parsed = JSON.parse(draft) as FileUploadResponse[];
          if (Array.isArray(parsed) && parsed.length > 0) {
            setUploadedFiles(parsed);
          }
        }
      }
    } catch {
      // Ignore corrupt data
    }
  }, [hydrated, filesStorageKey]);

  // Persist uploaded files whenever they change (after initial hydration).
  useEffect(() => {
    if (!filesHydratedRef.current) return;
    if (uploadedFiles.length > 0) {
      localStorage.setItem(filesStorageKey, JSON.stringify(uploadedFiles));
    } else {
      localStorage.removeItem(filesStorageKey);
    }
  }, [uploadedFiles, filesStorageKey]);

  // When a session is first created (null → id), migrate draft files to session key.
  const prevSessionIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (sessionId && !prevSessionIdRef.current) {
      const draft = localStorage.getItem(STORAGE_KEY_FILES_DRAFT);
      if (draft) {
        localStorage.setItem(storageKeyFilesForSession(sessionId), draft);
        localStorage.removeItem(STORAGE_KEY_FILES_DRAFT);
      }
    }
    prevSessionIdRef.current = sessionId;
  }, [sessionId]);

  // Auto-select / clean up selected providers when the provider list loads.
  // Removes stale IDs (deleted/renamed providers persisted in localStorage) and
  // IDs whose backing LLM provider is now offline/unconfigured (mirrors the
  // availability filter in ChatProviderSelector).
  // Guard: skip until localStorage has been read so we don't clobber the
  // persisted selection with an empty initial state.
  useEffect(() => {
    if (!hydrated) return;
    if (chatProviders.length === 0) return;
    // On the first run after hydration, if a selection was restored from
    // localStorage, skip the filter — React may not have flushed the
    // hydrated selection into selectedChatProviderIds yet.
    if (selectionHydratedRef.current) {
      selectionHydratedRef.current = false;
      return;
    }
    let availableIds: Set<string>;
    if (llmProviders.length > 0) {
      const providerMap = new Map(llmProviders.map((p) => [p.id, p]));
      availableIds = new Set(
        chatProviders
          .filter((cp) => {
            const info = providerMap.get(cp.llm_provider_id);
            return info && (info.status === "connected" || info.status === "configured");
          })
          .map((cp) => cp.id),
      );
    } else {
      // llmProviders not yet loaded — fall back to checking by chat-provider ID only
      availableIds = new Set(chatProviders.map((cp) => cp.id));
    }
    const filtered = selectedChatProviderIds.filter((id) => availableIds.has(id));
    if (filtered.length !== selectedChatProviderIds.length || filtered.length === 0) {
      const first = [...availableIds][0];
      setSelectedChatProviderIds(filtered.length > 0 ? filtered : first ? [first] : []);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chatProviders, llmProviders, hydrated]);

  // Load session messages and reconstruct turns
  useEffect(() => {
    if (!sessionId) return;
    if (skipSessionLoadRef.current) {
      skipSessionLoadRef.current = false;
      return;
    }
    getSession(sessionId)
      .then((session) => {
        // Reconstruct turns from session.conversations dict or flat messages
        if (session.conversations && Object.keys(session.conversations).length > 0) {
          // Group by conversation (chat provider)
          const turnMap = new Map<string, ChatTurn>();

          // Iterate through each chat provider's conversation
          for (const [cpId, messages] of Object.entries(session.conversations)) {
            for (const msg of messages) {
              let turnId: string | undefined;

              if (msg.role === "user") {
                // Find or create a turn with this user message
                const existing = Array.from(turnMap.values()).find(
                  (t) => t.userMessage.content === msg.content
                );
                turnId = existing?.id || crypto.randomUUID();
              } else {
                // Assistant message — find the corresponding turn
                const existing = Array.from(turnMap.values()).find(
                  (t) => !t.responses[cpId]
                );
                turnId = existing?.id;
              }

              if (!turnId) turnId = crypto.randomUUID();

              let turn = turnMap.get(turnId);
              if (!turn) {
                turn = {
                  id: turnId,
                  userMessage: { content: "", timestamp: "" },
                  responses: {},
                };
                turnMap.set(turnId, turn);
              }

              if (msg.role === "user") {
                turn.userMessage = {
                  content: msg.content,
                  timestamp: msg.timestamp,
                };
              } else {
                turn.responses[cpId] = {
                  chatProviderId: cpId,
                  chatProviderName: msg.chat_provider_name || cpId,
                  executionId: msg.execution_id || crypto.randomUUID(),
                  content: msg.content,
                  isStreaming: false,
                  metrics: msg.metrics ?? undefined,
                };
              }
            }
          }

          const reconstructed = Array.from(turnMap.values());
          setTurns(reconstructed);

          // Restore chat provider selection from the providers present in conversations
          const sessionProviderIds = Object.keys(session.conversations);
          if (sessionProviderIds.length > 0) {
            setSelectedChatProviderIds(sessionProviderIds);
          }

          // Hydrate execution steps from traces
          _hydrateStepsFromTraces(reconstructed);
        } else {
          // Fallback: reconstruct from flat messages list
          const newTurns: ChatTurn[] = [];
          let currentTurn: ChatTurn | null = null;

          for (const msg of session.messages) {
            if (msg.role === "user") {
              currentTurn = {
                id: crypto.randomUUID(),
                userMessage: {
                  content: msg.content,
                  timestamp: msg.timestamp,
                },
                responses: {},
              };
              newTurns.push(currentTurn);
            } else if (currentTurn && msg.chat_provider_id) {
              currentTurn.responses[msg.chat_provider_id] = {
                chatProviderId: msg.chat_provider_id,
                chatProviderName: msg.chat_provider_name || msg.chat_provider_id,
                executionId: msg.execution_id || msg.id,
                content: msg.content,
                isStreaming: false,
                metrics: msg.metrics ?? undefined,
              };
            }
          }

          setTurns(newTurns);

          // Restore provider selection from flat messages
          const flatProviderIds = [
            ...new Set(
              session.messages
                .filter((m) => m.chat_provider_id)
                .map((m) => m.chat_provider_id!),
            ),
          ];
          if (flatProviderIds.length > 0) {
            setSelectedChatProviderIds(flatProviderIds);
          }

          // Hydrate execution steps from traces
          _hydrateStepsFromTraces(newTurns);
        }
      })
      .catch((err) => {
        if (String(err).includes("404")) {
          setSessionId(null);
          setTurns([]);
        }
      });

    /**
     * Fetch execution traces for all responses in the reconstructed turns
     * and backfill the `steps` field that is missing from persisted sessions.
     */
    function _hydrateStepsFromTraces(loadedTurns: ChatTurn[]) {
      // Capture session generation so stale callbacks from a previous
      // session don't register pollers after cancelAllPollers() runs.
      const hydrateGen = sessionGenRef.current;

      for (const turn of loadedTurns) {
        for (const [cpId, resp] of Object.entries(turn.responses)) {
          if (!resp.executionId || resp.steps) continue;
          getTrace(resp.executionId)
            .then((trace) => {
              if (sessionGenRef.current !== hydrateGen) return;
              if (trace.status === "running") {
                // Execution is still in progress — mark as streaming and
                // restart polling so the result appears when it completes.
                // Register in execMap and increment pending count so the
                // streaming indicator and cancel logic work correctly.
                execMapRef.current[resp.executionId] = { turnId: turn.id, chatProviderId: cpId };
                pendingCountRef.current += 1;
                setIsAnyStreaming(true);
                setTurns((prev) =>
                  prev.map((t) =>
                    t.id === turn.id
                      ? {
                          ...t,
                          responses: {
                            ...t.responses,
                            [cpId]: { ...t.responses[cpId], isStreaming: true },
                          },
                        }
                      : t,
                  ),
                );
                pollForResult(resp.executionId, cpId, turn.id);
                return;
              }
              if (trace.status !== "complete" || !trace.steps?.length) return;
              setTurns((prev) =>
                prev.map((t) =>
                  t.id === turn.id
                    ? {
                        ...t,
                        responses: {
                          ...t.responses,
                          [cpId]: {
                            ...t.responses[cpId],
                            steps: trace.steps,
                            metrics: t.responses[cpId].metrics ?? {
                              input_tokens: trace.result.input_tokens ?? 0,
                              output_tokens: trace.result.output_tokens ?? 0,
                              total_tokens: trace.budget.tokens_used,
                              cost_usd: trace.result.total_cost ?? 0,
                              elapsed_seconds:
                                trace.completed_at && trace.started_at
                                  ? (new Date(trace.completed_at).getTime() -
                                      new Date(trace.started_at).getTime()) /
                                    1000
                                  : 0,
                              steps: trace.budget.steps_used,
                            },
                          },
                        },
                      }
                    : t,
                ),
              );
            })
            .catch(() => {
              // Trace not available (e.g. server restarted without persistence)
            });
        }
      }

      // Also check for running executions that have no response entry yet
      // (user message stored, but assistant message not yet persisted because
      // the execution is still in progress).
      if (sessionId) {
        getExecutions(100, undefined, sessionId).then((execs) => {
          if (sessionGenRef.current !== hydrateGen) return;
          const running = execs.filter((e) => e.status === "running");
          for (const exec of running) {
            const cpId = exec.chat_provider_id;
            if (!cpId) continue;
            // Find the latest turn matching this query that still needs
            // a response for this provider.  Searching from the end
            // handles repeated identical prompts correctly — older turns
            // already have their responses, so we match the newest one.
            const matchingTurn = [...loadedTurns]
              .reverse()
              .find((t) => t.userMessage.content === exec.query && !t.responses[cpId]);
            if (!matchingTurn) continue;
            // Register bookkeeping before polling
            execMapRef.current[exec.execution_id] = { turnId: matchingTurn.id, chatProviderId: cpId };
            pendingCountRef.current += 1;
            setIsAnyStreaming(true);
            // Add a streaming placeholder and start polling
            setTurns((prev) =>
              prev.map((t) =>
                t.id === matchingTurn.id
                  ? {
                      ...t,
                      responses: {
                        ...t.responses,
                        [cpId]: {
                          chatProviderId: cpId,
                          chatProviderName: exec.chat_provider_name || cpId,
                          executionId: exec.execution_id,
                          content: "",
                          isStreaming: true,
                        },
                      },
                    }
                  : t,
              ),
            );
            pollForResult(exec.execution_id, cpId, matchingTurn.id);
          }
        }).catch(() => {});
      }
    }
  }, [sessionId]);

  // Load evaluations when session changes
  useEffect(() => {
    if (!sessionId) {
      setRatings({});
      setBestPicks({});
      setJudgeScores({});
      return;
    }
    getSessionEvaluations(sessionId)
      .then((evals) => {
        const newRatings: Record<string, "up" | "down"> = {};
        for (const r of evals.thumb_ratings) {
          newRatings[r.execution_id] = r.rating;
        }
        setRatings(newRatings);

        const newPicks: Record<string, string> = {};
        for (const p of evals.best_picks) {
          newPicks[p.winner_execution_id] = p.winner_execution_id;
        }
        setBestPicks(newPicks);

        const newScores: Record<string, JudgeScoreData> = {};
        for (const s of evals.judge_scores) {
          newScores[s.execution_id] = s;
        }
        setJudgeScores(newScores);
      })
      .catch(() => {
        // Evaluations endpoint may not exist yet
      });
  }, [sessionId]);

  // Auto-scroll on new turns
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  // No eager WebSocket connection — all queries use REST + polling.
  // The WS endpoint exists for future streaming support but connecting
  // eagerly caused console errors on page load and backend restarts.
  // Close any stale connection when session changes.
  useEffect(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      wsReadyRef.current = false;
    }
  }, [sessionId]);

  // REST polling fallback — tracks timeouts so they can be cancelled on cleanup
  const pollForResult = useCallback(
    (executionId: string, chatProviderId: string, turnId: string) => {
      setIsAnyStreaming(true);
      // cancelled flag lets in-flight fetches bail out after cleanup
      let cancelled = false;
      const cancelToken = { cancel: () => { cancelled = true; } };
      // Store on execMap so handleNewSession can cancel
      const mapping = execMapRef.current[executionId];
      if (mapping) (mapping as ExecMapping & { cancelPoll?: () => void }).cancelPoll = cancelToken.cancel;

      const schedulePoll = (delayMs: number) => {
        const timer = setTimeout(poll, delayMs);
        pollTimersRef.current.add(timer);
      };

      const poll = async () => {
        if (cancelled) return;
        try {
          const trace = await getTrace(executionId);
          if (cancelled) return;

          if (trace.status === "complete") {
            delete execMapRef.current[executionId];
            pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);

            setTurns((prev) =>
              prev.map((turn) =>
                turn.id === turnId
                  ? {
                      ...turn,
                      responses: {
                        ...turn.responses,
                        [chatProviderId]: {
                          ...turn.responses[chatProviderId],
                          content: trace.result.answer,
                          isStreaming: false,
                          steps: trace.steps?.length ? trace.steps : undefined,
                          metrics: {
                            input_tokens: trace.result.input_tokens ?? trace.budget.tokens_used,
                            output_tokens: trace.result.output_tokens ?? 0,
                            total_tokens: trace.budget.tokens_used,
                            cost_usd: trace.result.total_cost ?? trace.budget.cost_used,
                            elapsed_seconds:
                              trace.completed_at && trace.started_at
                                ? (new Date(trace.completed_at).getTime() -
                                    new Date(trace.started_at).getTime()) /
                                  1000
                                : 0,
                            steps: trace.budget.steps_used,
                          },
                        },
                      },
                    }
                  : turn,
              ),
            );

            if (pendingCountRef.current === 0) {
              setIsAnyStreaming(false);
            }
            return;
          } else if (trace.status === "error") {
            delete execMapRef.current[executionId];
            pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);

            const errorAnswer =
              trace.result?.answer ||
              trace.result?.error ||
              "Error: Execution failed";
            setTurns((prev) =>
              prev.map((turn) =>
                turn.id === turnId
                  ? {
                      ...turn,
                      responses: {
                        ...turn.responses,
                        [chatProviderId]: {
                          ...turn.responses[chatProviderId],
                          content: errorAnswer,
                          isStreaming: false,
                          metrics: {
                            input_tokens: trace.result?.input_tokens ?? 0,
                            output_tokens: trace.result?.output_tokens ?? 0,
                            total_tokens: trace.result?.input_tokens ?? 0,
                            cost_usd: trace.result?.total_cost ?? 0,
                            elapsed_seconds:
                              trace.completed_at && trace.started_at
                                ? (new Date(trace.completed_at).getTime() -
                                    new Date(trace.started_at).getTime()) /
                                  1000
                                : 0,
                            steps: 0,
                          },
                        },
                      },
                    }
                  : turn,
              ),
            );

            if (pendingCountRef.current === 0) {
              setIsAnyStreaming(false);
            }
            return;
          }

          // Still running — poll again
          schedulePoll(1000);
        } catch (err) {
          if (cancelled) return;
          console.error("Polling error:", err);
          schedulePoll(2000);
        }
      };

      poll();
    },
    [],
  );

  const handleRate = useCallback(
    async (executionId: string, rating: "up" | "down" | null) => {
      if (!sessionId) return;
      if (rating === null) {
        // Toggle off
        setRatings((prev) => {
          const next = { ...prev };
          delete next[executionId];
          return next;
        });
        try { await removeThumbRating(executionId); } catch { /* ignore */ }
      } else {
        // Find chat_provider_id from turns
        let cpId = "";
        for (const turn of turns) {
          for (const [id, resp] of Object.entries(turn.responses)) {
            if (resp.executionId === executionId) {
              cpId = id;
              break;
            }
          }
          if (cpId) break;
        }
        setRatings((prev) => ({ ...prev, [executionId]: rating }));
        try {
          await submitThumbRating({
            execution_id: executionId,
            session_id: sessionId,
            chat_provider_id: cpId,
            rating,
          });
        } catch { /* ignore */ }
      }
    },
    [sessionId, turns],
  );

  const handlePickWinner = useCallback(
    async (winnerExecutionId: string) => {
      if (!sessionId) return;
      setBestPicks((prev) => ({ ...prev, [winnerExecutionId]: winnerExecutionId }));
      try {
        await submitBestPick({
          session_id: sessionId,
          winner_execution_id: winnerExecutionId,
        });
      } catch { /* ignore */ }
    },
    [sessionId],
  );

  const handleTriggerJudge = useCallback(
    async (executionIds: string[]) => {
      if (!sessionId || executionIds.length === 0) return;
      setIsJudging(true);
      try {
        const result = await triggerJudge({
          session_id: sessionId,
          execution_ids: executionIds,
          mode: "both",
        });
        const newScores: Record<string, JudgeScoreData> = { ...judgeScores };
        for (const s of result.pointwise) {
          newScores[s.execution_id] = s;
        }
        setJudgeScores(newScores);
      } catch (err) {
        console.error("Judge failed:", err);
        toast.error("Judge evaluation failed");
      } finally {
        setIsJudging(false);
      }
    },
    [sessionId, judgeScores],
  );

  const handleSend = useCallback(
    async (text: string) => {
      if (selectedChatProviderIds.length === 0) return;

      // Capture generation at send-time so stale callbacks can bail out
      const sendGen = sessionGenRef.current;

      // Add user message
      const turnId = crypto.randomUUID();
      const newTurn: ChatTurn = {
        id: turnId,
        userMessage: {
          content: text,
          timestamp: new Date().toISOString(),
        },
        responses: {},
      };

      // Initialize empty responses for each selected provider
      for (const cpId of selectedChatProviderIds) {
        const matchedCp = chatProviders.find((cp) => cp.id === cpId);
        newTurn.responses[cpId] = {
          chatProviderId: cpId,
          chatProviderName: matchedCp?.name || cpId,
          profileName: matchedCp?.profile_name,
          executionId: "",
          content: "",
          isStreaming: true,
        };
      }

      setTurns((prev) => [...prev, newTurn]);
      pendingCountRef.current += selectedChatProviderIds.length;
      setIsAnyStreaming(true);

      // Ensure all providers share one session. When sessionId is null,
      // send the first provider sequentially to establish a session, then
      // fan out the rest in parallel with that shared id.
      let effectiveSessionId = sessionId;

      const submitOne = async (cpId: string) => {
        try {
          const resp = await submitChat({
            query: text,
            content: uploadedFiles.length > 0 ? null : text || null,
            file_ids: uploadedFiles.length > 0 ? uploadedFiles.map((f) => f.id) : null,
            chat_provider_id: cpId,
            session_id: effectiveSessionId,
          });

          // If session changed while we were awaiting, discard this result
          if (sessionGenRef.current !== sendGen) return;

          // Capture session id from the first response
          if (!effectiveSessionId) {
            effectiveSessionId = resp.session_id;
            skipSessionLoadRef.current = true;
            setSessionId(resp.session_id);
          }

          // Register mapping so WS events can find the turn
          execMapRef.current[resp.execution_id] = { turnId, chatProviderId: cpId };

          // Update the turn with the execution ID
          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === turnId
                ? {
                    ...turn,
                    responses: {
                      ...turn.responses,
                      [cpId]: {
                        ...turn.responses[cpId],
                        executionId: resp.execution_id,
                      },
                    },
                  }
                : turn,
            ),
          );

          // Always poll: the WS backend only emits events for WS-initiated
          // queries, not REST-submitted ones. REST polling is the reliable path.
          pollForResult(resp.execution_id, cpId, turnId);
        } catch (err) {
          if (sessionGenRef.current !== sendGen) return;
          console.error(`Failed to submit chat for provider ${cpId}:`, err);
          pendingCountRef.current = Math.max(0, pendingCountRef.current - 1);
          setTurns((prev) =>
            prev.map((turn) =>
              turn.id === turnId
                ? {
                    ...turn,
                    responses: {
                      ...turn.responses,
                      [cpId]: {
                        ...turn.responses[cpId],
                        content: `Error: ${err instanceof Error ? err.message : "Failed to submit request"}`,
                        isStreaming: false,
                      },
                    },
                  }
                : turn,
            ),
          );
          if (pendingCountRef.current === 0) {
            setIsAnyStreaming(false);
          }
        }
      };

      if (!sessionId && selectedChatProviderIds.length > 1) {
        // Sequential first request to establish session, then parallel rest
        await submitOne(selectedChatProviderIds[0]);
        if (sessionGenRef.current !== sendGen) return;
        await Promise.allSettled(
          selectedChatProviderIds.slice(1).map((cpId) => submitOne(cpId)),
        );
      } else {
        await Promise.allSettled(
          selectedChatProviderIds.map((cpId) => submitOne(cpId)),
        );
      }
    },
    [sessionId, uploadedFiles, selectedChatProviderIds, chatProviders, pollForResult],
  );

  const handleFileUpload = useCallback(async (files: File[]) => {
    const remaining = MAX_FILES - uploadedFiles.length;
    if (remaining <= 0) {
      toast.error(`Maximum ${MAX_FILES} files per session reached`);
      return;
    }
    const batch = files.slice(0, remaining);
    if (batch.length < files.length) {
      toast.warning(`Only the first ${remaining} file(s) will be uploaded (${MAX_FILES}-file limit)`);
    }
    setUploadProgress(0);
    let anyFailed = false;
    // Track cumulative extracted-text bytes across already-attached files and this batch.
    // size_bytes on FileUploadResponse reflects server-side extracted text size.
    let cumulativeBytes = uploadedFiles.reduce((sum, f) => sum + f.text_size_bytes, 0);
    for (let i = 0; i < batch.length; i++) {
      const file = batch[i];
      // Refuse what the server would refuse (or what would blow the per-message
      // limit) before sending the bytes, not after a long upload.
      if (file.size === 0) {
        toast.error(`"${file.name}" is empty (0 bytes), so there is nothing to upload`);
        anyFailed = true;
        continue;
      }
      if (file.size > MAX_UPLOAD_FILE_BYTES) {
        toast.error(`"${file.name}" is larger than the ${MAX_UPLOAD_FILE_MB} MB upload limit`);
        anyFailed = true;
        continue;
      }
      // A plain-text file's extracted text is at least as large as the file,
      // so the per-message limit can be checked up front.
      const isPlainText = PLAIN_TEXT_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext));
      if (isPlainText && cumulativeBytes + file.size > MAX_BYTES) {
        toast.error(
          `"${file.name}" skipped — adding it would exceed the ${MAX_MESSAGE_MB} MB per-message limit`,
        );
        anyFailed = true;
        continue;
      }
      try {
        const resp = await uploadFile(file, (pct) => {
          // Show overall progress: (completed files + current file fraction) / total
          const overall = ((i + (pct >= 100 ? 1 : pct / 100)) / batch.length) * 100;
          setUploadProgress(overall >= 100 ? "processing" : Math.round(overall));
        });
        if (cumulativeBytes + resp.text_size_bytes > MAX_BYTES) {
          toast.error(
            `"${resp.name}" skipped — adding it would exceed the ${MAX_MESSAGE_MB} MB per-message limit`,
          );
          anyFailed = true;
          continue;
        }
        cumulativeBytes += resp.text_size_bytes;
        // Commit each file immediately so partial failures don't discard successes
        setUploadedFiles((prev) => [...prev, resp]);
      } catch (err) {
        console.error(`Failed to upload ${file.name}:`, err);
        toast.error(
          `Couldn't upload "${file.name}": ${err instanceof Error ? err.message : "the upload failed"}`,
        );
        anyFailed = true;
      }
    }
    setUploadProgress(null);
    if (anyFailed && batch.length > 1) {
      toast.warning("Some files failed to upload — successfully uploaded files are still attached");
    }
  }, [uploadedFiles]);

  const cancelAllPollers = useCallback(() => {
    sessionGenRef.current += 1;
    for (const mapping of Object.values(execMapRef.current)) {
      (mapping as ExecMapping & { cancelPoll?: () => void }).cancelPoll?.();
    }
    for (const timer of pollTimersRef.current) {
      clearTimeout(timer);
    }
    pollTimersRef.current.clear();
    execMapRef.current = {};
    pendingCountRef.current = 0;
    setIsAnyStreaming(false);
  }, []);

  // Cleanup poll timers on unmount
  useEffect(() => {
    return () => { cancelAllPollers(); };
  }, [cancelAllPollers]);

  const handleNewSession = useCallback(() => {
    cancelAllPollers();
    // Clear draft files (not session-keyed files — those stay for reselection).
    localStorage.removeItem(STORAGE_KEY_FILES_DRAFT);
    setSessionId(null);
    setTurns([]);
    setUploadedFiles([]);
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
      wsReadyRef.current = false;
    }
  }, [cancelAllPollers, sessionId]);

  const handleSelectSession = useCallback(
    (id: string) => {
      if (id === sessionId) return;
      cancelAllPollers();
      setTurns([]);
      setUploadedFiles([]);
      setSessionId(id);
    },
    [sessionId, cancelAllPollers],
  );

  return (
    <AppShell
      activeSessionId={sessionId}
      onSelectSession={handleSelectSession}
      onNewSession={handleNewSession}
    >
      <div className="flex h-full flex-col">
        {/* Toolbar */}
        <div className="flex items-center gap-3 border-b px-4 py-2" role="toolbar" aria-label="Chat options">
          <ChatProviderSelector
            chatProviders={chatProviders}
            llmProviders={llmProviders}
            selectedIds={selectedChatProviderIds}
            onSelectionChange={setSelectedChatProviderIds}
            orderedIds={chatProviderOrder}
            onOrderChange={setChatProviderOrder}
          />
        </div>

        {/* Upload / processing progress */}
        {uploadProgress !== null && (
          <div className="border-b bg-muted/30 px-4 py-3">
            <div className="flex items-center justify-between text-sm">
              <div className="flex items-center gap-2 text-muted-foreground">
                {uploadProgress === "processing" ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <Upload className="h-4 w-4" aria-hidden="true" />
                )}
                <span className="font-medium">
                  {uploadProgress === "processing" ? "Extracting text…" : "Uploading…"}
                </span>
              </div>
              {typeof uploadProgress === "number" && (
                <span className="tabular-nums text-muted-foreground">{uploadProgress}%</span>
              )}
            </div>
            <div className="mt-2">
              {uploadProgress === "processing" ? (
                /* Indeterminate shimmer while server extracts text */
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                  <div className="h-full w-1/3 animate-[shimmer_1.2s_ease-in-out_infinite] rounded-full bg-primary" />
                </div>
              ) : (
                <Progress value={uploadProgress} className="h-1.5" />
              )}
            </div>
          </div>
        )}

        {/* File attachment bar */}
        {uploadedFiles.length > 0 && (
          <div className="flex flex-wrap gap-2 border-b px-4 py-2">
            {uploadedFiles.map((f) => (
              <FileAttachment
                key={f.id}
                name={f.name}
                sizeBytes={f.size_bytes}
                tokenCount={f.token_count}
                onRemove={() => setUploadedFiles((prev) => prev.filter((x) => x.id !== f.id))}
              />
            ))}
          </div>
        )}

        {turns.length === 0 ? (
          /* Empty state: centered welcome + input */
          <div className="flex min-h-0 flex-1 flex-col items-center justify-center px-4">
            <div className="mb-8 text-center text-muted-foreground">
              <p className="text-lg font-medium">Welcome to RLM Studio</p>
              <p className="mt-2 text-sm">
                Upload a document and ask questions. The RLM engine will recursively
                explore the content using code generation to find answers efficiently.
              </p>
            </div>
            <div className="w-full max-w-3xl">
              <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={isAnyStreaming || selectedChatProviderIds.length === 0 || uploadProgress !== null} />
            </div>
          </div>
        ) : (
          /* Active conversation: turns scroll, input pinned at bottom */
          <>
            <div className="min-h-0 flex-1 overflow-auto" aria-label="Message history">
              <div className="w-full space-y-8 px-4 py-6" role="log" aria-live="polite">
                {turns.map((turn) => (
                  <div key={turn.id} className="space-y-4">
                    {/* User message - full width */}
                    <div className="flex gap-3">
                      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <User className="h-4 w-4" aria-hidden="true" />
                      </div>
                      <div className="flex flex-col gap-1">
                        <div className="rounded-lg bg-primary px-4 py-3 text-sm text-primary-foreground">
                          <p className="whitespace-pre-wrap">{turn.userMessage.content}</p>
                        </div>
                      </div>
                    </div>

                    {/* Provider responses - grid expands to full width; browser scrolls */}
                    <div
                      className="grid gap-4 mr-4"
                      style={{
                        gridTemplateColumns: `repeat(${orderedSelectedIds.length}, minmax(320px, 1fr))`,
                      }}
                    >
                      {orderedSelectedIds.map((cpId) => {
                        const resp = turn.responses[cpId];
                        return (
                          <div
                            key={cpId}
                            className="min-w-0 overflow-hidden rounded-lg border bg-muted/30 p-4"
                          >
                            {/* Provider name header */}
                            <div className="mb-2 flex items-center gap-2">
                              <div className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
                                <Bot className="h-3 w-3" aria-hidden="true" />
                              </div>
                              <span className="text-xs font-medium text-muted-foreground">
                                {resp?.chatProviderName}
                                {resp?.profileName && (
                                  <span className="ml-1 font-normal opacity-70">
                                    · {resp.profileName}
                                  </span>
                                )}
                              </span>
                            </div>

                            {/* Response content or loading indicator */}
                            {resp && resp.content ? (
                              <>
                                <div className="prose prose-sm dark:prose-invert max-w-none overflow-x-auto">
                                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                    {resp.content}
                                  </ReactMarkdown>
                                </div>
                                {resp.metrics && (
                                  <div className="mt-3 flex flex-col gap-2 border-t pt-2 text-xs text-muted-foreground">
                                    <div className="flex items-center gap-4">
                                      <span>{resp.metrics.total_tokens} tokens</span>
                                      <span>${resp.metrics.cost_usd.toFixed(4)}</span>
                                      <span>{resp.metrics.elapsed_seconds.toFixed(1)}s</span>
                                      <ResponseRating
                                        executionId={resp.executionId}
                                        sessionId={sessionId || ""}
                                        chatProviderId={cpId}
                                        currentRating={ratings[resp.executionId]}
                                        onRate={handleRate}
                                      />
                                      {resp.steps && resp.steps.length > 0 && (
                                        <button
                                          onClick={() => setStepsVisible((prev) => ({ ...prev, [resp.executionId]: !prev[resp.executionId] }))}
                                          className="ml-auto flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
                                          aria-label="Toggle steps"
                                        >
                                          <Terminal className="h-3 w-3" />
                                          <span>{resp.steps.length} steps</span>
                                          {stepsVisible[resp.executionId] ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                                        </button>
                                      )}
                                    </div>
                                    {judgeScores[resp.executionId] && judgeScores[resp.executionId].overall_score > 0 && (
                                      <JudgeScores
                                        dimensions={judgeScores[resp.executionId].dimensions}
                                        overallScore={judgeScores[resp.executionId].overall_score}
                                      />
                                    )}
                                    {resp.steps && stepsVisible[resp.executionId] && (
                                      <div className="mt-2 space-y-2 border-t pt-2">
                                        {resp.steps.map((step, i) => (
                                          <div key={i} className="rounded border border-muted bg-background p-2 font-mono text-xs">
                                            <div className="flex items-center gap-2 mb-1 text-muted-foreground">
                                              <span className="font-semibold uppercase tracking-wide text-[10px]">{step.action_type}</span>
                                              <span>step {step.index + 1}</span>
                                              {step.model && <span className="ml-auto opacity-60">{step.model}</span>}
                                              {step.duration_seconds > 0 && <span className="opacity-60">{step.duration_seconds.toFixed(1)}s</span>}
                                              {(step.input_tokens + step.output_tokens) > 0 && (
                                                <span className="opacity-60">{step.input_tokens + step.output_tokens} tok</span>
                                              )}
                                            </div>
                                            {step.code && (
                                              <pre className="mb-1 overflow-x-auto rounded bg-muted/50 p-1.5 text-[11px] whitespace-pre-wrap">{step.code}</pre>
                                            )}
                                            {step.output && (
                                              <div className="whitespace-pre-wrap opacity-80">{step.output}</div>
                                            )}
                                          </div>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </>
                            ) : resp ? (
                              <TypingIndicator />
                            ) : null}
                          </div>
                        );
                      })}
                    </div>

                    {/* Comparison summary for multi-provider turns */}
                    {(() => {
                      const completed = Object.entries(turn.responses).filter(
                        ([, r]) => r?.metrics,
                      );
                      if (completed.length < 2) return null;
                      // Separate usable responses from failures/errors.
                      // Failed responses start with ⚠️ or "Error:" — they
                      // should appear in the summary but not compete for
                      // Cheapest/Fastest/Best Quality rankings.
                      const isUsable = (r: typeof completed[0][1]) =>
                        r.content && !r.content.startsWith("⚠️") && !r.content.startsWith("Error:");
                      const usable = completed.filter(([, r]) => isUsable(r));
                      const cheapest = usable.length > 0
                        ? usable.reduce((a, b) => a[1].metrics!.cost_usd <= b[1].metrics!.cost_usd ? a : b)
                        : null;
                      const fastest = usable.length > 0
                        ? usable.reduce((a, b) => a[1].metrics!.elapsed_seconds <= b[1].metrics!.elapsed_seconds ? a : b)
                        : null;
                      return (
                        <div className="rounded-lg border bg-muted/30 p-3 text-xs text-muted-foreground">
                          <div className="flex gap-6">
                            {completed.map(([cpId, r]) => (
                              <div key={cpId} className="space-y-0.5">
                                <span className={`font-medium ${isUsable(r) ? "text-foreground" : "text-destructive"}`}>
                                  {r.chatProviderName}{!isUsable(r) ? " (failed)" : ""}
                                </span>
                                <div>
                                  {r.metrics!.total_tokens.toLocaleString()} tokens
                                  {" · "}${r.metrics!.cost_usd.toFixed(4)}
                                  {" · "}{r.metrics!.elapsed_seconds.toFixed(1)}s
                                </div>
                              </div>
                            ))}
                          </div>
                          {(cheapest || fastest) && (
                          <p className="mt-1">
                            {cheapest && (<>Cheapest:{" "}
                            <span className="font-medium text-emerald-600 dark:text-emerald-400">
                              {cheapest[1].chatProviderName}
                            </span></>)}
                            {cheapest && fastest && " · "}
                            {fastest && (<>Fastest:{" "}
                            <span className="font-medium text-blue-600 dark:text-blue-400">
                              {fastest[1].chatProviderName}
                            </span></>)}
                            {(() => {
                              // Find best quality from judge scores, excluding failures
                              const scored = usable
                                .filter(([, r]) => judgeScores[r.executionId])
                                .map(([, r]) => ({
                                  name: r.chatProviderName,
                                  score: judgeScores[r.executionId].overall_score,
                                }));
                              if (scored.length === 0) return null;
                              const best = scored.reduce((a, b) =>
                                a.score >= b.score ? a : b
                              );
                              return (
                                <>
                                  {" · "}Best Quality:{" "}
                                  <span className="font-medium text-violet-600 dark:text-violet-400">
                                    {best.name}
                                  </span>
                                </>
                              );
                            })()}
                          </p>
                          )}
                          <div className="mt-2 flex items-center gap-3">
                            <PickWinner
                              sessionId={sessionId || ""}
                              responses={completed.map(([cpId, r]) => ({
                                executionId: r.executionId,
                                cpId,
                                name: r.chatProviderName,
                              }))}
                              currentWinnerId={
                                completed.find(
                                  ([, r]) => bestPicks[r.executionId]
                                )?.[1]?.executionId
                              }
                              onPick={handlePickWinner}
                            />
                            {config?.judge_chat_provider_id && (
                              <Button
                                variant="outline"
                                size="sm"
                                className="h-6 gap-1 px-2 text-xs"
                                disabled={isJudging}
                                onClick={() =>
                                  handleTriggerJudge(
                                    completed.map(([, r]) => r.executionId)
                                  )
                                }
                              >
                                {isJudging ? (
                                  <Loader2 className="h-3 w-3 animate-spin" />
                                ) : (
                                  <Sparkles className="h-3 w-3" />
                                )}
                                Auto-Judge
                              </Button>
                            )}
                          </div>
                        </div>
                      );
                    })()}
                  </div>
                ))}
                {isAnyStreaming && <TypingIndicator />}
                <div ref={messagesEndRef} />
              </div>
            </div>
            <ChatInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={isAnyStreaming || selectedChatProviderIds.length === 0 || uploadProgress !== null} />
          </>
        )}
      </div>
    </AppShell>
  );
}
