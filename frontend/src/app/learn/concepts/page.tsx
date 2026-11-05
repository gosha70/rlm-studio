"use client";

import useSWR from "swr";
import { Shield, ShieldX } from "lucide-react";
import { AppShell } from "@/components/shared/app-shell";
import { BackToLearn } from "@/components/learn/back-to-learn";
import { DiagnosticsStrip } from "@/components/learn/diagnostics-strip";
import { MarkdownDoc } from "@/components/learn/markdown-doc";
import { ModeChooser } from "@/components/learn/mode-chooser";
import { ReplayWalkthrough } from "@/components/learn/replay-walkthrough";
import { getBundledReplay, getDiagnostics, type LearnReplay } from "@/lib/api";

/**
 * Concepts page — V1 scope.
 *
 * Content order is task-first, not architecture-first. The page
 * opens with the problem the user actually has, then the mode
 * decision, then the trust boundary, then a deep-dive pull of the
 * research-level rlm-concepts.md for anyone who wants the theory.
 */
export default function ConceptsPage() {
  const { data } = useSWR("learn-diagnostics", getDiagnostics, {
    refreshInterval: 30_000,
    dedupingInterval: 30_000,
    errorRetryCount: 2,
  });
  const { data: replay, error: replayError } = useSWR<LearnReplay>(
    "learn-bundled-replay",
    getBundledReplay,
    { revalidateOnFocus: false },
  );

  return (
    <AppShell>
      <div className="mx-auto max-w-5xl px-6 py-8">
        <BackToLearn className="mb-4" />
        <header className="mb-4">
          <h2 className="text-2xl font-semibold tracking-tight">Concepts</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            What RLM Studio solves, which mode to pick, and what the
            sandbox does (and doesn&apos;t) do.
          </p>
        </header>

        <DiagnosticsStrip data={data ?? null} className="mb-8" />

        {/* 1 — Lead with the user problem, not the architecture. */}
        <section
          id="problem"
          aria-labelledby="concepts-problem"
          className="mb-10"
        >
          <h3
            id="concepts-problem"
            className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground"
          >
            What problem does this solve?
          </h3>
          <div className="rounded-lg border bg-card p-5 shadow-sm">
            <p className="text-base font-medium">
              Some tasks are too large or too messy for one prompt.
            </p>
            <p className="mt-3 text-sm text-muted-foreground">
              RLM Studio lets the model inspect the content step by step
              instead of stuffing everything into context at once. For a
              long document, a dense log, or a multi-part question, that
              usually produces better answers at lower cost than trying
              to fit the whole input into a single request.
            </p>
            <p className="mt-3 text-sm text-muted-foreground">
              Each Chat Provider runs one mode — Direct, RLM, RAG or
              Official RLM. Not sure which fits? Select a Direct and an RLM
              Chat Provider together in Chat and compare their answers,
              tokens, cost and time side by side.
            </p>
          </div>
        </section>

        {/* 2 — Mode decision. Kept high; the "When not to use RLM"
            callout sits directly under the rows so users see it before
            committing to RLM for small tasks. */}
        <section
          id="mode-guide"
          aria-labelledby="concepts-mode-guide"
          className="mb-10"
        >
          <h3
            id="concepts-mode-guide"
            className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground"
          >
            Which mode should I use?
          </h3>
          <ModeChooser />

          <p className="mt-3 text-xs text-muted-foreground">
            These are the four modes a Chat Provider can run (Settings
            &rarr; Chat Providers); this is just a guided way to pick the
            right one for the task you&apos;re about to run.
          </p>

          <aside className="mt-4 rounded-md border border-amber-300/60 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-400/30 dark:bg-amber-500/10 dark:text-amber-200">
            <p className="font-semibold">When not to use RLM</p>
            <ul className="mt-2 ml-5 list-disc space-y-1">
              <li>Short note or single-turn chat — use Direct.</li>
              <li>
                Side-by-side benchmarking — select several Chat Providers in
                Chat, or use the LLM Tuner.
              </li>
              <li>
                Not sure — compare Direct and RLM side by side; don&apos;t
                force RLM.
              </li>
            </ul>
          </aside>
        </section>

        {/* 3 — Trust boundary. Surfaced above the theory deep-dive
            because "what can the sandbox actually do to my machine"
            is usually the first follow-up question after "what is RLM". */}
        <section
          id="sandbox"
          aria-labelledby="concepts-sandbox"
          className="mb-10"
        >
          <h3
            id="concepts-sandbox"
            className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground"
          >
            What the sandbox can and can&apos;t do
          </h3>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div className="rounded-lg border border-emerald-300/50 bg-emerald-50 p-4 text-sm text-emerald-950 shadow-sm dark:border-emerald-400/30 dark:bg-emerald-500/10 dark:text-emerald-100">
              <div className="flex items-center gap-2 font-semibold">
                <Shield className="h-4 w-4" aria-hidden="true" />
                Can do
              </div>
              <ul className="mt-2 ml-5 list-disc space-y-1">
                <li>Run small Python snippets written by the model.</li>
                <li>
                  Inspect the content you provided via helpers like{" "}
                  <code className="rounded bg-emerald-100/60 px-1 dark:bg-emerald-500/20">
                    peek
                  </code>
                  ,{" "}
                  <code className="rounded bg-emerald-100/60 px-1 dark:bg-emerald-500/20">
                    grep
                  </code>
                  ,{" "}
                  <code className="rounded bg-emerald-100/60 px-1 dark:bg-emerald-500/20">
                    chunk
                  </code>
                  .
                </li>
                <li>
                  Build up an answer step-by-step in REPL variables.
                </li>
              </ul>
            </div>
            <div className="rounded-lg border border-red-300/50 bg-red-50 p-4 text-sm text-red-950 shadow-sm dark:border-red-400/30 dark:bg-red-500/10 dark:text-red-100">
              <div className="flex items-center gap-2 font-semibold">
                <ShieldX className="h-4 w-4" aria-hidden="true" />
                Can&apos;t do
              </div>
              <ul className="mt-2 ml-5 list-disc space-y-1">
                <li>Read, write, or delete files on your machine.</li>
                <li>Open network connections or call external services.</li>
                <li>
                  Spawn subprocesses or import restricted modules
                  (<code className="rounded bg-red-100/60 px-1 dark:bg-red-500/20">
                    os
                  </code>
                  ,{" "}
                  <code className="rounded bg-red-100/60 px-1 dark:bg-red-500/20">
                    subprocess
                  </code>
                  ,{" "}
                  <code className="rounded bg-red-100/60 px-1 dark:bg-red-500/20">
                    socket
                  </code>
                  , etc.).
                </li>
              </ul>
            </div>
          </div>
          <p className="mt-3 text-xs text-muted-foreground">
            Code is compiled through RestrictedPython before execution;
            only an allowlisted set of safe builtins is in scope.
          </p>
        </section>

        {/* 4 — Replay walkthrough (V2). Sits between sandbox and the
            research deep dive: decision context first, trust boundary
            second, then mechanics through a real example. */}
        <section
          id="replay"
          aria-labelledby="concepts-replay"
          className="mb-10"
        >
          <h3
            id="concepts-replay"
            className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground"
          >
            See it in action
          </h3>
          {replayError ? (
            <div
              role="alert"
              className="rounded-md border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive"
            >
              Couldn’t load the bundled replay.
            </div>
          ) : !replay ? (
            <p className="text-sm text-muted-foreground">Loading replay…</p>
          ) : (
            <ReplayWalkthrough replay={replay} />
          )}
        </section>

        {/* 5 — Deep dive for readers who want the research-level
            explanation (kept, but last — not the landing content). */}
        <section
          id="deep-dive"
          aria-labelledby="concepts-deep-dive"
          className="mb-10"
        >
          <h3
            id="concepts-deep-dive"
            className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted-foreground"
          >
            Dive deeper — the RLM paper and loop mechanics
          </h3>
          <MarkdownDoc slug="rlm-concepts" />
        </section>
      </div>
    </AppShell>
  );
}
