"use client";

/**
 * Evidence-gated prompt refinement (#78) for one profile: pick failed or
 * degraded RLM runs, propose a new RLM supplement, evaluate it against the
 * current one on held-out cases, then accept, reject or roll back.
 *
 * Nothing here writes to the profile except Accept and Rollback, and both go
 * through the server's checked, locked decision endpoints.
 */

import { useMemo, useRef, useState } from "react";
import useSWR from "swr";
import { toast } from "sonner";
import { Download } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  acceptRefinement,
  apiErrorMessage,
  apiErrorStatus,
  createRefinement,
  evaluateRefinement,
  getHoldoutCases,
  getRefinement,
  getRefinementSourceRuns,
  listRefinements,
  refinementExportUrl,
  rejectRefinement,
  rollbackRefinement,
  type ChatProviderConfig,
  type Refinement,
  type RefinementEvaluation,
  type RunProfile,
} from "@/lib/api";
import {
  MODE_RLM,
  REFINEMENT_DECISION_REASON_MAX_CHARS,
  REFINEMENT_MAX_REPS,
  REFINEMENT_MIN_REPS,
  REFINEMENT_POLL_MS,
  REFINEMENT_REJECTABLE_STATUSES,
  REFINEMENT_STATUS_ACCEPTED,
  REFINEMENT_STATUS_EVALUATED,
  REFINEMENT_STATUS_EVALUATING,
  REFINEMENT_STATUS_LABELS,
  REFINEMENT_STATUS_PROPOSAL_READY,
  type RefinementStatus,
} from "@/lib/constants";

const NEW_REFINEMENT = "__new__";
/** Statuses from which an accept can still happen (the detach note applies). */
const ACCEPT_STILL_POSSIBLE: readonly RefinementStatus[] = [
  REFINEMENT_STATUS_PROPOSAL_READY,
  REFINEMENT_STATUS_EVALUATING,
  REFINEMENT_STATUS_EVALUATED,
];
const CONFIGURED_JUDGE = "__configured__";

/** Local date-time with seconds; never a relative "5m ago". */
function formatTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString();
}

function formatRate(rate: number | null): string {
  return rate === null ? "—" : `${Math.round(rate * 100)}%`;
}

function statusVariant(status: RefinementStatus): "default" | "secondary" | "destructive" | "outline" | "success" | "warning" {
  switch (status) {
    case REFINEMENT_STATUS_ACCEPTED:
      return "success";
    case REFINEMENT_STATUS_EVALUATING:
      return "warning";
    case REFINEMENT_STATUS_EVALUATED:
    case REFINEMENT_STATUS_PROPOSAL_READY:
      return "default";
    default:
      return "secondary";
  }
}

function StatusBadge({ status }: { status: RefinementStatus }) {
  return (
    <Badge variant={statusVariant(status)} className="text-xs">
      {REFINEMENT_STATUS_LABELS[status] ?? status}
    </Badge>
  );
}

interface RefinementPanelProps {
  profile: RunProfile;
  chatProviders: ChatProviderConfig[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** Called after Accept or Rollback changed the profile's RLM prompt. */
  onProfileChanged: () => void;
}

export function RefinementPanel({
  profile,
  chatProviders,
  open,
  onOpenChange,
  onProfileChanged,
}: RefinementPanelProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const {
    data: refinements = [],
    error,
    mutate: mutateList,
  } = useSWR(["refinements", profile.id], () => listRefinements(profile.id));

  const current = selected ?? (refinements.length === 0 ? NEW_REFINEMENT : refinements[0].id);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] max-w-5xl overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Refine RLM prompt: {profile.name}</DialogTitle>
          <DialogDescription>
            Propose a new RLM supplement from this profile&apos;s failed runs, test it against
            the current one on held-out cases, then decide. Nothing changes the profile until you
            accept.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 md:grid-cols-[14rem_1fr]">
          <nav aria-label="Refinement history" className="space-y-2">
            <Button
              size="sm"
              variant={current === NEW_REFINEMENT ? "default" : "outline"}
              className="w-full"
              onClick={() => setSelected(NEW_REFINEMENT)}
            >
              New refinement
            </Button>
            {error && (
              <p className="text-xs text-destructive" role="alert">
                {apiErrorMessage(error)}
              </p>
            )}
            <ul className="space-y-1">
              {refinements.map((r) => (
                <li key={r.id}>
                  <button
                    type="button"
                    onClick={() => setSelected(r.id)}
                    aria-current={current === r.id}
                    className={`w-full rounded-md border px-2 py-1.5 text-left text-xs ${
                      current === r.id ? "border-primary bg-muted" : "border-muted"
                    }`}
                  >
                    <StatusBadge status={r.status} />
                    <span className="mt-1 block text-muted-foreground">{formatTime(r.created_at)}</span>
                  </button>
                </li>
              ))}
            </ul>
            {refinements.length > 0 && (
              <a
                href={refinementExportUrl(profile.id)}
                className="inline-flex items-center gap-1 text-xs text-muted-foreground underline"
              >
                <Download className="h-3 w-3" aria-hidden="true" />
                Export history (JSONL)
              </a>
            )}
          </nav>

          <section aria-label="Refinement detail" className="min-w-0">
            {current === NEW_REFINEMENT ? (
              <NewProposal
                profile={profile}
                chatProviders={chatProviders}
                onCreated={(created) => {
                  void mutateList();
                  setSelected(created.id);
                }}
              />
            ) : (
              <RefinementDetail
                key={current}
                profile={profile}
                chatProviders={chatProviders}
                refinementId={current}
                onChanged={() => void mutateList()}
                onProfileChanged={onProfileChanged}
              />
            )}
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}

// ---------------------------------------------------------------------------
// New proposal
// ---------------------------------------------------------------------------

function NewProposal({
  profile,
  chatProviders,
  onCreated,
}: {
  profile: RunProfile;
  chatProviders: ChatProviderConfig[];
  onCreated: (created: Refinement) => void;
}) {
  const { data: runs = [], error, isLoading } = useSWR(["refinement-source-runs", profile.id], () =>
    getRefinementSourceRuns(profile.id),
  );
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [proposerId, setProposerId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  // Default proposer: the Chat Provider of the newest selected run.
  const defaultProposer = useMemo(() => {
    const newest = runs.find((r) => picked.has(r.id));
    return newest?.chat_provider_id ?? null;
  }, [runs, picked]);
  const effectiveProposer = proposerId ?? defaultProposer;
  const proposerName = chatProviders.find((cp) => cp.id === effectiveProposer)?.name;

  const toggle = (id: string) =>
    setPicked((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const propose = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const created = await createRefinement(profile.id, {
        source_run_ids: runs.filter((r) => picked.has(r.id)).map((r) => r.id),
        ...(effectiveProposer ? { proposal_chat_provider_id: effectiveProposer } : {}),
      });
      toast.success("Proposal ready");
      onCreated(created);
    } catch (err) {
      setMessage(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  if (isLoading) return <p className="text-sm text-muted-foreground">Loading runs…</p>;
  if (error) {
    return (
      <p className="text-sm text-destructive" role="alert">
        {apiErrorMessage(error)}
      </p>
    );
  }
  if (runs.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        No usable runs yet. Refinement learns from RLM runs of this profile recorded after
        document hashing was added; run the profile in RLM mode first.
      </p>
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-sm">Select the runs that went wrong. The newest 12 are used as evidence.</p>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-8">
              <span className="sr-only">Select</span>
            </TableHead>
            <TableHead>When</TableHead>
            <TableHead>Outcome</TableHead>
            <TableHead>Query</TableHead>
            <TableHead>Chat Provider</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow key={run.id}>
              <TableCell>
                <input
                  type="checkbox"
                  aria-label={`Use run ${run.id}`}
                  checked={picked.has(run.id)}
                  onChange={() => toggle(run.id)}
                />
              </TableCell>
              <TableCell className="whitespace-nowrap text-xs">{formatTime(run.created_at)}</TableCell>
              <TableCell className="text-xs">
                {run.outcome_category ?? (run.success ? "success" : "error")}
                {run.degraded && (
                  <Badge variant="warning" className="ml-1 text-xs">
                    degraded
                  </Badge>
                )}
              </TableCell>
              <TableCell className="max-w-[18rem] truncate text-xs" title={run.query}>
                {run.query}
              </TableCell>
              <TableCell className="text-xs">{run.chat_provider_name ?? "—"}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      <div className="space-y-1">
        <Label htmlFor={`proposer-${profile.id}`} className="text-xs">
          Proposal Chat Provider
        </Label>
        <Select
          value={effectiveProposer ?? undefined}
          onValueChange={(v) => setProposerId(v)}
        >
          <SelectTrigger id={`proposer-${profile.id}`} className="h-8 text-xs">
            <SelectValue placeholder="Newest selected run's Chat Provider" />
          </SelectTrigger>
          <SelectContent>
            {chatProviders.map((cp) => (
              <SelectItem key={cp.id} value={cp.id} className="text-xs">
                {cp.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          One call. The selected runs&apos; queries, answers and errors are sent to{" "}
          {proposerName ?? "this provider"} after known key shapes and key=value lines are
          redacted; secrets written in free text are not caught.
        </p>
      </div>

      {message && (
        <p className="text-xs text-destructive" role="alert">
          {message}
        </p>
      )}
      <Button size="sm" onClick={propose} disabled={busy || picked.size === 0}>
        {busy ? "Proposing…" : `Propose from ${picked.size} run${picked.size === 1 ? "" : "s"}`}
      </Button>
    </div>
  );
}

// ---------------------------------------------------------------------------
// One refinement
// ---------------------------------------------------------------------------

function RefinementDetail({
  profile,
  chatProviders,
  refinementId,
  onChanged,
  onProfileChanged,
}: {
  profile: RunProfile;
  chatProviders: ChatProviderConfig[];
  refinementId: string;
  onChanged: () => void;
  onProfileChanged: () => void;
}) {
  const lastStatus = useRef<RefinementStatus | null>(null);
  const { data: refinement, error, mutate } = useSWR(
    ["refinement", profile.id, refinementId],
    () => getRefinement(profile.id, refinementId),
    {
      refreshInterval: (latest) =>
        latest?.status === REFINEMENT_STATUS_EVALUATING ? REFINEMENT_POLL_MS : 0,
      // Only this view polls, so refresh the history list when a poll sees
      // the evaluation end; otherwise its badge stays on "Evaluating".
      onSuccess: (latest) => {
        if (lastStatus.current === REFINEMENT_STATUS_EVALUATING && latest.status !== REFINEMENT_STATUS_EVALUATING) {
          onChanged();
        }
        lastStatus.current = latest.status;
      },
    },
  );
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [reason, setReason] = useState("");
  const [acknowledge, setAcknowledge] = useState(false);

  if (error) {
    return (
      <p className="text-sm text-destructive" role="alert">
        {apiErrorMessage(error)}
      </p>
    );
  }
  if (!refinement) return <p className="text-sm text-muted-foreground">Loading…</p>;

  const { proposal, status } = refinement;
  const run = async (action: () => Promise<unknown>, success: string, changesProfile: boolean) => {
    setBusy(true);
    setMessage(null);
    // Refresh the profile before re-reading the refinement, so a stale edit
    // form on the card is closed whatever happens to that read. Stay busy
    // through the re-read so the buttons cannot send a second request.
    try {
      try {
        await action();
        toast.success(success);
        if (changesProfile) onProfileChanged();
      } catch (err) {
        setMessage(apiErrorMessage(err));
        // A split-brain accept or rollback (5xx after a successful save) did
        // change the profile, so refresh it. A 4xx was refused before any
        // write: leave the card, and any edit in progress, alone.
        if (changesProfile && apiErrorStatus(err) >= 500) onProfileChanged();
      }
      await mutate();
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const evaluation = refinement.evaluation;
  const canAccept =
    status === REFINEMENT_STATUS_EVALUATED &&
    evaluation !== null &&
    evaluation.complete &&
    evaluation.deterministic_case_count > 0 &&
    (!evaluation.regression || acknowledge);
  const currentCustom = profile.system_prompts?.[MODE_RLM];

  return (
    <div className="space-y-4 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge status={status} />
        <span className="text-xs text-muted-foreground">
          Proposed {formatTime(refinement.created_at)} by {proposal.provider}/{proposal.model} ·
          prompt v{proposal.prompt_version} · from {proposal.source_run_ids.length} run
          {proposal.source_run_ids.length === 1 ? "" : "s"}
        </span>
      </div>

      <div>
        <h3 className="text-xs font-medium">Rationale</h3>
        <p className="mt-1 whitespace-pre-wrap text-xs">{proposal.rationale}</p>
      </div>

      <div>
        <h3 className="text-xs font-medium">Change to the RLM supplement</h3>
        <pre className="mt-1 max-h-64 overflow-auto rounded-md border bg-muted p-2 text-xs">
          {proposal.diff || "(no textual change)"}
        </pre>
        <div className="mt-2 grid gap-2 md:grid-cols-2">
          <div className="min-w-0">
            <h4 className="text-xs text-muted-foreground">Current supplement (when proposed)</h4>
            <pre
              aria-label="Current supplement"
              className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded-md border p-2 text-xs"
            >
              {proposal.base.resolved || "(empty)"}
            </pre>
          </div>
          <div className="min-w-0">
            <h4 className="text-xs text-muted-foreground">Candidate supplement</h4>
            <pre
              aria-label="Candidate supplement"
              className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded-md border p-2 text-xs"
            >
              {proposal.candidate}
            </pre>
          </div>
        </div>
        {!proposal.base.rlm_key_present && ACCEPT_STILL_POSSIBLE.includes(status) && (
          <p className="mt-1 text-xs text-muted-foreground">
            This profile follows the prompt group &quot;{proposal.base.template_name ?? "global defaults"}
            &quot;. Accepting gives it a custom RLM prompt, so later edits to that group stop applying
            to RLM until you roll back.
          </p>
        )}
        <p className="mt-1 text-xs text-muted-foreground">
          The profile&apos;s RLM prompt now:{" "}
          {currentCustom ? "custom text" : `follows "${profile.prompt_template_name ?? "global defaults"}"`}
          {currentCustom === proposal.candidate ? " (this candidate)" : ""}.
        </p>
      </div>

      {status === REFINEMENT_STATUS_PROPOSAL_READY && (
        <EvaluateForm
          profile={profile}
          chatProviders={chatProviders}
          refinement={refinement}
          onStarted={async () => {
            await mutate();
            onChanged();
          }}
        />
      )}

      {status === REFINEMENT_STATUS_EVALUATING && <EvaluationProgress refinement={refinement} />}

      {evaluation && (
        <EvaluationResults
          evaluation={evaluation}
          decisionPending={status === REFINEMENT_STATUS_EVALUATED}
        />
      )}

      {refinement.error && (
        <p className="text-xs text-destructive" role="alert">
          {refinement.error}
        </p>
      )}

      {refinement.decision && (
        <p className="text-xs text-muted-foreground">
          {REFINEMENT_STATUS_LABELS[refinement.decision.decision as RefinementStatus] ??
            refinement.decision.decision}{" "}
          {formatTime(refinement.decision.decided_at)}
          {refinement.decision.acknowledge_regression ? " despite the flagged regression" : ""}
          {refinement.decision.reason ? `: ${refinement.decision.reason}` : ""}
          {refinement.decision.rollback
            ? ` · rolled back ${formatTime(refinement.decision.rollback.rolled_back_at)}`
            : ""}
        </p>
      )}

      {REFINEMENT_REJECTABLE_STATUSES.includes(status) && (
        <div className="space-y-2 rounded-md border p-3">
          <Label htmlFor={`reason-${refinementId}`} className="text-xs">
            Reason (optional)
          </Label>
          <Input
            id={`reason-${refinementId}`}
            className="h-8 text-xs"
            value={reason}
            maxLength={REFINEMENT_DECISION_REASON_MAX_CHARS}
            onChange={(e) => setReason(e.target.value)}
          />
          {status === REFINEMENT_STATUS_EVALUATED && evaluation?.regression && (
            <label className="flex items-center gap-2 text-xs">
              <input
                type="checkbox"
                checked={acknowledge}
                onChange={(e) => setAcknowledge(e.target.checked)}
              />
              Accept despite the regression on {evaluation.regression_case_ids.join(", ")}
            </label>
          )}
          <div className="flex gap-2">
            {status === REFINEMENT_STATUS_EVALUATED && (
              <Button
                size="sm"
                // A flagged regression makes Reject the default action (spec scenario 3).
                variant={evaluation?.regression ? "outline" : "default"}
                disabled={busy || !canAccept}
                onClick={() =>
                  run(
                    () =>
                      acceptRefinement(profile.id, refinementId, {
                        acknowledge_regression: acknowledge,
                        ...(reason.trim() ? { reason: reason.trim() } : {}),
                      }),
                    "Candidate accepted",
                    true,
                  )
                }
              >
                Accept
              </Button>
            )}
            <Button
              size="sm"
              variant={status === REFINEMENT_STATUS_EVALUATED && evaluation?.regression ? "default" : "outline"}
              disabled={busy}
              onClick={() =>
                run(
                  () => rejectRefinement(profile.id, refinementId, reason.trim() || undefined),
                  "Refinement rejected",
                  false,
                )
              }
            >
              Reject
            </Button>
          </div>
          {status === REFINEMENT_STATUS_EVALUATED && evaluation && !evaluation.complete && (
            <p className="text-xs text-muted-foreground">
              An incomplete evaluation cannot be accepted.
            </p>
          )}
          {status === REFINEMENT_STATUS_EVALUATED &&
            evaluation?.complete &&
            evaluation.deterministic_case_count === 0 && (
              <p className="text-xs text-muted-foreground">
                None of the evaluated cases has an expected answer, so this evaluation cannot be
                accepted.
              </p>
            )}
        </div>
      )}

      {status === REFINEMENT_STATUS_ACCEPTED && (
        <Button
          size="sm"
          variant="outline"
          disabled={busy}
          onClick={() =>
            run(() => rollbackRefinement(profile.id, refinementId), "Rolled back", true)
          }
        >
          Roll back
        </Button>
      )}

      {message && (
        <p className="text-xs text-destructive" role="alert">
          {message}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

function EvaluateForm({
  profile,
  chatProviders,
  refinement,
  onStarted,
}: {
  profile: RunProfile;
  chatProviders: ChatProviderConfig[];
  refinement: Refinement;
  onStarted: () => Promise<void>;
}) {
  const { data: cases = [], error } = useSWR(["refinement-holdout-cases", profile.id], () =>
    getHoldoutCases(profile.id),
  );
  const [picked, setPicked] = useState<Set<string> | null>(null);
  const [reps, setReps] = useState(String(REFINEMENT_MIN_REPS));
  const [evaluatorId, setEvaluatorId] = useState<string | null>(null);
  const [judgeId, setJudgeId] = useState(CONFIGURED_JUDGE);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const profileProviders = chatProviders.filter((cp) => cp.profile_id === profile.id);
  const proposalCp = refinement.proposal.proposal_chat_provider_id;
  const defaultEvaluator = profileProviders.some((cp) => cp.id === proposalCp)
    ? proposalCp
    : profileProviders[0]?.id ?? null;
  const evaluator = evaluatorId ?? defaultEvaluator;
  const evaluatorName = chatProviders.find((cp) => cp.id === evaluator)?.name;

  const selectedCases =
    picked ?? new Set(cases.filter((c) => c.holdout && c.materialisable).map((c) => c.id));
  // One list drives both the run count and the request, so they cannot disagree.
  const chosenCaseIds = cases
    .filter((c) => c.materialisable && selectedCases.has(c.id))
    .map((c) => c.id);
  const repCount = Number.parseInt(reps, 10);
  const repsValid =
    Number.isInteger(repCount) && repCount >= REFINEMENT_MIN_REPS && repCount <= REFINEMENT_MAX_REPS;
  const runCount = repsValid ? chosenCaseIds.length * repCount * 2 : 0;

  const toggle = (id: string) => {
    const next = new Set(selectedCases);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setPicked(next);
  };

  const start = async () => {
    if (!evaluator) return;
    setBusy(true);
    setMessage(null);
    try {
      await evaluateRefinement(profile.id, refinement.id, {
        case_ids: chosenCaseIds,
        reps: repCount,
        chat_provider_id: evaluator,
        ...(judgeId !== CONFIGURED_JUDGE ? { judge_chat_provider_id: judgeId } : {}),
      });
      toast.success("Evaluation started");
      await onStarted();
    } catch (err) {
      setMessage(apiErrorMessage(err));
    } finally {
      setBusy(false);
    }
  };

  if (error) {
    return (
      <p className="text-xs text-destructive" role="alert">
        {apiErrorMessage(error)}
      </p>
    );
  }

  return (
    <div className="space-y-3 rounded-md border p-3">
      <h3 className="text-xs font-medium">Evaluate on held-out cases</h3>
      <ul className="space-y-1">
        {cases.map((c) => (
          <li key={c.id} className="flex items-start gap-2 text-xs">
            <input
              type="checkbox"
              aria-label={`Evaluate on ${c.id}`}
              checked={selectedCases.has(c.id)}
              disabled={!c.materialisable}
              onChange={() => toggle(c.id)}
            />
            <span>
              <span className="font-mono">{c.id}</span>
              {c.holdout && (
                <Badge variant="outline" className="ml-1 text-xs">
                  holdout
                </Badge>
              )}
              {!c.deterministic && (
                <Badge variant="secondary" className="ml-1 text-xs">
                  judge only
                </Badge>
              )}
              {!c.materialisable && (
                <span className="ml-1 text-muted-foreground">({c.skip_reason})</span>
              )}
            </span>
          </li>
        ))}
      </ul>

      <div className="grid gap-3 sm:grid-cols-3">
        <div className="space-y-1">
          <Label htmlFor={`reps-${refinement.id}`} className="text-xs">
            Repetitions ({REFINEMENT_MIN_REPS}–{REFINEMENT_MAX_REPS})
          </Label>
          <Input
            id={`reps-${refinement.id}`}
            type="text"
            inputMode="numeric"
            className="h-8 text-xs"
            value={reps}
            onChange={(e) => setReps(e.target.value)}
          />
        </div>
        <div className="space-y-1">
          <Label htmlFor={`evaluator-${refinement.id}`} className="text-xs">
            Chat Provider (uses this profile)
          </Label>
          <Select value={evaluator ?? undefined} onValueChange={(v) => setEvaluatorId(v)}>
            <SelectTrigger id={`evaluator-${refinement.id}`} className="h-8 text-xs">
              <SelectValue placeholder="No Chat Provider uses this profile" />
            </SelectTrigger>
            <SelectContent>
              {profileProviders.map((cp) => (
                <SelectItem key={cp.id} value={cp.id} className="text-xs">
                  {cp.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <Label htmlFor={`judge-${refinement.id}`} className="text-xs">
            Judge (reported, never gates)
          </Label>
          <Select value={judgeId} onValueChange={setJudgeId}>
            <SelectTrigger id={`judge-${refinement.id}`} className="h-8 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={CONFIGURED_JUDGE} className="text-xs">
                Configured judge, if any
              </SelectItem>
              {chatProviders.map((cp) => (
                <SelectItem key={cp.id} value={cp.id} className="text-xs">
                  {cp.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <p className="text-xs text-muted-foreground">
        {repsValid
          ? `${runCount} RLM run${runCount === 1 ? "" : "s"} (${chosenCaseIds.length} cases × ${repCount} repetitions × baseline and candidate) on ${evaluatorName ?? "—"} at temperature 0. Cost appears in telemetry afterwards; it is not estimated here.`
          : `Repetitions must be between ${REFINEMENT_MIN_REPS} and ${REFINEMENT_MAX_REPS}.`}
      </p>
      {message && (
        <p className="text-xs text-destructive" role="alert">
          {message}
        </p>
      )}
      <Button
        size="sm"
        onClick={start}
        disabled={busy || !evaluator || !repsValid || chosenCaseIds.length === 0}
      >
        {busy ? "Starting…" : `Run ${runCount} evaluation runs`}
      </Button>
    </div>
  );
}

function EvaluationProgress({ refinement }: { refinement: Refinement }) {
  const progress = refinement.progress;
  // total can be 0 when every case was excluded or skipped: never divide by it.
  const text =
    progress && progress.total > 0
      ? `${progress.done} of ${progress.total} case × repetition pairs done`
      : "Preparing cases…";
  return (
    <p className="text-xs text-muted-foreground" role="status">
      Evaluating: {text}
    </p>
  );
}

function EvaluationResults({
  evaluation,
  decisionPending,
}: {
  evaluation: RefinementEvaluation;
  /** True while accept/reject is still open; only then is Reject "the default". */
  decisionPending: boolean;
}) {
  const { deltas } = evaluation;
  return (
    <div className="space-y-2">
      <h3 className="text-xs font-medium">
        Evaluation on {evaluation.setup.provider}/{evaluation.setup.model}, {evaluation.setup.reps}{" "}
        repetitions
      </h3>
      {!evaluation.complete && (
        <div className="rounded-md border border-amber-300 p-2 text-xs" role="alert">
          Incomplete, so it cannot be accepted:
          <ul className="ml-4 list-disc">
            {evaluation.incomplete_reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      )}
      {evaluation.regression && (
        <div
          className="rounded-md border border-destructive/60 bg-destructive/5 p-2 text-xs text-destructive"
          role="alert"
        >
          The candidate passed less often than the current supplement on{" "}
          {evaluation.regression_case_ids.join(", ")}.
          {decisionPending ? " Reject is the default." : ""}
        </div>
      )}
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Case</TableHead>
            <TableHead>Current</TableHead>
            <TableHead>Candidate</TableHead>
            <TableHead>Per repetition</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {evaluation.cases.map((c) => (
            <TableRow key={c.case_id}>
              <TableCell className="font-mono text-xs">
                {c.case_id}
                {c.regressed && (
                  <Badge variant="destructive" className="ml-1 text-xs">
                    regressed
                  </Badge>
                )}
              </TableCell>
              <TableCell className="text-xs">{formatRate(c.baseline_pass_rate)}</TableCell>
              <TableCell className="text-xs">{formatRate(c.candidate_pass_rate)}</TableCell>
              <TableCell className="text-xs">
                {c.reps
                  .map((r) => `${armMark(r.baseline)}/${armMark(r.candidate)}`)
                  .join("  ")}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {Object.keys(evaluation.setup.excluded_cases).length > 0 && (
        <p className="text-xs text-muted-foreground">
          Excluded (same document as a source run):{" "}
          {Object.keys(evaluation.setup.excluded_cases).join(", ")}
        </p>
      )}
      {deltas && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
          <dt className="text-muted-foreground">Accuracy</dt>
          <dd>
            {formatRate(deltas.accuracy_baseline)} → {formatRate(deltas.accuracy_candidate)}
          </dd>
          {deltas.judge_baseline && deltas.judge_candidate && (
            <>
              <dt className="text-muted-foreground">Judge (mean, min–max)</dt>
              <dd>
                {deltas.judge_baseline.mean.toFixed(2)} ({deltas.judge_baseline.min}–
                {deltas.judge_baseline.max}) → {deltas.judge_candidate.mean.toFixed(2)} (
                {deltas.judge_candidate.min}–{deltas.judge_candidate.max})
              </dd>
            </>
          )}
          <dt className="text-muted-foreground">Cost (USD)</dt>
          <dd>
            {deltas.cost_baseline.toFixed(4)} → {deltas.cost_candidate.toFixed(4)}
          </dd>
          <dt className="text-muted-foreground">Mean steps</dt>
          <dd>
            {deltas.steps_mean_baseline.toFixed(1)} → {deltas.steps_mean_candidate.toFixed(1)}
          </dd>
          <dt className="text-muted-foreground">Outcomes</dt>
          <dd>
            {formatOutcomes(deltas.outcomes_baseline)} → {formatOutcomes(deltas.outcomes_candidate)}
          </dd>
        </dl>
      )}
    </div>
  );
}

function armMark(arm: { passed: boolean | null; judge_overall: number | null } | null): string {
  if (arm === null) return "missing";
  if (arm.passed === null) return arm.judge_overall === null ? "—" : arm.judge_overall.toFixed(1);
  return arm.passed ? "pass" : "fail";
}

function formatOutcomes(outcomes: Record<string, number>): string {
  return Object.entries(outcomes)
    .map(([k, v]) => `${k} ${v}`)
    .join(", ");
}
