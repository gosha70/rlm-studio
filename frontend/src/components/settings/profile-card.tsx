"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { FieldLabel } from "./field-label";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  createProfile,
  deleteProfile,
  updateProfile,
  getPromptTemplates,
  apiErrorMessage,
  type RunProfile,
  type ChatProviderConfig,
  type SystemPromptTemplate,
} from "@/lib/api";
import useSWR from "swr";
import { toast } from "sonner";
import { Trash2, Lock, Edit2, Copy, Download, ChevronDown, ChevronUp, FlaskConical } from "lucide-react";
import {
  MAX_OUTPUT_TOKENS_LIMIT,
  MODE_RLM,
  RLM_DEFAULT_NUDGE_AT_FRACTION,
  displayModeName,
} from "@/lib/constants";
import { StrategySelectItems } from "./strategy-select-items";
import {
  BUDGET_MAX_COST_USD,
  BUDGET_MAX_RECURSION_DEPTH,
  BUDGET_MAX_REPEAT_LIMIT,
  BUDGET_MAX_STEPS,
  BUDGET_MAX_TIME_SECONDS,
  BUDGET_MAX_TOKENS,
  BUDGET_MIN_COST_USD,
  BUDGET_MIN_TOKENS,
  RUNTIME_MAX_TIMEOUT_SECONDS,
  decimalProblem,
  wholeNumberProblem,
} from "./input-limits";
import { RefinementPanel } from "./refinement-panel";

interface ProfileCardProps {
  profile: RunProfile;
  chatProviders?: ChatProviderConfig[];
  onDeleted?: () => void;
  onUpdated?: () => void;
  onCloned?: () => void;
}

export function ProfileCard({
  profile,
  chatProviders = [],
  onDeleted,
  onUpdated,
  onCloned,
}: ProfileCardProps) {
  const [deleting, setDeleting] = useState(false);
  const [cloning, setCloning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [editing, setEditing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [refining, setRefining] = useState(false);
  const { data: templates = [] } = useSWR<SystemPromptTemplate[]>("prompt-templates", getPromptTemplates);

  const [editData, setEditData] = useState({
    name: profile.name,
    strategy: profile.strategy,
    description: profile.description,
    // Numeric fields stored as strings during editing so users can clear
    // and retype freely.  Parsed back to numbers on save.
    temperature: String(profile.runtime_settings.temperature),
    top_p: String(profile.runtime_settings.top_p),
    max_output_tokens: String(profile.runtime_settings.max_output_tokens),
    timeout_seconds: String(profile.runtime_settings.timeout_seconds),
    max_steps: String(profile.budget.max_steps),
    max_tokens: String(profile.budget.max_tokens),
    max_cost_usd: String(profile.budget.max_cost_usd),
    max_time_seconds: String(profile.budget.max_time_seconds),
    max_recursion_depth: String(profile.budget.max_recursion_depth),
    repeat_limit: String(profile.budget.repeat_limit ?? 2),
    nudge_at_fraction: String(profile.budget.nudge_at_fraction ?? RLM_DEFAULT_NUDGE_AT_FRACTION),
    prompt_template_name: profile.prompt_template_name as string | null,
    rlm_prompt: profile.system_prompts?.[MODE_RLM] ?? "",
  });

  // Editing a field clears the last Save message: it described the old values.
  const updateEditData = (patch: Partial<typeof editData>) => {
    setMessage(null);
    setEditData((prev) => ({ ...prev, ...patch }));
  };

  const usedBy = chatProviders.filter((cp) => cp.profile_id === profile.id);

  const handleDelete = async () => {
    if (usedBy.length > 0) {
      const names = usedBy.map((cp) => cp.name).join(", ");
      setMessage(`Cannot delete: used by ${names}`);
      return;
    }
    if (!confirm(`Delete profile "${profile.name}"?`)) return;
    setDeleting(true);
    try {
      await deleteProfile(profile.id);
      onDeleted?.();
    } catch (err) {
      setMessage(`Profile not deleted: ${apiErrorMessage(err)}`);
      setDeleting(false);
    }
  };

  const handleClone = async () => {
    setCloning(true);
    setMessage(null);
    try {
      await createProfile({
        name: `Copy of ${profile.name}`,
        description: profile.description,
        strategy: profile.strategy,
        runtime_settings: { ...profile.runtime_settings },
        budget: { ...profile.budget },
        system_prompts: { ...(profile.system_prompts ?? {}) },
        prompt_template_name: profile.prompt_template_name,
      });
      onCloned?.();
    } catch (err) {
      setMessage(`Failed to clone profile: ${apiErrorMessage(err)}`);
    } finally {
      setCloning(false);
    }
  };

  const handleStartEdit = () => {
    setEditData({
      name: profile.name,
      strategy: profile.strategy,
      description: profile.description,
      temperature: String(profile.runtime_settings.temperature),
      top_p: String(profile.runtime_settings.top_p),
      max_output_tokens: String(profile.runtime_settings.max_output_tokens),
      timeout_seconds: String(profile.runtime_settings.timeout_seconds),
      max_steps: String(profile.budget.max_steps),
      max_tokens: String(profile.budget.max_tokens),
      max_cost_usd: String(profile.budget.max_cost_usd),
      max_time_seconds: String(profile.budget.max_time_seconds),
      max_recursion_depth: String(profile.budget.max_recursion_depth),
      repeat_limit: String(profile.budget.repeat_limit ?? 2),
      nudge_at_fraction: String(profile.budget.nudge_at_fraction ?? RLM_DEFAULT_NUDGE_AT_FRACTION),
      prompt_template_name: profile.prompt_template_name,
      rlm_prompt: profile.system_prompts?.[MODE_RLM] ?? "",
    });
    setEditing(true);
    setExpanded(true);
    setMessage(null);
  };

  const handleExport = () => {
    const exportData = {
      name: profile.name,
      description: profile.description,
      strategy: profile.strategy,
      runtime_settings: profile.runtime_settings,
      budget: profile.budget,
      system_prompts: profile.system_prompts ?? {},
      prompt_template_name: profile.prompt_template_name,
    };
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `profile-${profile.name.toLowerCase().replace(/\s+/g, "-")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const MIN_TIMEOUT_SECONDS = 5;
  const MIN_EXECUTION_TIME_SECONDS = 10;

  const handleSaveEdit = async () => {

    // Validate every edited field before saving; the first problem is shown.
    const problem =
      decimalProblem("Temperature", editData.temperature, 0, 2) ??
      decimalProblem("Top P", editData.top_p, 0, 1) ??
      wholeNumberProblem("Max Tokens", editData.max_output_tokens, 1, MAX_OUTPUT_TOKENS_LIMIT) ??
      wholeNumberProblem(
        "Timeout (s)",
        editData.timeout_seconds,
        MIN_TIMEOUT_SECONDS,
        RUNTIME_MAX_TIMEOUT_SECONDS,
      ) ??
      wholeNumberProblem("Max Steps", editData.max_steps, 1, BUDGET_MAX_STEPS) ??
      decimalProblem(
        "Max Cost ($)",
        editData.max_cost_usd,
        BUDGET_MIN_COST_USD,
        BUDGET_MAX_COST_USD,
      ) ??
      wholeNumberProblem("Repeat Limit", editData.repeat_limit, 1, BUDGET_MAX_REPEAT_LIMIT) ??
      decimalProblem("Nudge at Fraction", editData.nudge_at_fraction, 0.1, 1) ??
      wholeNumberProblem(
        "Max Execution Time (s)",
        editData.max_time_seconds,
        MIN_EXECUTION_TIME_SECONDS,
        BUDGET_MAX_TIME_SECONDS,
      ) ??
      // Editable since an older profile can hold a value the API now refuses;
      // a hidden field would block every save with no way to fix it.
      wholeNumberProblem(
        "Max Tokens (budget)",
        editData.max_tokens,
        BUDGET_MIN_TOKENS,
        BUDGET_MAX_TOKENS,
      ) ??
      wholeNumberProblem(
        "Max Recursion Depth",
        editData.max_recursion_depth,
        0,
        BUDGET_MAX_RECURSION_DEPTH,
      );
    if (problem) {
      setMessage(problem);
      return;
    }

    const temperature = Number(editData.temperature.trim());
    const topP = Number(editData.top_p.trim());
    const maxTokens = Number(editData.max_output_tokens.trim());
    const timeout = Number(editData.timeout_seconds.trim());
    const maxSteps = Number(editData.max_steps.trim());
    const maxCost = Number(editData.max_cost_usd.trim());
    const repeatLimit = Number(editData.repeat_limit.trim());
    const nudgeFraction = Number(editData.nudge_at_fraction.trim());
    const maxTime = Number(editData.max_time_seconds.trim());

    const maxTokensBudget = Number(editData.max_tokens.trim());
    const maxRecursion = Number(editData.max_recursion_depth.trim());

    // Keep every stored per-mode prompt; only the RLM entry is editable here.
    // An empty RLM field removes the key so the profile follows its prompt group.
    const systemPrompts: Record<string, string> = { ...(profile.system_prompts ?? {}) };
    if (editData.rlm_prompt.trim()) {
      systemPrompts[MODE_RLM] = editData.rlm_prompt;
    } else {
      delete systemPrompts[MODE_RLM];
    }

    setSaving(true);
    setMessage(null);
    try {
      await updateProfile(profile.id, {
        name: editData.name.trim() || profile.name,
        strategy: editData.strategy,
        description: editData.description,
        runtime_settings: {
          temperature,
          top_p: topP,
          max_output_tokens: maxTokens,
          timeout_seconds: timeout,
        },
        budget: {
          max_steps: maxSteps,
          max_tokens: maxTokensBudget,
          max_cost_usd: maxCost,
          max_time_seconds: maxTime,
          max_recursion_depth: maxRecursion,
          repeat_limit: repeatLimit,
          nudge_at_fraction: nudgeFraction,
        },
        prompt_template_name: editData.prompt_template_name,
        system_prompts: systemPrompts,
      });
      toast.success("Profile saved");
      onUpdated?.();
    } catch (err) {
      setMessage(`Failed to update profile: ${apiErrorMessage(err)}`);
    } finally {
      setSaving(false);
    }
  };

  return (
    <Collapsible open={expanded} onOpenChange={setExpanded}>
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2">
            <CardTitle className="min-w-0 text-sm wrap-anywhere">{profile.name}</CardTitle>
            {profile.is_builtin && (
              <Badge variant="secondary" className="text-xs">
                <Lock className="mr-1 h-3 w-3" aria-hidden="true" />
                Built-in
              </Badge>
            )}
            <Badge variant="outline" className="text-xs">
              {displayModeName(profile.strategy)}
            </Badge>
          </div>
          <div className="flex shrink-0 gap-1">
            {!profile.is_builtin && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setRefining(true)}
                aria-label={`Refine ${profile.name} RLM prompt`}
              >
                <FlaskConical className="h-4 w-4" aria-hidden="true" />
              </Button>
            )}
            {!profile.is_builtin && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleStartEdit}
                disabled={editing}
                aria-label={`Edit ${profile.name} profile`}
              >
                <Edit2 className="h-4 w-4" aria-hidden="true" />
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleExport}
              aria-label={`Export ${profile.name} profile`}
            >
              <Download className="h-4 w-4" aria-hidden="true" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClone}
              disabled={cloning}
              aria-label={`Clone ${profile.name} profile`}
            >
              <Copy className="h-4 w-4" aria-hidden="true" />
            </Button>
            {!profile.is_builtin && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDelete}
                disabled={deleting}
                aria-label={`Delete ${profile.name} profile`}
                className="text-destructive hover:text-destructive"
              >
                <Trash2 className="h-4 w-4" aria-hidden="true" />
              </Button>
            )}
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                size="sm"
                aria-label={expanded ? "Collapse profile" : "Expand profile"}
              >
                {expanded
                  ? <ChevronUp className="h-4 w-4" aria-hidden="true" />
                  : <ChevronDown className="h-4 w-4" aria-hidden="true" />}
              </Button>
            </CollapsibleTrigger>
          </div>
        </div>
      </CardHeader>
      <CardContent className="pb-3">
        {profile.description && (
          <p className="text-xs text-muted-foreground mb-2">{profile.description}</p>
        )}
        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
          <span>Temp: {profile.runtime_settings.temperature}</span>
          <span>Max tokens: {profile.runtime_settings.max_output_tokens}</span>
          <span>Steps: {profile.budget.max_steps}</span>
          <span>Prompts: {profile.prompt_template_name ?? "Global defaults"}</span>
        </div>

        {usedBy.length > 0 ? (
          <p className="text-xs text-muted-foreground mt-2 wrap-anywhere">
            Used by: {usedBy.map((cp) => cp.name).join(", ")}
          </p>
        ) : (
          <p className="text-xs text-muted-foreground/60 mt-2">
            Not used by any Chat Provider
          </p>
        )}

        <CollapsibleContent>
        {!editing && (
          <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1 rounded-lg border border-muted p-3 text-xs">
            <span className="text-muted-foreground">Temperature</span>
            <span>{profile.runtime_settings.temperature}</span>
            <span className="text-muted-foreground">Top P</span>
            <span>{profile.runtime_settings.top_p}</span>
            <span className="text-muted-foreground">Max Output Tokens</span>
            <span>{profile.runtime_settings.max_output_tokens}</span>
            <span className="text-muted-foreground">Timeout (s)</span>
            <span>{profile.runtime_settings.timeout_seconds}</span>
            <span className="text-muted-foreground">Max Steps</span>
            <span>{profile.budget.max_steps}</span>
            <span className="text-muted-foreground">Max Tokens (budget)</span>
            <span>{profile.budget.max_tokens}</span>
            <span className="text-muted-foreground">Max Cost ($)</span>
            <span>{profile.budget.max_cost_usd}</span>
            <span className="text-muted-foreground">Max Time (s)</span>
            <span>{profile.budget.max_time_seconds}</span>
            <span className="text-muted-foreground">Max Recursion Depth</span>
            <span>{profile.budget.max_recursion_depth}</span>
            <span className="text-muted-foreground">Repeat Limit</span>
            <span>{profile.budget.repeat_limit ?? 2}</span>
            <span className="text-muted-foreground">Nudge at Fraction</span>
            <span>{profile.budget.nudge_at_fraction ?? RLM_DEFAULT_NUDGE_AT_FRACTION}</span>
            <span className="text-muted-foreground">Prompt Group</span>
            <span>{profile.prompt_template_name ?? "Global defaults"}</span>
            <span className="text-muted-foreground">Custom RLM prompt</span>
            <span>{profile.system_prompts?.[MODE_RLM] ? `${profile.system_prompts[MODE_RLM].length} chars` : "None"}</span>
          </div>
        )}

        {editing && (
          <div className="mt-4 space-y-3 rounded-lg border border-muted p-3">
            <div className="space-y-1">
              <Label htmlFor={`edit-name-${profile.id}`} className="text-xs">Name</Label>
              <Input
                id={`edit-name-${profile.id}`}
                className="h-8 text-xs"
                value={editData.name}
                onChange={(e) => updateEditData({ name: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor={`edit-strategy-${profile.id}`} className="text-xs">Strategy</Label>
              <Select
                value={editData.strategy}
                onValueChange={(v) => updateEditData({ strategy: v })}
              >
                <SelectTrigger id={`edit-strategy-${profile.id}`} className="h-8 text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <StrategySelectItems />
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <Label htmlFor={`edit-desc-${profile.id}`} className="text-xs">Description</Label>
              <Input
                id={`edit-desc-${profile.id}`}
                className="h-8 text-xs"
                value={editData.description}
                onChange={(e) => updateEditData({ description: e.target.value })}
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-temp-${profile.id}`} className="text-xs" tooltip="Controls randomness. Lower values (0.1-0.3) give focused, deterministic outputs; higher values (0.7-1.0) increase creativity.">Temperature</FieldLabel>
                <Input
                  id={`edit-temp-${profile.id}`}
                  type="number"
                  step="0.1"
                  min="0"
                  max="2"
                  className="h-8 text-xs"
                  value={editData.temperature}
                  onChange={(e) => updateEditData({ temperature: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-topp-${profile.id}`} className="text-xs" tooltip="Nucleus sampling: only consider tokens whose cumulative probability reaches this threshold. 1.0 = no filtering.">Top P</FieldLabel>
                <Input
                  id={`edit-topp-${profile.id}`}
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  className="h-8 text-xs"
                  value={editData.top_p}
                  onChange={(e) => updateEditData({ top_p: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-maxtokens-${profile.id}`} className="text-xs" tooltip="Maximum output tokens per LLM call (not the total budget). Controls response length per step.">Max Tokens</FieldLabel>
                <Input
                  id={`edit-maxtokens-${profile.id}`}
                  type="text"
                  inputMode="numeric"
                  className="h-8 text-xs"
                  value={editData.max_output_tokens}
                  onChange={(e) => updateEditData({ max_output_tokens: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-timeout-${profile.id}`} className="text-xs" tooltip="Per-step timeout in seconds. If a single LLM call exceeds this, it is aborted. Minimum 5 seconds.">Timeout (s)</FieldLabel>
                <Input
                  id={`edit-timeout-${profile.id}`}
                  type="text"
                  inputMode="numeric"
                  className="h-8 text-xs"
                  value={editData.timeout_seconds}
                  onChange={(e) => updateEditData({ timeout_seconds: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-steps-${profile.id}`} className="text-xs" tooltip="Maximum reasoning steps in RLM mode. More steps allow deeper document exploration but cost more tokens.">Max Steps</FieldLabel>
                <Input
                  id={`edit-steps-${profile.id}`}
                  type="text"
                  inputMode="numeric"
                  className="h-8 text-xs"
                  value={editData.max_steps}
                  onChange={(e) => updateEditData({ max_steps: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-max-time-${profile.id}`} className="text-xs" tooltip="Total wall-clock time limit for the entire RLM execution (all steps combined). Not the per-step timeout — that is 'Timeout (s)' above. Set high (600+) for slow local models.">Max Execution Time (s)</FieldLabel>
                <Input
                  id={`edit-max-time-${profile.id}`}
                  type="text"
                  inputMode="numeric"
                  className="h-8 text-xs"
                  value={editData.max_time_seconds}
                  onChange={(e) => updateEditData({ max_time_seconds: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-max-tokens-budget-${profile.id}`} className="text-xs" tooltip="Total tokens one query may use across all steps. The run stops once it is reached.">Max Tokens (budget)</FieldLabel>
                <Input
                  id={`edit-max-tokens-budget-${profile.id}`}
                  type="text"
                  inputMode="numeric"
                  className="h-8 text-xs"
                  value={editData.max_tokens}
                  onChange={(e) => updateEditData({ max_tokens: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-max-recursion-${profile.id}`} className="text-xs" tooltip="How many levels of sub-RLM calls a run may nest. 0 allows none.">Max Recursion Depth</FieldLabel>
                <Input
                  id={`edit-max-recursion-${profile.id}`}
                  type="text"
                  inputMode="numeric"
                  className="h-8 text-xs"
                  value={editData.max_recursion_depth}
                  onChange={(e) => updateEditData({ max_recursion_depth: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-cost-${profile.id}`} className="text-xs" tooltip="Dollar limit per query. The execution is stopped if this cost is exceeded.">Max Cost ($)</FieldLabel>
                <Input
                  id={`edit-cost-${profile.id}`}
                  type="text"
                  inputMode="decimal"
                  className="h-8 text-xs"
                  value={editData.max_cost_usd}
                  onChange={(e) => updateEditData({ max_cost_usd: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-repeat-limit-${profile.id}`} className="text-xs" tooltip="RLM convergence: how many duplicate inspect actions before forcing finalization. Lower = faster convergence, higher = more retries.">Repeat Limit</FieldLabel>
                <Input
                  id={`edit-repeat-limit-${profile.id}`}
                  type="text"
                  inputMode="numeric"
                  className="h-8 text-xs"
                  value={editData.repeat_limit}
                  onChange={(e) => updateEditData({ repeat_limit: e.target.value })}
                />
              </div>
              <div className="space-y-1">
                <FieldLabel htmlFor={`edit-nudge-fraction-${profile.id}`} className="text-xs" tooltip="RLM convergence: fraction of max_steps at which a soft nudge is sent to encourage finalization. E.g., 0.5 = nudge at step 16 of 32.">Nudge at Fraction</FieldLabel>
                <Input
                  id={`edit-nudge-fraction-${profile.id}`}
                  type="number"
                  step="0.05"
                  min="0.1"
                  max="1.0"
                  className="h-8 text-xs"
                  value={editData.nudge_at_fraction}
                  onChange={(e) => updateEditData({ nudge_at_fraction: e.target.value })}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label className="text-xs font-medium">Prompt Group</Label>
              <p className="text-xs text-muted-foreground">
                Choose a prompt group. Updates to the group in the Prompts tab will
                automatically apply to this profile.
              </p>
              <Select
                value={editData.prompt_template_name ?? "__global__"}
                onValueChange={(value) => {
                  updateEditData({
                    prompt_template_name: value === "__global__" ? null : value,
                  });
                }}
              >
                <SelectTrigger className="h-8 text-xs" id={`edit-prompt-template-${profile.id}`}>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="__global__">
                    <span>Use global defaults</span>
                    <span className="ml-2 text-muted-foreground">— inherits from Prompts tab</span>
                  </SelectItem>
                  {templates.map((t) => (
                    <SelectItem key={t.name} value={t.name} className="text-xs">
                      <span>{t.name}</span>
                      {t.description && (
                        <span className="ml-2 text-muted-foreground">— {t.description}</span>
                      )}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1">
              <FieldLabel
                htmlFor={`edit-rlm-prompt-${profile.id}`}
                className="text-xs"
                tooltip="Supplement appended to the built-in RLM system prompt for this profile. When set, it replaces the prompt group's RLM text; leave it empty to follow the prompt group."
              >
                Custom RLM prompt
              </FieldLabel>
              <Textarea
                id={`edit-rlm-prompt-${profile.id}`}
                className="min-h-24 text-xs font-mono"
                value={editData.rlm_prompt}
                onChange={(e) => updateEditData({ rlm_prompt: e.target.value })}
              />
            </div>
            <div className="flex gap-2">
              <Button size="sm" onClick={handleSaveEdit} disabled={saving}>
                {saving ? "Saving..." : "Save"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setEditing(false)}
                disabled={saving}
              >
                Cancel
              </Button>
            </div>
          </div>
        )}

        </CollapsibleContent>
        {/* Outside the collapsible part: a refused delete must be visible
            on a collapsed card, which is where the Delete button is. */}
        {message && (
          <p className="text-xs text-destructive mt-2" role="status">
            {message}
          </p>
        )}
      </CardContent>
    </Card>
    {refining && (
      <RefinementPanel
        profile={profile}
        chatProviders={chatProviders}
        open={refining}
        onOpenChange={setRefining}
        onProfileChanged={() => {
          // An accept or rollback rewrote the RLM prompt: drop any open edit
          // form, whose save would send the prompt captured before it.
          setEditing(false);
          onUpdated?.();
        }}
      />
    )}
    </Collapsible>
  );
}
