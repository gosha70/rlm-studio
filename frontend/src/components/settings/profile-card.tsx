"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
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
  type RunProfile,
  type ChatProviderConfig,
  type SystemPromptTemplate,
} from "@/lib/api";
import useSWR from "swr";
import { toast } from "sonner";
import { Trash2, Lock, Edit2, Copy, Download, ChevronDown, ChevronUp } from "lucide-react";
import { displayModeName } from "@/lib/constants";
import { StrategySelectItems } from "./strategy-select-items";

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
    nudge_at_fraction: String(profile.budget.nudge_at_fraction ?? 0.6),
    prompt_template_name: profile.prompt_template_name as string | null,
  });

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
    } catch {
      setMessage("Failed to delete profile");
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
        prompt_template_name: profile.prompt_template_name,
      });
      onCloned?.();
    } catch {
      setMessage("Failed to clone profile");
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
      nudge_at_fraction: String(profile.budget.nudge_at_fraction ?? 0.6),
      prompt_template_name: profile.prompt_template_name,
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

  const handleSaveEdit = async () => {
    // Strict parsers that reject partial matches like "5s", "1.5foo", "1e3".
    const strictInt = (s: string): number => {
      const trimmed = s.trim();
      return /^-?\d+$/.test(trimmed) ? parseInt(trimmed, 10) : NaN;
    };
    const strictFloat = (s: string): number => {
      const trimmed = s.trim();
      return /^-?\d+(\.\d+)?$/.test(trimmed) ? parseFloat(trimmed) : NaN;
    };

    // Parse all string fields back to numbers and validate before saving.
    const temperature = strictFloat(editData.temperature);
    const topP = strictFloat(editData.top_p);
    const maxTokens = strictInt(editData.max_output_tokens);
    const timeout = strictInt(editData.timeout_seconds);
    const maxSteps = strictInt(editData.max_steps);
    const maxTokensBudget = strictInt(editData.max_tokens);
    const maxCost = strictFloat(editData.max_cost_usd);
    const maxTime = strictInt(editData.max_time_seconds);
    const maxRecursion = strictInt(editData.max_recursion_depth);
    const repeatLimit = strictInt(editData.repeat_limit);
    const nudgeFraction = strictFloat(editData.nudge_at_fraction);

    if (isNaN(temperature) || temperature < 0 || temperature > 2) {
      setMessage("Temperature must be between 0 and 2");
      return;
    }
    if (isNaN(topP) || topP < 0 || topP > 1) {
      setMessage("Top P must be between 0 and 1");
      return;
    }
    if (isNaN(maxTokens) || maxTokens < 1) {
      setMessage("Max Tokens must be at least 1");
      return;
    }
    if (isNaN(timeout) || timeout < MIN_TIMEOUT_SECONDS) {
      setMessage(`Timeout must be at least ${MIN_TIMEOUT_SECONDS} seconds`);
      return;
    }
    if (isNaN(maxSteps) || maxSteps < 1) {
      setMessage("Max Steps must be at least 1");
      return;
    }
    if (isNaN(maxCost) || maxCost < 0) {
      setMessage("Max Cost must be 0 or greater");
      return;
    }
    if (isNaN(repeatLimit) || repeatLimit < 1) {
      setMessage("Repeat Limit must be at least 1");
      return;
    }
    if (isNaN(nudgeFraction) || nudgeFraction < 0.1 || nudgeFraction > 1) {
      setMessage("Nudge at Fraction must be between 0.1 and 1.0");
      return;
    }
    if (isNaN(maxTime) || maxTime < 10) {
      setMessage("Max Execution Time must be at least 10 seconds");
      return;
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
          max_tokens: isNaN(maxTokensBudget) ? 0 : maxTokensBudget,
          max_cost_usd: maxCost,
          max_time_seconds: maxTime,
          max_recursion_depth: isNaN(maxRecursion) ? 0 : maxRecursion,
          repeat_limit: repeatLimit,
          nudge_at_fraction: nudgeFraction,
        },
        prompt_template_name: editData.prompt_template_name,
        system_prompts: {},
      });
      toast.success("Profile saved");
      onUpdated?.();
    } catch {
      setMessage("Failed to update profile");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Collapsible open={expanded} onOpenChange={setExpanded}>
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CardTitle className="text-sm">{profile.name}</CardTitle>
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
          <div className="flex gap-1">
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
          <p className="text-xs text-muted-foreground mt-2">
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
            <span>{profile.budget.nudge_at_fraction ?? 0.6}</span>
            <span className="text-muted-foreground">Prompt Group</span>
            <span>{profile.prompt_template_name ?? "Global defaults"}</span>
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
                onChange={(e) => setEditData({ ...editData, name: e.target.value })}
              />
            </div>
            <div className="space-y-1">
              <Label htmlFor={`edit-strategy-${profile.id}`} className="text-xs">Strategy</Label>
              <Select
                value={editData.strategy}
                onValueChange={(v) => setEditData({ ...editData, strategy: v })}
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
                onChange={(e) => setEditData({ ...editData, description: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, temperature: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, top_p: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, max_output_tokens: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, timeout_seconds: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, max_steps: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, max_time_seconds: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, max_cost_usd: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, repeat_limit: e.target.value })}
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
                  onChange={(e) => setEditData({ ...editData, nudge_at_fraction: e.target.value })}
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
                  setEditData({
                    ...editData,
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

        {message && (
          <p className="text-xs text-destructive mt-2" role="status">
            {message}
          </p>
        )}
        </CollapsibleContent>
      </CardContent>
    </Card>
    </Collapsible>
  );
}
