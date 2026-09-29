"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ModeConfig } from "@/lib/api";
import { Save, RotateCcw } from "lucide-react";
import {
  ALL_EXECUTION_MODES,
  DEFAULT_ENABLED_MODES,
  MODE_AUTO,
  MODE_DIRECT,
  MODE_RLM,
  MODE_RAG,
  MODE_RLM_OFFICIAL,
} from "@/lib/constants";

const DEFAULT_CONFIG: ModeConfig = {
  enabled_modes: [...DEFAULT_ENABLED_MODES],
  default_mode: MODE_AUTO,
  rag_config: {
    chunk_size: 1000,
    chunk_overlap: 200,
    top_k: 5,
    embedding_model: "text-embedding-3-small",
  },
  rlm_max_steps: 16,
  rlm_timeout_seconds: 600,
};

const EMBEDDING_MODELS = [
  "text-embedding-3-small",
  "text-embedding-3-large",
  "text-embedding-ada-002",
];

interface ExecutionConfigProps {
  config: ModeConfig;
  onChange: (config: ModeConfig) => void;
}

export function ExecutionConfig({ config, onChange }: ExecutionConfigProps) {
  const [local, setLocal] = useState<ModeConfig>(config);
  const [saving, setSaving] = useState(false);

  const ragEnabled = local.enabled_modes.includes(MODE_RAG);

  const toggleMode = (mode: string) => {
    const current = local.enabled_modes;
    if (current.includes(mode)) {
      if (current.length === 1) return; // Keep at least one
      setLocal({ ...local, enabled_modes: current.filter((m) => m !== mode) });
    } else {
      setLocal({ ...local, enabled_modes: [...current, mode] });
    }
  };

  const handleSave = () => {
    setSaving(true);
    onChange(local);
    setTimeout(() => setSaving(false), 300);
  };

  const handleReset = () => {
    setLocal(DEFAULT_CONFIG);
  };

  return (
    <div className="space-y-6">
      {/* Mode Selection */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Enabled Modes</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Select which execution modes are available for chat queries.
          </p>
          <div className="flex flex-col gap-3">
            {ALL_EXECUTION_MODES.map((mode) => (
              <div key={mode} className="flex items-center justify-between">
                <div>
                  <Label htmlFor={`mode-${mode}`} className="capitalize font-medium">
                    {mode}
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    {mode === MODE_DIRECT && "Send query directly to LLM"}
                    {mode === MODE_RLM && "Recursive Language Model with code execution"}
                    {mode === MODE_RAG && "Retrieval-Augmented Generation from uploaded files"}
                    {mode === MODE_RLM_OFFICIAL &&
                      "The paper authors' reference implementation (rlms); needs the interop extra"}
                  </p>
                </div>
                <Switch
                  id={`mode-${mode}`}
                  checked={local.enabled_modes.includes(mode)}
                  onCheckedChange={() => toggleMode(mode)}
                  aria-label={`Enable ${mode} mode`}
                />
              </div>
            ))}
          </div>

          <div className="pt-2">
            <Label>Default Mode</Label>
            <Select
              value={local.default_mode}
              onValueChange={(v) => setLocal({ ...local, default_mode: v })}
            >
              <SelectTrigger className="w-full mt-1" aria-label="Default mode">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={MODE_AUTO}>Auto (system decides)</SelectItem>
                {ALL_EXECUTION_MODES.map((mode) => (
                  <SelectItem key={mode} value={mode} className="capitalize">
                    {mode}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* RLM Settings */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">RLM Settings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Max Steps: {local.rlm_max_steps}</Label>
            <Slider
              value={[local.rlm_max_steps]}
              onValueChange={([v]) => setLocal({ ...local, rlm_max_steps: v })}
              min={1}
              max={64}
              step={1}
              aria-label="RLM max steps"
            />
            <p className="text-xs text-muted-foreground">
              Maximum reasoning-execution loop iterations
            </p>
          </div>

          <div className="space-y-2">
            <Label htmlFor="rlm-timeout">RLM Timeout (seconds)</Label>
            <Input
              id="rlm-timeout"
              type="number"
              value={local.rlm_timeout_seconds}
              onChange={(e) =>
                setLocal({ ...local, rlm_timeout_seconds: parseInt(e.target.value) || 600 })
              }
              min={10}
              max={3600}
              aria-label="RLM timeout in seconds"
            />
          </div>
        </CardContent>
      </Card>

      {/* RAG Settings */}
      <Card className={!ragEnabled ? "opacity-50" : undefined}>
        <CardHeader>
          <CardTitle className="text-base">
            RAG Settings
            {!ragEnabled && (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                (enable RAG mode above)
              </span>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>Chunk Size: {local.rag_config.chunk_size}</Label>
            <Slider
              value={[local.rag_config.chunk_size]}
              onValueChange={([v]) =>
                setLocal({ ...local, rag_config: { ...local.rag_config, chunk_size: v } })
              }
              min={200}
              max={4000}
              step={100}
              disabled={!ragEnabled}
              aria-label="Chunk size"
            />
          </div>

          <div className="space-y-2">
            <Label>Chunk Overlap: {local.rag_config.chunk_overlap}</Label>
            <Slider
              value={[local.rag_config.chunk_overlap]}
              onValueChange={([v]) =>
                setLocal({ ...local, rag_config: { ...local.rag_config, chunk_overlap: v } })
              }
              min={0}
              max={1000}
              step={50}
              disabled={!ragEnabled}
              aria-label="Chunk overlap"
            />
          </div>

          <div className="space-y-2">
            <Label>Top-K Results: {local.rag_config.top_k}</Label>
            <Slider
              value={[local.rag_config.top_k]}
              onValueChange={([v]) =>
                setLocal({ ...local, rag_config: { ...local.rag_config, top_k: v } })
              }
              min={1}
              max={20}
              step={1}
              disabled={!ragEnabled}
              aria-label="Top K results"
            />
          </div>

          <div className="space-y-2">
            <Label>Embedding Model</Label>
            <Select
              value={local.rag_config.embedding_model}
              onValueChange={(v) =>
                setLocal({ ...local, rag_config: { ...local.rag_config, embedding_model: v } })
              }
              disabled={!ragEnabled}
            >
              <SelectTrigger className="w-full" aria-label="Embedding model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {EMBEDDING_MODELS.map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Save/Reset */}
      <div className="flex gap-2">
        <Button onClick={handleSave} disabled={saving}>
          <Save className="mr-1 h-4 w-4" aria-hidden="true" />
          {saving ? "Saving..." : "Save"}
        </Button>
        <Button variant="outline" onClick={handleReset}>
          <RotateCcw className="mr-1 h-4 w-4" aria-hidden="true" />
          Reset to Defaults
        </Button>
      </div>
    </div>
  );
}
