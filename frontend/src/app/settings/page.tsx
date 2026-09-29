"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import useSWR from "swr";
import { useTheme } from "next-themes";
import { AppShell } from "@/components/shared/app-shell";
import { Badge } from "@/components/ui/badge";
import { BudgetConfig } from "@/components/settings/budget-config";
import { ProfileCard } from "@/components/settings/profile-card";
import { SystemPromptEditor } from "@/components/settings/system-prompt-editor";
import { FieldLabel } from "@/components/settings/field-label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
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
  getConfig,
  getProfiles,
  createProfile,
  updateConfig,
  getChatProviders,
  createChatProvider,
  updateChatProvider,
  deleteChatProvider,
  getLLMProviders,
  createLLMProvider,
  updateLLMProvider,
  deleteLLMProvider,
  testLLMProvider,
  testLLMProviderConfig,
  getProviderModels,
  type AppConfig,
  type RunProfile,
  type ChatProviderConfig,
  type ChatProviderCreateRequest,
  type RAGConfig,
  type LLMProviderConfig,
  type LLMProviderCreateRequest,
  type ModelInfo,
} from "@/lib/api";
import { Plus, Edit2, Trash2, Upload, Check, X, Wifi, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import {
  DeepLinkBanner,
  parseDeepLinkFromParams,
  type DeepLinkValues,
} from "@/components/settings/deep-link-banner";
import {
  BACKEND_OPENAI,
  BACKEND_ANTHROPIC,
  BACKEND_OLLAMA,
  BACKEND_LMSTUDIO,
  BACKEND_VLLM,
  CLOUD_BACKENDS,
  LOCAL_BACKENDS,
  BACKEND_DISPLAY_NAMES,
  MODEL_PLACEHOLDER,
  ENDPOINT_PLACEHOLDER,
  MODE_DIRECT,
  MODE_RLM,
  MODE_RAG,
} from "@/lib/constants";
import { StrategySelectItems } from "@/components/settings/strategy-select-items";

// Backend keys the Settings form accepts. The deep-link banner only
// renders when `?provider=` matches one of these, so a random or
// malicious query param cannot trigger the banner UI.
const DEEP_LINK_ALLOWED_BACKENDS: ReadonlySet<string> = new Set([
  BACKEND_OPENAI,
  BACKEND_ANTHROPIC,
  BACKEND_OLLAMA,
  BACKEND_LMSTUDIO,
  BACKEND_VLLM,
]);

// Backends for which a deep-link may carry a baseUrl. Cloud backends
// are excluded because the Settings form does not expose an endpoint
// field for them — a crafted ?provider=openai&baseUrl=… would
// otherwise stash the override in hidden form state and route the
// user's API key to the attacker's host when they save. See the
// pitch's Deep-link security rabbit hole.
const DEEP_LINK_BASE_URL_ALLOWED_BACKENDS: ReadonlySet<string> = new Set([
  BACKEND_OLLAMA,
  BACKEND_LMSTUDIO,
  BACKEND_VLLM,
]);

/**
 * Default export wraps the real page in a Suspense boundary because
 * the inner component calls ``useSearchParams()`` for the Cookbook
 * deep-link banner. Next's static generation would otherwise fail
 * the CSR-bailout check during ``next build``. The fallback is
 * minimal — the Suspense window is only open until router params
 * are available, which is a single tick on the client.
 */
export default function SettingsPage() {
  return (
    <Suspense fallback={null}>
      <SettingsPageInner />
    </Suspense>
  );
}

function SettingsPageInner() {
  const { theme, setTheme } = useTheme();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [saving, setSaving] = useState(false);
  const [newProfileName, setNewProfileName] = useState("");
  const [newProfileStrategy, setNewProfileStrategy] = useState<string>(MODE_DIRECT);
  const [newProfileDescription, setNewProfileDescription] = useState("");
  const [creatingProfile, setCreatingProfile] = useState(false);
  const [profileError, setProfileError] = useState<string | null>(null);
  const importFileRef = useRef<HTMLInputElement>(null);

  // Chat Providers state
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [formData, setFormData] = useState<{
    name: string;
    llm_provider_id: string;
    profile_id: string;
    rag_config: RAGConfig;
  }>({
    name: "",
    llm_provider_id: "",
    profile_id: "",
    rag_config: {
      chunk_size: 512,
      chunk_overlap: 64,
      top_k: 5,
      embedding_model: "text-embedding-3-small",
    },
  });
  const [savingProvider, setSavingProvider] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // LLM Providers state
  // Sentinel key in `testResults` that holds the form-level Test
  // Connection result (distinct from per-row saved-provider results).
  // Referenced by the invalidation effect, the save path (signal for
  // post-save status stamp), and the form banner renderer.
  const FORM_TEST_RESULT_KEY = "__form__";
  const [showLLMProviderForm, setShowLLMProviderForm] = useState(false);
  const [editingLLMProviderId, setEditingLLMProviderId] = useState<string | null>(null);
  const [llmProviderForm, setLLMProviderForm] = useState<{
    name: string;
    backend: string;
    model: string;
    api_key: string;
    endpoint: string;
    context_window: string;  // stored as string for input; parsed to int on save
  }>({
    name: "",
    backend: BACKEND_OPENAI,
    model: "",
    api_key: "",
    endpoint: "",
    context_window: "",
  });
  const [savingLLMProvider, setSavingLLMProvider] = useState(false);
  const [llmProviderModels, setLLMProviderModels] = useState<ModelInfo[]>([]);
  const [loadingLLMModels, setLoadingLLMModels] = useState(false);
  const llmModelFetchSeqRef = useRef(0);
  const [deletingLLMProviderId, setDeletingLLMProviderId] = useState<string | null>(null);
  const [testingLLMProviderId, setTestingLLMProviderId] = useState<string | null>(null);
  const [testResults, setTestResults] = useState<Record<string, { connected: boolean; error?: string | null; latency_ms?: number | null }>>({});

  // Cookbook → Settings deep-link state. Parsing lives in a pure
  // helper so the security-critical validation (allowed providers,
  // cloud-baseUrl rejection, api_key never read) is unit-testable
  // without mounting the whole Settings page.
  const deepLinkValues: DeepLinkValues | null = useMemo(
    () =>
      parseDeepLinkFromParams(
        searchParams,
        DEEP_LINK_ALLOWED_BACKENDS,
        DEEP_LINK_BASE_URL_ALLOWED_BACKENDS,
      ),
    [searchParams],
  );

  const clearDeepLink = () => router.replace("/settings");

  const applyDeepLink = () => {
    if (!deepLinkValues) return;
    // Populate the LLM Provider creation form. API key stays blank —
    // the deep link never carries secrets, and the confirmation
    // banner's reassurance line codifies that contract.
    setLLMProviderForm({
      name: "",
      backend: deepLinkValues.provider,
      model: deepLinkValues.model ?? "",
      api_key: "",
      endpoint: deepLinkValues.baseUrl ?? "",
      context_window: "",
    });
    setEditingLLMProviderId(null);
    setShowLLMProviderForm(true);
    clearDeepLink();
  };

  // Fetch available models when backend or endpoint changes in the LLM Provider form.
  // Uses a sequence counter to discard stale responses from concurrent fetches.
  const fetchLLMProviderModels = async (backend: string, ep?: string) => {
    const seq = ++llmModelFetchSeqRef.current;
    setLLMProviderModels([]);
    setLoadingLLMModels(true);
    try {
      const models = await getProviderModels(backend, ep || undefined);
      if (seq === llmModelFetchSeqRef.current) setLLMProviderModels(models);
    } catch {
      if (seq === llmModelFetchSeqRef.current) setLLMProviderModels([]);
    } finally {
      if (seq === llmModelFetchSeqRef.current) setLoadingLLMModels(false);
    }
  };

  useEffect(() => {
    if (showLLMProviderForm && llmProviderForm.backend) {
      const ep = (LOCAL_BACKENDS as readonly string[]).includes(llmProviderForm.backend) ? llmProviderForm.endpoint : undefined;
      fetchLLMProviderModels(llmProviderForm.backend, ep);
    }
  }, [showLLMProviderForm, llmProviderForm.backend, llmProviderForm.endpoint]);

  const { data: config, mutate: mutateConfig } = useSWR<AppConfig>("config", getConfig);
  const { data: profiles = [], mutate: mutateProfiles } = useSWR<RunProfile[]>("profiles", getProfiles);
  const { data: chatProviders = [], mutate: mutateChatProviders } = useSWR<ChatProviderConfig[]>(
    "chat-providers",
    getChatProviders
  );
  const { data: llmProviders = [], mutate: mutateLLMProviders } = useSWR<LLMProviderConfig[]>(
    "llm-providers",
    getLLMProviders
  );

  const handleSaveBudget = async (budget: AppConfig["budget"]) => {
    if (!config) return;
    setSaving(true);
    try {
      const updated = await updateConfig({ budget });
      mutateConfig(updated, false);
    } catch (err) {
      console.error("Failed to save budget:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleThemeChange = async (newTheme: string) => {
    setTheme(newTheme);
    try {
      await updateConfig({ appearance: { theme: newTheme, sidebar_collapsed: false } });
    } catch (err) {
      console.error("Failed to save theme preference:", err);
    }
  };

  const handleSaveConnectionTestInterval = async (minutes: number) => {
    setSaving(true);
    try {
      const updated = await updateConfig({
        connection_test_interval_minutes: minutes,
      });
      mutateConfig(updated, false);
    } catch (err) {
      console.error("Failed to save connection-test interval:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleCreateProfile = async () => {
    if (!newProfileName.trim()) return;
    const duplicate = profiles.some(
      (p) => p.name.toLowerCase() === newProfileName.trim().toLowerCase(),
    );
    if (duplicate) {
      setProfileError(`Profile "${newProfileName.trim()}" already exists`);
      return;
    }
    setProfileError(null);
    setCreatingProfile(true);
    try {
      await createProfile({
        name: newProfileName.trim(),
        strategy: newProfileStrategy,
        description: newProfileDescription.trim() || undefined,
      });
      setNewProfileName("");
      setNewProfileStrategy(MODE_DIRECT);
      setNewProfileDescription("");
      mutateProfiles();
    } catch (err) {
      console.error("Failed to create profile:", err);
      toast.error("Failed to create profile");
    } finally {
      setCreatingProfile(false);
    }
  };

  const handleEditChatProvider = (provider: ChatProviderConfig) => {
    setEditingId(provider.id);
    setFormData({
      name: provider.name,
      llm_provider_id: provider.llm_provider_id,
      profile_id: provider.profile_id || "",
      rag_config: provider.rag_config || {
        chunk_size: 512,
        chunk_overlap: 64,
        top_k: 5,
        embedding_model: "text-embedding-3-small",
      },
    });
  };

  const handleSaveChatProvider = async () => {
    if (!formData.name.trim() || !formData.llm_provider_id || !formData.profile_id) {
      console.error("Missing required fields");
      return;
    }

    setSavingProvider(true);
    try {
      const selectedProfile = profiles.find((p) => p.id === formData.profile_id);
      const payload: ChatProviderCreateRequest = {
        name: formData.name,
        llm_provider_id: formData.llm_provider_id,
        profile_id: formData.profile_id,
      };

      if (selectedProfile?.strategy === MODE_RAG) {
        payload.rag_config = formData.rag_config;
      }

      const result = editingId
        ? await updateChatProvider(editingId, payload)
        : await createChatProvider(payload);

      if (result.context_window_warning) {
        toast.warning(result.context_window_warning, { duration: 10000 });
      }

      setShowCreateForm(false);
      setEditingId(null);
      resetForm();
      mutateChatProviders();
    } catch (err) {
      console.error("Failed to save chat provider:", err);
      toast.error("Failed to save Chat Provider");
    } finally {
      setSavingProvider(false);
    }
  };

  const handleDeleteChatProvider = async (id: string) => {
    if (!confirm("Are you sure you want to delete this Chat Provider?")) return;

    setDeletingId(id);
    try {
      await deleteChatProvider(id);
      mutateChatProviders();
      // Backend clears judge_chat_provider_id when the deleted provider was
      // the active judge — revalidate config so the UI reflects this immediately.
      if (config?.judge_chat_provider_id === id) {
        mutateConfig();
      }
    } catch (err) {
      console.error("Failed to delete chat provider:", err);
      toast.error("Failed to delete Chat Provider");
    } finally {
      setDeletingId(null);
    }
  };

  const handleImportProfile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setProfileError(null);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      // Auto-rename if name already exists
      let name = data.name || "Imported Profile";
      const nameExists = (n: string) =>
        profiles.some((p) => p.name.toLowerCase() === n.toLowerCase());
      if (nameExists(name)) {
        let i = 2;
        while (nameExists(`${name} (${i})`)) i++;
        name = `${name} (${i})`;
      }
      await createProfile({
        name,
        description: data.description || "",
        strategy: data.strategy || MODE_DIRECT,
        runtime_settings: data.runtime_settings,
        budget: data.budget,
        system_prompts: data.system_prompts || {},
      });
      mutateProfiles();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to import profile";
      setProfileError(msg);
    }
    // Reset file input so same file can be re-imported
    e.target.value = "";
  };

  // Resolve the selected profile for the form
  const selectedProfile = profiles.find((p) => p.id === formData.profile_id);

  const resetForm = () => {
    setFormData({
      name: "",
      llm_provider_id: "",
      profile_id: "",
      rag_config: {
        chunk_size: 512,
        chunk_overlap: 64,
        top_k: 5,
        embedding_model: "text-embedding-3-small",
      },
    });
  };

  const resetLLMProviderForm = () => {
    setLLMProviderForm({ name: "", backend: BACKEND_OPENAI, model: "", api_key: "", endpoint: "", context_window: "" });
    setEditingLLMProviderId(null);
    setLLMProviderModels([]);
    // Clear the form-mode test banner so a stale result from a prior
    // edit doesn't render when the form reopens for a new provider.
    setTestResults((prev) => {
      if (!(FORM_TEST_RESULT_KEY in prev)) return prev;
      const rest = { ...prev };
      delete rest[FORM_TEST_RESULT_KEY];
      return rest;
    });
  };

  const handleEditLLMProvider = (lp: LLMProviderConfig) => {
    setEditingLLMProviderId(lp.id);
    setLLMProviderModels([]);
    setLLMProviderForm({
      name: lp.name,
      backend: lp.backend,
      model: lp.model,
      api_key: "",
      endpoint: lp.endpoint || "",
      context_window: lp.context_window ? String(lp.context_window) : "",
    });
    // A banner from a previous form session (create flow or a
    // different provider's edit) must not leak into this one — it
    // would imply the just-loaded values were tested, which they
    // weren't, and would also drive the post-save status stamp.
    setTestResults((prev) => {
      if (!(FORM_TEST_RESULT_KEY in prev)) return prev;
      const rest = { ...prev };
      delete rest[FORM_TEST_RESULT_KEY];
      return rest;
    });
    setShowLLMProviderForm(true);
  };

  // Invalidate the form-level test result whenever the user edits a
  // field that actually affects the probe outcome. Keeping a stale
  // green banner while the form has been mutated would both mislead
  // the user (implying the current values tested OK) and trigger a
  // bogus post-save status stamp in handleSaveLLMProvider, which
  // uses the banner's presence as its "propagate status" signal.
  useEffect(() => {
    setTestResults((prev) => {
      if (!(FORM_TEST_RESULT_KEY in prev)) return prev;
      const rest = { ...prev };
      delete rest[FORM_TEST_RESULT_KEY];
      return rest;
    });
    // Intentionally depends only on fields that change the probe
    // (backend, model, endpoint, api_key). `name` and
    // `context_window` do not affect the outcome, so typing them
    // shouldn't clear a fresh result.
  }, [
    llmProviderForm.backend,
    llmProviderForm.model,
    llmProviderForm.endpoint,
    llmProviderForm.api_key,
  ]);

  // Build the request payload from the current form state. Used by
  // both save and test paths so the two always see the same values.
  const buildLLMProviderRequest = (): LLMProviderCreateRequest => {
    const parsedCtxWindow = llmProviderForm.context_window.trim()
      ? parseInt(llmProviderForm.context_window.trim(), 10)
      : null;
    return {
      name: llmProviderForm.name.trim(),
      backend: llmProviderForm.backend,
      model: llmProviderForm.model.trim(),
      api_key: llmProviderForm.api_key.trim() || null,
      endpoint: llmProviderForm.endpoint.trim() || null,
      context_window:
        parsedCtxWindow && parsedCtxWindow > 0 ? parsedCtxWindow : null,
    };
  };

  const handleSaveLLMProvider = async () => {
    if (!llmProviderForm.name.trim() || !llmProviderForm.model.trim()) return;
    setSavingLLMProvider(true);
    try {
      const req = buildLLMProviderRequest();
      // If the user ran Test Connection in the form before saving, we
      // already have a fresh result for these exact values. Propagate
      // that status to the saved record so the list row's colored dot
      // matches — otherwise a successful form test followed by Save
      // leaves the row at the default "configured" (orange) despite
      // the user just seeing a green "Connected" banner.
      const formTestResult = testResults[FORM_TEST_RESULT_KEY];
      let savedId: string | null = null;
      if (editingLLMProviderId) {
        await updateLLMProvider(editingLLMProviderId, req);
        savedId = editingLLMProviderId;
      } else {
        const created = await createLLMProvider(req);
        savedId = created.id;
      }
      if (savedId && formTestResult) {
        // Re-run the id-keyed test endpoint so the backend persists
        // status + last_tested_at on the saved record. Cheapest path
        // to unify "test banner" with "list-row dot" without adding
        // a save-time status flag on the API.
        try {
          await testLLMProvider(savedId);
        } catch (err) {
          // Status-stamping is best-effort — save already succeeded.
          console.warn("Post-save status stamp failed:", err);
        }
      }
      setShowLLMProviderForm(false);
      resetLLMProviderForm();
      mutateLLMProviders();
      toast.success(editingLLMProviderId ? "LLM Provider updated" : "LLM Provider created");
    } catch (err) {
      console.error("Failed to save LLM provider:", err);
      toast.error("Failed to save LLM Provider");
    } finally {
      setSavingLLMProvider(false);
    }
  };

  // Form-mode Test Connection. Sends the CURRENT form payload
  // (unsaved changes included) to the stateless
  // `POST /api/llm-providers/test` endpoint. This fixes the
  // long-standing bug where clicking Test Connection in Edit mode
  // tested the last-saved record and silently ignored in-flight edits
  // to api_key / model / endpoint. Result renders in the same banner
  // used by the saved-row Test button.
  const handleTestLLMProviderForm = async () => {
    if (!llmProviderForm.name.trim() || !llmProviderForm.model.trim()) return;
    setTestingLLMProviderId(FORM_TEST_RESULT_KEY);
    try {
      // In edit mode, pass the provider id so the backend can fall
      // back to the saved API key when the form's api_key field is
      // blank (mirrors the "leave blank to keep current key" save
      // contract). Create mode omits the id → no saved-key fallback.
      const req = {
        ...buildLLMProviderRequest(),
        llm_provider_id: editingLLMProviderId,
      };
      const result = await testLLMProviderConfig(req);
      setTestResults((prev) => ({
        ...prev,
        [FORM_TEST_RESULT_KEY]: {
          connected: result.connected,
          error: result.error,
          latency_ms: result.latency_ms,
        },
      }));
    } catch (err) {
      console.error("Test failed:", err);
      setTestResults((prev) => ({
        ...prev,
        [FORM_TEST_RESULT_KEY]: { connected: false, error: String(err) },
      }));
    } finally {
      setTestingLLMProviderId(null);
    }
  };

  const handleDeleteLLMProvider = async (id: string) => {
    if (!confirm("Delete this LLM Provider? Any Chat Providers referencing it will stop working.")) return;
    setDeletingLLMProviderId(id);
    try {
      await deleteLLMProvider(id);
      mutateLLMProviders();
    } catch (err) {
      console.error("Failed to delete LLM provider:", err);
      toast.error("Failed to delete LLM Provider");
    } finally {
      setDeletingLLMProviderId(null);
    }
  };

  const handleTestLLMProvider = async (id: string) => {
    setTestingLLMProviderId(id);
    try {
      const result = await testLLMProvider(id);
      setTestResults((prev) => ({
        ...prev,
        [id]: { connected: result.connected, error: result.error, latency_ms: result.latency_ms },
      }));
      mutateLLMProviders();
    } catch (err) {
      console.error("Test failed:", err);
      setTestResults((prev) => ({
        ...prev,
        [id]: { connected: false, error: String(err) },
      }));
    } finally {
      setTestingLLMProviderId(null);
    }
  };

  const STATUS_COLORS: Record<string, string> = {
    connected: "bg-emerald-500",
    configured: "bg-amber-500",
    offline: "bg-gray-400",
    error: "bg-red-500",
    not_configured: "bg-gray-400",
  };

  const formatRelativeTime = (iso: string | null | undefined): string => {
    if (!iso) return "never";
    const then = new Date(iso).getTime();
    if (!Number.isFinite(then)) return "never";
    const delta = Math.max(0, Date.now() - then) / 1000;
    if (delta < 60) return `${Math.floor(delta)}s ago`;
    if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
    if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
    return `${Math.floor(delta / 86400)}d ago`;
  };

  return (
    <AppShell>
      <div className="mx-auto max-w-[1200px] space-y-6 p-6">
        <h2 className="text-2xl font-semibold">Settings</h2>

        {deepLinkValues && (
          <DeepLinkBanner
            values={deepLinkValues}
            providerDisplayName={BACKEND_DISPLAY_NAMES[deepLinkValues.provider]}
            onCancel={clearDeepLink}
            onUseValues={applyDeepLink}
          />
        )}

        <Tabs defaultValue="llm-providers">
          <TabsList>
            <TabsTrigger value="llm-providers">LLM Providers</TabsTrigger>
            <TabsTrigger value="chat-providers">Chat Providers</TabsTrigger>
            <TabsTrigger value="budget">Budget</TabsTrigger>
            <TabsTrigger value="profiles">Profiles</TabsTrigger>
            <TabsTrigger value="prompts">Prompts</TabsTrigger>
            <TabsTrigger value="general">General</TabsTrigger>
            <TabsTrigger value="appearance">Appearance</TabsTrigger>
          </TabsList>

          <TabsContent value="llm-providers" className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-muted-foreground">Named LLM Provider instances</h3>
              {!showLLMProviderForm && (
                <Button
                  size="sm"
                  onClick={() => { resetLLMProviderForm(); setShowLLMProviderForm(true); }}
                >
                  <Plus className="mr-2 h-4 w-4" aria-hidden="true" />
                  Add LLM Provider
                </Button>
              )}
            </div>
            <div className="rounded-md border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
              {(config?.connection_test_interval_minutes ?? 0) > 0 ? (
                <>
                  Auto-testing every {config?.connection_test_interval_minutes}m — change
                  the interval on the <strong>General</strong> tab.
                </>
              ) : (
                <>
                  Auto-testing of connections is disabled. Enable it on the{" "}
                  <strong>General</strong> tab to catch provider outages without a manual
                  click.
                </>
              )}
            </div>

            {showLLMProviderForm && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    {editingLLMProviderId ? "Edit LLM Provider" : "Add LLM Provider"}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <FieldLabel htmlFor="llmp-name" tooltip="A friendly display name to identify this LLM provider in dropdowns and logs.">Name</FieldLabel>
                    <Input
                      id="llmp-name"
                      placeholder="e.g., My GPT-4o"
                      value={llmProviderForm.name}
                      onChange={(e) => setLLMProviderForm({ ...llmProviderForm, name: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <FieldLabel htmlFor="llmp-backend" tooltip="The inference backend. Cloud providers (OpenAI, Anthropic) require an API key; local providers (Ollama, LM Studio, vLLM) connect to a local endpoint.">Backend</FieldLabel>
                    <Select
                      value={llmProviderForm.backend}
                      onValueChange={(v) => { setLLMProviderForm({ ...llmProviderForm, backend: v, model: "", endpoint: "" }); setLLMProviderModels([]); }}
                      disabled={!!editingLLMProviderId}
                    >
                      <SelectTrigger id="llmp-backend">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value={BACKEND_OPENAI}>OpenAI</SelectItem>
                        <SelectItem value={BACKEND_ANTHROPIC}>Anthropic</SelectItem>
                        <SelectItem value={BACKEND_OLLAMA}>Ollama</SelectItem>
                        <SelectItem value={BACKEND_LMSTUDIO}>LM Studio</SelectItem>
                        <SelectItem value={BACKEND_VLLM}>vLLM (Self-hosted)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <FieldLabel htmlFor="llmp-model" tooltip="The model identifier. For cloud providers, use the official model name (e.g., gpt-4o). For local, use the model name as loaded in your server.">Model</FieldLabel>
                    {llmProviderModels.length > 0 ? (
                      <div className="flex gap-2">
                        <Select
                          value={llmProviderForm.model || undefined}
                          onValueChange={(v) => setLLMProviderForm({ ...llmProviderForm, model: v })}
                        >
                          <SelectTrigger id="llmp-model" className="flex-1">
                            <SelectValue placeholder={MODEL_PLACEHOLDER[llmProviderForm.backend] ?? "e.g., model-name"} />
                          </SelectTrigger>
                          <SelectContent>
                            {llmProviderModels.map((m) => (
                              <SelectItem key={m.name} value={m.name}>
                                {m.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => fetchLLMProviderModels(llmProviderForm.backend, llmProviderForm.endpoint || undefined)}
                          disabled={loadingLLMModels}
                          aria-label="Refresh model list"
                          className="shrink-0"
                        >
                          <RefreshCw className={cn("h-4 w-4", loadingLLMModels && "animate-spin")} aria-hidden="true" />
                        </Button>
                      </div>
                    ) : (
                      <div className="flex gap-2">
                        <Input
                          id="llmp-model"
                          className="flex-1"
                          placeholder={MODEL_PLACEHOLDER[llmProviderForm.backend] ?? "e.g., model-name"}
                          value={llmProviderForm.model}
                          onChange={(e) => setLLMProviderForm({ ...llmProviderForm, model: e.target.value })}
                        />
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => fetchLLMProviderModels(llmProviderForm.backend, llmProviderForm.endpoint || undefined)}
                          disabled={loadingLLMModels}
                          aria-label="Fetch available models"
                          className="shrink-0"
                        >
                          <RefreshCw className={cn("h-4 w-4", loadingLLMModels && "animate-spin")} aria-hidden="true" />
                        </Button>
                      </div>
                    )}
                  </div>
                  {/*
                    Endpoint field renders for local backends by default, but
                    also whenever form.endpoint already holds a value — even
                    for cloud backends. That way any endpoint override (from
                    legacy data, an external config import, or a would-be
                    deep-link attacker) is always visible to the user before
                    they save, never hidden in form state.
                  */}
                  {((LOCAL_BACKENDS as readonly string[]).includes(llmProviderForm.backend) ||
                    llmProviderForm.endpoint !== "") && (
                    <div className="space-y-2">
                      <FieldLabel htmlFor="llmp-endpoint" tooltip="The HTTP endpoint of your local inference server. Default ports: Ollama 11434, LM Studio 1234, vLLM 8000.">Endpoint</FieldLabel>
                      <Input
                        id="llmp-endpoint"
                        placeholder={ENDPOINT_PLACEHOLDER[llmProviderForm.backend] ?? "http://localhost:8000"}
                        value={llmProviderForm.endpoint}
                        onChange={(e) => setLLMProviderForm({ ...llmProviderForm, endpoint: e.target.value })}
                      />
                    </div>
                  )}
                  {(CLOUD_BACKENDS as readonly string[]).includes(llmProviderForm.backend) && (
                    <div className="space-y-2">
                      <FieldLabel htmlFor="llmp-apikey" tooltip="Your provider API key. Stored in the OS keyring when available, otherwise in a local file. Leave blank to keep the current key.">API Key</FieldLabel>
                      <Input
                        id="llmp-apikey"
                        type="password"
                        placeholder="sk-..."
                        value={llmProviderForm.api_key}
                        onChange={(e) => setLLMProviderForm({ ...llmProviderForm, api_key: e.target.value })}
                      />
                    </div>
                  )}
                  <div className="space-y-2">
                    <FieldLabel htmlFor="llmp-context-window" tooltip="Total context window in tokens (input + output). Auto-discovered from the model endpoint when left blank. Set manually to override, e.g. 8192 for Qwen 7B. Profile max_tokens will be capped to 75% of this value.">Context Window (tokens)</FieldLabel>
                    <Input
                      id="llmp-context-window"
                      type="number"
                      placeholder="Auto-discover (e.g. 8192)"
                      value={llmProviderForm.context_window}
                      onChange={(e) => setLLMProviderForm({ ...llmProviderForm, context_window: e.target.value })}
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={handleSaveLLMProvider}
                      disabled={savingLLMProvider || !llmProviderForm.name.trim() || !llmProviderForm.model.trim()}
                    >
                      {editingLLMProviderId ? "Save" : "Create"}
                    </Button>
                    {/* Test Connection runs against the current form
                        values in both Create and Edit modes. Previously
                        Edit mode tested the last-saved record, so
                        unsaved changes to api_key / model / endpoint
                        were silently ignored. Now identical in both
                        modes: form payload → stateless test endpoint. */}
                    <Button
                      variant="secondary"
                      onClick={handleTestLLMProviderForm}
                      disabled={
                        testingLLMProviderId === FORM_TEST_RESULT_KEY ||
                        savingLLMProvider ||
                        !llmProviderForm.name.trim() ||
                        !llmProviderForm.model.trim()
                      }
                    >
                      {testingLLMProviderId === FORM_TEST_RESULT_KEY ? "Testing…" : "Test Connection"}
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => { setShowLLMProviderForm(false); resetLLMProviderForm(); }}
                      disabled={savingLLMProvider}
                    >
                      Cancel
                    </Button>
                  </div>
                  {testResults[FORM_TEST_RESULT_KEY] && (
                    <div className={`mt-2 rounded-md p-2 text-sm ${testResults[FORM_TEST_RESULT_KEY].connected ? "bg-emerald-500/10 text-emerald-400" : "bg-destructive/10 text-destructive"}`}>
                      {testResults[FORM_TEST_RESULT_KEY].connected
                        ? `Connected (${testResults[FORM_TEST_RESULT_KEY].latency_ms ?? 0}ms)`
                        : `Failed: ${testResults[FORM_TEST_RESULT_KEY].error || "Connection failed"}`}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}

            {llmProviders.map((lp) => {
              const testResult = testResults[lp.id];
              const statusColor = STATUS_COLORS[lp.status] || "bg-gray-400";
              return (
                <Card key={lp.id}>
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <CardTitle className="text-base">{lp.name}</CardTitle>
                          <span
                            className={cn("inline-block h-2 w-2 rounded-full", statusColor)}
                            title={lp.status}
                          />
                          <Badge variant="secondary" className="text-xs capitalize">
                            {lp.status.replace("_", " ")}
                          </Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">
                          {BACKEND_DISPLAY_NAMES[lp.backend] || lp.backend} &middot; {lp.model}
                          {lp.endpoint && <> &middot; {lp.endpoint}</>}
                          {lp.context_window && <> &middot; {lp.context_window.toLocaleString()} tokens</>}
                        </p>
                        {(lp.last_tested_at || (lp.consecutive_failures ?? 0) > 0) && (
                          <p className="text-xs text-muted-foreground">
                            {lp.last_tested_at && (
                              <>
                                Last tested {formatRelativeTime(lp.last_tested_at)}
                                {lp.last_tested_by && (
                                  <> ({lp.last_tested_by === "background" ? "auto" : "you"})</>
                                )}
                              </>
                            )}
                            {(lp.consecutive_failures ?? 0) > 0 &&
                              lp.status === "connected" && (
                                <span className="ml-2 text-amber-500">
                                  &middot; {lp.consecutive_failures} recent failure
                                  {lp.consecutive_failures === 1 ? "" : "s"} — monitoring
                                </span>
                              )}
                          </p>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => handleTestLLMProvider(lp.id)}
                          disabled={testingLLMProviderId === lp.id}
                        >
                          <Wifi className="mr-1 h-3 w-3" aria-hidden="true" />
                          {testingLLMProviderId === lp.id ? "Testing..." : "Test"}
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleEditLLMProvider(lp)}
                          disabled={showLLMProviderForm && editingLLMProviderId === lp.id}
                        >
                          <Edit2 className="h-4 w-4" aria-hidden="true" />
                          <span className="sr-only">Edit</span>
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteLLMProvider(lp.id)}
                          disabled={deletingLLMProviderId === lp.id}
                        >
                          <Trash2 className="h-4 w-4" aria-hidden="true" />
                          <span className="sr-only">Delete</span>
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  {testResult && (
                    <CardContent className="pt-0 pb-3">
                      <div className={cn("flex items-center gap-2 text-xs", testResult.connected ? "text-emerald-600" : "text-destructive")}>
                        {testResult.connected
                          ? <><Check className="h-3 w-3" /> Connected{testResult.latency_ms != null && ` (${testResult.latency_ms}ms)`}</>
                          : <><X className="h-3 w-3" /> {testResult.error || "Connection failed"}</>
                        }
                      </div>
                    </CardContent>
                  )}
                </Card>
              );
            })}

            {llmProviders.length === 0 && !showLLMProviderForm && (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  No LLM Providers configured yet. Add one to get started.
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="chat-providers" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Judge Provider</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="mb-2 text-sm text-muted-foreground">
                  Select a Chat Provider to use as the LLM-as-Judge for auto-scoring responses.
                </p>
                <Select
                  value={config?.judge_chat_provider_id || "__none__"}
                  onValueChange={async (value) => {
                    try {
                      const updated = await updateConfig({
                        judge_chat_provider_id: value === "__none__" ? "" : value,
                      });
                      mutateConfig(updated, false);
                    } catch (err) {
                      console.error("Failed to save judge provider:", err);
                    }
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="None (judging disabled)" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">None (judging disabled)</SelectItem>
                    {chatProviders.map((cp) => (
                      <SelectItem key={cp.id} value={cp.id}>
                        {cp.name} ({cp.llm_provider_name || cp.llm_provider_id})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            {!showCreateForm && (
              <Button
                onClick={() => {
                  setEditingId(null);
                  resetForm();
                  setShowCreateForm(true);
                }}
                size="sm"
                className="mb-4"
              >
                <Plus className="mr-2 h-4 w-4" aria-hidden="true" />
                Create Chat Provider
              </Button>
            )}

            {showCreateForm && !editingId && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Create Chat Provider</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-2">
                    <FieldLabel htmlFor="cp-name" tooltip="A unique name for this chat provider configuration, shown in the chat toolbar.">Name</FieldLabel>
                    <Input
                      id="cp-name"
                      placeholder="e.g., My GPT-4 Config"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    />
                  </div>

                  <div className="space-y-2">
                    <FieldLabel htmlFor="cp-llm-provider" tooltip="The LLM provider instance to use for inference. Must be configured and connected in the LLM Providers tab.">LLM Provider</FieldLabel>
                    <Select
                      value={formData.llm_provider_id}
                      onValueChange={(value) => setFormData({ ...formData, llm_provider_id: value })}
                    >
                      <SelectTrigger id="cp-llm-provider">
                        <SelectValue placeholder="Select an LLM Provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {llmProviders.map((lp) => (
                          <SelectItem key={lp.id} value={lp.id}>
                            <span>{lp.name}</span>
                            <span className="ml-2 text-xs text-muted-foreground">
                              {BACKEND_DISPLAY_NAMES[lp.backend] || lp.backend} &middot; {lp.model}
                            </span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <FieldLabel htmlFor="cp-profile" tooltip="The execution profile that defines the strategy (Direct/RLM/RAG), runtime settings (temperature, tokens), and budget limits.">Profile</FieldLabel>
                    <Select
                      value={formData.profile_id}
                      onValueChange={(value) => setFormData({ ...formData, profile_id: value })}
                    >
                      <SelectTrigger id="cp-profile">
                        <SelectValue placeholder="Select a profile" />
                      </SelectTrigger>
                      <SelectContent>
                        {profiles.map((p) => (
                          <SelectItem key={p.id} value={p.id}>
                            {p.name}{p.is_builtin ? " (built-in)" : ""}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  {selectedProfile && (
                    <div className="rounded-lg border border-muted bg-muted/30 p-4 space-y-1">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground capitalize">
                          {selectedProfile.strategy}
                        </span>
                        {selectedProfile.description && (
                          <span className="text-xs text-muted-foreground">{selectedProfile.description}</span>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground">
                        Temp {selectedProfile.runtime_settings.temperature} · Top P {selectedProfile.runtime_settings.top_p} · Max tokens {selectedProfile.runtime_settings.max_output_tokens} · Timeout {selectedProfile.runtime_settings.timeout_seconds}s
                      </p>
                      <p className="text-xs text-muted-foreground">
                        Steps {selectedProfile.budget.max_steps} · Max cost ${selectedProfile.budget.max_cost_usd} · Time {selectedProfile.budget.max_time_seconds}s
                      </p>
                    </div>
                  )}

                  {selectedProfile?.strategy === MODE_RAG && (
                    <div className="space-y-4 rounded-lg border border-muted p-4">
                      <h4 className="font-medium text-sm">RAG Settings</h4>
                      <div className="space-y-2">
                        <FieldLabel htmlFor="cp-chunk-size" tooltip="Number of tokens per text chunk. Smaller chunks give more precise retrieval; larger chunks preserve more context. Typical: 256-1024.">Chunk Size</FieldLabel>
                        <Input
                          id="cp-chunk-size"
                          type="number"
                          min="1"
                          value={formData.rag_config.chunk_size}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rag_config: {
                                ...formData.rag_config,
                                chunk_size: parseInt(e.target.value) || 512,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <FieldLabel htmlFor="cp-chunk-overlap" tooltip="Number of tokens overlapping between adjacent chunks. Prevents information loss at chunk boundaries. Aim for 10-15% of chunk size.">Chunk Overlap</FieldLabel>
                        <Input
                          id="cp-chunk-overlap"
                          type="number"
                          min="0"
                          value={formData.rag_config.chunk_overlap}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rag_config: {
                                ...formData.rag_config,
                                chunk_overlap: parseInt(e.target.value) || 64,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <FieldLabel htmlFor="cp-top-k" tooltip="Number of most relevant chunks to retrieve. Higher values provide more context but increase token usage and cost.">Top K</FieldLabel>
                        <Input
                          id="cp-top-k"
                          type="number"
                          min="1"
                          value={formData.rag_config.top_k}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rag_config: {
                                ...formData.rag_config,
                                top_k: parseInt(e.target.value) || 5,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <FieldLabel htmlFor="cp-embedding-model" tooltip="The embedding model used to vectorize text chunks for semantic search. Must match the model used at indexing time.">Embedding Model</FieldLabel>
                        <Input
                          id="cp-embedding-model"
                          placeholder="e.g., text-embedding-3-small"
                          value={formData.rag_config.embedding_model}
                          onChange={(e) =>
                            setFormData({
                              ...formData,
                              rag_config: {
                                ...formData.rag_config,
                                embedding_model: e.target.value,
                              },
                            })
                          }
                        />
                      </div>
                    </div>
                  )}

                  <div className="flex gap-2">
                    <Button
                      onClick={handleSaveChatProvider}
                      disabled={
                        savingProvider ||
                        !formData.name.trim() ||
                        !formData.llm_provider_id ||
                        !formData.profile_id
                      }
                    >
                      Create
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => {
                        setShowCreateForm(false);
                        setEditingId(null);
                        resetForm();
                      }}
                      disabled={savingProvider}
                    >
                      Cancel
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {chatProviders.map((provider) => {
              const providerProfile = profiles.find((p) => p.id === provider.profile_id);
              const isEditing = editingId === provider.id && !showCreateForm;
              const editProfile = isEditing ? profiles.find((p) => p.id === formData.profile_id) : null;
              return (
              <Card key={provider.id}>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div className="space-y-1">
                      <CardTitle className="text-base">{provider.name}</CardTitle>
                      <p className="text-sm text-muted-foreground">
                        {provider.llm_provider_name || provider.llm_provider_id}
                        {providerProfile && ` · ${providerProfile.name}`}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="inline-flex items-center rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground capitalize">
                        {provider.execution_mode}
                      </span>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => isEditing ? setEditingId(null) : handleEditChatProvider(provider)}
                        disabled={showCreateForm}
                      >
                        <Edit2 className="h-4 w-4" aria-hidden="true" />
                        <span className="sr-only">{isEditing ? "Cancel edit" : "Edit"}</span>
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDeleteChatProvider(provider.id)}
                        disabled={deletingId === provider.id}
                      >
                        <Trash2 className="h-4 w-4" aria-hidden="true" />
                        <span className="sr-only">Delete</span>
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                {!isEditing && (
                  <CardContent className="pt-0 pb-3">
                    <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                      <span>Temp: {provider.runtime_settings.temperature}</span>
                      <span>Max tokens: {provider.runtime_settings.max_output_tokens}</span>
                      <span>Timeout: {provider.runtime_settings.timeout_seconds}s</span>
                      {provider.execution_mode === MODE_RLM && <span>Steps: {provider.rlm_max_steps}</span>}
                      {provider.execution_mode === MODE_RAG && provider.rag_config && (
                        <span>Top K: {provider.rag_config.top_k}</span>
                      )}
                    </div>
                  </CardContent>
                )}
                {isEditing && (
                  <CardContent className="space-y-4 border-t pt-4">
                    <div className="space-y-2">
                      <FieldLabel htmlFor={`edit-cp-name-${provider.id}`} tooltip="A unique name for this chat provider configuration, shown in the chat toolbar.">Name</FieldLabel>
                      <Input
                        id={`edit-cp-name-${provider.id}`}
                        value={formData.name}
                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      />
                    </div>
                    <div className="space-y-2">
                      <FieldLabel htmlFor={`edit-cp-llm-provider-${provider.id}`} tooltip="The LLM provider instance to use for inference.">LLM Provider</FieldLabel>
                      <Select
                        value={formData.llm_provider_id}
                        onValueChange={(value) => setFormData({ ...formData, llm_provider_id: value })}
                      >
                        <SelectTrigger id={`edit-cp-llm-provider-${provider.id}`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {llmProviders.map((lp) => (
                            <SelectItem key={lp.id} value={lp.id}>
                              {lp.name} &middot; {BACKEND_DISPLAY_NAMES[lp.backend] || lp.backend} &middot; {lp.model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <FieldLabel htmlFor={`edit-cp-profile-${provider.id}`} tooltip="The execution profile with strategy, runtime settings, and budget limits.">Profile</FieldLabel>
                      <Select
                        value={formData.profile_id}
                        onValueChange={(value) => setFormData({ ...formData, profile_id: value })}
                      >
                        <SelectTrigger id={`edit-cp-profile-${provider.id}`}>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {profiles.map((p) => (
                            <SelectItem key={p.id} value={p.id}>
                              {p.name}{p.is_builtin ? " (built-in)" : ""}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    {editProfile && (
                      <div className="rounded-lg border border-muted bg-muted/30 p-3 text-xs text-muted-foreground space-y-1">
                        <span className="inline-flex items-center rounded-full bg-secondary px-2 py-0.5 text-xs font-medium capitalize">
                          {editProfile.strategy}
                        </span>
                        <p>Temp {editProfile.runtime_settings.temperature} · Max tokens {editProfile.runtime_settings.max_output_tokens} · Steps {editProfile.budget.max_steps}</p>
                      </div>
                    )}
                    {editProfile?.strategy === MODE_RAG && (
                      <div className="space-y-3 rounded-lg border border-muted p-3">
                        <h4 className="font-medium text-sm">RAG Settings</h4>
                        <div className="grid grid-cols-2 gap-3">
                          <div className="space-y-1">
                            <FieldLabel htmlFor={`edit-cp-chunk-${provider.id}`} className="text-xs" tooltip="Number of tokens per text chunk for retrieval.">Chunk Size</FieldLabel>
                            <Input id={`edit-cp-chunk-${provider.id}`} type="number" min="1" className="h-8 text-xs"
                              value={formData.rag_config.chunk_size}
                              onChange={(e) => setFormData({ ...formData, rag_config: { ...formData.rag_config, chunk_size: parseInt(e.target.value) || 512 }})}
                            />
                          </div>
                          <div className="space-y-1">
                            <FieldLabel htmlFor={`edit-cp-overlap-${provider.id}`} className="text-xs" tooltip="Token overlap between adjacent chunks. Aim for 10-15% of chunk size.">Overlap</FieldLabel>
                            <Input id={`edit-cp-overlap-${provider.id}`} type="number" min="0" className="h-8 text-xs"
                              value={formData.rag_config.chunk_overlap}
                              onChange={(e) => setFormData({ ...formData, rag_config: { ...formData.rag_config, chunk_overlap: parseInt(e.target.value) || 64 }})}
                            />
                          </div>
                          <div className="space-y-1">
                            <FieldLabel htmlFor={`edit-cp-topk-${provider.id}`} className="text-xs" tooltip="Number of most relevant chunks to retrieve per query.">Top K</FieldLabel>
                            <Input id={`edit-cp-topk-${provider.id}`} type="number" min="1" className="h-8 text-xs"
                              value={formData.rag_config.top_k}
                              onChange={(e) => setFormData({ ...formData, rag_config: { ...formData.rag_config, top_k: parseInt(e.target.value) || 5 }})}
                            />
                          </div>
                          <div className="space-y-1">
                            <FieldLabel htmlFor={`edit-cp-embed-${provider.id}`} className="text-xs" tooltip="The embedding model for vectorizing text chunks.">Embedding Model</FieldLabel>
                            <Input id={`edit-cp-embed-${provider.id}`} className="h-8 text-xs"
                              value={formData.rag_config.embedding_model}
                              onChange={(e) => setFormData({ ...formData, rag_config: { ...formData.rag_config, embedding_model: e.target.value }})}
                            />
                          </div>
                        </div>
                      </div>
                    )}
                    <div className="flex gap-2">
                      <Button size="sm" onClick={handleSaveChatProvider}
                        disabled={savingProvider || !formData.name.trim() || !formData.llm_provider_id || !formData.profile_id}
                      >
                        {savingProvider ? "Saving..." : "Save"}
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => setEditingId(null)} disabled={savingProvider}>
                        Cancel
                      </Button>
                    </div>
                  </CardContent>
                )}
              </Card>
              );
            })}

            {chatProviders.length === 0 && !showCreateForm && (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  No Chat Providers configured yet. Create one to get started.
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="budget">
            {config ? (
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Budget Limits</CardTitle>
                </CardHeader>
                <CardContent>
                  <BudgetConfig config={config.budget} onChange={handleSaveBudget} />
                  {saving && (
                    <p className="mt-2 text-sm text-muted-foreground">Saving...</p>
                  )}
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  Loading configuration...
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="profiles" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Create Profile</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex gap-2">
                  <Input
                    placeholder="Profile name"
                    value={newProfileName}
                    onChange={(e) => {
                      setNewProfileName(e.target.value);
                      setProfileError(null);
                    }}
                    onKeyDown={(e) => e.key === "Enter" && handleCreateProfile()}
                    aria-label="New profile name"
                    className="flex-1"
                  />
                  <Select value={newProfileStrategy} onValueChange={setNewProfileStrategy}>
                    <SelectTrigger className="w-[120px]" aria-label="Strategy">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <StrategySelectItems />
                    </SelectContent>
                  </Select>
                  <Button
                    onClick={handleCreateProfile}
                    disabled={creatingProfile || !newProfileName.trim()}
                    size="sm"
                  >
                    <Plus className="mr-1 h-4 w-4" aria-hidden="true" />
                    Create
                  </Button>
                </div>
                <div className="flex gap-2 items-center">
                  <Input
                    placeholder="Description (optional)"
                    value={newProfileDescription}
                    onChange={(e) => setNewProfileDescription(e.target.value)}
                    aria-label="Profile description"
                    className="flex-1"
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => importFileRef.current?.click()}
                  >
                    <Upload className="mr-1 h-4 w-4" aria-hidden="true" />
                    Import
                  </Button>
                  <input
                    ref={importFileRef}
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={handleImportProfile}
                  />
                </div>
                {profileError && (
                  <p className="text-xs text-destructive">{profileError}</p>
                )}
              </CardContent>
            </Card>

            {profiles.map((profile) => (
              <ProfileCard
                key={profile.id}
                profile={profile}
                chatProviders={chatProviders}
                onDeleted={() => mutateProfiles()}
                onUpdated={() => {
                  mutateProfiles();
                  mutateChatProviders();
                }}
                onCloned={() => mutateProfiles()}
              />
            ))}

            {profiles.length === 0 && (
              <Card>
                <CardContent className="py-10 text-center text-muted-foreground">
                  No profiles yet. Create one above or use built-in presets.
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="prompts">
            <SystemPromptEditor />
          </TabsContent>

          <TabsContent value="general" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Scheduled Connection Testing</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  How often RLM Studio should auto-test LLM Provider connections. When enabled,
                  a background task probes each provider at this interval and updates its
                  status. Set to <strong>0</strong> to disable auto-testing (manual testing
                  via the Test button still works).
                </p>
                <div className="flex items-center gap-3">
                  <Label htmlFor="connection-test-interval" className="w-48">
                    Interval (minutes)
                  </Label>
                  <Input
                    id="connection-test-interval"
                    type="number"
                    min={0}
                    max={1440}
                    step={1}
                    className="w-32"
                    defaultValue={config?.connection_test_interval_minutes ?? 0}
                    onBlur={(e) => {
                      const raw = Number(e.target.value);
                      const clamped = Number.isFinite(raw)
                        ? Math.max(0, Math.min(1440, Math.floor(raw)))
                        : 0;
                      if (clamped !== (config?.connection_test_interval_minutes ?? 0)) {
                        handleSaveConnectionTestInterval(clamped);
                      }
                    }}
                    disabled={saving}
                  />
                  <span className="text-sm text-muted-foreground">
                    {(config?.connection_test_interval_minutes ?? 0) === 0
                      ? "disabled"
                      : `every ${config?.connection_test_interval_minutes}m`}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Range: 0-1440 minutes (24 hours). The first cycle runs after the full
                  interval elapses, not at server startup.
                </p>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="appearance">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Theme</CardTitle>
              </CardHeader>
              <CardContent>
                <RadioGroup
                  value={theme ?? "system"}
                  onValueChange={handleThemeChange}
                  className="flex gap-4"
                  aria-label="Theme selection"
                >
                  {(["system", "light", "dark"] as const).map((t) => (
                    <div key={t} className="flex items-center gap-2">
                      <RadioGroupItem value={t} id={`theme-${t}`} />
                      <Label htmlFor={`theme-${t}`} className="capitalize">
                        {t}
                      </Label>
                    </div>
                  ))}
                </RadioGroup>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </AppShell>
  );
}
