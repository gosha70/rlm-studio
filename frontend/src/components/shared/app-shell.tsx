"use client";

import { useState, useCallback } from "react";
import useSWR from "swr";
import { getSessions, getHealth, deleteSession as apiDeleteSession, renameSession as apiRenameSession, type SessionSummary } from "@/lib/api";
import { Sidebar } from "./sidebar";
import { ThemeToggle } from "./theme-toggle";
import { StatusIndicator } from "./status-indicator";
import { HEALTH_POLL_MS, HEALTH_RETRY_MS, SESSIONS_SWR_KEY } from "@/lib/constants";

interface AppShellProps {
  children: React.ReactNode;
  activeSessionId?: string | null;
  onSelectSession?: (id: string) => void;
  onNewSession?: () => void;
}

export function AppShell({
  children,
  activeSessionId = null,
  onSelectSession,
  onNewSession,
}: AppShellProps) {
  const { data: health, error: healthError } = useSWR("health", getHealth, {
    refreshInterval: HEALTH_POLL_MS,
    // SWR skips interval refreshes while the key holds an error, so retry on
    // a fixed beat instead; otherwise the header never noticed the backend
    // coming back.
    onErrorRetry: (_err, _key, _config, revalidate) => {
      setTimeout(() => revalidate(), HEALTH_RETRY_MS);
    },
  });
  // SWR keeps the last good response when a request fails, so the error, not
  // the data, says the backend is gone.
  const backendDown = Boolean(healthError);
  const headerStatus = backendDown ? "offline" : health?.status === "ok" ? "connected" : "loading";
  const [collapsed, setCollapsed] = useState(false);

  const { data: sessions = [], mutate: mutateSessions } = useSWR<SessionSummary[]>(
    SESSIONS_SWR_KEY,
    () => getSessions(),
  );

  const handleDeleteSession = useCallback(
    async (id: string) => {
      await apiDeleteSession(id);
      await mutateSessions();
    },
    [mutateSessions],
  );

  const handleRenameSession = useCallback(
    async (id: string, name: string) => {
      await apiRenameSession(id, name);
      await mutateSessions();
    },
    [mutateSessions],
  );

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed((c) => !c)}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewSession={onNewSession}
        onDeleteSession={handleDeleteSession}
        onSelectSession={onSelectSession}
        onRenameSession={handleRenameSession}
      />

      <div className="flex flex-1 flex-col overflow-hidden">
        <header className="flex h-14 shrink-0 items-center justify-between border-b bg-card px-4">
          <h1 className="text-lg font-semibold">RLM Studio</h1>
          <div className="flex items-center gap-3">
            <StatusIndicator status={headerStatus} />
            <ThemeToggle />
          </div>
        </header>
        {backendDown && (
          <div
            role="alert"
            className="shrink-0 border-b border-destructive/30 bg-destructive/10 px-4 py-2 text-sm text-destructive"
          >
            Can&apos;t reach the RLM Studio server. What this page shows may be out of date or empty
            until it is back; the page reconnects on its own.
          </div>
        )}

        <main className="h-0 flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  );
}
