"use client";

import { useRef, useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import useSWR from "swr";
import {
  MessageSquare,
  Gauge,
  AudioWaveform,
  SlidersHorizontal,
  BookMarked,
  Settings,
  PanelLeftClose,
  PanelLeft,
  Plus,
  Trash2,
  Pencil,
  Search,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { BASE_PATH, PRODUCT_NAME } from "@/lib/branding";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import {
  getDiagnostics,
  type DiagnosticsResponse,
  type SessionSummary,
} from "@/lib/api";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  sessions?: SessionSummary[];
  activeSessionId?: string | null;
  onNewSession?: () => void;
  onDeleteSession?: (id: string) => void;
  onSelectSession?: (id: string) => void;
  onRenameSession?: (id: string, name: string) => void;
}

const LOGO_SIZE_PX = 28;

const NAV_ITEMS = [
  { href: "/", label: "Chat", icon: MessageSquare },
  { href: "/compare", label: "LLM Tuner", icon: SlidersHorizontal },
  { href: "/dashboard", label: "Dashboard", icon: Gauge },
  { href: "/traces", label: "Traces", icon: AudioWaveform },
  { href: "/learn", label: "Learn", icon: BookMarked },
  { href: "/settings", label: "Settings", icon: Settings },
];

/**
 * Red-dot badge policy for the Learn nav item: fires only when at
 * least one diagnostic check is in `error` state. `warn` does NOT
 * badge — see pitch §Diagnostics red-dot badge (decision #6).
 */
export function hasLearnErrorBadge(
  diagnostics: DiagnosticsResponse | null | undefined,
): boolean {
  if (!diagnostics) return false;
  return (
    diagnostics.backend.status === "error" ||
    diagnostics.provider.status === "error" ||
    diagnostics.judge.status === "error" ||
    diagnostics.storage.status === "error"
  );
}

/** Confirmation text for deleting a session; its runs stay in Traces. */
function deleteSessionPrompt(name: string): string {
  return `Delete session "${name}"? Its conversation is removed; its runs stay in Traces. This cannot be undone.`;
}

export function Sidebar({
  collapsed,
  onToggle,
  sessions = [],
  activeSessionId,
  onNewSession,
  onDeleteSession,
  onSelectSession,
  onRenameSession,
}: SidebarProps) {
  const pathname = usePathname();
  const [searchQuery, setSearchQuery] = useState("");
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editName, setEditName] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);
  // Diagnostics fetch uses the same SWR key as the Learn pages, so
  // this is a shared, deduped read — one poll per 30s regardless of
  // which page is mounted.
  const { data: diagnostics } = useSWR<DiagnosticsResponse>(
    "learn-diagnostics",
    getDiagnostics,
    {
      refreshInterval: 30_000,
      dedupingInterval: 30_000,
      errorRetryCount: 2,
    },
  );
  const learnBadge = hasLearnErrorBadge(diagnostics);

  const filteredSessions = searchQuery
    ? sessions.filter((s) => s.name.toLowerCase().includes(searchQuery.toLowerCase()))
    : sessions;

  const startRename = (session: SessionSummary) => {
    setEditingSessionId(session.id);
    setEditName(session.name);
    setTimeout(() => editInputRef.current?.select(), 0);
  };

  const commitRename = () => {
    if (editingSessionId && editName.trim()) {
      onRenameSession?.(editingSessionId, editName.trim());
    }
    setEditingSessionId(null);
  };

  return (
    <aside
      className={cn(
        "flex h-full flex-col border-r bg-card transition-all duration-200",
        collapsed ? "w-16" : "w-60",
      )}
    >
      <div className="flex h-14 items-center justify-between border-b px-3">
        <div className="flex items-center gap-2">
          <Image
            src={`${BASE_PATH}/logo.png`}
            alt={PRODUCT_NAME}
            width={LOGO_SIZE_PX}
            height={LOGO_SIZE_PX}
            className="shrink-0 rounded-full"
            // Fixed display width: the unoptimized bundle serves the full-size
            // file, which "auto" would render at its natural 671 px.
            style={{ width: LOGO_SIZE_PX, height: "auto" }}
          />
          {!collapsed && <span className="text-sm font-semibold">{PRODUCT_NAME}</span>}
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="ml-auto"
        >
          {collapsed ? <PanelLeft className="h-4 w-4" aria-hidden="true" /> : <PanelLeftClose className="h-4 w-4" aria-hidden="true" />}
        </Button>
      </div>

      <nav aria-label="Main navigation" className="flex flex-col gap-1 px-2 pt-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const isActive = href === "/" ? pathname === "/" : pathname.startsWith(href);
          const showBadge = href === "/learn" && learnBadge;
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "relative flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground",
                isActive
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground",
                collapsed && "justify-center px-0",
              )}
              aria-current={isActive ? "page" : undefined}
            >
              <span className="relative inline-flex shrink-0">
                <Icon className="h-4 w-4" aria-hidden="true" />
                {showBadge && collapsed && (
                  <span
                    aria-label="Learn has unresolved errors"
                    className="absolute -right-1 -top-1 inline-block h-2 w-2 rounded-full bg-red-500 ring-2 ring-card"
                  />
                )}
              </span>
              {!collapsed && (
                <span className="flex items-center gap-2">
                  {label}
                  {showBadge && (
                    <span
                      aria-label="Learn has unresolved errors"
                      className="inline-block h-2 w-2 rounded-full bg-red-500"
                    />
                  )}
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {!collapsed && pathname === "/" && (
        <>
          <Separator className="my-3" />
          <div className="flex items-center justify-between px-4">
            <span className="text-xs font-semibold text-muted-foreground">Sessions</span>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={onNewSession}
              aria-label="New session"
            >
              <Plus className="h-3.5 w-3.5" aria-hidden="true" />
            </Button>
          </div>
          <div className="px-2 py-1">
            <div className="relative">
              <Search className="absolute left-2 top-1/2 h-3 w-3 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
              <Input
                placeholder="Search sessions..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-7 pl-7 text-xs"
                aria-label="Search sessions"
              />
            </div>
          </div>
          <ScrollArea className="flex-1 px-2 py-1">
            <div className="flex flex-col gap-0.5">
              {filteredSessions.map((s) => (
                <div
                  key={s.id}
                  className={cn(
                    "group flex items-center justify-between rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-accent cursor-pointer",
                    s.id === activeSessionId && "bg-accent text-accent-foreground",
                  )}
                >
                  {editingSessionId === s.id ? (
                    <input
                      ref={editInputRef}
                      value={editName}
                      onChange={(e) => setEditName(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitRename();
                        if (e.key === "Escape") setEditingSessionId(null);
                      }}
                      className="flex-1 truncate bg-transparent text-sm outline-none ring-1 ring-primary rounded px-1"
                      aria-label="Rename session"
                    />
                  ) : (
                    <button
                      type="button"
                      className="min-w-0 flex-1 truncate text-left"
                      onClick={() => onSelectSession?.(s.id)}
                      onDoubleClick={() => startRename(s)}
                    >
                      {s.name}
                    </button>
                  )}
                  {editingSessionId !== s.id && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                        "h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                        s.id === activeSessionId && "opacity-100",
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        startRename(s);
                      }}
                      aria-label={`Rename session ${s.name}`}
                      title="Rename"
                    >
                      <Pencil className="h-3 w-3" aria-hidden="true" />
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                      "h-6 w-6 shrink-0 opacity-0 group-hover:opacity-100 focus-visible:opacity-100",
                      s.id === activeSessionId && "opacity-100",
                    )}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm(deleteSessionPrompt(s.name))) onDeleteSession?.(s.id);
                    }}
                    aria-label={`Delete session ${s.name}`}
                    title="Delete"
                  >
                    <Trash2 className="h-3 w-3" aria-hidden="true" />
                  </Button>
                </div>
              ))}
              {filteredSessions.length === 0 && (
                <p className="px-3 py-2 text-xs text-muted-foreground">
                  {searchQuery ? "No matching sessions" : "No sessions yet"}
                </p>
              )}
            </div>
          </ScrollArea>
        </>
      )}
    </aside>
  );
}
