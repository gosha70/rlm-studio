/**
 * Single source of product identity for the frontend — mirrors
 * `src/rlmstudio/branding.py` on the backend.
 *
 * Product name, browser-storage keys, and build-time feature flags live
 * here so a rename is a constants change, not a search-and-replace.
 */

/** Human-facing product name (nav, titles, copy). */
export const PRODUCT_NAME = "RLM Studio";

// ---------------------------------------------------------------------------
// Build-time feature flags
// ---------------------------------------------------------------------------

/**
 * "Performance shape" UI (Compare perf ranking metrics, Traces Performance
 * tab). Read at build time — Next.js inlines literal
 * `process.env.NEXT_PUBLIC_*` accesses. The legacy `NEXT_PUBLIC_RLMKIT_PERF_UI`
 * name is honoured for one release cycle.
 */
export const PERF_UI_ENABLED: boolean =
  process.env.NEXT_PUBLIC_RLM_STUDIO_PERF_UI === "1" ||
  process.env.NEXT_PUBLIC_RLMKIT_PERF_UI === "1";

/**
 * URL prefix the UI is served under: `/studio` in the bundled build (set in
 * next.config.ts), empty otherwise. Prepend it to files fetched from
 * `public/`; Next adds it to `<Link>` and its own assets by itself.
 */
export const BASE_PATH: string = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

// ---------------------------------------------------------------------------
// localStorage keys
// ---------------------------------------------------------------------------

const STORAGE_PREFIX = "rlm_studio_";
const LEGACY_STORAGE_PREFIXES = ["rlmkit_", "rlmkit-"] as const;

export const STORAGE_KEY_ACTIVE_SESSION = `${STORAGE_PREFIX}active_session`;
export const STORAGE_KEY_SELECTED_CHAT_PROVIDERS = `${STORAGE_PREFIX}selected_chat_providers`;
export const STORAGE_KEY_CHAT_PROVIDER_ORDER = `${STORAGE_PREFIX}cp_order`;
export const STORAGE_KEY_FILES_DRAFT = `${STORAGE_PREFIX}files_draft`;
/** Per-session uploaded-files key. */
export const storageKeyFilesForSession = (sessionId: string): string =>
  `${STORAGE_PREFIX}files_${sessionId}`;

/**
 * Legacy → canonical key map for the fixed keys. Per-session file keys are
 * matched by prefix (`rlmkit_files_<id>` → `rlm_studio_files_<id>`).
 */
const LEGACY_KEY_MAP: Record<string, string> = {
  rlmkit_active_session: STORAGE_KEY_ACTIVE_SESSION,
  rlmkit_selected_chat_providers: STORAGE_KEY_SELECTED_CHAT_PROVIDERS,
  "rlmkit-cp-order": STORAGE_KEY_CHAT_PROVIDER_ORDER,
  rlmkit_files_draft: STORAGE_KEY_FILES_DRAFT,
};

/**
 * One-shot migration of pre-rename `localStorage` keys. Copies each legacy
 * value to its canonical key (only if the canonical key is not already
 * set), then removes the legacy key. Safe to call on every page load —
 * it is a no-op once nothing legacy remains. Returns the number of keys
 * migrated.
 */
export function migrateLegacyStorageKeys(storage: Storage | undefined = globalThis.localStorage): number {
  if (!storage) return 0;
  let migrated = 0;
  const legacyKeys: string[] = [];
  for (let i = 0; i < storage.length; i++) {
    const key = storage.key(i);
    if (key && LEGACY_STORAGE_PREFIXES.some((p) => key.startsWith(p))) {
      legacyKeys.push(key);
    }
  }
  for (const key of legacyKeys) {
    let target = LEGACY_KEY_MAP[key];
    if (!target && key.startsWith("rlmkit_files_")) {
      target = storageKeyFilesForSession(key.slice("rlmkit_files_".length));
    }
    if (!target) continue; // unknown legacy key: leave it alone
    const value = storage.getItem(key);
    if (value !== null && storage.getItem(target) === null) {
      storage.setItem(target, value);
    }
    storage.removeItem(key);
    migrated++;
  }
  return migrated;
}
