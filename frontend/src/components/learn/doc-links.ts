/**
 * Links inside a Learn markdown doc.
 *
 * The docs are written for the repository, so they link to other files
 * by relative path (`README.md#…`, `dgx-spark-vllm.md`). Rendered under
 * `/learn/…` those paths resolve against the page URL and open "Page not
 * found". `resolveDocHref` rewrites them: a doc that has a Learn page
 * opens that page; any other file opens on GitHub.
 */

import { docSlugForProvider, getProviderById } from "./provider-catalog";

/** Where repository files are browsed on GitHub (no trailing slash). */
export const GITHUB_BLOB_BASE = "https://github.com/gosha70/rlm-studio/blob/master";

/** Repository folder the Learn docs live in (`docs/` in the repo). */
const DOCS_DIR = "docs";
const HOSTS_DIR = "hosts";
const HOSTS_SLUG_PREFIX = docSlugForProvider("");

/** Docs with their own Learn page, by repository path. */
const LEARN_PAGES: Readonly<Record<string, string>> = {
  [`${DOCS_DIR}/rlm-concepts.md`]: "/learn/concepts",
  [`${DOCS_DIR}/troubleshoot.yaml`]: "/learn/troubleshoot",
};

/** Matches an href with a scheme (`https:`, `mailto:`) or a protocol-relative one. */
const EXTERNAL_HREF = /^([a-z][a-z0-9+.-]*:|\/\/)/i;

/** The repository folder a Learn doc slug's file sits in. */
export function docDirForSlug(slug: string): string {
  return slug.startsWith(HOSTS_SLUG_PREFIX) ? `${DOCS_DIR}/${HOSTS_DIR}` : DOCS_DIR;
}

/** Join `relative` onto `dir` and fold `.` and `..` segments. */
function resolvePath(dir: string, relative: string): string {
  const parts = relative.startsWith("/") ? [] : dir.split("/");
  for (const segment of relative.split("/")) {
    if (segment === "" || segment === ".") continue;
    if (segment === "..") parts.pop();
    else parts.push(segment);
  }
  return parts.join("/");
}

/** The Learn page for a repository path, if it has one. */
function learnPageFor(repoPath: string): string | undefined {
  const page = LEARN_PAGES[repoPath];
  if (page) return page;
  const hostsPrefix = `${DOCS_DIR}/${HOSTS_DIR}/`;
  if (repoPath.startsWith(hostsPrefix) && repoPath.endsWith(".md")) {
    const id = repoPath.slice(hostsPrefix.length, -".md".length);
    if (getProviderById(id)) return `/learn/cookbook?provider=${encodeURIComponent(id)}`;
  }
  return undefined;
}

export interface ResolvedDocHref {
  href: string;
  /** True for a Learn page (route inside the app); false for anything else. */
  internal: boolean;
}

/**
 * Rewrite a link found in the doc whose file sits in `docDir` (see
 * `docDirForSlug`). Absolute URLs and same-page `#anchors` are left alone.
 */
export function resolveDocHref(href: string, docDir: string): ResolvedDocHref {
  if (!href || href.startsWith("#") || EXTERNAL_HREF.test(href)) {
    return { href, internal: false };
  }
  const hashAt = href.indexOf("#");
  const path = hashAt === -1 ? href : href.slice(0, hashAt);
  const hash = hashAt === -1 ? "" : href.slice(hashAt);
  const repoPath = resolvePath(docDir, path);
  const page = learnPageFor(repoPath);
  if (page) return { href: page + hash, internal: true };
  return { href: `${GITHUB_BLOB_BASE}/${repoPath}${hash}`, internal: false };
}
