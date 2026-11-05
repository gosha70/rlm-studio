"use client";

/**
 * Markdown loader for the Learn tab.
 *
 * Fetches an allowlisted doc from GET /api/docs/{slug} and renders it
 * through the existing ReactMarkdown + remarkGfm pipeline. The render
 * wrapper matches the chat/compare prose styling so long-form content
 * reads the same everywhere.
 *
 * Heading IDs: H2 and H3 elements get deterministic slug IDs via the
 * shared slugifyHeading helper. That keeps anchor targets in sync
 * with any table of contents (e.g. the Cookbook provider guide rail)
 * without introducing a rehype-slug dependency.
 */

import type { ReactNode } from "react";
import Link from "next/link";
import useSWR from "swr";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getDoc, type DocResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { slugifyHeading } from "./markdown-toc";
import { docDirForSlug, resolveDocHref } from "./doc-links";

interface MarkdownDocProps {
  slug: string;
  className?: string;
}

function headingTextOf(children: ReactNode): string {
  if (typeof children === "string") return children;
  if (typeof children === "number") return String(children);
  if (Array.isArray(children)) return children.map(headingTextOf).join("");
  if (
    children !== null &&
    typeof children === "object" &&
    "props" in children &&
    // @ts-expect-error — React element typing surfaces `props` at runtime
    children.props?.children !== undefined
  ) {
    // @ts-expect-error — same as above
    return headingTextOf(children.props.children);
  }
  return "";
}

/** Links to other repo files open their Learn page or GitHub, not a 404. */
function docLink(docDir: string) {
  return function DocLink({ href = "", children }: { href?: string; children?: ReactNode }) {
    const resolved = resolveDocHref(href, docDir);
    if (resolved.internal) return <Link href={resolved.href}>{children}</Link>;
    // Anything but a same-page anchor leaves the app: open it in a new tab.
    const external = resolved.href !== "" && !resolved.href.startsWith("#");
    return (
      <a href={resolved.href} {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}>
        {children}
      </a>
    );
  };
}

const markdownComponents = {
  h2: ({ children, ...rest }: { children?: ReactNode }) => (
    <h2 id={slugifyHeading(headingTextOf(children))} {...rest}>
      {children}
    </h2>
  ),
  h3: ({ children, ...rest }: { children?: ReactNode }) => (
    <h3 id={slugifyHeading(headingTextOf(children))} {...rest}>
      {children}
    </h3>
  ),
};

export function MarkdownDoc({ slug, className }: MarkdownDocProps) {
  const { data, error, isLoading } = useSWR<DocResponse>(
    ["learn-doc", slug],
    () => getDoc(slug),
    { revalidateOnFocus: false },
  );

  if (isLoading || (!data && !error)) {
    return (
      <div
        role="status"
        aria-label="Loading document"
        className={cn("text-sm text-muted-foreground", className)}
      >
        Loading…
      </div>
    );
  }

  if (error || !data) {
    return (
      <div
        role="alert"
        className={cn(
          "rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm text-destructive",
          className,
        )}
      >
        Couldn’t load this guide.
      </div>
    );
  }

  return (
    <div
      data-slug={data.slug}
      className={cn(
        "prose prose-sm dark:prose-invert max-w-none overflow-x-auto",
        className,
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{ ...markdownComponents, a: docLink(docDirForSlug(slug)) }}
      >
        {data.content}
      </ReactMarkdown>
    </div>
  );
}
