"use client";

/**
 * Single Troubleshoot entry card.
 *
 * Renders title, category badge, symptom, cause, ordered fix list,
 * and see-also links. `seealso` entries of the form "cookbook/<id>"
 * route into the Cookbook provider guide pages; unknown shapes are
 * rendered as plain text (defensive — the YAML is author-controlled
 * but we still don't want to promise routes that don't exist).
 */

import type { ReactNode } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type {
  TroubleshootCategory,
  TroubleshootEntry as TroubleshootEntryData,
} from "@/lib/api";
import { getProviderById } from "./provider-catalog";

const CATEGORY_VARIANT: Record<
  TroubleshootCategory,
  "success" | "warning" | "secondary" | "outline" | "destructive" | "default"
> = {
  Setup: "secondary",
  Provider: "default",
  Compare: "warning",
  Judge: "success",
  Budget: "warning",
  Runtime: "destructive",
};

interface TroubleshootEntryProps {
  entry: TroubleshootEntryData;
  className?: string;
}

export function TroubleshootEntry({
  entry,
  className,
}: TroubleshootEntryProps) {
  return (
    <Card
      id={entry.id}
      aria-labelledby={`troubleshoot-title-${entry.id}`}
      className={cn("h-full", className)}
    >
      <CardHeader className="p-4 pb-2">
        <div className="flex items-start justify-between gap-2">
          <h3
            id={`troubleshoot-title-${entry.id}`}
            className="text-base font-semibold leading-tight"
          >
            {entry.title}
          </h3>
          <Badge variant={CATEGORY_VARIANT[entry.category]}>
            {entry.category}
          </Badge>
        </div>
        <p className="mt-2 text-sm text-foreground/90">
          <span className="font-semibold">Symptom: </span>
          {entry.symptom}
        </p>
        <p className="mt-1 text-sm text-muted-foreground">
          <span className="font-semibold">Cause: </span>
          {entry.cause}
        </p>
      </CardHeader>
      <CardContent className="p-4 pt-0">
        {entry.fix.length > 0 && (
          <>
            <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Fix
            </p>
            <ol className="ml-5 list-decimal space-y-1 text-sm">
              {entry.fix.map((step, i) => (
                <li key={i}>{withInlineCode(step)}</li>
              ))}
            </ol>
          </>
        )}
        {entry.seealso.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs">
            <span className="font-semibold text-muted-foreground">See also:</span>
            {entry.seealso.map((ref) => (
              <SeeAlsoLink key={ref} target={ref} />
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

const INLINE_CODE_MARK = "`";

/**
 * Render `code` spans in a YAML fix step as code instead of literal
 * backticks. Text with an unmatched backtick is left as written.
 */
function withInlineCode(text: string): ReactNode {
  const parts = text.split(INLINE_CODE_MARK);
  if (parts.length % 2 === 0) return text;
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <code key={i} className="rounded bg-muted px-1 font-mono text-[0.85em]">
        {part}
      </code>
    ) : (
      part
    ),
  );
}

const COOKBOOK_REF = /^cookbook\/([a-z0-9-]+)$/;

function SeeAlsoLink({ target }: { target: string }) {
  const cookbookMatch = COOKBOOK_REF.exec(target);
  // Syntax match is necessary but not sufficient: the provider id must
  // also be in the current Cookbook catalog. Otherwise a stale YAML
  // entry (e.g. "cookbook/groq" after the Groq removal in 5d1c29c) or
  // a typo would become a live link that lands on the "Provider not
  // found" alert at /learn/cookbook?provider=<id>. Fall through to
  // plain text for any unknown id.
  if (cookbookMatch && getProviderById(cookbookMatch[1]) !== undefined) {
    const providerId = cookbookMatch[1];
    return (
      <Link
        href={`/learn/cookbook?provider=${encodeURIComponent(providerId)}`}
        className="rounded border border-input bg-background px-2 py-0.5 hover:bg-muted"
      >
        Cookbook: {providerId}
      </Link>
    );
  }
  // Unknown shape or unknown id — render as plain text, not a dead link.
  return <span className="text-muted-foreground">{target}</span>;
}
