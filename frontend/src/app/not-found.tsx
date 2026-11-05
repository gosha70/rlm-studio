import Link from "next/link";
import { AppShell } from "@/components/shared/app-shell";

/** Unknown routes keep the app shell (navigation) and offer a way back to Chat. */
export default function NotFound() {
  return (
    <AppShell>
      <div className="mx-auto max-w-xl px-6 py-16 text-center">
        <h2 className="text-2xl font-semibold">Page not found</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          There is no page at this address. Use the navigation on the left, or go back to Chat.
        </p>
        <Link
          href="/"
          className="mt-6 inline-flex items-center rounded-md border px-4 py-2 text-sm font-medium hover:bg-accent"
        >
          Go to Chat
        </Link>
      </div>
    </AppShell>
  );
}
