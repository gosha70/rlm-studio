import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { CodeBlock } from "./code-block";
import type { TraceStep } from "@/lib/api";

interface StepDetailProps {
  step: TraceStep;
}

export function StepDetail({ step }: StepDetailProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <CardTitle className="text-base">Step {step.index + 1}</CardTitle>
          <Badge variant="outline">{step.action_type}</Badge>
          {step.model && (
            <Badge variant="secondary" className="text-xs">
              {step.model}
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {step.code && <CodeBlock code={step.code} output={step.output} />}

        {!step.code && step.output && (
          <div>
            <p className="text-xs font-medium text-muted-foreground mb-1">Output</p>
            <p className="text-sm whitespace-pre-wrap">{step.output}</p>
          </div>
        )}

        <Separator />

        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
          <div>
            <p className="text-xs text-muted-foreground">Input Tokens</p>
            <p className="font-medium">{step.input_tokens.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Output Tokens</p>
            <p className="font-medium">{step.output_tokens.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Cost</p>
            {step.cost_usd == null ? (
              <p className="font-medium text-muted-foreground">Unavailable</p>
            ) : (
              <p className="font-medium">${step.cost_usd.toFixed(4)}</p>
            )}
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Duration</p>
            <p className="font-medium">{step.duration_seconds.toFixed(2)}s</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Recursion Depth</p>
            <p className="font-medium">{step.recursion_depth}</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
