import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { fn } from "storybook/test";
import { Timeline } from "./timeline";
import type { TraceStep } from "@/lib/api";

const sampleSteps: TraceStep[] = [
  { index: 0, action_type: "inspect", code: "len(content)", output: "4200", input_tokens: 200, output_tokens: 50, cost_usd: null, duration_seconds: 0.8, recursion_depth: 0 },
  { index: 1, action_type: "inspect", code: "content[:500]", output: "...", input_tokens: 250, output_tokens: 80, cost_usd: null, duration_seconds: 1.2, recursion_depth: 0 },
  { index: 2, action_type: "subcall", code: null, output: null, input_tokens: 300, output_tokens: 100, cost_usd: null, duration_seconds: 2.5, recursion_depth: 1 },
  { index: 3, action_type: "final", code: null, output: "The document describes...", input_tokens: 400, output_tokens: 200, cost_usd: null, duration_seconds: 1.0, recursion_depth: 0 },
];

const meta: Meta<typeof Timeline> = {
  title: "Trace/Timeline",
  component: Timeline,
  args: { steps: sampleSteps, onSelect: fn() },
};

export default meta;
type Story = StoryObj<typeof Timeline>;

export const Default: Story = {};
export const WithSelection: Story = { args: { selectedIndex: 1 } };
export const SingleStep: Story = {
  args: {
    steps: [sampleSteps[3]],
  },
};
