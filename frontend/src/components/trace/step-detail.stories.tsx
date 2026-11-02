import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { StepDetail } from "./step-detail";
import type { TraceStep } from "@/lib/api";

/**
 * Covers the three cost states the backend can now produce (issue #94):
 * the model is unpriced ("Unavailable"), the row genuinely cost nothing
 * (subcall / runtime fingerprint), and a priced step.
 */
const baseStep: TraceStep = {
  index: 0,
  action_type: "inspect",
  code: "len(content)",
  output: "4200",
  input_tokens: 200,
  output_tokens: 50,
  cost_usd: null,
  duration_seconds: 0.8,
  recursion_depth: 0,
};

const meta: Meta<typeof StepDetail> = {
  title: "Trace/StepDetail",
  component: StepDetail,
};

export default meta;
type Story = StoryObj<typeof StepDetail>;

export const CostUnavailable: Story = {
  args: { step: { ...baseStep, cost_usd: null } },
};

export const GenuineZeroCost: Story = {
  args: {
    step: {
      ...baseStep,
      action_type: "subcall",
      input_tokens: 0,
      output_tokens: 0,
      cost_usd: 0,
    },
  },
};

export const PositiveCost: Story = {
  args: { step: { ...baseStep, cost_usd: 0.0075 } },
};
