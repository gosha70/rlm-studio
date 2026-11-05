"""Combined quality scoring engine for provider recommendations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rlmstudio.application.sandbox_vars import EPHEMERAL_CP_PREFIX
from rlmstudio.application.services.outcome_classifier import classify_execution_outcome
from rlmstudio.server.models import ProviderQualityScore

if TYPE_CHECKING:
    from rlmstudio.server.dependencies import AppState

# Minimum data to make a recommendation
_MIN_RATINGS = 3
_MIN_JUDGE_SCORES = 1


def _collect_metrics_from_messages(
    state: AppState, session_id: str, cp_id: str
) -> tuple[list[float], list[float]]:
    """Collect cost and latency values from session messages for a provider.

    Reads from persisted session messages (metrics field) so data survives
    restarts, then falls back to in-memory ExecutionRecords.

    Non-usable runs (failures, degraded warnings) are excluded so they
    don't skew provider cost/speed averages.
    """
    costs: list[float] = []
    speeds: list[float] = []

    session = state.sessions.get(session_id)
    if session:
        for msg in session.messages:
            if msg.get("role") == "assistant" and msg.get("chat_provider_id") == cp_id:
                metrics = msg.get("metrics")
                if isinstance(metrics, dict):
                    # Check usability via the classifier
                    content = msg.get("content", "")
                    msg_success = bool(metrics.get("success", True))
                    msg_error = metrics.get("error")
                    outcome = classify_execution_outcome(msg_success, msg_error, content)
                    if not outcome.is_usable:
                        continue

                    cost = metrics.get("cost_usd", 0.0)
                    elapsed = metrics.get("elapsed_seconds", 0.0)
                    if cost or elapsed:
                        costs.append(float(cost))
                        speeds.append(float(elapsed))

    # Fallback: also check in-memory executions for data not yet in messages
    for rec in state.executions.values():
        if rec.session_id == session_id and rec.chat_provider_id == cp_id and rec.result:
            eid = rec.execution_id
            # Skip if already captured from messages
            already_captured = session and any(
                msg.get("execution_id") == eid
                and msg.get("role") == "assistant"
                and isinstance(msg.get("metrics"), dict)
                for msg in session.messages
            )
            if not already_captured:
                # Check usability via the classifier
                rec_success = bool(rec.result.get("success", True))
                rec_error = rec.result.get("error")
                rec_answer = rec.result.get("answer", "")
                outcome = classify_execution_outcome(rec_success, rec_error, rec_answer)
                if not outcome.is_usable:
                    continue

                costs.append(float(rec.result.get("total_cost", rec.result.get("cost_used", 0))))
                speeds.append(
                    float(rec.result.get("elapsed_time", rec.result.get("elapsed_seconds", 0)))
                )

    return costs, speeds


def _collect_exec_ids_from_messages(state: AppState, session_id: str, cp_id: str) -> set[str]:
    """Get all execution_ids for a provider in a session from persisted messages."""
    result: set[str] = set()
    session = state.sessions.get(session_id)
    if session:
        for msg in session.messages:
            if (
                msg.get("role") == "assistant"
                and msg.get("chat_provider_id") == cp_id
                and msg.get("execution_id")
            ):
                result.add(msg["execution_id"])
    # Also include in-memory executions
    for eid, rec in state.executions.items():
        if rec.session_id == session_id and rec.chat_provider_id == cp_id:
            result.add(eid)
    return result


def _collect_cp_ids_from_session(state: AppState, session_id: str) -> set[str]:
    """Get all chat_provider_ids that appear in a session."""
    cp_ids: set[str] = set()
    session = state.sessions.get(session_id)
    if session:
        for msg in session.messages:
            cpid = msg.get("chat_provider_id")
            if cpid and msg.get("role") == "assistant":
                cp_ids.add(cpid)
    # Also check in-memory executions
    for rec in state.executions.values():
        if rec.session_id == session_id and rec.chat_provider_id:
            cp_ids.add(rec.chat_provider_id)
    return cp_ids


def _is_tuner_provider(state: AppState, session_id: str, cp_id: str) -> bool:
    """True for an LLM Tuner's temporary Chat Provider.

    While it exists it is marked ephemeral; once gone (they are never saved),
    the name it ran under is still on its messages and executions.
    """
    cp = state.get_chat_provider(cp_id)
    if cp is not None:
        return cp.ephemeral or cp.name.startswith(EPHEMERAL_CP_PREFIX)
    names: list[str] = []
    session = state.sessions.get(session_id)
    if session:
        names.extend(
            msg.get("chat_provider_name") or ""
            for msg in session.messages
            if msg.get("chat_provider_id") == cp_id
        )
    names.extend(
        rec.chat_provider_name or ""
        for rec in state.executions.values()
        if rec.session_id == session_id and rec.chat_provider_id == cp_id
    )
    return any(name.startswith(EPHEMERAL_CP_PREFIX) for name in names)


class QualityEngine:
    WEIGHTS = {
        "manual_thumb": 0.20,
        "manual_best_pick": 0.20,
        "judge_pointwise": 0.30,
        "judge_pairwise": 0.10,
        "cost": 0.10,
        "speed": 0.10,
    }

    def compute_scores(self, session_id: str, state: AppState) -> dict[str, ProviderQualityScore]:
        """Compute per-provider quality scores for a session."""
        evals = state.evaluations

        # Filter to this session
        thumbs = [r for r in evals["thumb_ratings"] if r["session_id"] == session_id]
        picks = [p for p in evals["best_picks"] if p["session_id"] == session_id]
        judge_scores = [s for s in evals["judge_scores"] if s["session_id"] == session_id]
        judge_pairs = [p for p in evals["judge_pairwise"] if p["session_id"] == session_id]

        # Collect all chat_provider_ids from persisted messages + in-memory
        cp_ids = _collect_cp_ids_from_session(state, session_id)

        # Also include any from evaluations
        for t in thumbs:
            cp_ids.add(t["chat_provider_id"])
        for s in judge_scores:
            cp_ids.add(s["chat_provider_id"])

        # LLM Tuner runs are not Chat Providers a user can pick, so they get
        # no card and are never recommended (the Dashboard's other metrics
        # leave them out too).
        cp_ids = {cp_id for cp_id in cp_ids if not _is_tuner_provider(state, session_id, cp_id)}

        # Pre-compute per-provider cost/speed for normalization
        all_costs: dict[str, float] = {}
        all_speeds: dict[str, float] = {}
        for cp_id in cp_ids:
            costs, speeds = _collect_metrics_from_messages(state, session_id, cp_id)
            all_costs[cp_id] = sum(costs) / len(costs) if costs else 0.0
            all_speeds[cp_id] = sum(speeds) / len(speeds) if speeds else 0.0

        # Normalization ranges for cost and speed (lower is better → invert)
        cost_values = [v for v in all_costs.values() if v > 0]
        speed_values = [v for v in all_speeds.values() if v > 0]
        max_cost = max(cost_values) if cost_values else 1.0
        max_speed = max(speed_values) if speed_values else 1.0

        result: dict[str, ProviderQualityScore] = {}

        for cp_id in cp_ids:
            cp = state.get_chat_provider(cp_id)
            cp_name = cp.name if cp else cp_id

            # Thumb ratings
            cp_thumbs = [t for t in thumbs if t["chat_provider_id"] == cp_id]
            thumb_up = sum(1 for t in cp_thumbs if t["rating"] == "up")
            thumb_down = sum(1 for t in cp_thumbs if t["rating"] == "down")
            total_thumbs = thumb_up + thumb_down
            thumb_score = thumb_up / total_thumbs if total_thumbs > 0 else 0.0

            # Best picks — use persisted execution_ids
            winner_exec_ids = {p["winner_execution_id"] for p in picks}
            cp_exec_ids = _collect_exec_ids_from_messages(state, session_id, cp_id)
            best_picks = len(winner_exec_ids & cp_exec_ids)
            total_picks = len(picks)
            pick_rate = best_picks / total_picks if total_picks > 0 else 0.0

            # Judge pointwise average
            cp_judge = [s for s in judge_scores if s["chat_provider_id"] == cp_id]
            judge_avg: float | None = None
            if cp_judge:
                judge_avg = sum(s["overall_score"] for s in cp_judge) / len(cp_judge)

            # Judge pairwise wins
            pairwise_wins = 0
            pairwise_total = 0
            for p in judge_pairs:
                if p["execution_id_a"] in cp_exec_ids:
                    pairwise_total += 1
                    if p["winner"] == "a":
                        pairwise_wins += 1
                elif p["execution_id_b"] in cp_exec_ids:
                    pairwise_total += 1
                    if p["winner"] == "b":
                        pairwise_wins += 1

            # Combined score (0-1 scale)
            combined = 0.0
            has_data = False

            if total_thumbs > 0:
                combined += self.WEIGHTS["manual_thumb"] * thumb_score
                has_data = True

            if total_picks > 0:
                combined += self.WEIGHTS["manual_best_pick"] * pick_rate
                has_data = True

            if judge_avg is not None:
                # Normalize from 1-5 scale to 0-1
                combined += self.WEIGHTS["judge_pointwise"] * ((judge_avg - 1.0) / 4.0)
                has_data = True

            if pairwise_total > 0:
                combined += self.WEIGHTS["judge_pairwise"] * (pairwise_wins / pairwise_total)
                has_data = True

            # Performance signals: lower cost/speed is better → invert
            avg_cost = all_costs.get(cp_id, 0.0)
            avg_speed = all_speeds.get(cp_id, 0.0)
            if avg_cost > 0 and max_cost > 0:
                cost_score = 1.0 - (avg_cost / max_cost)
                combined += self.WEIGHTS["cost"] * cost_score
                has_data = True
            if avg_speed > 0 and max_speed > 0:
                speed_score = 1.0 - (avg_speed / max_speed)
                combined += self.WEIGHTS["speed"] * speed_score
                has_data = True

            if not has_data:
                combined = 0.0

            result[cp_id] = ProviderQualityScore(
                chat_provider_id=cp_id,
                chat_provider_name=cp_name,
                thumb_up=thumb_up,
                thumb_down=thumb_down,
                thumb_score=round(thumb_score, 3),
                best_picks=best_picks,
                pick_rate=round(pick_rate, 3),
                judge_avg_score=round(judge_avg, 2) if judge_avg is not None else None,
                combined_score=round(combined, 3),
            )

        return result

    def get_recommendation(self, scores: dict[str, ProviderQualityScore]) -> tuple[str | None, str]:
        """Pick the best provider from computed scores.

        Returns (provider_id, reason). Returns (None, reason) if insufficient data.
        """
        if not scores:
            return None, "No evaluation data available"

        # Check minimum data threshold — thumbs, best-picks, and judge all count
        total_ratings = sum(s.thumb_up + s.thumb_down for s in scores.values())
        total_picks = sum(s.best_picks for s in scores.values())
        total_judge = sum(1 for s in scores.values() if s.judge_avg_score is not None)
        total_manual = total_ratings + total_picks

        if total_manual < _MIN_RATINGS and total_judge < _MIN_JUDGE_SCORES:
            return None, (
                f"Need at least {_MIN_RATINGS} manual ratings/picks or "
                f"{_MIN_JUDGE_SCORES} judge score to recommend"
            )

        # Primary key: judge_avg_score (explicit quality signal).
        # Providers without a judge score sort below those with one.
        # Secondary key: combined_score (speed/cost tiebreaker).
        def _sort_key(s: ProviderQualityScore) -> tuple[float, float]:
            return (s.judge_avg_score if s.judge_avg_score is not None else -1.0, s.combined_score)

        best = max(scores.values(), key=_sort_key)
        if best.combined_score <= 0 and best.judge_avg_score is None:
            return None, "No positive quality signals yet"

        parts: list[str] = []
        if best.thumb_score > 0:
            parts.append(f"{best.thumb_up}/{best.thumb_up + best.thumb_down} thumbs up")
        if best.judge_avg_score is not None:
            parts.append(f"judge score {best.judge_avg_score}/5")
        if best.best_picks > 0:
            parts.append(f"picked best {best.best_picks} times")

        reason = f"Best quality: {', '.join(parts)}" if parts else "Highest combined score"
        return best.chat_provider_id, reason
