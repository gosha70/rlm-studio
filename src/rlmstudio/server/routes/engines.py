"""Engine availability — ``GET /api/engines``.

Lets the frontend enable or disable the ``rlm_official`` option (Compare slot
picker, Chat Provider mode select) and show *why* it is unavailable, instead
of failing a run later.  Availability here is package-level (is the
``interop`` extra installed); per-provider mapping errors surface as a 400
on the run itself.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import EnginesResponse, EngineStatus

router = APIRouter()


@router.get("/api/engines")
async def get_engines(
    state: AppState = Depends(get_state),  # noqa: B008
) -> EnginesResponse:
    """Report which third-party engines can run in this process."""
    available, reason, version = state.rlm_engine_availability()
    return EnginesResponse(
        rlm_official=EngineStatus(available=available, reason=reason, version=version),
    )
