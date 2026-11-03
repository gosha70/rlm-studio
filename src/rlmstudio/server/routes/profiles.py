"""Run profile management endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException

from rlmstudio.application.sandbox_vars import RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS
from rlmstudio.server.dependencies import AppState, get_state
from rlmstudio.server.models import (
    BudgetConfig,
    RunProfile,
    RunProfileCreate,
    RunProfileUpdate,
    RuntimeSettings,
)

router = APIRouter()

# Built-in profiles (read-only, always available)
BUILTIN_PROFILES: list[RunProfile] = [
    RunProfile(
        id="builtin-fast",
        name="Fast & cheap",
        description="Low-cost, fast responses using Direct mode with conservative token limits.",
        strategy="direct",
        is_builtin=True,
        runtime_settings=RuntimeSettings(
            temperature=0.5, top_p=1.0, max_output_tokens=1000, timeout_seconds=15
        ),
        budget=BudgetConfig(
            max_steps=8,
            max_tokens=20000,
            max_cost_usd=0.5,
            max_time_seconds=15,
            max_recursion_depth=3,
        ),
    ),
    RunProfile(
        id="builtin-accurate",
        name="Accurate",
        description="High-quality responses with lower temperature for precision.",
        strategy="direct",
        is_builtin=True,
        runtime_settings=RuntimeSettings(
            temperature=0.2, top_p=0.95, max_output_tokens=4096, timeout_seconds=30
        ),
        budget=BudgetConfig(
            max_steps=16,
            max_tokens=50000,
            max_cost_usd=2.0,
            max_time_seconds=RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
            max_recursion_depth=5,
        ),
    ),
    RunProfile(
        id="builtin-rlm-deep",
        name="RLM deep",
        description="Deep recursive reasoning with high step budget for complex problems.",
        strategy="rlm",
        is_builtin=True,
        runtime_settings=RuntimeSettings(
            temperature=0.4, top_p=1.0, max_output_tokens=4096, timeout_seconds=120
        ),
        budget=BudgetConfig(
            max_steps=32,
            max_tokens=100000,
            max_cost_usd=5.0,
            max_time_seconds=RLM_DEFAULT_WALL_CLOCK_BUDGET_SECONDS,
            max_recursion_depth=8,
        ),
    ),
    RunProfile(
        id="builtin-rag",
        name="RAG retrieval",
        description="Retrieval-augmented generation with moderate token budget for grounded answers.",
        strategy="rag",
        is_builtin=True,
        runtime_settings=RuntimeSettings(
            temperature=0.3, top_p=0.95, max_output_tokens=4096, timeout_seconds=120
        ),
        budget=BudgetConfig(
            max_steps=8,
            max_tokens=50000,
            max_cost_usd=2.0,
            max_time_seconds=120,
            max_recursion_depth=3,
        ),
    ),
]


@router.get("/api/profiles")
async def list_profiles(
    state: AppState = Depends(get_state),
) -> list[RunProfile]:
    """List all profiles (builtins + user-created)."""
    return list(BUILTIN_PROFILES) + list(state.user_profiles)


@router.post("/api/profiles", status_code=201)
async def create_profile(
    req: RunProfileCreate,
    state: AppState = Depends(get_state),
) -> RunProfile:
    """Create a new user profile."""
    all_profiles = list(BUILTIN_PROFILES) + list(state.user_profiles)
    for existing in all_profiles:
        if existing.name.lower() == req.name.strip().lower():
            raise HTTPException(
                status_code=409,
                detail=f"Profile with name '{req.name}' already exists",
            )

    profile = RunProfile(
        id=str(uuid.uuid4()),
        name=req.name,
        description=req.description,
        strategy=req.strategy,
        default_provider=req.default_provider,
        providers_enabled=req.providers_enabled,
        runtime_settings=req.runtime_settings,
        budget=req.budget,
        system_prompts=req.system_prompts,
        prompt_template_name=req.prompt_template_name,
        is_builtin=False,
    )
    state.user_profiles.append(profile)
    state.save_config()
    return profile


@router.put("/api/profiles/{profile_id}")
async def update_profile(
    profile_id: str,
    req: RunProfileUpdate,
    state: AppState = Depends(get_state),
) -> RunProfile:
    """Update a user profile. Cannot modify builtins."""
    # Check builtins
    for bp in BUILTIN_PROFILES:
        if bp.id == profile_id:
            raise HTTPException(status_code=400, detail="Cannot modify built-in profiles")

    # Check name uniqueness if renaming
    if req.name is not None:
        all_profiles = list(BUILTIN_PROFILES) + list(state.user_profiles)
        for existing in all_profiles:
            if existing.id != profile_id and existing.name.lower() == req.name.strip().lower():
                raise HTTPException(
                    status_code=409,
                    detail=f"Profile with name '{req.name}' already exists",
                )

    # Find user profile.  Mutate and save under the config lock so an edit
    # cannot interleave with a prompt-refinement accept or rollback (#78
    # FR-9/FR-10a), which hold the same lock across their own check and save.
    with state._config_lock:
        for profile in state.user_profiles:
            if profile.id == profile_id:
                if req.name is not None:
                    profile.name = req.name
                if req.description is not None:
                    profile.description = req.description
                if req.strategy is not None:
                    profile.strategy = req.strategy
                if req.default_provider is not None:
                    profile.default_provider = req.default_provider
                if req.providers_enabled is not None:
                    profile.providers_enabled = req.providers_enabled
                if req.runtime_settings is not None:
                    profile.runtime_settings = req.runtime_settings
                if req.budget is not None:
                    profile.budget = req.budget
                if req.system_prompts is not None:
                    profile.system_prompts = req.system_prompts
                if req.prompt_template_name is not None:
                    profile.prompt_template_name = req.prompt_template_name
                state.save_config()
                return profile

    raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")


@router.delete("/api/profiles/{profile_id}", status_code=204)
async def delete_profile(
    profile_id: str,
    state: AppState = Depends(get_state),
) -> None:
    """Delete a user profile. Cannot delete builtins or profiles in use."""
    for bp in BUILTIN_PROFILES:
        if bp.id == profile_id:
            raise HTTPException(status_code=400, detail="Cannot delete built-in profiles")

    referencing = [cp.name for cp in state.config.chat_providers if cp.profile_id == profile_id]
    if referencing:
        names = ", ".join(referencing)
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete: profile is used by Chat Providers: {names}",
        )

    with state._config_lock:
        for i, profile in enumerate(state.user_profiles):
            if profile.id == profile_id:
                state.user_profiles.pop(i)
                state.save_config()
                return

    raise HTTPException(status_code=404, detail=f"Profile not found: {profile_id}")
