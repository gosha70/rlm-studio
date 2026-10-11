# Copyright (c) 2026 EGOGE
# SPDX-License-Identifier: Apache-2.0

"""Shared UI-facing services and data for RLM Studio.

Historical note: this package once hosted a Streamlit prototype. That UI
was removed in favour of the Next.js RLM Studio front-end (``frontend/``,
served by FastAPI); what remains here is pure-Python code the server
depends on — ``services/`` (secret / profile stores, chat manager,
analytics) and ``data/`` (the provider catalog).
"""
