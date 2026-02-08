"""Real FastAPI server for the DeepPDCFR solver API.

Loads the trained V2 model on startup and serves Nash-equilibrium strategies
through the same API contract as the mock server.

Run with::

    uvicorn api.server:app --port 8000

Then open http://localhost:8000/docs for Swagger UI.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.converters import (
    build_card_config,
    build_response,
    build_tree_config,
    convert_history,
)
from api.schemas import (
    ErrorDetail,
    HealthResponse,
    SolveRequest,
    SolveResponse,
    parse_board_cards,
)
from deeppdcfr.card_tools import card_tools
from deeppdcfr.query import SolverQuery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals (populated at startup)
# ---------------------------------------------------------------------------

_solver: SolverQuery | None = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _PROJECT_ROOT / (os.environ.get("CONFIG") or "configs/NLHEGeneralized.yaml")

# Model kwargs extracted from config (only architecture-related keys)
_MODEL_KWARG_KEYS = {
    "card_embed_dim",
    "situation_hidden",
    "combo_hidden",
    "fusion_hidden",
    "fusion_residual_blocks",
}


def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _init_solver(config_path: Path | None = None) -> SolverQuery:
    config_path = config_path or _DEFAULT_CONFIG

    cfg = _load_config(config_path)
    model_kwargs = {k: cfg[k] for k in _MODEL_KWARG_KEYS if k in cfg}

    # Resolve model dir from config's save_dir
    model_dir = _PROJECT_ROOT / cfg.get("save_dir", "models/NLHEGeneralized")

    # Use CPU for inference by default (fast enough for single queries,
    # avoids GPU memory contention with training)
    device = cfg.get("inference_device", "cpu")

    logger.info("Loading model from %s (device=%s) with kwargs: %s", model_dir, device, model_kwargs)
    return SolverQuery(
        model_path=str(model_dir),
        device=device,
        **model_kwargs,
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _solver
    _solver = _init_solver()
    logger.info("Solver model loaded successfully")
    yield
    _solver = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DeepPDCFR Solver API",
    description=(
        "REST API for querying Nash-equilibrium strategies in No-Limit Hold'em. "
        "Uses PioSOLVER syntax for bet sizes and hand ranges."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_solver is not None,
        version="0.1.0",
    )


@app.post(
    "/v1/solve",
    response_model=SolveResponse,
    responses={
        422: {"description": "Validation error", "model": ErrorDetail},
        500: {"description": "Internal error", "model": ErrorDetail},
    },
    summary="Solve a game state",
    tags=["Solver"],
)
async def solve(req: SolveRequest) -> SolveResponse:
    """Compute Nash-equilibrium strategy for the given game state."""
    if _solver is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_not_loaded", "message": "Solver model is not loaded"},
        )

    try:
        # Build configs from request
        tree_config = build_tree_config(req)
        card_config = build_card_config(req)

        # Parse board into card IDs
        board_strs = parse_board_cards(req.board)
        board_ids = [card_tools.card_to_id(c) for c in board_strs]

        # Convert betting history
        engine_history, current_pot = convert_history(
            req.betting_history,
            req.starting_pot,
            req.effective_stack,
        )

        # Query solver
        result = _solver.query(
            board=board_ids,
            betting_history=engine_history if engine_history else None,
            tree_config=tree_config,
            card_config=card_config,
        )

        # Build response
        return build_response(req, result, current_pot)

    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid_request", "message": str(e)},
        )
    except Exception as e:
        logger.exception("Solver error")
        raise HTTPException(
            status_code=500,
            detail={"error": "solver_error", "message": str(e)},
        )
