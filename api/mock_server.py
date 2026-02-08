"""Mock FastAPI server for the DeepPDCFR solver API.

Run with::

    uvicorn api.mock_server:app --port 8000 --reload

Then open http://localhost:8000/docs for Swagger UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    ActionInfo,
    ErrorDetail,
    HandStrategy,
    HealthResponse,
    SolveRequest,
    SolveResponse,
)

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
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mock scenario: OOP opening on flop Ah Kd Qc
#   pot = 20bb, effective_stack = 100bb, bet_sizes = default (33, 67, a)
#   oop_range = AA,AKs,AKo,KK,QQ:0.5,JJ-99,AQs-ATs,KQs
#   betting_history = null (root of street, OOP acts first)
#   Board blockers: Ah, Kd, Qc
#
# Every combo below is in the OOP range and not blocked by the board.
# Strategies are [Check, Bet 33%, Bet 67%, All-in] — all sum to 1.0.
# No Fold at the root (OOP can only check or bet when opening action).
# ---------------------------------------------------------------------------

MOCK_ACTIONS = [
    ActionInfo(name="Check", type="check", amount_big_blinds=0, amount_percent=0),
    ActionInfo(name="Bet 33%", type="bet", amount_big_blinds=6.6, amount_percent=33.0),
    ActionInfo(name="Bet 67%", type="bet", amount_big_blinds=13.4, amount_percent=67.0),
    ActionInfo(name="All-in", type="allin", amount_big_blinds=100.0, amount_percent=500.0),
]

MOCK_COMBOS = [
    # --- AA (3 combos) — trips aces, mix trap + value bet ---
    HandStrategy(hand="AcAd", hand_id=1320, strategy=[0.15, 0.20, 0.50, 0.15]),
    HandStrategy(hand="AcAs", hand_id=1322, strategy=[0.15, 0.20, 0.50, 0.15]),
    HandStrategy(hand="AdAs", hand_id=1324, strategy=[0.15, 0.20, 0.50, 0.15]),

    # --- AKs (2 combos) — two pair aces and kings ---
    HandStrategy(hand="AcKc", hand_id=1301, strategy=[0.20, 0.45, 0.30, 0.05]),
    HandStrategy(hand="AsKs", hand_id=1319, strategy=[0.20, 0.45, 0.30, 0.05]),

    # --- AKo (6 combos) — two pair aces and kings ---
    HandStrategy(hand="AcKh", hand_id=1312, strategy=[0.25, 0.40, 0.30, 0.05]),
    HandStrategy(hand="AcKs", hand_id=1316, strategy=[0.25, 0.40, 0.30, 0.05]),
    HandStrategy(hand="AdKc", hand_id=1302, strategy=[0.25, 0.40, 0.30, 0.05]),
    HandStrategy(hand="AdKh", hand_id=1313, strategy=[0.25, 0.40, 0.30, 0.05]),
    HandStrategy(hand="AdKs", hand_id=1317, strategy=[0.25, 0.40, 0.30, 0.05]),
    HandStrategy(hand="AsKc", hand_id=1304, strategy=[0.25, 0.40, 0.30, 0.05]),
    HandStrategy(hand="AsKh", hand_id=1315, strategy=[0.25, 0.40, 0.30, 0.05]),

    # --- KK (3 combos) — set of kings ---
    HandStrategy(hand="KcKh", hand_id=1299, strategy=[0.20, 0.25, 0.40, 0.15]),
    HandStrategy(hand="KcKs", hand_id=1300, strategy=[0.20, 0.25, 0.40, 0.15]),
    HandStrategy(hand="KhKs", hand_id=1311, strategy=[0.20, 0.25, 0.40, 0.15]),

    # --- QQ:0.5 (3 combos) — set of queens ---
    HandStrategy(hand="QdQh", hand_id=1271, strategy=[0.10, 0.20, 0.50, 0.20]),
    HandStrategy(hand="QdQs", hand_id=1272, strategy=[0.10, 0.20, 0.50, 0.20]),
    HandStrategy(hand="QhQs", hand_id=1281, strategy=[0.10, 0.20, 0.50, 0.20]),

    # --- AQs (2 combos) — two pair aces and queens ---
    HandStrategy(hand="AdQd", hand_id=1278, strategy=[0.25, 0.40, 0.25, 0.10]),
    HandStrategy(hand="AsQs", hand_id=1297, strategy=[0.25, 0.40, 0.25, 0.10]),

    # --- AJs (3 combos) — top pair + jack kicker ---
    HandStrategy(hand="AcJc", hand_id=1217, strategy=[0.45, 0.35, 0.15, 0.05]),
    HandStrategy(hand="AdJd", hand_id=1232, strategy=[0.45, 0.35, 0.15, 0.05]),
    HandStrategy(hand="AsJs", hand_id=1259, strategy=[0.45, 0.35, 0.15, 0.05]),

    # --- ATs (3 combos) — top pair + ten kicker ---
    HandStrategy(hand="AcTc", hand_id=1151, strategy=[0.55, 0.30, 0.12, 0.03]),
    HandStrategy(hand="AdTd", hand_id=1170, strategy=[0.55, 0.30, 0.12, 0.03]),
    HandStrategy(hand="AsTs", hand_id=1205, strategy=[0.55, 0.30, 0.12, 0.03]),

    # --- KQs (2 combos) — two pair kings and queens ---
    HandStrategy(hand="KhQh", hand_id=1284, strategy=[0.30, 0.35, 0.30, 0.05]),
    HandStrategy(hand="KsQs", hand_id=1293, strategy=[0.30, 0.35, 0.30, 0.05]),

    # --- JJ (6 combos) — underpair ---
    HandStrategy(hand="JcJd", hand_id=1206, strategy=[0.65, 0.22, 0.10, 0.03]),
    HandStrategy(hand="JcJh", hand_id=1207, strategy=[0.65, 0.22, 0.10, 0.03]),
    HandStrategy(hand="JcJs", hand_id=1208, strategy=[0.65, 0.22, 0.10, 0.03]),
    HandStrategy(hand="JdJh", hand_id=1221, strategy=[0.65, 0.22, 0.10, 0.03]),
    HandStrategy(hand="JdJs", hand_id=1222, strategy=[0.65, 0.22, 0.10, 0.03]),
    HandStrategy(hand="JhJs", hand_id=1235, strategy=[0.65, 0.22, 0.10, 0.03]),

    # --- TT (6 combos) — underpair ---
    HandStrategy(hand="TcTd", hand_id=1136, strategy=[0.70, 0.18, 0.09, 0.03]),
    HandStrategy(hand="TcTh", hand_id=1137, strategy=[0.70, 0.18, 0.09, 0.03]),
    HandStrategy(hand="TcTs", hand_id=1138, strategy=[0.70, 0.18, 0.09, 0.03]),
    HandStrategy(hand="TdTh", hand_id=1155, strategy=[0.70, 0.18, 0.09, 0.03]),
    HandStrategy(hand="TdTs", hand_id=1156, strategy=[0.70, 0.18, 0.09, 0.03]),
    HandStrategy(hand="ThTs", hand_id=1173, strategy=[0.70, 0.18, 0.09, 0.03]),

    # --- 99 (6 combos) — underpair ---
    HandStrategy(hand="9c9d", hand_id=1050, strategy=[0.75, 0.15, 0.08, 0.02]),
    HandStrategy(hand="9c9h", hand_id=1051, strategy=[0.75, 0.15, 0.08, 0.02]),
    HandStrategy(hand="9c9s", hand_id=1052, strategy=[0.75, 0.15, 0.08, 0.02]),
    HandStrategy(hand="9d9h", hand_id=1073, strategy=[0.75, 0.15, 0.08, 0.02]),
    HandStrategy(hand="9d9s", hand_id=1074, strategy=[0.75, 0.15, 0.08, 0.02]),
    HandStrategy(hand="9h9s", hand_id=1095, strategy=[0.75, 0.15, 0.08, 0.02]),
]

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
    """Returns service status, model availability, and version."""
    return HealthResponse(status="ok", model_loaded=True, version="0.1.0")


@app.post(
    "/v1/solve",
    response_model=SolveResponse,
    responses={
        422: {
            "description": "Validation error",
            "model": ErrorDetail,
        },
    },
    summary="Solve a game state",
    tags=["Solver"],
)
async def solve(req: SolveRequest) -> SolveResponse:
    """Compute Nash-equilibrium strategy for the given game state.

    Currently returns **mock data** — all 46 combos from the example
    OOP range ``AA,AKs,AKo,KK,QQ:0.5,JJ-99,AQs-ATs,KQs`` on board
    ``Ah Kd Qc``.  The real solver will be wired in later without
    changing the API shape.
    """
    return SolveResponse(
        player=req.player,
        board=req.board,
        pot=req.starting_pot,
        effective_stack=req.effective_stack,
        num_combos=len(MOCK_COMBOS),
        actions=MOCK_ACTIONS,
        combos=MOCK_COMBOS,
    )
