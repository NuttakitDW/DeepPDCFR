"""Pydantic models defining the API contract for the DeepPDCFR solver.

Uses PioSOLVER syntax conventions for bet sizes and hand ranges.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_RANKS = set("23456789TJQKA")
VALID_SUITS = set("cdhs")
VALID_CARDS = {f"{r}{s}" for r in VALID_RANKS for s in VALID_SUITS}

RANK_ORDER = "23456789TJQKA"
RANK_INDEX = {r: i for i, r in enumerate(RANK_ORDER)}

# Pio bet-size token pattern:
#   N or N%  (pot-relative)
#   Nx       (previous-bet-relative)
#   Nc       (constant chips)
#   e, Ne, Ne%  (geometric)
#   a        (all-in)
BET_TOKEN_RE = re.compile(
    r"^(?:"
    r"a"                                  # all-in
    r"|(?:\d+(?:\.\d+)?)?e(?:\d+(?:\.\d+)?%?)?"  # geometric: e, 2e, 3e200%
    r"|\d+(?:\.\d+)?x"                   # prev-bet relative: 2.5x
    r"|\d+(?:\.\d+)?c"                   # constant chips: 100c
    r"|\d+(?:\.\d+)?%?"                  # pot-relative: 33, 67%
    r")$"
)

# Pio range token patterns
_RANK = r"[2-9TJQKA]"
_SUIT = r"[cdhs]"
_WEIGHT = r"(?::(?:0(?:\.\d+)?|1(?:\.0+)?))"  # :0.5, :1, :0.75 etc.

# Specific combo: AhKd
SPECIFIC_COMBO_RE = re.compile(
    rf"^{_RANK}{_SUIT}{_RANK}{_SUIT}{_WEIGHT}?$"
)
# Pair: AA, AA:0.5
PAIR_RE = re.compile(
    rf"^({_RANK})\1{_WEIGHT}?$"
)
# Suited/offsuit with optional +/dash: AKs, AKs+, ATs-A6s, AK, AKo
HAND_RE = re.compile(
    rf"^{_RANK}{_RANK}[so]?{_WEIGHT}?$"
)
PLUS_RE = re.compile(
    rf"^{_RANK}{_RANK}[so]?\+{_WEIGHT}?$"
)
DASH_RE = re.compile(
    rf"^{_RANK}{_RANK}[so]?-{_RANK}{_RANK}[so]?{_WEIGHT}?$"
)

# Pair ranges: 88+, 88-55
PAIR_PLUS_RE = re.compile(
    rf"^({_RANK})\1\+{_WEIGHT}?$"
)
PAIR_DASH_RE = re.compile(
    rf"^({_RANK})\1-({_RANK})\2{_WEIGHT}?$"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_board_cards(board: str) -> list[str]:
    """Parse board string into list of 2-char card strings.

    Accepts both ``"AhKdQc"`` (no spaces) and ``"Ah Kd Qc"`` (space-separated).
    """
    board = board.strip()
    if " " in board:
        return board.split()
    # No spaces – split every 2 chars
    if len(board) % 2 != 0:
        raise ValueError(f"Board string length must be even, got '{board}'")
    return [board[i : i + 2] for i in range(0, len(board), 2)]


def validate_card(card: str) -> str:
    """Validate a single 2-char card string like 'Ah'."""
    if len(card) != 2:
        raise ValueError(f"Card must be 2 characters, got '{card}'")
    if card[0] not in VALID_RANKS:
        raise ValueError(f"Invalid rank '{card[0]}' in card '{card}'")
    if card[1] not in VALID_SUITS:
        raise ValueError(f"Invalid suit '{card[1]}' in card '{card}'")
    return card


def validate_bet_size_string(s: str) -> str:
    """Validate a Pio-format bet size string like ``'33, 67, a'``."""
    tokens = [t.strip() for t in s.split(",")]
    for token in tokens:
        if not token:
            raise ValueError("Empty bet size token")
        if not BET_TOKEN_RE.match(token):
            raise ValueError(
                f"Invalid bet size token '{token}'. "
                "Expected: N, N%, Nx, Nc, e, Ne, Ne%, or a"
            )
    return s


def validate_range_string(s: str) -> str:
    """Validate a Pio-format range string like ``'AA,AKs,KK,QQ:0.5'``."""
    tokens = [t.strip() for t in s.split(",")]
    for token in tokens:
        if not token:
            raise ValueError("Empty range token")
        if any(
            pat.match(token)
            for pat in (
                SPECIFIC_COMBO_RE,
                PAIR_RE,
                PAIR_PLUS_RE,
                PAIR_DASH_RE,
                HAND_RE,
                PLUS_RE,
                DASH_RE,
            )
        ):
            continue
        raise ValueError(
            f"Invalid range token '{token}'. "
            "Expected Pio syntax: AA, AKs, AKo, AK, 88+, 88-55, ATs+, "
            "ATs-A6s, AhKd, AA:0.5, etc."
        )
    return s


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class BetSizes(BaseModel):
    """Bet sizing configuration in PioSOLVER syntax.

    Each field is a comma-separated string of size tokens.
    """

    oop_bet: str = Field(
        default="33, 67, a",
        description="OOP bet sizes (Pio syntax). e.g. '33, 67, a'",
        json_schema_extra={"example": "33, 67, a"},
    )
    oop_raise: str = Field(
        default="50, a",
        description="OOP raise sizes (Pio syntax). e.g. '50, a'",
        json_schema_extra={"example": "50, a"},
    )
    ip_bet: str = Field(
        default="33, 67, a",
        description="IP bet sizes (Pio syntax). e.g. '33, 67, a'",
        json_schema_extra={"example": "33, 67, a"},
    )
    ip_raise: str = Field(
        default="50, a",
        description="IP raise sizes (Pio syntax). e.g. '50, a'",
        json_schema_extra={"example": "50, a"},
    )

    @field_validator("oop_bet", "oop_raise", "ip_bet", "ip_raise")
    @classmethod
    def _validate_bet_sizes(cls, v: str) -> str:
        return validate_bet_size_string(v)


class HistoryAction(BaseModel):
    """A single action in the betting history.

    Each entry explicitly identifies the sequence order and which position
    acted, so the history is self-documenting.
    """

    order: int = Field(
        ...,
        ge=1,
        description="1-based sequence number of this action",
        json_schema_extra={"example": 1},
    )
    position: Literal["OOP", "IP"] = Field(
        ...,
        description="Which position made this action",
        json_schema_extra={"example": "OOP"},
    )
    action: Literal["check", "call", "fold", "bet", "raise", "allin", "deal"] = Field(
        ...,
        description="Action type",
        json_schema_extra={"example": "check"},
    )
    amount_percent: Optional[float] = Field(
        default=None,
        description="Bet/raise size as percentage of pot (required for bet/raise, omit otherwise)",
    )
    card: Optional[str] = Field(
        default=None,
        description="Card dealt (required for deal action only). e.g. '9h'",
    )

    @model_validator(mode="after")
    def _check_fields(self) -> "HistoryAction":
        if self.action in ("bet", "raise") and self.amount_percent is None:
            raise ValueError(
                f"'amount_percent' is required for action '{self.action}'"
            )
        if self.action == "deal" and self.card is None:
            raise ValueError("'card' is required for action 'deal'")
        if self.card is not None:
            validate_card(self.card)
        return self


class SolveRequest(BaseModel):
    """Request body for the ``POST /v1/solve`` endpoint.

    Uses PioSOLVER syntax for board, bet sizes, and ranges.
    """

    player: Literal["OOP", "IP"] = Field(
        ...,
        description="Which player's strategy to return",
        json_schema_extra={"example": "OOP"},
    )
    board: str = Field(
        ...,
        description=(
            'Board cards. Space-separated or concatenated. '
            'e.g. "Ah Kd Qc" or "AhKdQc"'
        ),
        json_schema_extra={"example": "Ah Kd Qc"},
    )
    effective_stack: int = Field(
        ...,
        gt=0,
        description="Effective stack size in big blinds (bb)",
        json_schema_extra={"example": 100},
    )
    starting_pot: int = Field(
        ...,
        gt=0,
        description="Pot size at the start of the current street in big blinds (bb)",
        json_schema_extra={"example": 20},
    )
    bet_sizes: Optional[BetSizes] = Field(
        default=None,
        description="Bet sizing configuration (Pio syntax). Defaults apply if omitted.",
    )
    betting_history: Optional[list[HistoryAction]] = Field(
        default=None,
        description=(
            "Betting actions to replay to reach the target node. "
            "Null or empty = root of the street (first to act). "
            "Each entry has order (1-based), position (OOP/IP), and action detail."
        ),
        json_schema_extra={
            "example": [
                {"order": 1, "position": "OOP", "action": "check"},
                {"order": 2, "position": "IP", "action": "bet", "amount_percent": 67},
            ]
        },
    )
    oop_range: Optional[str] = Field(
        default=None,
        description=(
            "OOP range in Pio syntax. e.g. 'AA,AKs,KK,QQ:0.5'. "
            "Null = all combos (uniform)."
        ),
        json_schema_extra={
            "example": "AA,AKs,AKo,KK,QQ:0.5,JJ-99,AQs-ATs,KQs"
        },
    )
    ip_range: Optional[str] = Field(
        default=None,
        description=(
            "IP range in Pio syntax. "
            "Null = all combos (uniform)."
        ),
        json_schema_extra={
            "example": "22+,A2s+,K9s+,Q9s+,J9s+,T8s+,97s+,87s,76s,65s,ATo+,KJo+"
        },
    )

    @field_validator("board")
    @classmethod
    def _validate_board(cls, v: str) -> str:
        cards = parse_board_cards(v)
        if len(cards) < 3 or len(cards) > 5:
            raise ValueError(
                f"Board must have 3-5 cards, got {len(cards)}: {cards}"
            )
        seen: set[str] = set()
        for card in cards:
            validate_card(card)
            if card in seen:
                raise ValueError(f"Duplicate card on board: '{card}'")
            seen.add(card)
        return v

    @field_validator("oop_range", "ip_range")
    @classmethod
    def _validate_range(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            validate_range_string(v)
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ActionInfo(BaseModel):
    """Description of an available action at a decision node."""

    name: str = Field(
        ...,
        description="Human-readable action name",
        json_schema_extra={"example": "Bet 33%"},
    )
    type: Literal["fold", "check", "call", "bet", "raise", "allin"] = Field(
        ...,
        description="Semantic action type",
        json_schema_extra={"example": "bet"},
    )
    amount_big_blinds: float = Field(
        ...,
        description="Amount in big blinds (0 for check/fold)",
        json_schema_extra={"example": 6.6},
    )
    amount_percent: float = Field(
        ...,
        description="Amount as percentage of pot (0 for check/fold)",
        json_schema_extra={"example": 33.0},
    )


class HandStrategy(BaseModel):
    """Strategy for a single combo (hand)."""

    hand: str = Field(
        ...,
        description="Hand in card notation. e.g. 'AhKd'",
        json_schema_extra={"example": "QdQs"},
    )
    hand_id: int = Field(
        ...,
        description="Internal combo ID (0-1325)",
        json_schema_extra={"example": 820},
    )
    strategy: list[float] = Field(
        ...,
        description=(
            "Action probabilities matching the 'actions' array order. Sums to 1.0."
        ),
        json_schema_extra={"example": [0.05, 0.25, 0.55, 0.15]},
    )


class SolveResponse(BaseModel):
    """Response body for ``POST /v1/solve``."""

    player: Literal["OOP", "IP"] = Field(
        ...,
        description="Acting player at this node",
        json_schema_extra={"example": "OOP"},
    )
    board: str = Field(
        ...,
        description="Board cards (space-separated)",
        json_schema_extra={"example": "Ah Kd Qc"},
    )
    pot: int = Field(
        ...,
        description="Current pot size in big blinds (bb)",
        json_schema_extra={"example": 20},
    )
    effective_stack: int = Field(
        ...,
        description="Effective stack in big blinds (bb)",
        json_schema_extra={"example": 100},
    )
    num_combos: int = Field(
        ...,
        description="Number of combos returned",
        json_schema_extra={"example": 15},
    )
    actions: list[ActionInfo] = Field(
        ...,
        description="Available actions at this node",
    )
    combos: list[HandStrategy] = Field(
        ...,
        description="Per-combo strategy",
    )


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str = Field(
        ...,
        description="Service status",
        json_schema_extra={"example": "ok"},
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the solver model is loaded",
        json_schema_extra={"example": True},
    )
    version: str = Field(
        ...,
        description="API version string",
        json_schema_extra={"example": "0.1.0"},
    )


class ErrorDetail(BaseModel):
    """Error response body."""

    error: str = Field(
        ...,
        description="Error code / category",
        json_schema_extra={"example": "invalid_board"},
    )
    message: str = Field(
        ...,
        description="Human-readable error message",
        json_schema_extra={"example": "Board must have 3-5 cards"},
    )
