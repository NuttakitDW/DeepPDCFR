"""Converters between Pio-syntax API inputs and the solver engine's internal types.

Handles range parsing, bet size parsing, config building, history conversion,
and response formatting.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from deeppdcfr.card_tools import card_tools
from deeppdcfr.nlhe_game import (
    AllIn,
    CardConfig,
    PotRelative,
    Street,
    StreetBetConfig,
    TreeConfig,
)
from deeppdcfr.query import QueryResult

from api.schemas import (
    ActionInfo,
    BetSizes,
    HandStrategy,
    HistoryAction,
    RANK_INDEX,
    RANK_ORDER,
    SolveRequest,
    SolveResponse,
    parse_board_cards,
)


# ---------------------------------------------------------------------------
# Range parser
# ---------------------------------------------------------------------------

_SUIT_CHARS = "cdhs"
_SUIT_INDEX = {s: i for i, s in enumerate(_SUIT_CHARS)}


def _card_id(rank_idx: int, suit_idx: int) -> int:
    """Card ID from rank index (0=2 .. 12=A) and suit index (0=c,1=d,2=h,3=s)."""
    return rank_idx * 4 + suit_idx


def _pair_combos(rank_idx: int) -> list[int]:
    """All 6 pair combos for a given rank."""
    ids = []
    for s1 in range(4):
        for s2 in range(s1 + 1, 4):
            c1 = _card_id(rank_idx, s1)
            c2 = _card_id(rank_idx, s2)
            ids.append(card_tools.hand_to_id((card_tools.id_to_card(c1), card_tools.id_to_card(c2))))
    return ids


def _suited_combos(r1: int, r2: int) -> list[int]:
    """4 suited combos for two different ranks."""
    ids = []
    for s in range(4):
        c1 = _card_id(r1, s)
        c2 = _card_id(r2, s)
        card_str_1 = card_tools.id_to_card(min(c1, c2))
        card_str_2 = card_tools.id_to_card(max(c1, c2))
        ids.append(card_tools.hand_to_id((card_str_1, card_str_2)))
    return ids


def _offsuit_combos(r1: int, r2: int) -> list[int]:
    """12 offsuit combos for two different ranks."""
    ids: set[int] = set()
    for s1 in range(4):
        for s2 in range(4):
            if s1 == s2:
                continue
            c1 = _card_id(r1, s1)
            c2 = _card_id(r2, s2)
            card_str_1 = card_tools.id_to_card(min(c1, c2))
            card_str_2 = card_tools.id_to_card(max(c1, c2))
            ids.add(card_tools.hand_to_id((card_str_1, card_str_2)))
    return list(ids)


def _parse_weight(token: str) -> tuple[str, float]:
    """Split off trailing :weight from a token. Returns (base_token, weight)."""
    if ":" in token:
        parts = token.rsplit(":", 1)
        return parts[0], float(parts[1])
    return token, 1.0


def _expand_token(token: str) -> list[tuple[int, float]]:
    """Expand a single range token into list of (hand_id, weight) pairs."""
    base, weight = _parse_weight(token)

    # Specific combo: AhKd
    if len(base) == 4 and base[1] in _SUIT_CHARS and base[3] in _SUIT_CHARS:
        c1_str = base[:2]
        c2_str = base[2:]
        c1 = card_tools.card_to_id(c1_str)
        c2 = card_tools.card_to_id(c2_str)
        card_str_1 = card_tools.id_to_card(min(c1, c2))
        card_str_2 = card_tools.id_to_card(max(c1, c2))
        hid = card_tools.hand_to_id((card_str_1, card_str_2))
        return [(hid, weight)]

    # Pair-plus: 88+
    if len(base) == 3 and base[0] == base[1] and base[2] == "+":
        r = RANK_INDEX[base[0]]
        result = []
        for ri in range(r, 13):  # r through A (12)
            result.extend((hid, weight) for hid in _pair_combos(ri))
        return result

    # Pair-dash: 88-55
    if "-" in base and not base.endswith("s") and not base.endswith("o"):
        parts = base.split("-")
        if len(parts) == 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
            r0_0, r0_1 = parts[0][0], parts[0][1]
            r1_0, r1_1 = parts[1][0], parts[1][1]
            # Pair dash: AA-55
            if r0_0 == r0_1 and r1_0 == r1_1:
                hi = RANK_INDEX[r0_0]
                lo = RANK_INDEX[r1_0]
                if lo > hi:
                    hi, lo = lo, hi
                result = []
                for ri in range(lo, hi + 1):
                    result.extend((hid, weight) for hid in _pair_combos(ri))
                return result

    # Dash range with suit qualifier: ATs-A6s or ATo-A6o
    if "-" in base:
        parts = base.split("-")
        if len(parts) == 2:
            left, right = parts[0], parts[1]
            # Determine suit qualifier from either side
            suit_q = None
            left_base = left
            right_base = right
            if left.endswith("s") or left.endswith("o"):
                suit_q = left[-1]
                left_base = left[:-1]
            if right.endswith("s") or right.endswith("o"):
                suit_q = right[-1]
                right_base = right[:-1]

            if len(left_base) == 2 and len(right_base) == 2:
                # Both must share the high card
                hi_rank = left_base[0]
                lo_kicker_left = RANK_INDEX[left_base[1]]
                lo_kicker_right = RANK_INDEX[right_base[1]]
                hi = RANK_INDEX[hi_rank]

                lo_k = min(lo_kicker_left, lo_kicker_right)
                hi_k = max(lo_kicker_left, lo_kicker_right)

                result = []
                for ki in range(lo_k, hi_k + 1):
                    if ki == hi:
                        continue  # Skip pairs
                    if suit_q == "s":
                        result.extend((hid, weight) for hid in _suited_combos(hi, ki))
                    elif suit_q == "o":
                        result.extend((hid, weight) for hid in _offsuit_combos(hi, ki))
                    else:
                        result.extend((hid, weight) for hid in _suited_combos(hi, ki))
                        result.extend((hid, weight) for hid in _offsuit_combos(hi, ki))
                return result

    # Plus range: ATs+, ATo+, AT+
    if base.endswith("+"):
        inner = base[:-1]
        suit_q = None
        if inner.endswith("s") or inner.endswith("o"):
            suit_q = inner[-1]
            inner = inner[:-1]
        if len(inner) == 2:
            hi = RANK_INDEX[inner[0]]
            lo = RANK_INDEX[inner[1]]
            if hi == lo:
                # Pair-plus already handled above, but just in case
                result = []
                for ri in range(hi, 13):
                    result.extend((hid, weight) for hid in _pair_combos(ri))
                return result
            # Kicker goes UP from lo to hi-1
            result = []
            for ki in range(lo, hi):
                if suit_q == "s":
                    result.extend((hid, weight) for hid in _suited_combos(hi, ki))
                elif suit_q == "o":
                    result.extend((hid, weight) for hid in _offsuit_combos(hi, ki))
                else:
                    result.extend((hid, weight) for hid in _suited_combos(hi, ki))
                    result.extend((hid, weight) for hid in _offsuit_combos(hi, ki))
            return result

    # Pair: AA
    if len(base) == 2 and base[0] == base[1]:
        return [(hid, weight) for hid in _pair_combos(RANK_INDEX[base[0]])]

    # Suited/offsuit/both: AKs, AKo, AK
    if len(base) == 2 or (len(base) == 3 and base[2] in "so"):
        r1 = RANK_INDEX[base[0]]
        r2 = RANK_INDEX[base[1]]
        suit_q = base[2] if len(base) == 3 else None

        if suit_q == "s":
            return [(hid, weight) for hid in _suited_combos(r1, r2)]
        elif suit_q == "o":
            return [(hid, weight) for hid in _offsuit_combos(r1, r2)]
        else:
            # Both suited + offsuit
            result = [(hid, weight) for hid in _suited_combos(r1, r2)]
            result.extend((hid, weight) for hid in _offsuit_combos(r1, r2))
            return result

    raise ValueError(f"Cannot parse range token: '{token}'")


def parse_range(range_str: Optional[str], board_cards: list[int]) -> np.ndarray:
    """Parse a Pio range string into a 1326-element weight array.

    Args:
        range_str: Pio-format range (e.g. "AA,AKs,88+,ATs-A6s,AhKd:0.5").
                   None → uniform (all 1.0).
        board_cards: List of card IDs on the board (used to zero out conflicts).

    Returns:
        np.ndarray of shape (1326,) with weights for each combo.
    """
    weights = np.zeros(1326, dtype=np.float64)

    if range_str is None:
        weights[:] = 1.0
    else:
        tokens = [t.strip() for t in range_str.split(",")]
        for token in tokens:
            for hid, w in _expand_token(token):
                weights[hid] = w

    # Zero out combos that conflict with board cards
    board_set = set(board_cards)
    for hid in range(1326):
        c1 = int(card_tools.hand_ids[hid][0])
        c2 = int(card_tools.hand_ids[hid][1])
        if c1 in board_set or c2 in board_set:
            weights[hid] = 0.0

    return weights


# ---------------------------------------------------------------------------
# Bet size parser
# ---------------------------------------------------------------------------


def parse_pio_bet_sizes(bet_str: str) -> list:
    """Parse Pio bet size string into list of BetSize objects.

    Args:
        bet_str: e.g. "33, 67, a"

    Returns:
        List of PotRelative or AllIn objects.

    Raises:
        ValueError: For unsupported size types (Nx, Nc, e).
    """
    tokens = [t.strip() for t in bet_str.split(",")]
    result = []
    for token in tokens:
        if not token:
            continue
        if token == "a":
            result.append(AllIn())
        elif token.endswith("x"):
            raise ValueError(f"Previous-bet-relative sizes ('{token}') not supported")
        elif token.endswith("c"):
            raise ValueError(f"Constant chip sizes ('{token}') not supported")
        elif "e" in token:
            raise ValueError(f"Geometric sizes ('{token}') not supported")
        else:
            # Pot-relative: "33" or "33%"
            num_str = token.rstrip("%")
            fraction = float(num_str) / 100.0
            result.append(PotRelative(fraction))
    return result


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def _street_from_board_count(n: int) -> Street:
    """Determine street from number of board cards."""
    if n == 3:
        return Street.FLOP
    elif n == 4:
        return Street.TURN
    elif n == 5:
        return Street.RIVER
    raise ValueError(f"Invalid board card count: {n}")


def build_tree_config(req: SolveRequest) -> TreeConfig:
    """Build TreeConfig from a SolveRequest."""
    board_cards = parse_board_cards(req.board)
    street = _street_from_board_count(len(board_cards))

    bet_sizes = req.bet_sizes or BetSizes()

    oop_bet = parse_pio_bet_sizes(bet_sizes.oop_bet)
    oop_raise = parse_pio_bet_sizes(bet_sizes.oop_raise)
    ip_bet = parse_pio_bet_sizes(bet_sizes.ip_bet)
    ip_raise = parse_pio_bet_sizes(bet_sizes.ip_raise)

    oop_config = StreetBetConfig(bet=oop_bet, raise_=oop_raise)
    ip_config = StreetBetConfig(bet=ip_bet, raise_=ip_raise)

    # Same config applied to all streets
    street_sizes = [oop_config, ip_config]

    return TreeConfig(
        effective_stack=req.effective_stack,
        starting_pot=req.starting_pot,
        initial_state=street,
        add_allin_threshold=1.5,
        force_allin_threshold=0.2,
        max_raises_per_street=0,
        merging_threshold=0.1,
        rake_rate=0.0,
        rake_cap=0.0,
        flop_bet_sizes=street_sizes,
        turn_bet_sizes=street_sizes,
        river_bet_sizes=street_sizes,
    )


def build_card_config(req: SolveRequest) -> CardConfig:
    """Build CardConfig from a SolveRequest."""
    board_strs = parse_board_cards(req.board)
    board_ids = [card_tools.card_to_id(c) for c in board_strs]

    flop = board_ids[:3]
    turn = board_ids[3] if len(board_ids) > 3 else None
    river = board_ids[4] if len(board_ids) > 4 else None

    oop_range = parse_range(req.oop_range, board_ids)
    ip_range = parse_range(req.ip_range, board_ids)

    return CardConfig(
        flop=flop,
        turn=turn,
        river=river,
        oop_range=oop_range,
        ip_range=ip_range,
    )


# ---------------------------------------------------------------------------
# History adapter
# ---------------------------------------------------------------------------


def convert_history(
    history: Optional[list[HistoryAction]],
    starting_pot: int,
    effective_stack: int,
) -> tuple[list, int]:
    """Convert API HistoryActions to engine replay format.

    Args:
        history: List of HistoryAction from the request. None/empty = root.
        starting_pot: Starting pot size in bb.
        effective_stack: Effective stack in bb.

    Returns:
        Tuple of (engine_history, current_pot):
            engine_history: List suitable for SolverQuery._replay_history()
            current_pot: Pot size after replaying (for response builder).
    """
    if not history:
        return [], starting_pot

    engine_actions = []
    pot = starting_pot
    half_pot = starting_pot // 2
    stacks = [effective_stack - half_pot, effective_stack - half_pot]
    bets_this_street = [0, 0]
    player_map = {"OOP": 0, "IP": 1}

    for ha in sorted(history, key=lambda x: x.order):
        p = player_map[ha.position]
        opp = 1 - p

        if ha.action == "check":
            engine_actions.append("Check")

        elif ha.action == "call":
            engine_actions.append("Call")
            call_amount = bets_this_street[opp] - bets_this_street[p]
            bets_this_street[p] += call_amount
            stacks[p] -= call_amount
            # After call, pot absorbs bets and street ends
            pot += bets_this_street[0] + bets_this_street[1]
            bets_this_street = [0, 0]

        elif ha.action == "fold":
            engine_actions.append("Fold")

        elif ha.action == "allin":
            remaining = stacks[p]
            engine_actions.append({"AllIn": remaining})
            bets_this_street[p] += remaining
            stacks[p] = 0

        elif ha.action == "bet":
            total_pot = pot + bets_this_street[0] + bets_this_street[1]
            amount = round(total_pot * ha.amount_percent / 100.0)
            amount = min(amount, stacks[p])
            engine_actions.append({"Bet": amount})
            bets_this_street[p] += amount
            stacks[p] -= amount

        elif ha.action == "raise":
            call_amount = bets_this_street[opp] - bets_this_street[p]
            pot_after_call = pot + bets_this_street[0] + bets_this_street[1] + call_amount
            raise_portion = round(pot_after_call * ha.amount_percent / 100.0)
            total_amount = call_amount + raise_portion
            total_amount = min(total_amount, stacks[p])
            engine_actions.append({"Raise": total_amount})
            bets_this_street[p] += total_amount
            stacks[p] -= total_amount

        elif ha.action == "deal":
            card_id = card_tools.card_to_id(ha.card)
            engine_actions.append({"Deal": card_id})
            # After a deal, pot already absorbed from call; reset street bets
            bets_this_street = [0, 0]

    current_pot = pot + bets_this_street[0] + bets_this_street[1]
    return engine_actions, current_pot


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------


def _action_type(name: str) -> str:
    """Map engine action name like 'Bet:12' to API action type."""
    base = name.split(":")[0]
    return {
        "Fold": "fold",
        "Check": "check",
        "Call": "call",
        "Bet": "bet",
        "Raise": "raise",
        "AllIn": "allin",
    }.get(base, "bet")


def build_response(
    req: SolveRequest,
    result: QueryResult,
    current_pot: int,
) -> SolveResponse:
    """Convert QueryResult into an API SolveResponse.

    Filters strategy to legal actions only and formats action/combo info.
    """
    # Build action info from result
    actions_info = []
    for i, action_name in enumerate(result.actions):
        atype = _action_type(action_name)
        amount_bb = 0.0
        amount_pct = 0.0

        if ":" in action_name:
            amount_bb = float(action_name.split(":")[1])
            if current_pot > 0:
                amount_pct = round(amount_bb / current_pot * 100, 1)

        if atype == "allin":
            # AllIn: amount = remaining stack
            half_pot = req.starting_pot // 2
            amount_bb = float(req.effective_stack - half_pot)
            if current_pot > 0:
                amount_pct = round(amount_bb / current_pot * 100, 1)

        # Human-readable name
        if atype == "fold":
            display = "Fold"
        elif atype == "check":
            display = "Check"
        elif atype == "call":
            display = f"Call {amount_bb:.0f}" if amount_bb > 0 else "Call"
        elif atype == "allin":
            display = "All-in"
        elif amount_pct > 0:
            display = f"{atype.capitalize()} {amount_pct:.0f}%"
        else:
            display = action_name

        actions_info.append(ActionInfo(
            name=display,
            type=atype,
            amount_big_blinds=amount_bb,
            amount_percent=amount_pct,
        ))

    # Build per-combo strategies
    # result.strategy shape: [C, MAX_ACTIONS] but we only want legal action columns
    legal_indices = result.action_indices
    combos_out = []
    for i, (combo, hid) in enumerate(zip(result.combos, result.combo_ids)):
        # Extract only legal action probabilities
        strat_row = result.strategy[i]
        legal_probs = [float(strat_row[ai]) for ai in legal_indices]

        # Normalize (softmax output should already sum to ~1.0, but guard against degenerate cases)
        total = sum(legal_probs)
        if total > 1e-8:
            legal_probs = [p / total for p in legal_probs]
        else:
            # Degenerate: uniform over legal actions
            n = len(legal_probs)
            legal_probs = [1.0 / n] * n

        hand_str = combo[0] + combo[1]
        combos_out.append(HandStrategy(
            hand=hand_str,
            hand_id=hid,
            strategy=legal_probs,
        ))

    # Compute average frequency per action across all combos
    num_actions = len(actions_info)
    if combos_out:
        freq_sums = [0.0] * num_actions
        for combo in combos_out:
            for j, p in enumerate(combo.strategy):
                freq_sums[j] += p
        n = len(combos_out)
        for j in range(num_actions):
            actions_info[j].frequency = round(freq_sums[j] / n, 4)

    # Format board as space-separated
    board_cards = parse_board_cards(req.board)
    board_display = " ".join(board_cards)

    return SolveResponse(
        player=req.player,
        board=board_display,
        pot=current_pot,
        effective_stack=req.effective_stack,
        num_combos=len(combos_out),
        actions=actions_info,
        combos=combos_out,
    )
