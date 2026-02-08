"""
Custom No-Limit Hold'em game engine for Neural CFR training.

Implements NLHEGame and NLHEState that parse tree_config format (from solver
header.json) and generate the same game tree structure. Provides the same
interface that DeepCumuAdv.dfs() uses (current_player, legal_actions,
legal_actions_mask, child, is_terminal, returns, chance_outcomes).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Union

import numpy as np

from deeppdcfr.card_tools import card_tools

MAX_ACTIONS = 10


class Street(IntEnum):
    FLOP = 0
    TURN = 1
    RIVER = 2


class Player(IntEnum):
    OOP = 0  # Out of position
    IP = 1   # In position
    CHANCE = -1
    TERMINAL = -4


# ---------------------------------------------------------------------------
# Bet size types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PotRelative:
    fraction: float

@dataclass(frozen=True)
class AllIn:
    pass

BetSize = Union[PotRelative, AllIn]


def parse_bet_size(raw) -> BetSize:
    if raw == "AllIn":
        return AllIn()
    if isinstance(raw, dict) and "PotRelative" in raw:
        return PotRelative(raw["PotRelative"])
    raise ValueError(f"Unknown bet size format: {raw}")


# ---------------------------------------------------------------------------
# Per-street bet configuration
# ---------------------------------------------------------------------------

@dataclass
class StreetBetConfig:
    bet: list[BetSize] = field(default_factory=list)
    raise_: list[BetSize] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "StreetBetConfig":
        return StreetBetConfig(
            bet=[parse_bet_size(s) for s in d.get("bet", [])],
            raise_=[parse_bet_size(s) for s in d.get("raise", [])],
        )


# ---------------------------------------------------------------------------
# Tree configuration
# ---------------------------------------------------------------------------

@dataclass
class TreeConfig:
    effective_stack: int
    starting_pot: int
    initial_state: Street
    add_allin_threshold: float
    force_allin_threshold: float
    max_raises_per_street: int
    merging_threshold: float
    rake_rate: float
    rake_cap: float
    # Per-street configs: index 0 = OOP, index 1 = IP
    flop_bet_sizes: list[list[StreetBetConfig]]  # [street][player]
    turn_bet_sizes: list[list[StreetBetConfig]]
    river_bet_sizes: list[list[StreetBetConfig]]
    turn_donk_sizes: Optional[list] = None
    river_donk_sizes: Optional[list] = None

    @staticmethod
    def from_dict(d: dict) -> "TreeConfig":
        def parse_street_sizes(raw_list):
            if raw_list is None:
                return []
            return [StreetBetConfig.from_dict(entry) for entry in raw_list]

        return TreeConfig(
            effective_stack=d["effective_stack"],
            starting_pot=d["starting_pot"],
            initial_state=_parse_street(d["initial_state"]),
            add_allin_threshold=d.get("add_allin_threshold", 1.5),
            force_allin_threshold=d.get("force_allin_threshold", 0.2),
            max_raises_per_street=d.get("max_raises_per_street", 0),
            merging_threshold=d.get("merging_threshold", 0.1),
            rake_rate=d.get("rake_rate", 0.0),
            rake_cap=d.get("rake_cap", 0.0),
            flop_bet_sizes=parse_street_sizes(d.get("flop_bet_sizes")),
            turn_bet_sizes=parse_street_sizes(d.get("turn_bet_sizes")),
            river_bet_sizes=parse_street_sizes(d.get("river_bet_sizes")),
            turn_donk_sizes=d.get("turn_donk_sizes"),
            river_donk_sizes=d.get("river_donk_sizes"),
        )


# ---------------------------------------------------------------------------
# Card configuration
# ---------------------------------------------------------------------------

@dataclass
class CardConfig:
    flop: list[int]  # 3 card IDs
    turn: Optional[int] = None
    river: Optional[int] = None
    oop_range: Optional[np.ndarray] = None  # 1326
    ip_range: Optional[np.ndarray] = None   # 1326

    @staticmethod
    def from_dict(d: dict) -> "CardConfig":
        return CardConfig(
            flop=d["flop"],
            turn=d.get("turn"),
            river=d.get("river"),
            oop_range=np.array(d["oop_range"], dtype=np.float64) if "oop_range" in d else None,
            ip_range=np.array(d["ip_range"], dtype=np.float64) if "ip_range" in d else None,
        )


def _parse_street(s: str) -> Street:
    return {"flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}[s]


# ---------------------------------------------------------------------------
# Action representation
# ---------------------------------------------------------------------------

# Fixed action space indices
ACTION_FOLD = 0
ACTION_CHECK_CALL = 1
ACTION_BET_START = 2  # Indices 2..8 for bet/raise sizes
ACTION_BET_END = 8
ACTION_ALLIN = 9


@dataclass(frozen=True)
class Action:
    """Represents a concrete action with its index and semantics."""
    index: int
    name: str       # "Fold", "Check", "Call", "Bet", "Raise", "AllIn"
    amount: int = 0  # chip amount (0 for fold/check)


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class NLHEState:
    """
    Immutable game state for NLHE. Implements the interface expected by DFS:
    - current_player() -> int
    - legal_actions() -> list[int]
    - legal_actions_mask() -> list[int]
    - child(action_index) -> NLHEState
    - is_terminal() -> bool
    - returns() -> list[float]
    - chance_outcomes() -> list[tuple[int, float]]
    """

    def __init__(
        self,
        game: "NLHEGame",
        street: Street,
        pot: int,
        stacks: list[int],  # [oop_remaining, ip_remaining]
        board: list[int],   # card IDs on board
        player: int,        # Player enum value
        history: list,      # list of action dicts for replay
        bets_this_street: list[int],  # [oop_bet, ip_bet] this street
        num_raises_this_street: int,
        is_first_action_on_street: bool,
        last_action: Optional[dict] = None,
    ):
        self._game = game
        self._street = street
        self._pot = pot
        self._stacks = list(stacks)
        self._board = list(board)
        self._player = player
        self._history = list(history)
        self._bets_this_street = list(bets_this_street)
        self._num_raises = num_raises_this_street
        self._is_first_action = is_first_action_on_street
        self._last_action = last_action

        # Cached computations
        self._legal_actions_cache = None
        self._action_map_cache = None

    def current_player(self) -> int:
        return self._player

    def is_terminal(self) -> bool:
        return self._player == Player.TERMINAL

    def is_chance(self) -> bool:
        return self._player == Player.CHANCE

    def num_distinct_actions(self) -> int:
        return MAX_ACTIONS

    def legal_actions(self) -> list[int]:
        if self._legal_actions_cache is None:
            self._compute_legal_actions()
        return self._legal_actions_cache

    def legal_actions_mask(self) -> list[int]:
        mask = [0] * MAX_ACTIONS
        for a in self.legal_actions():
            mask[a] = 1
        return mask

    def _compute_legal_actions(self):
        """Compute legal actions and their concrete meanings."""
        if self._player == Player.TERMINAL or self._player == Player.CHANCE:
            self._legal_actions_cache = []
            self._action_map_cache = {}
            return

        actions = []
        action_map = {}
        player = self._player
        opponent = 1 - player
        my_bet = self._bets_this_street[player]
        opp_bet = self._bets_this_street[opponent]
        my_stack = self._stacks[player]
        total_pot = self._pot + self._bets_this_street[0] + self._bets_this_street[1]
        facing_bet = opp_bet > my_bet

        if facing_bet:
            # Facing a bet: can Fold, Call, Raise, AllIn
            actions.append(ACTION_FOLD)
            action_map[ACTION_FOLD] = Action(ACTION_FOLD, "Fold")

            call_amount = opp_bet - my_bet
            if call_amount >= my_stack:
                # Call is effectively all-in
                actions.append(ACTION_CHECK_CALL)
                action_map[ACTION_CHECK_CALL] = Action(
                    ACTION_CHECK_CALL, "Call", my_stack
                )
            else:
                actions.append(ACTION_CHECK_CALL)
                action_map[ACTION_CHECK_CALL] = Action(
                    ACTION_CHECK_CALL, "Call", call_amount
                )

                # Raise sizes
                raise_sizes = self._get_raise_sizes(player, total_pot, call_amount)
                for idx, amount in raise_sizes:
                    actions.append(idx)
                    action_map[idx] = Action(idx, "Raise", amount)
        else:
            # No bet facing: can Check, Bet, AllIn
            actions.append(ACTION_CHECK_CALL)
            action_map[ACTION_CHECK_CALL] = Action(ACTION_CHECK_CALL, "Check")

            bet_sizes = self._get_bet_sizes(player, total_pot)
            for idx, amount in bet_sizes:
                actions.append(idx)
                action_map[idx] = Action(idx, "Bet", amount)

        self._legal_actions_cache = sorted(actions)
        self._action_map_cache = action_map

    def _get_bet_config(self, player: int) -> StreetBetConfig:
        """Get bet config for current street and player."""
        tc = self._game.tree_config
        if self._street == Street.FLOP:
            configs = tc.flop_bet_sizes
        elif self._street == Street.TURN:
            configs = tc.turn_bet_sizes
        else:
            configs = tc.river_bet_sizes
        if player < len(configs):
            return configs[player]
        return StreetBetConfig()

    def _get_bet_sizes(self, player: int, total_pot: int) -> list[tuple[int, int]]:
        """Get bet sizes (not facing a bet). Returns (action_index, chip_amount) pairs."""
        config = self._get_bet_config(player)
        my_stack = self._stacks[player]
        result = []
        slot = ACTION_BET_START
        has_allin = False

        for bs in config.bet:
            if slot > ACTION_BET_END:
                break
            if isinstance(bs, AllIn):
                has_allin = True
                continue
            amount = self._compute_bet_amount(bs, total_pot)
            if amount >= my_stack:
                has_allin = True
                continue
            # Check force_allin_threshold
            remaining_after = my_stack - amount
            if remaining_after < self._game.tree_config.force_allin_threshold * total_pot:
                has_allin = True
                continue
            # Check add_allin_threshold
            if remaining_after < self._game.tree_config.add_allin_threshold * total_pot:
                has_allin = True
            result.append((slot, amount))
            slot += 1

        # Merge close sizes
        result = self._merge_sizes(result, total_pot)

        if has_allin and my_stack > 0:
            result.append((ACTION_ALLIN, my_stack))

        return result

    def _get_raise_sizes(
        self, player: int, total_pot: int, call_amount: int
    ) -> list[tuple[int, int]]:
        """Get raise sizes (facing a bet). Returns (action_index, chip_amount) pairs."""
        tc = self._game.tree_config
        if tc.max_raises_per_street > 0 and self._num_raises >= tc.max_raises_per_street:
            return []

        config = self._get_bet_config(player)
        my_stack = self._stacks[player]
        pot_after_call = total_pot + call_amount
        result = []
        slot = ACTION_BET_START
        has_allin = False

        for bs in config.raise_:
            if slot > ACTION_BET_END:
                break
            if isinstance(bs, AllIn):
                has_allin = True
                continue
            # Raise amount = call + raise_size * pot_after_call
            raise_portion = self._compute_bet_amount(bs, pot_after_call)
            total_amount = call_amount + raise_portion
            if total_amount >= my_stack:
                has_allin = True
                continue
            remaining_after = my_stack - total_amount
            if remaining_after < tc.force_allin_threshold * pot_after_call:
                has_allin = True
                continue
            if remaining_after < tc.add_allin_threshold * pot_after_call:
                has_allin = True
            result.append((slot, total_amount))
            slot += 1

        result = self._merge_sizes(result, total_pot)

        if has_allin and my_stack > call_amount:
            result.append((ACTION_ALLIN, my_stack))

        return result

    def _compute_bet_amount(self, bs: BetSize, pot: int) -> int:
        if isinstance(bs, PotRelative):
            return max(1, round(pot * bs.fraction))
        return 0

    def _merge_sizes(
        self, sizes: list[tuple[int, int]], pot: int
    ) -> list[tuple[int, int]]:
        """Merge bet sizes that are too close together."""
        if len(sizes) <= 1:
            return sizes
        threshold = self._game.tree_config.merging_threshold
        merged = [sizes[0]]
        for i in range(1, len(sizes)):
            prev_amount = merged[-1][1]
            curr_amount = sizes[i][1]
            if pot > 0 and abs(curr_amount - prev_amount) / pot < threshold:
                continue
            merged.append(sizes[i])
        # Re-index slots
        return [(ACTION_BET_START + i, amount) for i, (_, amount) in enumerate(merged)]

    def child(self, action_index: int) -> "NLHEState":
        """Return new state after taking action. Immutable — creates a copy."""
        if self._action_map_cache is None:
            self._compute_legal_actions()
        action = self._action_map_cache[action_index]
        return self._apply_action(action)

    def _apply_action(self, action: Action) -> "NLHEState":
        player = self._player
        opponent = 1 - player

        if action.name == "Fold":
            return self._make_terminal(winner=opponent)

        if action.name == "Check":
            if self._is_ip_closing_action(player):
                return self._advance_street()
            else:
                # OOP checks, IP to act
                return NLHEState(
                    game=self._game,
                    street=self._street,
                    pot=self._pot,
                    stacks=list(self._stacks),
                    board=list(self._board),
                    player=opponent,
                    history=self._history + [{"Check": None}],
                    bets_this_street=list(self._bets_this_street),
                    num_raises_this_street=self._num_raises,
                    is_first_action_on_street=False,
                    last_action={"Check": None},
                )

        if action.name == "Call":
            call_amount = action.amount
            new_stacks = list(self._stacks)
            new_stacks[player] -= call_amount
            new_bets = list(self._bets_this_street)
            new_bets[player] += call_amount

            # Is this an all-in call?
            if new_stacks[player] <= 0 or new_stacks[opponent] <= 0:
                # All-in call → showdown (deal remaining cards if needed)
                new_pot = self._pot + new_bets[0] + new_bets[1]
                return self._make_allin_runout(new_pot, new_stacks)

            return self._advance_street(
                stacks=new_stacks,
                extra_pot=new_bets[0] + new_bets[1],
            )

        if action.name in ("Bet", "Raise"):
            amount = action.amount
            new_stacks = list(self._stacks)
            new_stacks[player] -= amount
            new_bets = list(self._bets_this_street)
            new_bets[player] += amount
            num_raises = self._num_raises + (1 if action.name == "Raise" else 1)

            return NLHEState(
                game=self._game,
                street=self._street,
                pot=self._pot,
                stacks=new_stacks,
                board=list(self._board),
                player=opponent,
                history=self._history + [{action.name: amount}],
                bets_this_street=new_bets,
                num_raises_this_street=num_raises,
                is_first_action_on_street=False,
                last_action={action.name: amount},
            )

        if action.name == "AllIn":
            amount = action.amount
            new_stacks = list(self._stacks)
            new_stacks[player] -= amount
            new_bets = list(self._bets_this_street)
            new_bets[player] += amount

            opp_bet = self._bets_this_street[opponent]
            if new_bets[player] <= opp_bet:
                # All-in for less than or equal to opponent's bet — just a call
                new_pot = self._pot + new_bets[0] + new_bets[1]
                return self._make_allin_runout(new_pot, new_stacks)

            return NLHEState(
                game=self._game,
                street=self._street,
                pot=self._pot,
                stacks=new_stacks,
                board=list(self._board),
                player=opponent,
                history=self._history + [{"AllIn": amount}],
                bets_this_street=new_bets,
                num_raises_this_street=self._num_raises + 1,
                is_first_action_on_street=False,
                last_action={"AllIn": amount},
            )

        raise ValueError(f"Unknown action: {action}")

    def _is_ip_closing_action(self, acting_player: int) -> bool:
        """Check if this is IP closing action (check-check or IP action after OOP)."""
        return acting_player == Player.IP and not self._is_first_action

    def _advance_street(
        self,
        stacks: Optional[list[int]] = None,
        extra_pot: int = 0,
    ) -> "NLHEState":
        """Advance to next street (or showdown if on river)."""
        new_stacks = stacks if stacks is not None else list(self._stacks)
        new_pot = self._pot + extra_pot if extra_pot else (
            self._pot + self._bets_this_street[0] + self._bets_this_street[1]
        )

        if self._street == Street.RIVER:
            # Showdown
            return self._make_showdown(new_pot, new_stacks)

        # Create chance node to deal next card
        return NLHEState(
            game=self._game,
            street=self._street,
            pot=new_pot,
            stacks=new_stacks,
            board=list(self._board),
            player=Player.CHANCE,
            history=self._history + [{"StreetEnd": None}],
            bets_this_street=[0, 0],
            num_raises_this_street=0,
            is_first_action_on_street=True,
            last_action={"StreetEnd": None},
        )

    def _make_allin_runout(self, pot: int, stacks: list[int]) -> "NLHEState":
        """All-in: need to deal remaining community cards then showdown."""
        if self._street == Street.RIVER and len(self._board) == 5:
            return self._make_showdown(pot, stacks)

        # Chance node that deals remaining cards
        return NLHEState(
            game=self._game,
            street=self._street,
            pot=pot,
            stacks=stacks,
            board=list(self._board),
            player=Player.CHANCE,
            history=self._history + [{"AllinRunout": None}],
            bets_this_street=[0, 0],
            num_raises_this_street=0,
            is_first_action_on_street=True,
            last_action={"AllinRunout": None},
        )

    def _make_terminal(self, winner: int) -> "NLHEState":
        """Create terminal state where one player folded."""
        total_in_pot = self._pot + self._bets_this_street[0] + self._bets_this_street[1]
        eff = self._game.tree_config.effective_stack
        loser = 1 - winner

        # Each player's total investment = effective_stack - remaining_stack
        # (stacks already decremented by bets_this_street in _apply_action)
        # Winner net = total_pot - winner_investment = loser_investment
        # Loser net  = -loser_investment
        payoffs = [0.0, 0.0]
        winner_invested = eff - self._stacks[winner]
        loser_invested = eff - self._stacks[loser]
        payoffs[winner] = float(total_in_pot - winner_invested)
        payoffs[loser] = float(-loser_invested)

        state = NLHEState(
            game=self._game,
            street=self._street,
            pot=total_in_pot,
            stacks=list(self._stacks),
            board=list(self._board),
            player=Player.TERMINAL,
            history=self._history + [{"Fold": winner}],
            bets_this_street=list(self._bets_this_street),
            num_raises_this_street=0,
            is_first_action_on_street=False,
            last_action={"Fold": winner},
        )
        state._payoffs = payoffs
        return state

    def _make_showdown(self, pot: int, stacks: list[int]) -> "NLHEState":
        """Create terminal showdown state. Payoffs computed during traversal."""
        state = NLHEState(
            game=self._game,
            street=self._street,
            pot=pot,
            stacks=stacks,
            board=list(self._board),
            player=Player.TERMINAL,
            history=self._history + [{"Showdown": None}],
            bets_this_street=[0, 0],
            num_raises_this_street=0,
            is_first_action_on_street=False,
            last_action={"Showdown": None},
        )
        state._is_showdown = True
        state._payoffs = None  # Computed per hand pair
        return state

    def returns(self, oop_hand: Optional[tuple] = None, ip_hand: Optional[tuple] = None) -> list[float]:
        """
        Terminal payoffs.
        For fold nodes, payoffs are pre-computed.
        For showdown nodes, requires hand information.
        """
        if hasattr(self, "_payoffs") and self._payoffs is not None:
            return self._payoffs

        if not hasattr(self, "_is_showdown") or not self._is_showdown:
            raise ValueError("returns() called on non-terminal state")

        if oop_hand is None or ip_hand is None:
            raise ValueError("Showdown requires hand information")

        return self._compute_showdown_payoff(oop_hand, ip_hand)

    def _compute_showdown_payoff(
        self, oop_hand: tuple[int, int], ip_hand: tuple[int, int]
    ) -> list[float]:
        """Compute showdown payoff given both players' hands."""
        board_arr = np.array(self._board, dtype=int)
        oop_cards = np.array([[*self._board, oop_hand[0], oop_hand[1]]], dtype=int)
        ip_cards = np.array([[*self._board, ip_hand[0], ip_hand[1]]], dtype=int)

        oop_strength = card_tools.evaluate(oop_cards)[0]
        ip_strength = card_tools.evaluate(ip_cards)[0]

        eff = self._game.tree_config.effective_stack
        oop_invested = eff - self._stacks[0]
        ip_invested = eff - self._stacks[1]
        pot = self._pot

        # Apply rake if configured
        rake = 0.0
        tc = self._game.tree_config
        if tc.rake_rate > 0:
            rake = min(pot * tc.rake_rate, tc.rake_cap)

        if oop_strength > ip_strength:
            # OOP wins
            return [float(ip_invested - rake / 2), float(-ip_invested - rake / 2)]
        elif ip_strength > oop_strength:
            # IP wins
            return [float(-oop_invested - rake / 2), float(oop_invested - rake / 2)]
        else:
            # Tie — split pot
            return [-rake / 2, -rake / 2]

    def chance_outcomes(self) -> list[tuple[int, float]]:
        """Return possible chance outcomes: (card_id, probability)."""
        if self._player != Player.CHANCE:
            raise ValueError("chance_outcomes called on non-chance state")

        # Cards already dealt
        dead_cards = set(self._board)
        # Note: hand-specific dead cards are handled during traversal

        deck = [c for c in range(52) if c not in dead_cards]
        prob = 1.0 / len(deck)
        return [(c, prob) for c in deck]

    def deal_card(self, card: int) -> "NLHEState":
        """Deal a card (chance action). Returns new state."""
        new_board = self._board + [card]
        next_street = Street(self._street + 1) if len(new_board) > len(self._game.card_config.flop) else self._street

        # Determine if we need more cards (all-in runout)
        is_allin_runout = (
            self._last_action is not None and
            "AllinRunout" in self._last_action
        )

        if is_allin_runout and len(new_board) < 5:
            # Need more cards
            return NLHEState(
                game=self._game,
                street=next_street,
                pot=self._pot,
                stacks=list(self._stacks),
                board=new_board,
                player=Player.CHANCE,
                history=self._history + [{"Deal": card}],
                bets_this_street=[0, 0],
                num_raises_this_street=0,
                is_first_action_on_street=True,
                last_action={"AllinRunout": None},
            )

        if is_allin_runout and len(new_board) == 5:
            # All cards dealt, showdown
            return self._make_showdown_from_deal(new_board)

        # Normal street transition: OOP acts first
        return NLHEState(
            game=self._game,
            street=next_street,
            pot=self._pot,
            stacks=list(self._stacks),
            board=new_board,
            player=Player.OOP,
            history=self._history + [{"Deal": card}],
            bets_this_street=[0, 0],
            num_raises_this_street=0,
            is_first_action_on_street=True,
            last_action={"Deal": card},
        )

    def _make_showdown_from_deal(self, board: list[int]) -> "NLHEState":
        state = NLHEState(
            game=self._game,
            street=Street.RIVER,
            pot=self._pot,
            stacks=list(self._stacks),
            board=board,
            player=Player.TERMINAL,
            history=self._history + [{"Showdown": None}],
            bets_this_street=[0, 0],
            num_raises_this_street=0,
            is_first_action_on_street=False,
            last_action={"Showdown": None},
        )
        state._is_showdown = True
        state._payoffs = None
        return state

    # ---- Convenience properties ----

    @property
    def street(self) -> Street:
        return self._street

    @property
    def pot(self) -> int:
        return self._pot

    @property
    def stacks(self) -> list[int]:
        return list(self._stacks)

    @property
    def board(self) -> list[int]:
        return list(self._board)

    @property
    def history(self) -> list:
        return list(self._history)

    @property
    def bets_this_street(self) -> list[int]:
        return list(self._bets_this_street)

    @property
    def is_fold_terminal(self) -> bool:
        """True if this is a terminal state caused by a fold (not showdown)."""
        if not self.is_terminal():
            return False
        return hasattr(self, "_payoffs") and self._payoffs is not None and not getattr(self, "_is_showdown", False)

    @property
    def is_showdown_terminal(self) -> bool:
        """True if this is a showdown terminal state."""
        return self.is_terminal() and getattr(self, "_is_showdown", False)

    @property
    def fold_payoffs(self) -> list[float]:
        """Pre-computed fold payoffs. Only valid if is_fold_terminal is True."""
        if not self.is_fold_terminal:
            raise ValueError("fold_payoffs only valid for fold terminal nodes")
        return list(self._payoffs)

    def action_name(self, action_index: int) -> str:
        """Get human-readable name for an action index."""
        if self._action_map_cache is None:
            self._compute_legal_actions()
        if action_index in self._action_map_cache:
            a = self._action_map_cache[action_index]
            if a.amount > 0:
                return f"{a.name}:{a.amount}"
            return a.name
        return f"Action_{action_index}"

    def action_amount(self, action_index: int) -> int:
        """Get chip amount for an action."""
        if self._action_map_cache is None:
            self._compute_legal_actions()
        if action_index in self._action_map_cache:
            return self._action_map_cache[action_index].amount
        return 0

    def action_as_dict(self, action_index: int) -> dict:
        """Convert action index to solver-format dict (for matching game-export)."""
        if self._action_map_cache is None:
            self._compute_legal_actions()
        action = self._action_map_cache[action_index]
        if action.name == "Fold":
            return "Fold"
        if action.name == "Check":
            return "Check"
        if action.name == "Call":
            return "Call"
        if action.name == "AllIn":
            return {"AllIn": self._game.tree_config.effective_stack}
        return {action.name: action.amount}


# ---------------------------------------------------------------------------
# Game wrapper
# ---------------------------------------------------------------------------

class NLHEGame:
    """
    Game object that creates initial states from TreeConfig + CardConfig.
    Mirrors the OpenSpiel game interface used by DeepCumuAdv.
    """

    def __init__(self, tree_config: TreeConfig, card_config: CardConfig):
        self.tree_config = tree_config
        self.card_config = card_config

        # Precompute valid hand combos for each player
        board_set = set(card_config.flop)
        if card_config.turn is not None:
            board_set.add(card_config.turn)
        if card_config.river is not None:
            board_set.add(card_config.river)

        self._valid_combos = [[], []]  # [oop_combos, ip_combos]
        ranges = [card_config.oop_range, card_config.ip_range]
        for p in range(2):
            if ranges[p] is not None:
                for hid in range(1326):
                    c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
                    if c1 in board_set or c2 in board_set:
                        continue
                    if ranges[p][hid] > 0:
                        self._valid_combos[p].append((hid, c1, c2))

    def new_initial_state(self) -> NLHEState:
        tc = self.tree_config
        cc = self.card_config
        board = list(cc.flop)
        if cc.turn is not None:
            board.append(cc.turn)
        if cc.river is not None:
            board.append(cc.river)

        # Initial stacks: effective_stack minus what's already in pot
        # starting_pot is total pot; each player contributed half
        half_pot = tc.starting_pot // 2
        initial_stacks = [tc.effective_stack - half_pot, tc.effective_stack - half_pot]

        return NLHEState(
            game=self,
            street=tc.initial_state,
            pot=tc.starting_pot,
            stacks=initial_stacks,
            board=board,
            player=Player.OOP,
            history=[],
            bets_this_street=[0, 0],
            num_raises_this_street=0,
            is_first_action_on_street=True,
        )

    def num_distinct_actions(self) -> int:
        return MAX_ACTIONS

    def max_utility(self) -> float:
        return float(self.tree_config.effective_stack)

    def min_utility(self) -> float:
        return float(-self.tree_config.effective_stack)

    def num_players(self) -> int:
        return 2

    def valid_combos(self, player: int) -> list[tuple[int, int, int]]:
        """Return list of (hand_id, card1, card2) for valid combos."""
        return self._valid_combos[player]

    @staticmethod
    def from_header(header_path: str) -> "NLHEGame":
        """Load game from a solver export header.json file."""
        with open(header_path) as f:
            header = json.load(f)
        config = header["configuration"]
        tree_config = TreeConfig.from_dict(config["tree_config"])
        card_config = CardConfig.from_dict(config["card_config"])
        return NLHEGame(tree_config, card_config)
