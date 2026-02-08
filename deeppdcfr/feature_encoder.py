"""
Feature encoding for NLHE Neural CFR (V2).

FastFeatureEncoder produces:
- Situation: board_ids[5], board_mask[5], numerical[98]
  (44 base features + 54 range histogram features)
- Combos: card_ids[C,2], hand_features[C,2] (strength + flush_draw)

BoardStrengthCache pre-computes hand strength for ALL 1326 combos per board.
HandClassifier classifies hands into made/draw/equity categories per board.
RangeEncoder compresses 1326-dim ranges into 27-dim histograms.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from deeppdcfr.card_tools import card_tools
from deeppdcfr.nlhe_game import (
    MAX_ACTIONS,
    NLHEGame,
    NLHEState,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_RANKS = 13
NUM_SUITS = 4
MAX_BET_HISTORY = 20

# V2 feature dimensions (base situation features)
SITUATION_BASE_DIM = 44       # street[3]+player[1]+pot_ratio[1]+SPR[1]+committed[2]+pot_odds[1]+bet_history[20]+actions[10]+board_mask[5]
COMBO_HAND_FEATURES_DIM = 2   # hand_strength[1] + flush_draw[1]

# Range encoding dimensions
NUM_MADE_HAND_CATEGORIES = 10  # nothing, bottom/mid/top pair, overpair, two pair, set, straight, flush, fullhouse+
NUM_DRAW_CATEGORIES = 7        # no draw, gutshot, OESD, flush draw, combo draw, backdoor flush, backdoor straight
NUM_EQUITY_BUCKETS = 10        # [0-0.1), [0.1-0.2), ..., [0.9-1.0]
RANGE_FEATURES_PER_PLAYER = NUM_MADE_HAND_CATEGORIES + NUM_DRAW_CATEGORIES + NUM_EQUITY_BUCKETS  # 27
RANGE_FEATURES_TOTAL = RANGE_FEATURES_PER_PLAYER * 2  # 54 (OOP + IP)
SITUATION_NUMERICAL_DIM = SITUATION_BASE_DIM + RANGE_FEATURES_TOTAL  # 98


# ---------------------------------------------------------------------------
# Suit isomorphism (from docs/isomorphism.md)
# ---------------------------------------------------------------------------

CANONICAL_SUITS = ["h", "d", "c", "s"]
SUIT_INDEX = {"h": 0, "d": 1, "c": 2, "s": 3}


def card_id_to_rank_suit(card_id: int) -> tuple[int, int]:
    """Convert card ID (0-51) to (rank_index 0-12, suit_index 0-3)."""
    rank = card_id // 4
    suit = card_id % 4
    return rank, suit


def rank_suit_to_card_id(rank: int, suit: int) -> int:
    return rank * 4 + suit


def canonicalize_board(board: list[int]) -> tuple[list[int], dict[int, int]]:
    """
    Convert board card IDs to canonical form.
    Returns (canonical_board, swap_map) where swap_map maps original suit -> canonical suit.
    """
    suits_seen = []
    swap_map = {}

    for card_id in board:
        _, suit = card_id_to_rank_suit(card_id)
        if suit not in suits_seen:
            suits_seen.append(suit)
            canonical_idx = len(suits_seen) - 1
            swap_map[suit] = canonical_idx

    # Fill in remaining suits
    used = set(swap_map.values())
    remaining = [i for i in range(4) if i not in used]
    for s in range(4):
        if s not in swap_map:
            swap_map[s] = remaining.pop(0)

    canonical_board = []
    for card_id in board:
        rank, suit = card_id_to_rank_suit(card_id)
        canonical_board.append(rank_suit_to_card_id(rank, swap_map[suit]))

    return canonical_board, swap_map


def apply_swap_to_card(card_id: int, swap_map: dict[int, int]) -> int:
    rank, suit = card_id_to_rank_suit(card_id)
    return rank_suit_to_card_id(rank, swap_map[suit])


def apply_swap_to_hand(hand: tuple[int, int], swap_map: dict[int, int]) -> tuple[int, int]:
    c1 = apply_swap_to_card(hand[0], swap_map)
    c2 = apply_swap_to_card(hand[1], swap_map)
    return (min(c1, c2), max(c1, c2))


def apply_swap_to_range(
    player_range: np.ndarray, swap_map: dict[int, int]
) -> np.ndarray:
    """Transform a 1326-dim range vector by applying suit swap."""
    new_range = np.zeros(1326, dtype=np.float64)
    for hid in range(1326):
        c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
        new_c1 = apply_swap_to_card(c1, swap_map)
        new_c2 = apply_swap_to_card(c2, swap_map)
        # Map to canonical hand ID
        new_hand = (min(new_c1, new_c2), max(new_c1, new_c2))
        new_hand_str = (card_tools.id_to_card(new_hand[0]), card_tools.id_to_card(new_hand[1]))
        new_hid = card_tools.hand_to_id(new_hand_str)
        new_range[new_hid] = player_range[hid]
    return new_range


def reverse_swap_map(swap_map: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in swap_map.items()}


# ---------------------------------------------------------------------------
# Hand classifier (per board, cached)
# ---------------------------------------------------------------------------

# 10 possible straights: A-5, 2-6, 3-7, ..., T-A (rank indices)
_STRAIGHT_WINDOWS = []
for low in range(10):  # low rank index: 0(2) through 9(J) for T-A straight
    _STRAIGHT_WINDOWS.append(set(range(low, low + 5)))
# Wheel: A(12), 2(0), 3(1), 4(2), 5(3)
_STRAIGHT_WINDOWS.append({12, 0, 1, 2, 3})


class HandClassifier:
    """
    Classify all 1326 hands on a given board into made hand, draw, and
    equity bucket categories. Results are cached per board.
    """

    def __init__(self, strength_cache: 'BoardStrengthCache'):
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._strength_cache = strength_cache

    def get(
        self, board: list[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (made_labels[1326], draw_labels[1326], eq_buckets[1326]).
        Invalid combos (conflicting with board) get label 0.
        """
        key = tuple(sorted(board))
        if key in self._cache:
            return self._cache[key]

        board_set = set(board)
        board_ranks = [card_id // 4 for card_id in board]
        board_suits = [card_id % 4 for card_id in board]
        board_rank_set = set(board_ranks)

        # Suit counts on board
        board_suit_counts = {}
        for s in board_suits:
            board_suit_counts[s] = board_suit_counts.get(s, 0) + 1

        # Board rank counts for pair detection
        board_rank_counts = {}
        for r in board_ranks:
            board_rank_counts[r] = board_rank_counts.get(r, 0) + 1

        # Sorted board ranks (descending) for top/mid/bottom pair classification
        sorted_board_ranks = sorted(set(board_ranks), reverse=True)

        # Get strength data for equity buckets
        if len(board) >= 3:
            _, relative_strengths, valid_mask = self._strength_cache.get(board)
        else:
            relative_strengths = np.full(1326, 0.5, dtype=np.float32)
            valid_mask = np.ones(1326, dtype=bool)
            for hid in range(1326):
                c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
                if c1 in board_set or c2 in board_set:
                    valid_mask[hid] = False

        made_labels = np.zeros(1326, dtype=np.int32)
        draw_labels = np.zeros(1326, dtype=np.int32)
        eq_buckets = np.zeros(1326, dtype=np.int32)

        is_river = len(board) == 5
        is_flop = len(board) == 3

        for hid in range(1326):
            c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
            if c1 in board_set or c2 in board_set:
                continue  # invalid combo, labels stay 0

            h_r1, h_s1 = c1 // 4, c1 % 4
            h_r2, h_s2 = c2 // 4, c2 % 4

            # All cards: board + hole
            all_ranks = board_ranks + [h_r1, h_r2]
            all_suits = board_suits + [h_s1, h_s2]

            # Rank counts for all 7 cards
            rank_counts = dict(board_rank_counts)
            rank_counts[h_r1] = rank_counts.get(h_r1, 0) + 1
            rank_counts[h_r2] = rank_counts.get(h_r2, 0) + 1

            # Suit counts for all 7 cards
            suit_counts = dict(board_suit_counts)
            suit_counts[h_s1] = suit_counts.get(h_s1, 0) + 1
            suit_counts[h_s2] = suit_counts.get(h_s2, 0) + 1

            # --- Made hand classification (check strongest first) ---
            counts_vals = sorted(rank_counts.values(), reverse=True)
            has_trips = counts_vals[0] >= 3
            has_quads = counts_vals[0] >= 4
            num_pairs = sum(1 for v in rank_counts.values() if v >= 2)

            # Full house+ (9): quads, or trips + pair
            if has_quads or (has_trips and num_pairs >= 2):
                made_labels[hid] = 9
            # Flush (8): 5+ same suit
            elif any(v >= 5 for v in suit_counts.values()):
                made_labels[hid] = 8
            # Straight (7): 5 consecutive ranks
            elif self._has_straight(set(all_ranks)):
                made_labels[hid] = 7
            # Set/Trips (6): 3 of a kind where hole card contributes
            elif has_trips and (rank_counts.get(h_r1, 0) >= 3 or rank_counts.get(h_r2, 0) >= 3) and \
                    (h_r1 == h_r2 or h_r1 in board_rank_set or h_r2 in board_rank_set):
                # Hole card must contribute: pocket pair hitting board, or hole card matches board pair
                made_labels[hid] = 6
            # Two pair (5): 2+ pairs, at least one hole card contributes
            elif num_pairs >= 2 and self._hole_contributes_pair(h_r1, h_r2, board_rank_counts):
                made_labels[hid] = 5
            # Overpair (4): pocket pair above all board ranks
            elif h_r1 == h_r2 and len(sorted_board_ranks) > 0 and h_r1 > sorted_board_ranks[0]:
                made_labels[hid] = 4
            # Top pair (3): hole card matches highest board rank
            elif len(sorted_board_ranks) > 0 and (h_r1 == sorted_board_ranks[0] or h_r2 == sorted_board_ranks[0]):
                made_labels[hid] = 3
            # Middle pair / underpair (2): hole card matches non-top board rank, or pocket pair below top
            elif self._is_mid_pair(h_r1, h_r2, sorted_board_ranks, board_rank_set):
                made_labels[hid] = 2
            # Bottom pair (1): hole card matches lowest board rank
            elif len(sorted_board_ranks) > 0 and (h_r1 == sorted_board_ranks[-1] or h_r2 == sorted_board_ranks[-1]):
                made_labels[hid] = 1
            # Nothing (0): default

            # --- Draw classification (flop/turn only) ---
            if not is_river:
                draw_labels[hid] = self._classify_draw(
                    h_r1, h_s1, h_r2, h_s2,
                    board_ranks, board_suit_counts, suit_counts,
                    all_ranks, is_flop,
                )

            # --- Equity bucket ---
            eq_buckets[hid] = min(int(relative_strengths[hid] * 10), 9)

        self._cache[key] = (made_labels, draw_labels, eq_buckets)
        return made_labels, draw_labels, eq_buckets

    @staticmethod
    def _has_straight(rank_set: set[int]) -> bool:
        """Check if any 5-card straight exists in the rank set."""
        for window in _STRAIGHT_WINDOWS:
            if window.issubset(rank_set):
                return True
        return False

    @staticmethod
    def _hole_contributes_pair(h_r1: int, h_r2: int, board_rank_counts: dict) -> bool:
        """Check if at least one hole card contributes to a pair."""
        # Pocket pair counts
        if h_r1 == h_r2:
            return True
        # Hole card pairs with board card
        if board_rank_counts.get(h_r1, 0) >= 1:
            return True
        if board_rank_counts.get(h_r2, 0) >= 1:
            return True
        return False

    @staticmethod
    def _is_mid_pair(h_r1: int, h_r2: int, sorted_board_ranks: list, board_rank_set: set) -> bool:
        """Middle pair: matches a non-top, non-bottom board rank, or underpair."""
        if len(sorted_board_ranks) < 2:
            return False
        mid_ranks = set(sorted_board_ranks[1:-1]) if len(sorted_board_ranks) > 2 else set()
        # Hole card matches middle board rank
        if h_r1 in mid_ranks or h_r2 in mid_ranks:
            return True
        # Underpair: pocket pair between top and bottom board rank
        if h_r1 == h_r2 and sorted_board_ranks[-1] < h_r1 < sorted_board_ranks[0]:
            return True
        return False

    @staticmethod
    def _classify_draw(
        h_r1: int, h_s1: int, h_r2: int, h_s2: int,
        board_ranks: list[int], board_suit_counts: dict,
        suit_counts: dict, all_ranks: list[int], is_flop: bool,
    ) -> int:
        """
        Classify draw type. Returns draw category index (0-6).
        Checks combo draw first, then individual draws, then backdoors.
        """
        all_rank_set = set(all_ranks)

        # Count flush draw cards (4 same suit)
        has_flush_draw = any(v == 4 for v in suit_counts.values())

        # Count straight outs: windows missing exactly 1 rank
        straight_outs = 0
        for window in _STRAIGHT_WINDOWS:
            missing = window - all_rank_set
            if len(missing) == 1:
                straight_outs += 1

        has_oesd = straight_outs >= 2  # 8+ outs (2+ completing ranks)
        has_gutshot = straight_outs == 1  # 4 outs (1 completing rank)

        # Combo draw (4): flush draw + straight draw
        if has_flush_draw and (has_oesd or has_gutshot):
            return 4
        # Flush draw (3)
        if has_flush_draw:
            return 3
        # OESD (2)
        if has_oesd:
            return 2
        # Gutshot (1)
        if has_gutshot:
            return 1

        # Backdoor draws (flop only)
        if is_flop:
            # Backdoor flush (5): 3 same suit
            if any(v == 3 for v in suit_counts.values()):
                return 5
            # Backdoor straight (6): 3+ connected ranks in a 5-window
            for window in _STRAIGHT_WINDOWS:
                if len(window & all_rank_set) >= 3:
                    return 6

        return 0  # No draw

    def clear(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# Range encoder
# ---------------------------------------------------------------------------


class RangeEncoder:
    """
    Compress a 1326-dim range into histogram features using hand
    classification labels.
    """

    @staticmethod
    def encode(
        player_range: np.ndarray,
        made_labels: np.ndarray,
        draw_labels: np.ndarray,
        eq_buckets: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Encode a range into RANGE_FEATURES_PER_PLAYER (27) features.

        Args:
            player_range: [1326] range weights
            made_labels:  [1326] made hand category (0-9)
            draw_labels:  [1326] draw category (0-6)
            eq_buckets:   [1326] equity bucket (0-9)
            valid_mask:   [1326] bool, True for combos not conflicting with board

        Returns: [27] float32 histogram features (3 normalized sub-histograms)
        """
        features = np.zeros(RANGE_FEATURES_PER_PLAYER, dtype=np.float32)

        # Mask range to valid combos only
        masked_range = player_range * valid_mask

        total = masked_range.sum()
        if total < 1e-10:
            return features

        # Made hand histogram (10 bins)
        offset = 0
        for cat in range(NUM_MADE_HAND_CATEGORIES):
            features[offset + cat] = masked_range[made_labels == cat].sum()
        made_sum = features[offset:offset + NUM_MADE_HAND_CATEGORIES].sum()
        if made_sum > 1e-10:
            features[offset:offset + NUM_MADE_HAND_CATEGORIES] /= made_sum

        # Draw histogram (7 bins)
        offset = NUM_MADE_HAND_CATEGORIES
        for cat in range(NUM_DRAW_CATEGORIES):
            features[offset + cat] = masked_range[draw_labels == cat].sum()
        draw_sum = features[offset:offset + NUM_DRAW_CATEGORIES].sum()
        if draw_sum > 1e-10:
            features[offset:offset + NUM_DRAW_CATEGORIES] /= draw_sum

        # Equity bucket histogram (10 bins)
        offset = NUM_MADE_HAND_CATEGORIES + NUM_DRAW_CATEGORIES
        for cat in range(NUM_EQUITY_BUCKETS):
            features[offset + cat] = masked_range[eq_buckets == cat].sum()
        eq_sum = features[offset:offset + NUM_EQUITY_BUCKETS].sum()
        if eq_sum > 1e-10:
            features[offset:offset + NUM_EQUITY_BUCKETS] /= eq_sum

        return features


# ---------------------------------------------------------------------------
# Board strength cache (V2)
# ---------------------------------------------------------------------------


class BoardStrengthCache:
    """
    Pre-compute hand strength for ALL 1326 combos given a board.
    Evaluates once, then lookups are O(1).

    Stores:
        strengths[1326]: raw hand rank (higher = better)
        relative_strengths[1326]: win % vs random opponent (0-1)
    """

    def __init__(self):
        self._cache = {}  # board_key -> (strengths, relative_strengths, valid_mask)

    def get(self, board: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get (strengths, relative_strengths, valid_mask) for a board.
        Returns cached result or computes fresh.

        strengths:          [1326] raw hand rank
        relative_strengths: [1326] win % vs random (0-1), 0 for invalid combos
        valid_mask:         [1326] bool, True where combo doesn't conflict with board
        """
        key = tuple(sorted(board))
        if key in self._cache:
            return self._cache[key]

        board_set = set(board)
        board_arr = np.array(board, dtype=int)

        # Build all 1326 hands with board
        all_hands = np.tile(board_arr, (1326, 1))  # [1326, len(board)]
        all_hands = np.concatenate([all_hands, card_tools.hand_ids.astype(int)], axis=1)  # [1326, len(board)+2]
        strengths = card_tools.evaluate(all_hands)  # [1326]

        # Valid mask: no card conflicts with board
        valid_mask = np.ones(1326, dtype=bool)
        for hid in range(1326):
            c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
            if c1 in board_set or c2 in board_set:
                valid_mask[hid] = False

        # Relative strengths: for each valid combo, compute win % vs all other valid combos
        relative_strengths = np.zeros(1326, dtype=np.float32)
        valid_strengths = strengths[valid_mask]
        if len(valid_strengths) > 0:
            for hid in range(1326):
                if not valid_mask[hid]:
                    continue
                c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
                my_dead = board_set | {c1, c2}
                # Opponents that don't conflict with our hand
                opp_valid = valid_mask.copy()
                for ohid in range(1326):
                    if opp_valid[ohid]:
                        oc1, oc2 = int(card_tools.hand_ids[ohid][0]), int(card_tools.hand_ids[ohid][1])
                        if oc1 in my_dead or oc2 in my_dead:
                            opp_valid[ohid] = False
                opp_s = strengths[opp_valid]
                if len(opp_s) > 0:
                    wins = np.sum(strengths[hid] > opp_s)
                    ties = np.sum(strengths[hid] == opp_s) * 0.5
                    relative_strengths[hid] = float((wins + ties) / len(opp_s))

        self._cache[key] = (strengths, relative_strengths, valid_mask)
        return strengths, relative_strengths, valid_mask

    def clear(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# Fast feature encoder (V2)
# ---------------------------------------------------------------------------


class FastFeatureEncoder:
    """
    Encodes NLHEState into V2 feature format:
    - Situation: board_ids[5] + board_mask[5] + numerical[98]
    - Combos: card_ids[C,2] + hand_features[C,2]

    Uses BoardStrengthCache for hand strength and HandClassifier for range encoding.
    """

    def __init__(self, game: NLHEGame):
        self.game = game
        self.effective_stack = game.tree_config.effective_stack
        self.strength_cache = BoardStrengthCache()
        self.hand_classifier = HandClassifier(self.strength_cache)

    def encode_situation(
        self,
        state: NLHEState,
        oop_range: np.ndarray,
        ip_range: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode situation features for V2 model.

        Args:
            state: current game state
            oop_range: [1326] OOP player range weights (required)
            ip_range:  [1326] IP player range weights (required)

        Returns:
            board_ids:     [5] int64 (padded with 52)
            board_mask:    [5] float32
            sit_numerical: [SITUATION_NUMERICAL_DIM] float32 (98 dims: 44 base + 54 range)
        """
        board = state.board

        # Board IDs (pad to 5 with 52)
        board_ids = np.full(5, 52, dtype=np.int64)
        board_mask = np.zeros(5, dtype=np.float32)
        for i, card_id in enumerate(board):
            board_ids[i] = card_id
            board_mask[i] = 1.0

        # Numerical features (98 dims total)
        numerical = np.zeros(SITUATION_NUMERICAL_DIM, dtype=np.float32)
        idx = 0

        # Street one-hot [F, T, R] (3 dims)
        numerical[idx + int(state.street)] = 1.0
        idx += 3

        # Player to act (1 dim)
        numerical[idx] = float(state.current_player())
        idx += 1

        # Pot ratio: pot / (effective_stack * 2) (1 dim)
        pot = state.pot + state.bets_this_street[0] + state.bets_this_street[1]
        pot_ratio = pot / (self.effective_stack * 2) if self.effective_stack > 0 else 0
        numerical[idx] = pot_ratio
        idx += 1

        # SPR: log(remaining_stack / pot + 1) / 4 (1 dim)
        remaining = min(state.stacks[0], state.stacks[1])
        spr = math.log(remaining / max(pot, 1) + 1) / 4.0
        numerical[idx] = spr
        idx += 1

        # OOP committed this street (1 dim)
        numerical[idx] = state.bets_this_street[0] / max(pot, 1)
        idx += 1

        # IP committed this street (1 dim)
        numerical[idx] = state.bets_this_street[1] / max(pot, 1)
        idx += 1

        # Pot odds (1 dim)
        player = state.current_player()
        if player >= 0:
            opp = 1 - player
            bet_to_call = max(state.bets_this_street[opp] - state.bets_this_street[player], 0)
            pot_odds = bet_to_call / max(pot + bet_to_call, 1)
        else:
            pot_odds = 0.0
        numerical[idx] = pot_odds
        idx += 1

        # Bet history: last MAX_BET_HISTORY actions as bet/pot fractions (20 dims)
        hist_idx = 0
        for h in state.history:
            if hist_idx >= MAX_BET_HISTORY:
                break
            if isinstance(h, dict):
                for key, val in h.items():
                    if key in ("Bet", "Raise", "AllIn") and val is not None:
                        numerical[idx + hist_idx] = val / max(pot, 1)
                        hist_idx += 1
                    elif key == "Call":
                        numerical[idx + hist_idx] = 0.0
                        hist_idx += 1
                    elif key == "Check":
                        numerical[idx + hist_idx] = 0.0
                        hist_idx += 1
        idx += MAX_BET_HISTORY  # 20

        # Available actions: pot-fraction for each slot (10 dims)
        if player >= 0:
            for a in state.legal_actions():
                if a == 0:  # Fold
                    numerical[idx + a] = -1.0
                elif a == 1:  # Check/Call
                    opp = 1 - player
                    call_amt = max(state.bets_this_street[opp] - state.bets_this_street[player], 0)
                    numerical[idx + a] = call_amt / max(pot, 1)
                else:
                    amt = state.action_amount(a)
                    numerical[idx + a] = amt / max(pot, 1)
        idx += MAX_ACTIONS  # 10

        # Board mask (repeated for numerical context) (5 dims)
        numerical[idx:idx + 5] = board_mask
        idx += 5

        assert idx == SITUATION_BASE_DIM, f"Expected {SITUATION_BASE_DIM}, got {idx}"

        # --- Range features (54 dims: 27 OOP + 27 IP) ---
        if len(board) >= 3:
            made_labels, draw_labels, eq_buckets = self.hand_classifier.get(board)
            _, _, valid_mask = self.strength_cache.get(board)
        else:
            # Preflop: no meaningful hand classification
            made_labels = np.zeros(1326, dtype=np.int32)
            draw_labels = np.zeros(1326, dtype=np.int32)
            eq_buckets = np.full(1326, 5, dtype=np.int32)  # middle bucket
            valid_mask = np.ones(1326, dtype=bool)
            board_set = set(board)
            for hid in range(1326):
                c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
                if c1 in board_set or c2 in board_set:
                    valid_mask[hid] = False

        oop_features = RangeEncoder.encode(oop_range, made_labels, draw_labels, eq_buckets, valid_mask)
        ip_features = RangeEncoder.encode(ip_range, made_labels, draw_labels, eq_buckets, valid_mask)

        numerical[idx:idx + RANGE_FEATURES_PER_PLAYER] = oop_features
        idx += RANGE_FEATURES_PER_PLAYER
        numerical[idx:idx + RANGE_FEATURES_PER_PLAYER] = ip_features
        idx += RANGE_FEATURES_PER_PLAYER

        assert idx == SITUATION_NUMERICAL_DIM, f"Expected {SITUATION_NUMERICAL_DIM}, got {idx}"

        return board_ids, board_mask, numerical

    def encode_all_combos(
        self,
        player: int,
        board: list[int],
        exclude_cards: Optional[set[int]] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int, int]]]:
        """
        Encode combo features for all valid combos of a player (V2 format).

        Returns:
            combo_card_ids: [C, 2] int64 (hole card IDs)
            hand_features:  [C, COMBO_HAND_FEATURES_DIM] float32 (strength, flush_draw)
            valid_combos:   list of (hand_id, card1, card2)
        """
        cc = self.game.card_config
        player_range = cc.oop_range if player == 0 else cc.ip_range
        board_set = set(board)
        dead = board_set | (exclude_cards or set())

        # Get cached strength data
        if len(board) >= 3:
            _, relative_strengths, _ = self.strength_cache.get(board)
        else:
            relative_strengths = np.full(1326, 0.5, dtype=np.float32)

        # Pre-compute flush info for the board
        board_suit_counts = {}
        for card_id in board:
            _, suit = card_id_to_rank_suit(card_id)
            board_suit_counts[suit] = board_suit_counts.get(suit, 0) + 1
        flush_suit = max(board_suit_counts, key=board_suit_counts.get) if board_suit_counts else -1
        flush_count = board_suit_counts.get(flush_suit, 0) if flush_suit >= 0 else 0

        combos = []
        card_ids_list = []
        features_list = []

        for hid in range(1326):
            c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
            if c1 in dead or c2 in dead:
                continue
            if player_range is not None and player_range[hid] <= 0:
                continue

            combos.append((hid, c1, c2))
            card_ids_list.append([c1, c2])

            # Hand features
            strength = relative_strengths[hid]

            # Flush draw indicator
            flush_draw = 0.0
            if flush_count >= 2:
                matching = 0
                _, s1 = card_id_to_rank_suit(c1)
                _, s2 = card_id_to_rank_suit(c2)
                if s1 == flush_suit:
                    matching += 1
                if s2 == flush_suit:
                    matching += 1
                flush_draw = matching / 2.0

            features_list.append([strength, flush_draw])

        if combos:
            combo_card_ids = np.array(card_ids_list, dtype=np.int64)
            hand_features = np.array(features_list, dtype=np.float32)
        else:
            combo_card_ids = np.zeros((0, 2), dtype=np.int64)
            hand_features = np.zeros((0, COMBO_HAND_FEATURES_DIM), dtype=np.float32)

        return combo_card_ids, hand_features, combos

