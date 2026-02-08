"""
SolverQuery: Inference API for the generalized neural CFR engine (V2).

Given a board, betting history, tree config, and card config, queries
the trained average policy model (BulkPolicyModelV2) to get strategy
for all combos at that node via a single forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from deeppdcfr.bulk_model import BulkPolicyModelV2
from deeppdcfr.card_tools import card_tools
from deeppdcfr.feature_encoder import (
    FastFeatureEncoder,
    canonicalize_board,
    apply_swap_to_range,
    reverse_swap_map,
)
from deeppdcfr.nlhe_game import (
    ACTION_FOLD,
    ACTION_CHECK_CALL,
    ACTION_ALLIN,
    CardConfig,
    NLHEGame,
    NLHEState,
    TreeConfig,
)


@dataclass
class QueryResult:
    """Result of a solver query."""
    strategy: np.ndarray          # [C, A] action probabilities per combo
    combos: list[tuple[str, str]] # List of (card1_str, card2_str) in original suit space
    combo_ids: list[int]          # Hand IDs in original space
    actions: list[str]            # Action names (only legal ones)
    action_indices: list[int]     # Action indices (only legal ones)


class SolverQuery:
    """
    Query interface for the trained V2 neural CFR model.

    Loads the average policy model (ave_policy.pt) and returns Nash
    equilibrium strategy for any board/stack/bet size combination.

    Usage:
        solver = SolverQuery(model_path="models/NLHEGeneralized/")
        result = solver.query(
            board=[18, 29, 33],
            betting_history=["Check", {"Bet": 18}],
            tree_config=TreeConfig(...),
            card_config=CardConfig(...),
        )
        # result.strategy: [C, A] action probabilities per combo
        # result.combos: list of (card1, card2) tuples
        # result.actions: list of action names
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        **model_kwargs,
    ):
        self.device = device
        self.model_path = Path(model_path)
        self.model_kwargs = model_kwargs

        self.model = BulkPolicyModelV2(**model_kwargs).to(device)
        path = self.model_path / "ave_policy.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {path}. "
                f"Train a model first or check the path."
            )
        state_dict = torch.load(path, map_location=device, weights_only=True)
        # Strip _orig_mod. prefix added by torch.compile()
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def query(
        self,
        board: list[int],
        betting_history: Optional[list] = None,
        tree_config: Optional[TreeConfig] = None,
        card_config: Optional[CardConfig] = None,
        header_path: Optional[str] = None,
    ) -> QueryResult:
        """
        Query the model for strategy at a given game state.

        Args:
            board: List of card IDs on the board
            betting_history: List of actions to replay (e.g., ["Check", {"Bet": 18}])
            tree_config: Tree configuration (or loaded from header)
            card_config: Card configuration with ranges
            header_path: Path to header.json (alternative to tree/card config)

        Returns: QueryResult with strategy, combos, actions
        """
        # Load config from header if not provided
        if header_path and (tree_config is None or card_config is None):
            game = NLHEGame.from_header(header_path)
            tree_config = game.tree_config
            card_config = game.card_config

        if tree_config is None or card_config is None:
            raise ValueError("Must provide tree_config+card_config or header_path")

        # Canonicalize board
        canonical_board, swap_map = canonicalize_board(board)
        reverse_map = reverse_swap_map(swap_map)

        # Apply swap to card config
        canonical_card_config = CardConfig(
            flop=[canonical_board[i] for i in range(min(3, len(canonical_board)))],
            turn=canonical_board[3] if len(canonical_board) > 3 else None,
            river=canonical_board[4] if len(canonical_board) > 4 else None,
            oop_range=(
                apply_swap_to_range(card_config.oop_range, swap_map)
                if card_config.oop_range is not None else None
            ),
            ip_range=(
                apply_swap_to_range(card_config.ip_range, swap_map)
                if card_config.ip_range is not None else None
            ),
        )

        # Build game with canonical config
        game = NLHEGame(tree_config, canonical_card_config)
        encoder = FastFeatureEncoder(game)

        # Replay betting history to get current state
        state = game.new_initial_state()
        if betting_history:
            state = self._replay_history(state, betting_history)

        if state.is_terminal() or state.is_chance():
            raise ValueError(f"Cannot query terminal/chance state (player={state.current_player()})")

        player = state.current_player()

        # Build current ranges for range encoding
        oop_range = canonical_card_config.oop_range
        ip_range = canonical_card_config.ip_range
        if oop_range is None:
            oop_range = np.ones(1326, dtype=np.float64)
        if ip_range is None:
            ip_range = np.ones(1326, dtype=np.float64)

        # Encode situation and combos (V2 format)
        board_ids, board_mask, sit_numerical = encoder.encode_situation(
            state, oop_range=oop_range, ip_range=ip_range
        )
        combo_card_ids, hand_features, valid_combos = encoder.encode_all_combos(
            player, state.board
        )
        if len(valid_combos) == 0:
            raise ValueError("No valid combos for this player")

        mask = np.array(state.legal_actions_mask(), dtype=np.float32)

        # Forward pass
        with torch.no_grad():
            bid = torch.tensor(board_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            bmask = torch.tensor(board_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            sit = torch.tensor(sit_numerical, dtype=torch.float32, device=self.device).unsqueeze(0)
            cids = torch.tensor(combo_card_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            hf = torch.tensor(hand_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            amask = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            strategy = self.model(bid, bmask, sit, cids, hf, amask).squeeze(0).cpu().numpy()

        # Map canonical combos back to original suits
        original_combos = []
        original_ids = []
        for _, c1, c2 in valid_combos:
            orig_c1 = _apply_swap(c1, reverse_map)
            orig_c2 = _apply_swap(c2, reverse_map)
            orig_hand = (min(orig_c1, orig_c2), max(orig_c1, orig_c2))
            original_combos.append((
                card_tools.id_to_card(orig_hand[0]),
                card_tools.id_to_card(orig_hand[1]),
            ))
            original_ids.append(
                card_tools.hand_to_id((
                    card_tools.id_to_card(orig_hand[0]),
                    card_tools.id_to_card(orig_hand[1]),
                ))
            )

        # Legal action names
        legal = state.legal_actions()
        action_names = [state.action_name(a) for a in legal]

        return QueryResult(
            strategy=strategy,
            combos=original_combos,
            combo_ids=original_ids,
            actions=action_names,
            action_indices=legal,
        )

    def _replay_history(self, state: NLHEState, history: list) -> NLHEState:
        """Replay a sequence of actions to reach a target state."""
        for action in history:
            if state.is_chance():
                if isinstance(action, int):
                    state = state.deal_card(action)
                elif isinstance(action, dict) and "Deal" in action:
                    state = state.deal_card(action["Deal"])
                else:
                    raise ValueError(f"Expected card to deal at chance node, got: {action}")
                continue

            action_idx = self._map_action(state, action)
            if action_idx is None:
                raise ValueError(f"Cannot map action {action} at state with legal actions: "
                                f"{[state.action_name(a) for a in state.legal_actions()]}")
            state = state.child(action_idx)

        return state

    def _map_action(self, state: NLHEState, action) -> Optional[int]:
        """Map a human-readable action to action index."""
        if action == "Fold":
            return ACTION_FOLD if ACTION_FOLD in state.legal_actions() else None
        if action in ("Check", "Call"):
            return ACTION_CHECK_CALL
        if isinstance(action, dict):
            if "AllIn" in action:
                return ACTION_ALLIN
            amount = action.get("Bet") or action.get("Raise")
            if amount is not None:
                best = None
                best_diff = float("inf")
                for a in state.legal_actions():
                    diff = abs(state.action_amount(a) - amount)
                    if diff < best_diff:
                        best_diff = diff
                        best = a
                return best
        return None


def _apply_swap(card_id: int, swap_map: dict[int, int]) -> int:
    """Apply suit swap to a card ID."""
    rank = card_id // 4
    suit = card_id % 4
    return rank * 4 + swap_map.get(suit, suit)


def query_from_header(
    model_path: str,
    header_path: str,
    betting_history: Optional[list] = None,
    device: str = "cpu",
) -> QueryResult:
    """Convenience function to query a model using a header file."""
    import json

    with open(header_path) as f:
        header = json.load(f)

    config = header["configuration"]
    tree_config = TreeConfig.from_dict(config["tree_config"])
    card_config = CardConfig.from_dict(config["card_config"])

    board = list(card_config.flop)
    if card_config.turn is not None:
        board.append(card_config.turn)
    if card_config.river is not None:
        board.append(card_config.river)

    solver = SolverQuery(model_path=model_path, device=device)
    return solver.query(
        board=board,
        betting_history=betting_history,
        tree_config=tree_config,
        card_config=card_config,
    )
