"""
PostflopVRDeepPDCFR: GPU-parallel VR-PDCFR+ training pipeline.

Uses External Sampling MCCFR: at each decision node, compute regrets for ALL
combos in a single forward pass. One action is sampled per node (shared across
combos), and opponent reach probabilities are tracked as a vector.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import multiprocessing as mp

import numpy as np
import torch

from deeppdcfr.bulk_model import (
    BulkPolicyTrainer,
    BulkRegretTrainer,
)
from deeppdcfr.feature_encoder import (
    BoardStrengthCache,
    FastFeatureEncoder,
    canonicalize_board,
)
from deeppdcfr.logger import Logger
from deeppdcfr.nlhe_game import (
    MAX_ACTIONS,
    AllIn as AllInSize,
    CardConfig,
    NLHEGame,
    NLHEState,
    PotRelative,
    Street,
    StreetBetConfig,
    TreeConfig,
    parse_bet_size,
)
from deeppdcfr.card_tools import card_tools
from deeppdcfr.utils import set_seed


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

# Default tree config templates
DEFAULT_BET_SIZES_SMALL = {
    "bet": [{"PotRelative": 0.33}, {"PotRelative": 0.67}, "AllIn"],
    "raise": [{"PotRelative": 0.5}, "AllIn"],
}

DEFAULT_BET_SIZES_MEDIUM = {
    "bet": [{"PotRelative": 0.25}, {"PotRelative": 0.5}, {"PotRelative": 0.75}, "AllIn"],
    "raise": [{"PotRelative": 0.33}, {"PotRelative": 0.67}, "AllIn"],
}

DEFAULT_BET_SIZES_LARGE = {
    "bet": [
        {"PotRelative": 0.2}, {"PotRelative": 0.33},
        {"PotRelative": 0.55}, {"PotRelative": 0.83},
        {"PotRelative": 1.25}, "AllIn",
    ],
    "raise": [{"PotRelative": 0.33}, {"PotRelative": 0.55}, "AllIn"],
}


@dataclass
class Scenario:
    """A randomly generated training scenario."""
    tree_config: TreeConfig
    card_config: CardConfig
    canonical_board: list[int]
    swap_map: dict[int, int]


class ScenarioGenerator:
    """Generates diverse training scenarios for generalized neural CFR."""

    def __init__(
        self,
        stack_range: tuple[int, int] = (20, 200),
        pot_range: tuple[int, int] = (3, 60),
        streets: list[str] = None,
        seed: int = 0,
        fixed_oop_bet_config: Optional[dict] = None,
        fixed_ip_bet_config: Optional[dict] = None,
    ):
        self.stack_range = stack_range
        self.pot_range = pot_range
        self.streets = streets or ["flop", "turn", "river"]
        self.rng = np.random.RandomState(seed)

        # Parse fixed bet configs if provided
        self.fixed_oop_bet_config = self._parse_bet_config(fixed_oop_bet_config) if fixed_oop_bet_config else None
        self.fixed_ip_bet_config = self._parse_bet_config(fixed_ip_bet_config) if fixed_ip_bet_config else None

        # Pre-enumerate canonical flop combos (sample on demand)
        self._all_flops = list(self._enumerate_canonical_flops())

    @staticmethod
    def _parse_bet_config(raw: dict) -> StreetBetConfig:
        """Parse a raw dict like {"bet": [...], "raise": [...]} into a StreetBetConfig."""
        return StreetBetConfig(
            bet=[parse_bet_size(s) for s in raw.get("bet", [])],
            raise_=[parse_bet_size(s) for s in raw.get("raise", [])],
        )

    def _enumerate_canonical_flops(self):
        """Yield all unique canonical flop combinations."""
        seen = set()
        for c1 in range(52):
            for c2 in range(c1 + 1, 52):
                for c3 in range(c2 + 1, 52):
                    board = [c1, c2, c3]
                    canonical, _ = canonicalize_board(board)
                    key = tuple(sorted(canonical))
                    if key not in seen:
                        seen.add(key)
                        yield canonical

    def sample(self) -> Scenario:
        """Sample a random training scenario."""
        # Random canonical flop
        flop = list(random.choice(self._all_flops))
        canonical_board, swap_map = canonicalize_board(flop)

        # Random street
        street = random.choice(self.streets)

        # Deal turn/river if needed
        board = list(canonical_board)
        dead = set(board)
        deck = [c for c in range(52) if c not in dead]

        turn = None
        river = None
        if street in ("turn", "river"):
            turn = random.choice(deck)
            board.append(turn)
            deck.remove(turn)
        if street == "river":
            river = random.choice(deck)
            board.append(river)
            deck.remove(river)

        # Random stack and pot
        eff_stack = self.rng.randint(self.stack_range[0], self.stack_range[1] + 1)
        starting_pot = self.rng.randint(
            min(self.pot_range[0], eff_stack),
            min(self.pot_range[1], eff_stack) + 1,
        )

        # Bet sizes: use fixed configs if provided, else random
        if self.fixed_oop_bet_config and self.fixed_ip_bet_config:
            oop_config = self.fixed_oop_bet_config
            ip_config = self.fixed_ip_bet_config
        elif self.fixed_oop_bet_config or self.fixed_ip_bet_config:
            # One fixed, one random
            oop_config = self.fixed_oop_bet_config or self._generate_random_bet_config()
            ip_config = self.fixed_ip_bet_config or self._generate_random_bet_config()
        else:
            # Random bet sizes: 50% from templates, 50% fully randomized continuous
            if random.random() < 0.5:
                bet_template = random.choice([
                    DEFAULT_BET_SIZES_SMALL, DEFAULT_BET_SIZES_MEDIUM, DEFAULT_BET_SIZES_LARGE
                ])
                street_config = StreetBetConfig(
                    bet=[parse_bet_size(s) for s in bet_template["bet"]],
                    raise_=[parse_bet_size(s) for s in bet_template["raise"]],
                )
            else:
                street_config = self._generate_random_bet_config()
            oop_config = street_config
            ip_config = street_config

        tree_config = TreeConfig(
            effective_stack=eff_stack,
            starting_pot=starting_pot,
            initial_state={"flop": Street.FLOP, "turn": Street.TURN, "river": Street.RIVER}[street],
            add_allin_threshold=1.5,
            force_allin_threshold=0.2,
            max_raises_per_street=0,
            merging_threshold=0.1,
            rake_rate=0.0,
            rake_cap=0.0,
            flop_bet_sizes=[oop_config, ip_config],
            turn_bet_sizes=[oop_config, ip_config],
            river_bet_sizes=[oop_config, ip_config],
        )

        # Random ranges (diverse strategies for range-aware training)
        oop_range = self._generate_random_range(dead, board)
        ip_range = self._generate_random_range(dead, board)

        card_config = CardConfig(
            flop=canonical_board,
            turn=turn,
            river=river,
            oop_range=oop_range,
            ip_range=ip_range,
        )

        return Scenario(
            tree_config=tree_config,
            card_config=card_config,
            canonical_board=board,
            swap_map=swap_map,
        )

    def _generate_random_range(
        self, dead_cards: set[int], board: list[int]
    ) -> np.ndarray:
        """
        Generate a random range [1326] with diverse strategies.

        Strategies (randomly chosen):
        - 30% Uniform: all valid combos equal weight
        - 30% Random zero-out: remove 20-80% of combos randomly
        - 20% Equity-based: keep hands above/below a random threshold
        - 20% Random weights: each hand gets random [0, 1] weight
        """
        player_range = np.zeros(1326, dtype=np.float64)

        # Set valid combos (not conflicting with dead cards)
        for hid in range(1326):
            c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
            if c1 not in dead_cards and c2 not in dead_cards:
                player_range[hid] = 1.0

        strategy = random.random()

        if strategy < 0.3:
            # Uniform: keep as-is
            pass
        elif strategy < 0.6:
            # Random zero-out: remove 20-80% of combos
            valid_indices = np.where(player_range > 0)[0]
            remove_frac = 0.2 + random.random() * 0.6  # 20-80%
            num_remove = int(len(valid_indices) * remove_frac)
            if num_remove > 0 and num_remove < len(valid_indices):
                remove_idx = np.random.choice(valid_indices, size=num_remove, replace=False)
                player_range[remove_idx] = 0.0
        elif strategy < 0.8:
            # Equity-based: keep hands above/below a random threshold
            if len(board) >= 3:
                if not hasattr(self, '_strength_cache'):
                    self._strength_cache = BoardStrengthCache()
                _, rel_strengths, _ = self._strength_cache.get(board)
                threshold = 0.2 + random.random() * 0.6  # threshold in [0.2, 0.8]
                if random.random() < 0.5:
                    # Keep strong hands
                    player_range *= (rel_strengths >= threshold)
                else:
                    # Keep weak hands
                    player_range *= (rel_strengths <= threshold)
                # Ensure at least some combos survive
                if player_range.sum() < 1e-10:
                    for hid in range(1326):
                        c1, c2 = int(card_tools.hand_ids[hid][0]), int(card_tools.hand_ids[hid][1])
                        if c1 not in dead_cards and c2 not in dead_cards:
                            player_range[hid] = 1.0
        else:
            # Random weights: each hand gets random [0, 1] weight
            valid_mask = player_range > 0
            player_range[valid_mask] = np.random.random(valid_mask.sum())

        # Normalize
        total = player_range.sum()
        if total > 0:
            player_range /= total

        return player_range

    def _generate_random_bet_config(self) -> 'StreetBetConfig':
        """
        Generate a StreetBetConfig with random continuous bet sizes.

        Bet sizes: 1-5 random pot-relative values in [0.15, 2.0] + AllIn
        Raise sizes: 1-3 random pot-relative values in [0.2, 1.5] + AllIn
        Sizes are sorted ascending and spaced to avoid merging.
        """
        # Random number of bet sizes (1-5)
        num_bets = random.randint(1, 5)
        bet_fracs = sorted(random.uniform(0.15, 2.0) for _ in range(num_bets))
        # Ensure minimum spacing of 0.08 to avoid merging
        spaced_bets = [bet_fracs[0]]
        for f in bet_fracs[1:]:
            if f - spaced_bets[-1] >= 0.08:
                spaced_bets.append(f)
        bet_sizes = [PotRelative(round(f, 3)) for f in spaced_bets] + [AllInSize()]

        # Random number of raise sizes (1-3)
        num_raises = random.randint(1, 3)
        raise_fracs = sorted(random.uniform(0.2, 1.5) for _ in range(num_raises))
        spaced_raises = [raise_fracs[0]]
        for f in raise_fracs[1:]:
            if f - spaced_raises[-1] >= 0.08:
                spaced_raises.append(f)
        raise_sizes = [PotRelative(round(f, 3)) for f in spaced_raises] + [AllInSize()]

        return StreetBetConfig(bet=bet_sizes, raise_=raise_sizes)


# ---------------------------------------------------------------------------
# Terminal value computation
# ---------------------------------------------------------------------------


def compute_showdown_values(
    trav_combos: list[tuple[int, int, int]],
    opp_combos: list[tuple[int, int, int]],
    opp_reach: np.ndarray,
    board: list[int],
    pot: int,
    trav_stacks_invested: float,
    opp_stacks_invested: float,
    rake_rate: float = 0.0,
    rake_cap: float = 0.0,
) -> np.ndarray:
    """
    Compute showdown values for ALL traverser combos against ALL opponent combos
    weighted by opp_reach.

    Returns: [C_trav] values for traverser (in chips, not normalized)
    """
    C_trav = len(trav_combos)
    C_opp = len(opp_combos)

    if C_trav == 0 or C_opp == 0:
        return np.zeros(C_trav, dtype=np.float32)

    board_arr = np.array(board, dtype=int)

    # Evaluate all traverser hands
    trav_cards = np.zeros((C_trav, len(board) + 2), dtype=int)
    trav_cards[:, :len(board)] = board_arr
    for i, (_, c1, c2) in enumerate(trav_combos):
        trav_cards[i, len(board)] = c1
        trav_cards[i, len(board) + 1] = c2
    trav_strengths = card_tools.evaluate(trav_cards)  # [C_trav]

    # Evaluate all opponent hands
    opp_cards = np.zeros((C_opp, len(board) + 2), dtype=int)
    opp_cards[:, :len(board)] = board_arr
    for i, (_, c1, c2) in enumerate(opp_combos):
        opp_cards[i, len(board)] = c1
        opp_cards[i, len(board) + 1] = c2
    opp_strengths = card_tools.evaluate(opp_cards)  # [C_opp]

    # Build card conflict mask: [C_trav, C_opp]
    # True where cards don't conflict (valid matchup)
    trav_card_set = [(c1, c2) for (_, c1, c2) in trav_combos]
    opp_card_set = [(c1, c2) for (_, c1, c2) in opp_combos]

    valid_mask = np.ones((C_trav, C_opp), dtype=np.float32)
    for i, (tc1, tc2) in enumerate(trav_card_set):
        for j, (oc1, oc2) in enumerate(opp_card_set):
            if tc1 == oc1 or tc1 == oc2 or tc2 == oc1 or tc2 == oc2:
                valid_mask[i, j] = 0.0

    # Payoff matrix: [C_trav, C_opp]
    # Compare strengths: trav wins (+opp_invested), loses (-trav_invested), ties (0)
    # trav_strengths: [C_trav], opp_strengths: [C_opp]
    win_matrix = (trav_strengths[:, None] > opp_strengths[None, :]).astype(np.float32)
    lose_matrix = (trav_strengths[:, None] < opp_strengths[None, :]).astype(np.float32)

    rake = 0.0
    if rake_rate > 0:
        rake = min(pot * rake_rate, rake_cap)

    payoff_matrix = (
        win_matrix * (opp_stacks_invested - rake / 2)
        - lose_matrix * (trav_stacks_invested + rake / 2)
    )

    # Apply card conflict mask and opponent reach weighting
    weighted_payoffs = payoff_matrix * valid_mask * opp_reach[None, :]  # [C_trav, C_opp]

    # Normalize by total valid reach weight per traverser combo
    valid_reach = (valid_mask * opp_reach[None, :]).sum(axis=1)  # [C_trav]
    valid_reach = np.maximum(valid_reach, 1e-10)

    # EV = sum(payoff * opp_reach * valid) / sum(opp_reach * valid)
    values = weighted_payoffs.sum(axis=1) / valid_reach  # [C_trav]

    return values.astype(np.float32)


# ---------------------------------------------------------------------------
# PostflopVRDeepPDCFR training pipeline (V2 — External Sampling)
# ---------------------------------------------------------------------------


class PostflopVRDeepPDCFR:
    """
    Neural CFR training pipeline using External Sampling MCCFR.

    At each decision node, computes strategy for ALL combos in one GPU forward
    pass. Samples ONE action (shared across combos). Tracks opponent reach
    probabilities as a vector.
    """

    def __init__(
        self,
        num_iterations: int = 300,
        num_traversals: int = 500,
        advantage_buffer_size: int = 10_000,
        ave_policy_buffer_size: int = 20_000,
        learning_rate: float = 3e-4,
        advantage_network_train_steps: int = 500,
        ave_policy_network_train_steps: int = 2000,
        advantage_batch_size: int = 64,
        ave_policy_batch_size: int = 64,
        evaluation_frequency: int = 3,
        epsilon: float = 0.6,
        alpha: float = 2.3,
        gamma: float = 2.0,
        use_regret_matching_argmax: bool = True,
        reinitialize_advantage_networks: bool = False,
        reinitialize_imm_regret_networks: bool = True,
        stack_range: tuple[int, int] = (20, 200),
        pot_range: tuple[int, int] = (3, 60),
        fixed_oop_bet_config: Optional[dict] = None,
        fixed_ip_bet_config: Optional[dict] = None,
        device: str = "cpu",
        seed: int = 0,
        max_time: int = 0,
        save_interval: int = 600,
        save_dir: str = "models/NLHEGeneralized",
        resume: bool = False,
        traversal_workers: int = 1,
        traversal_device: Optional[str] = None,
        traversal_mp_context: str = "spawn",
        traversal_chunk_size: int = 0,
        logger: Optional[Logger] = None,
        # V2 model architecture params
        card_embed_dim: int = 64,
        situation_hidden: int = 384,
        combo_hidden: int = 192,
        fusion_hidden: int = 384,
        fusion_residual_blocks: int = 2,
    ):
        self.num_iterations = num_iterations
        self.num_traversals = num_traversals
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.evaluation_frequency = evaluation_frequency
        self.reinitialize_advantage_networks = reinitialize_advantage_networks
        self.device = device
        self.max_time = max_time
        self.save_interval = save_interval
        self.save_dir = Path(save_dir)
        self.logger = logger or Logger(writer_strings=[])

        set_seed(seed)
        self.seed = seed

        self.traversal_workers = max(1, int(traversal_workers))
        self.traversal_device = traversal_device or device
        self.traversal_mp_context = traversal_mp_context
        self.traversal_chunk_size = max(0, int(traversal_chunk_size))

        # Initialize scenario generator
        self.scenario_gen = ScenarioGenerator(
            stack_range=stack_range, pot_range=pot_range, seed=seed,
            fixed_oop_bet_config=fixed_oop_bet_config,
            fixed_ip_bet_config=fixed_ip_bet_config,
        )

        # V2 model kwargs
        model_kwargs = dict(
            card_embed_dim=card_embed_dim,
            situation_hidden=situation_hidden,
            combo_hidden=combo_hidden,
            fusion_hidden=fusion_hidden,
            fusion_residual_blocks=fusion_residual_blocks,
        )

        # Initialize trainers with V2 architecture
        self.regret_trainers = [
            BulkRegretTrainer(
                learning_rate=learning_rate,
                buffer_size=advantage_buffer_size,
                batch_size=advantage_batch_size,
                train_steps=advantage_network_train_steps,
                alpha=alpha,
                use_regret_matching_argmax=use_regret_matching_argmax,
                reinitialize_imm=reinitialize_imm_regret_networks,
                device=device,
                **model_kwargs,
            )
            for _ in range(2)
        ]

        self.ave_policy_trainer = BulkPolicyTrainer(
            learning_rate=learning_rate,
            buffer_size=ave_policy_buffer_size,
            batch_size=ave_policy_batch_size,
            train_steps=ave_policy_network_train_steps,
            gamma=gamma,
            device=device,
            **model_kwargs,
        )

        # Tracking
        self.num_iteration = 0
        self.nodes_touched = 0
        self.episode = 0

        # Resume from checkpoint
        if resume:
            self._load_checkpoint()

    def _mp_ctx(self):
        try:
            return mp.get_context(self.traversal_mp_context)
        except ValueError:
            self.logger.info(
                f"Invalid traversal_mp_context={self.traversal_mp_context!r}; falling back to 'spawn'."
            )
            return mp.get_context("spawn")

    def solve(self):
        """Main training loop."""
        self.start_time = time.time()
        self.last_save_time = self.start_time

        iteration = 0
        while self.num_iterations == 0 or iteration < self.num_iterations:
            if self.max_time > 0 and time.time() - self.start_time >= self.max_time:
                self.logger.info(f"Time limit reached ({time.time() - self.start_time:.0f}s). Stopping.")
                break

            self.iteration()
            iteration += 1

            if self.save_interval > 0 and time.time() - self.last_save_time >= self.save_interval:
                self.save_models(checkpoint=True)
                self.last_save_time = time.time()

        self.save_models()

    def iteration(self):
        """One iteration: collect data via vectorized DFS, train regrets, train policy."""
        self.num_iteration += 1

        # Get scenario for this iteration
        scenario = self._get_scenario()
        game = NLHEGame(scenario.tree_config, scenario.card_config)
        encoder = FastFeatureEncoder(game)

        for player in range(2):
            self.regret_trainers[player].reset_buffer()
            self.collect_training_data(game, encoder, scenario, player)
            self.train_regret(player)

        should_eval = (
            self.num_iteration % self.evaluation_frequency == 0 or
            (self.num_iteration < self.evaluation_frequency and
             self.num_iteration % max(self.evaluation_frequency // 3, 1) == 0)
        )
        if should_eval:
            self.train_average_policy()
            self.evaluate()

    def collect_training_data(
        self, game: NLHEGame, encoder: FastFeatureEncoder,
        scenario: Scenario, traverser: int
    ):
        """Run vectorized traversals to collect regret data for all combos."""
        opp = 1 - traverser
        self.logger.info(
            f"[it {self.num_iteration}][p{traverser}] traverse start | n={self.num_traversals}"
        )
        nodes_before = self.nodes_touched
        episode_before = self.episode

        # If using multiprocessing, shard traversals across workers and merge buffers back.
        if self.traversal_workers > 1:
            self._collect_training_data_mp(scenario, traverser)
            self.logger.info(
                f"[it {self.num_iteration}][p{traverser}] traverse done | +ep={self.episode - episode_before} | +nodes={self.nodes_touched - nodes_before}"
            )
            return

        # Pre-encode combos for both players (shared across traversals on same board)
        trav_combos = game.valid_combos(traverser)
        opp_combos = game.valid_combos(opp)
        progress_interval = max(1, self.num_traversals // 4)

        for tid in range(1, self.num_traversals + 1):
            self.episode += 1
            root = game.new_initial_state()

            # Initial opponent reach: uniform over range
            cc = game.card_config
            opp_range = cc.ip_range if traverser == 0 else cc.oop_range
            # Build initial opp_reach from range (only valid combos)
            opp_reach = np.zeros(len(opp_combos), dtype=np.float32)
            for i, (hid, _, _) in enumerate(opp_combos):
                if opp_range is not None:
                    opp_reach[i] = opp_range[hid]
                else:
                    opp_reach[i] = 1.0

            # Normalize
            reach_sum = opp_reach.sum()
            if reach_sum > 0:
                opp_reach /= reach_sum

            self.vectorized_dfs(
                root, traverser, encoder, game,
                trav_combos, opp_combos, opp_reach,
            )
            if tid % progress_interval == 0 or tid == self.num_traversals:
                self.logger.info(
                    f"[it {self.num_iteration}][p{traverser}] traverse {tid}/{self.num_traversals}"
                )
        self.logger.info(
            f"[it {self.num_iteration}][p{traverser}] traverse done | +ep={self.episode - episode_before} | +nodes={self.nodes_touched - nodes_before}"
        )

    def _collect_training_data_mp(self, scenario: Scenario, traverser: int):
        def _cpu_state_dict(m: torch.nn.Module) -> dict:
            # torch.compile wraps modules in an OptimizedModule that prefixes keys with "_orig_mod.".
            # Workers disable compile, so we always serialize the original module's keys.
            orig = getattr(m, "_orig_mod", None)
            sd = (orig.state_dict() if orig is not None else m.state_dict())
            # Defensive: if someone passed in a prefixed dict, strip it.
            if sd and all(k.startswith("_orig_mod.") for k in sd.keys()):
                sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
            return {k: v.detach().cpu() for k, v in sd.items()}

        # Snapshot model weights once per traversal phase.
        reg_state = []
        imm_state = []
        for p in range(2):
            reg_state.append(_cpu_state_dict(self.regret_trainers[p].model))
            imm_state.append(_cpu_state_dict(self.regret_trainers[p].imm_model))

        # Split traversals into chunks so we can stream progress while workers run.
        workers = self.traversal_workers
        n = int(self.num_traversals)
        chunk = self.traversal_chunk_size if self.traversal_chunk_size > 0 else max(1, n // max(1, workers * 4))

        # Worker payload keeps config minimal; workers build their own game/encoder.
        payloads = []
        remaining = n
        task_id = 0
        while remaining > 0:
            w_n = min(chunk, remaining)
            wid = task_id % workers
            payloads.append(
                dict(
                    task_id=task_id,
                    wid=wid,
                    n_traversals=w_n,
                    traverser=traverser,
                    scenario=scenario,
                    num_iteration=self.num_iteration,
                    traversal_device=self.traversal_device,
                    seed=self.seed,
                )
            )
            remaining -= w_n
            task_id += 1

        t0 = time.perf_counter()
        ctx = self._mp_ctx()
        self.logger.info(
            f"[it {self.num_iteration}][p{traverser}] mp traverse | workers={workers} device={self.traversal_device} chunks={len(payloads)} chunk_size={chunk}"
        )
        # NOTE: With spawn, sending large tensors per-task is extremely slow (and kills parallelism).
        # Push common state once via the pool initializer.
        init_args = (
            reg_state,
            imm_state,
            dict(
                epsilon=self.epsilon,
                alpha=self.alpha,
                use_regret_matching_argmax=self.regret_trainers[0].use_regret_matching_argmax,
                # Buffer sizes: make them large enough to avoid local reservoir replacement.
                advantage_buffer_size=self.regret_trainers[0].buffer.buffer_size * max(1, workers),
                ave_policy_buffer_size=self.ave_policy_trainer.buffer.buffer_size * max(1, workers),
                # Model architecture + LR only needed to construct the models.
                learning_rate=self.regret_trainers[0].learning_rate,
                card_embed_dim=self.regret_trainers[0].model_kwargs["card_embed_dim"],
                situation_hidden=self.regret_trainers[0].model_kwargs["situation_hidden"],
                combo_hidden=self.regret_trainers[0].model_kwargs["combo_hidden"],
                fusion_hidden=self.regret_trainers[0].model_kwargs["fusion_hidden"],
                fusion_residual_blocks=self.regret_trainers[0].model_kwargs["fusion_residual_blocks"],
            ),
        )
        total_nodes = 0
        total_ep = 0
        done_tasks = 0
        total_tasks = len(payloads)
        progress_interval = max(1, total_tasks // 8)
        with ctx.Pool(processes=workers, initializer=_mp_worker_init, initargs=init_args) as pool:
            for r in pool.imap_unordered(_mp_collect_traversals, payloads):
                total_ep += r["episodes"]
                self.nodes_touched += r["nodes_touched"]
                total_nodes += r["nodes_touched"]

                for node in r["regret_nodes"]:
                    self.regret_trainers[traverser].add_node_data(
                        node["board_ids"],
                        node["sit_numerical"],
                        node["combo_card_ids"],
                        node["hand_features"],
                        node["values"],
                        node["action_mask"],
                        node["iteration"],
                    )
                for node in r["policy_nodes"]:
                    self.ave_policy_trainer.add_node_data(
                        node["board_ids"],
                        node["sit_numerical"],
                        node["combo_card_ids"],
                        node["hand_features"],
                        node["values"],
                        node["action_mask"],
                        node["iteration"],
                    )

                done_tasks += 1
                if done_tasks % progress_interval == 0 or done_tasks == total_tasks:
                    self.logger.info(
                        f"[it {self.num_iteration}][p{traverser}] mp traverse progress | tasks={done_tasks}/{total_tasks} ep={total_ep}/{n}"
                    )

        self.episode += total_ep
        dt = (time.perf_counter() - t0) * 1000.0
        self.logger.info(
            f"[it {self.num_iteration}][p{traverser}] mp merge done | +ep={total_ep} | workers={workers} | dt_ms={dt:.1f}"
        )

    def vectorized_dfs(
        self,
        state: NLHEState,
        traverser: int,
        encoder: FastFeatureEncoder,
        game: NLHEGame,
        trav_combos: list[tuple[int, int, int]],
        opp_combos: list[tuple[int, int, int]],
        opp_reach: np.ndarray,
    ) -> np.ndarray:
        """
        External sampling DFS that processes ALL combos at each node.

        Args:
            state: current game state
            traverser: player 0 or 1
            encoder: FastFeatureEncoder instance
            game: NLHEGame instance
            trav_combos: list of (hid, c1, c2) for traverser
            opp_combos: list of (hid, c1, c2) for opponent
            opp_reach: [C_opp] opponent reach probabilities

        Returns: [C_trav] values for traverser (in chips)
        """
        self.nodes_touched += 1
        C_trav = len(trav_combos)
        max_utility = game.tree_config.effective_stack

        # Terminal
        if state.is_terminal():
            return self._compute_terminal_values(
                state, traverser, game, trav_combos, opp_combos, opp_reach
            )

        # Chance node
        if state.is_chance():
            outcomes = state.chance_outcomes()
            # Sample one card, excluding cards that are in BOTH players' ranges
            # (we can't exclude per-combo cards in external sampling, just board)
            cards, probs = zip(*outcomes)
            probs = np.array(probs)
            probs /= probs.sum()
            idx = np.random.choice(len(cards), p=probs)
            card = cards[idx]

            next_state = state.deal_card(card)

            # Filter out combos that conflict with the new card
            new_trav_combos = [(h, c1, c2) for h, c1, c2 in trav_combos if c1 != card and c2 != card]
            new_opp_combos = []
            new_opp_reach = []
            for i, (h, c1, c2) in enumerate(opp_combos):
                if c1 != card and c2 != card:
                    new_opp_combos.append((h, c1, c2))
                    new_opp_reach.append(opp_reach[i])
            new_opp_reach = np.array(new_opp_reach, dtype=np.float32) if new_opp_reach else np.zeros(0, dtype=np.float32)

            # Renormalize opp reach
            reach_sum = new_opp_reach.sum()
            if reach_sum > 0:
                new_opp_reach /= reach_sum

            if len(new_trav_combos) == 0:
                return np.zeros(C_trav, dtype=np.float32)

            # Recurse and map back to original combo indices
            child_values = self.vectorized_dfs(
                next_state, traverser, encoder, game,
                new_trav_combos, new_opp_combos, new_opp_reach,
            )

            # Map child values back to original trav_combos ordering
            result = np.zeros(C_trav, dtype=np.float32)
            new_idx = 0
            for orig_idx, (h, c1, c2) in enumerate(trav_combos):
                if c1 != card and c2 != card:
                    result[orig_idx] = child_values[new_idx]
                    new_idx += 1
            return result

        # Decision node
        player = state.current_player()
        is_traverser = (player == traverser)

        # Get the acting player's combos and data
        if is_traverser:
            acting_combos = trav_combos
        else:
            acting_combos = opp_combos

        C_acting = len(acting_combos)
        if C_acting == 0:
            return np.zeros(C_trav, dtype=np.float32)

        # Build current evolved ranges [1326] for range encoding
        cc = game.card_config
        trav_range_full = np.zeros(1326, dtype=np.float64)
        for hid, c1, c2 in trav_combos:
            trav_range_full[hid] = cc.oop_range[hid] if traverser == 0 else cc.ip_range[hid]

        opp_range_full = np.zeros(1326, dtype=np.float64)
        for i, (hid, c1, c2) in enumerate(opp_combos):
            opp_range_full[hid] = opp_reach[i]

        # Assign to OOP/IP
        if traverser == 0:
            current_oop_range, current_ip_range = trav_range_full, opp_range_full
        else:
            current_oop_range, current_ip_range = opp_range_full, trav_range_full

        # Encode situation and combos for acting player
        board_ids, board_mask, sit_numerical = encoder.encode_situation(
            state, oop_range=current_oop_range, ip_range=current_ip_range
        )
        combo_card_ids, hand_features, _ = encoder.encode_all_combos(
            player, state.board
        )

        mask = np.array(state.legal_actions_mask(), dtype=np.float32)
        legal_actions = state.legal_actions()

        # Get strategy from regret trainer for ALL combos
        strategy = self.regret_trainers[player].get_policy(
            board_ids, board_mask, sit_numerical,
            combo_card_ids, hand_features, mask,
            self.num_iteration,
        )  # [C_acting, MAX_ACTIONS]

        # Sample ONE action (from mean strategy across combos)
        mean_strategy = strategy.mean(axis=0) * mask
        sp_sum = mean_strategy.sum()
        if sp_sum > 0:
            mean_strategy /= sp_sum
        else:
            mean_strategy = mask / max(mask.sum(), 1)

        # Exploration for traverser's nodes
        if is_traverser:
            uniform = mask / max(mask.sum(), 1)
            sample_strategy = uniform * self.epsilon + mean_strategy * (1 - self.epsilon)
            sample_strategy /= max(sample_strategy.sum(), 1e-10)
        else:
            sample_strategy = mean_strategy.copy()

        # Sample from legal actions only
        legal_probs = sample_strategy[legal_actions]
        legal_probs = np.maximum(legal_probs, 0)
        lp_sum = legal_probs.sum()
        if lp_sum > 0:
            legal_probs /= lp_sum
        else:
            legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
        chosen_idx = np.random.choice(len(legal_actions), p=legal_probs)
        action = legal_actions[chosen_idx]
        sample_prob = sample_strategy[action]

        # Update opponent reach if this is opponent's node
        if not is_traverser:
            # Map strategy from encode_all_combos order to opp_combos order
            # The encoder produces combos in the same hid order as game.valid_combos()
            # so they should align, but we need to be careful
            new_opp_reach = opp_reach.copy()
            # strategy is [C_acting, MAX_ACTIONS] for acting_combos (= opp_combos)
            for i in range(len(opp_combos)):
                if i < strategy.shape[0]:
                    new_opp_reach[i] *= strategy[i, action]
                else:
                    new_opp_reach[i] = 0.0

            # Record opponent's strategy for average policy
            self.ave_policy_trainer.add_node_data(
                board_ids, sit_numerical, combo_card_ids,
                hand_features, strategy, mask, self.num_iteration,
            )
        else:
            new_opp_reach = opp_reach

        # Recurse
        next_state = state.child(action)
        child_values = self.vectorized_dfs(
            next_state, traverser, encoder, game,
            trav_combos, opp_combos, new_opp_reach,
        )  # [C_trav]

        # Compute regrets for traverser's nodes
        if is_traverser:
            # Build q_values for the sampled action
            q_values = np.zeros((C_trav, MAX_ACTIONS), dtype=np.float32)
            q_values[:, action] = child_values / max(sample_prob, 1e-10)

            # Node values: v(c) = sum_a strategy(c,a) * q(c,a)
            node_values = (q_values * strategy[:C_trav]).sum(axis=1)  # [C_trav]

            # Counterfactual regrets for ALL legal actions
            cf_regrets = np.zeros((C_trav, MAX_ACTIONS), dtype=np.float32)
            for a in legal_actions:
                cf_regrets[:, a] = q_values[:, a] - node_values

            # Normalize regrets by max_utility for numerical stability
            cf_regrets /= max(max_utility, 1)

            # Store regret data for all combos at this node
            self.regret_trainers[player].add_node_data(
                board_ids, sit_numerical, combo_card_ids,
                hand_features, cf_regrets, mask, self.num_iteration,
            )

            return node_values
        else:
            return child_values

    def _compute_terminal_values(
        self,
        state: NLHEState,
        traverser: int,
        game: NLHEGame,
        trav_combos: list[tuple[int, int, int]],
        opp_combos: list[tuple[int, int, int]],
        opp_reach: np.ndarray,
    ) -> np.ndarray:
        """Compute terminal values for ALL traverser combos."""
        C_trav = len(trav_combos)
        eff = game.tree_config.effective_stack

        if state.is_fold_terminal:
            # Fold: same payoff for all combos
            payoffs = state.fold_payoffs
            return np.full(C_trav, payoffs[traverser], dtype=np.float32)

        if state.is_showdown_terminal:
            # Showdown: evaluate all matchups
            trav_invested = eff - state.stacks[traverser]
            opp_invested = eff - state.stacks[1 - traverser]

            return compute_showdown_values(
                trav_combos, opp_combos, opp_reach,
                state.board, state.pot,
                trav_invested, opp_invested,
                game.tree_config.rake_rate, game.tree_config.rake_cap,
            )

        # Shouldn't reach here
        return np.zeros(C_trav, dtype=np.float32)

    def train_regret(self, player: int):
        if self.reinitialize_advantage_networks:
            self.regret_trainers[player].reset()
        loss = self.regret_trainers[player].train_model(self.num_iteration, self.logger)
        self.logger.record(f"regret_loss_{player}", loss)

    def train_average_policy(self):
        self.ave_policy_trainer.reset()
        loss = self.ave_policy_trainer.train_model(self.num_iteration, self.logger)
        self.logger.info(f"average policy loss: {loss}")

    def evaluate(self):
        self.logger.record("nodes_touched", self.nodes_touched)
        self.logger.record("iteration", self.num_iteration)
        self.logger.record("episode", self.episode)
        self.logger.dump(step=self.episode)

    def _load_checkpoint(self):
        """Load model weights from save_dir to resume training."""
        save_dir = self.save_dir
        if not save_dir.exists():
            self.logger.info(f"No checkpoint directory found at {save_dir}. Starting fresh.")
            return

        # Find best checkpoint: prefer final models, fall back to latest checkpoint
        loaded = 0
        for i in range(2):
            regret_path = save_dir / f"regret_player_{i}.pt"
            imm_path = save_dir / f"imm_regret_player_{i}.pt"

            # Fall back to latest checkpoint file if final model not found
            if not regret_path.exists():
                candidates = sorted(save_dir.glob(f"regret_player_{i}_checkpoint_*.pt"))
                regret_path = candidates[-1] if candidates else None
            if not imm_path.exists():
                candidates = sorted(save_dir.glob(f"imm_regret_player_{i}_checkpoint_*.pt"))
                imm_path = candidates[-1] if candidates else None

            if regret_path and regret_path.exists():
                self.regret_trainers[i].model.load_state_dict(
                    torch.load(regret_path, map_location=self.device, weights_only=True)
                )
                loaded += 1
                self.logger.info(f"Loaded regret model player {i} from {regret_path.name}")
            if imm_path and imm_path.exists():
                self.regret_trainers[i].imm_model.load_state_dict(
                    torch.load(imm_path, map_location=self.device, weights_only=True)
                )
                loaded += 1
                self.logger.info(f"Loaded imm regret model player {i} from {imm_path.name}")

        policy_path = save_dir / "ave_policy.pt"
        if not policy_path.exists():
            candidates = sorted(save_dir.glob("ave_policy_checkpoint_*.pt"))
            policy_path = candidates[-1] if candidates else None
        if policy_path and policy_path.exists():
            self.ave_policy_trainer.model.load_state_dict(
                torch.load(policy_path, map_location=self.device, weights_only=True)
            )
            loaded += 1
            self.logger.info(f"Loaded policy model from {policy_path.name}")

        if loaded > 0:
            self.logger.info(f"Resumed training: loaded {loaded} model(s) from {save_dir.absolute()}")
        else:
            self.logger.info("No checkpoint files found. Starting fresh.")

    def save_models(self, checkpoint: bool = False):
        save_dir = self.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_checkpoint_{int(time.time() - self.start_time)}s" if checkpoint else ""

        for i in range(2):
            torch.save(
                self.regret_trainers[i].model.state_dict(),
                save_dir / f"regret_player_{i}{suffix}.pt",
            )
            torch.save(
                self.regret_trainers[i].imm_model.state_dict(),
                save_dir / f"imm_regret_player_{i}{suffix}.pt",
            )

        torch.save(
            self.ave_policy_trainer.model.state_dict(),
            save_dir / f"ave_policy{suffix}.pt",
        )

        label = "Checkpoint" if checkpoint else "Final models"
        self.logger.info(f"{label} saved to {save_dir.absolute()}")

    def _get_scenario(self) -> Scenario:
        return self.scenario_gen.sample()


def _mp_collect_traversals(payload: dict) -> dict:
    # Common state is configured by _mp_worker_init.
    task_id = int(payload["task_id"])
    wid = int(payload["wid"])
    seed = int(payload["seed"])
    traverser = int(payload["traverser"])
    n_traversals = int(payload["n_traversals"])
    scenario = payload["scenario"]
    num_iteration = int(payload["num_iteration"])
    device = payload["traversal_device"]

    # Per-worker deterministic seed (best-effort; scheduling still affects reservoir replacement order).
    worker_seed = seed + 10_000_000 * (wid + 1) + 1000 * num_iteration + 10 * traverser + task_id
    set_seed(worker_seed)

    # Build a local solver instance for traversal-only, load snapshot weights.
    solver = PostflopVRDeepPDCFR(
        num_iterations=1,
        num_traversals=n_traversals,
        advantage_buffer_size=int(_MP_COMMON["advantage_buffer_size"]),
        ave_policy_buffer_size=int(_MP_COMMON["ave_policy_buffer_size"]),
        learning_rate=float(_MP_COMMON["learning_rate"]),
        advantage_network_train_steps=0,
        ave_policy_network_train_steps=0,
        advantage_batch_size=1,
        ave_policy_batch_size=1,
        evaluation_frequency=1,
        epsilon=float(_MP_COMMON["epsilon"]),
        alpha=float(_MP_COMMON["alpha"]),
        gamma=2.0,
        use_regret_matching_argmax=bool(_MP_COMMON["use_regret_matching_argmax"]),
        reinitialize_advantage_networks=False,
        reinitialize_imm_regret_networks=False,
        stack_range=(20, 200),
        pot_range=(3, 60),
        device=device,
        seed=worker_seed,
        max_time=0,
        save_interval=0,
        save_dir="",
        resume=False,
        logger=Logger(writer_strings=[]),
        card_embed_dim=int(_MP_COMMON["card_embed_dim"]),
        situation_hidden=int(_MP_COMMON["situation_hidden"]),
        combo_hidden=int(_MP_COMMON["combo_hidden"]),
        fusion_hidden=int(_MP_COMMON["fusion_hidden"]),
        fusion_residual_blocks=int(_MP_COMMON["fusion_residual_blocks"]),
    )

    for p in range(2):
        solver.regret_trainers[p].model.load_state_dict(_MP_REG_STATE[p], strict=True)
        solver.regret_trainers[p].imm_model.load_state_dict(_MP_IMM_STATE[p], strict=True)

    game = NLHEGame(scenario.tree_config, scenario.card_config)
    encoder = FastFeatureEncoder(game)

    nodes_before = solver.nodes_touched
    ep_before = solver.episode
    solver.num_iteration = num_iteration
    solver.collect_training_data(game, encoder, scenario, traverser)

    return dict(
        episodes=solver.episode - ep_before,
        nodes_touched=solver.nodes_touched - nodes_before,
        regret_nodes=solver.regret_trainers[traverser].buffer.nodes,
        policy_nodes=solver.ave_policy_trainer.buffer.nodes,
    )


_MP_REG_STATE = None
_MP_IMM_STATE = None
_MP_COMMON = None


def _mp_worker_init(reg_state, imm_state, common: dict):
    # Ensure we don't pay torch.compile overhead in workers.
    import os

    os.environ["DEEPPDCFR_DISABLE_COMPILE"] = "1"
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    global _MP_REG_STATE, _MP_IMM_STATE, _MP_COMMON
    _MP_REG_STATE = reg_state
    _MP_IMM_STATE = imm_state
    _MP_COMMON = common
