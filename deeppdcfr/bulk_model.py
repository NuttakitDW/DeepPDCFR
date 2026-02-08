"""
Bulk neural network architecture for NLHE Neural CFR (V2).

BulkRegretModelV2 and BulkPolicyModelV2 use learned card embeddings and
residual blocks. The situation encoder computes once per node, then fuses
with per-combo encodings for all C combos in a single forward pass.

Also includes NodeReservoirBuffer (node-level storage), BulkRegretTrainer,
and BulkPolicyTrainer for VR-PDCFR+ training.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats

from deeppdcfr.nlhe_game import MAX_ACTIONS

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU optimisation helpers
# ---------------------------------------------------------------------------

_HAS_COMPILE = hasattr(torch, "compile")
# torch.amp (unified API, PyTorch 2.3+) preferred over deprecated torch.cuda.amp
_HAS_AMP = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")


def _try_compile(model: nn.Module, device: str = "cpu") -> nn.Module:
    """Wrap *model* with ``torch.compile`` if available (PyTorch 2.0+ & CUDA).

    Only compiles on CUDA — the inductor backend has limited CPU/macOS support
    and the compile overhead isn't worthwhile for CPU anyway.
    """
    if _HAS_COMPILE and "cuda" in device and torch.cuda.is_available():
        try:
            return torch.compile(model)
        except Exception as exc:  # noqa: BLE001
            _log.warning("torch.compile failed, falling back to eager: %s", exc)
    return model


def _make_amp(device: str):
    """Return (autocast_ctx_factory, GradScaler | None) appropriate for *device*.

    Uses the unified ``torch.amp`` API (PyTorch 2.3+).  Falls back to a no-op
    context on CPU or older PyTorch.
    """
    use_amp = "cuda" in device and _HAS_AMP and torch.cuda.is_available()
    if use_amp:
        return (
            lambda: torch.amp.autocast("cuda"),
            torch.amp.GradScaler("cuda"),
        )
    # CPU / no-amp: identity context + no scaler
    from contextlib import nullcontext
    return (lambda: nullcontext(), None)


def _to_device(np_arr: np.ndarray, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Convert numpy array to tensor on *device*, using pinned memory for CUDA."""
    t = torch.tensor(np_arr, dtype=dtype)
    if "cuda" in device:
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)

# ---------------------------------------------------------------------------
# Feature dimensions for V2 (card IDs + numerical)
# ---------------------------------------------------------------------------

# Situation numerical: street[3] + player[1] + pot_ratio[1] + SPR[1] +
#   committed[2] + pot_odds[1] + bet_history[20] + actions[10] + board_mask[5]
#   + range_features[54] (27 OOP + 27 IP)
SITUATION_NUMERICAL_DIM = 98

# Per-combo: hand_strength[1] + flush_draw[1]
COMBO_HAND_FEATURES_DIM = 2

# ---------------------------------------------------------------------------
# Core layers (reuse SonnetLinear / ZeroInitLinear patterns)
# ---------------------------------------------------------------------------


class SonnetLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        stddev = 1 / math.sqrt(self.in_features)
        self.weight = nn.Parameter(
            torch.Tensor(
                stats.truncnorm.rvs(
                    -2, 2, loc=0, scale=stddev,
                    size=[self.out_features, self.in_features],
                )
            )
        )
        self.bias = nn.Parameter(torch.zeros([self.out_features]))

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.activation:
            y = F.relu(y)
        return y


class ZeroInitLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = nn.Parameter(torch.zeros([self.out_features, self.in_features]))
        self.bias = nn.Parameter(torch.zeros([self.out_features]))

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if self.activation:
            y = F.relu(y)
        return y


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """x + ReLU(linear2(ReLU(linear1(x))))"""

    def __init__(self, dim: int):
        super().__init__()
        self.linear1 = SonnetLinear(dim, dim, activation=True)
        self.linear2 = SonnetLinear(dim, dim, activation=False)

    def forward(self, x):
        return F.relu(x + self.linear2(self.linear1(x)))

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()


# ---------------------------------------------------------------------------
# V2 bulk regret model (~1.2M params)
# ---------------------------------------------------------------------------


class BulkRegretModelV2(nn.Module):
    """
    Predicts regret values for all combos at a node in one forward pass.

    Uses learned card embeddings instead of one-hot encodings:
    - Board cards → embed → concat with numerical situation features
    - Hole cards → embed → concat with hand features (strength, flush draw)
    - Fusion with residual blocks → output [B, C, MAX_ACTIONS]

    Input:
        board_ids:     [B, 5]  card IDs (0-51, padded with -1)
        board_mask:    [B, 5]  1.0 where card present
        sit_numerical: [B, SITUATION_NUMERICAL_DIM]
        combo_card_ids: [B, C, 2]  hole card IDs
        hand_features:  [B, C, COMBO_HAND_FEATURES_DIM]
        action_mask:    [B, MAX_ACTIONS]

    Output: [B, C, MAX_ACTIONS] regret values
    """

    def __init__(
        self,
        card_embed_dim: int = 64,
        situation_hidden: int = 384,
        situation_embed: int = 256,
        combo_hidden: int = 192,
        combo_embed: int = 128,
        fusion_hidden: int = 384,
        fusion_residual_blocks: int = 2,
        output_dim: int = MAX_ACTIONS,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.card_embed_dim = card_embed_dim

        # Shared card embedding (52 cards + 1 padding token at index 52)
        self.card_embedding = nn.Embedding(53, card_embed_dim, padding_idx=52)

        # Situation encoder: board embeds + board_mask + numerical
        # board: 5 × card_embed_dim + 5 (mask) + SITUATION_NUMERICAL_DIM
        sit_input_dim = 5 * card_embed_dim + 5 + SITUATION_NUMERICAL_DIM
        self.situation_encoder = nn.Sequential(
            SonnetLinear(sit_input_dim, situation_hidden),
            SonnetLinear(situation_hidden, situation_hidden),
            SonnetLinear(situation_hidden, situation_embed, activation=False),
        )

        # Combo encoder: 2 × card_embed_dim + hand_features
        combo_input_dim = 2 * card_embed_dim + COMBO_HAND_FEATURES_DIM
        self.combo_encoder = nn.Sequential(
            SonnetLinear(combo_input_dim, combo_hidden),
            SonnetLinear(combo_hidden, combo_embed, activation=False),
        )

        # Fusion: concat [sit_embed, combo_embed] → residual blocks → output
        fusion_input = situation_embed + combo_embed
        self.fusion_input_proj = SonnetLinear(fusion_input, fusion_hidden)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(fusion_hidden) for _ in range(fusion_residual_blocks)
        ])
        self.output_head = ZeroInitLinear(fusion_hidden, output_dim, activation=False)

    def forward(
        self,
        board_ids: torch.Tensor,
        board_mask: torch.Tensor,
        sit_numerical: torch.Tensor,
        combo_card_ids: torch.Tensor,
        hand_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            board_ids:      [B, 5] long (padded cards use index 52)
            board_mask:     [B, 5] float
            sit_numerical:  [B, SITUATION_NUMERICAL_DIM] float
            combo_card_ids: [B, C, 2] long
            hand_features:  [B, C, COMBO_HAND_FEATURES_DIM] float
            action_mask:    [B, MAX_ACTIONS] float

        Returns: [B, C, MAX_ACTIONS]
        """
        B, C, _ = combo_card_ids.shape

        # Board card embeddings
        board_embeds = self.card_embedding(board_ids)  # [B, 5, embed_dim]
        board_flat = board_embeds.view(B, -1)  # [B, 5 * embed_dim]

        # Situation encoding
        sit_input = torch.cat([board_flat, board_mask, sit_numerical], dim=-1)
        h_sit = self.situation_encoder(sit_input)  # [B, situation_embed]

        # Combo card embeddings
        combo_embeds = self.card_embedding(combo_card_ids)  # [B, C, 2, embed_dim]
        combo_flat = combo_embeds.view(B, C, -1)  # [B, C, 2 * embed_dim]

        # Combo encoding
        combo_input = torch.cat([combo_flat, hand_features], dim=-1)
        h_combo = self.combo_encoder(combo_input)  # [B, C, combo_embed]

        # Broadcast situation to all combos and fuse
        h_sit_expanded = h_sit.unsqueeze(1).expand(-1, C, -1)  # [B, C, sit_embed]
        h_fused = torch.cat([h_sit_expanded, h_combo], dim=-1)  # [B, C, fusion_input]

        # Fusion with residual blocks
        h = self.fusion_input_proj(h_fused)  # [B, C, fusion_hidden]
        for block in self.residual_blocks:
            h = block(h)
        output = self.output_head(h)  # [B, C, MAX_ACTIONS]

        # Mask illegal actions
        mask_expanded = action_mask.unsqueeze(1).expand(-1, C, -1)
        output = output * mask_expanded

        return output

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, 'reset_parameters') and module is not self:
                module.reset_parameters()


class BulkPolicyModelV2(nn.Module):
    """Same architecture as BulkRegretModelV2 but outputs softmax probabilities."""

    def __init__(self, **kwargs):
        super().__init__()
        self.regret_model = BulkRegretModelV2(**kwargs)
        self.output_dim = self.regret_model.output_dim

    def forward(
        self,
        board_ids: torch.Tensor,
        board_mask: torch.Tensor,
        sit_numerical: torch.Tensor,
        combo_card_ids: torch.Tensor,
        hand_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [B, C, MAX_ACTIONS] action probabilities."""
        # Reuse the regret model's forward to get raw logits, then softmax
        logits = self.regret_model(
            board_ids, board_mask, sit_numerical,
            combo_card_ids, hand_features, action_mask,
        )
        # Re-mask for softmax (logits are already zeroed, but we need -inf for softmax)
        mask_expanded = action_mask.unsqueeze(1).expand_as(logits)
        logits = torch.where(mask_expanded > 0, logits, torch.tensor(-1e20, device=logits.device))
        return F.softmax(logits, dim=-1)

    def reset_parameters(self):
        self.regret_model.reset_parameters()


# ---------------------------------------------------------------------------
# Node-level reservoir buffer
# ---------------------------------------------------------------------------


class NodeReservoirBuffer:
    """
    Stores training data at the node level. Each entry contains data for
    ALL combos at one decision node.

    Fields per node:
        board_ids:        [5] int  (padded with 52)
        sit_numerical:    [SITUATION_NUMERICAL_DIM] float
        combo_card_ids:   [C, 2] int
        hand_features:    [C, COMBO_HAND_FEATURES_DIM] float
        values:           [C, MAX_ACTIONS] float (regrets or policy)
        action_mask:      [MAX_ACTIONS] float
        num_combos:       int
        iteration:        int
    """

    def __init__(self, buffer_size: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.device = device
        self.nodes = []  # list of dicts
        self.cur_id = 0

    def reset(self):
        self.nodes.clear()
        self.cur_id = 0

    def add(
        self,
        board_ids: np.ndarray,
        sit_numerical: np.ndarray,
        combo_card_ids: np.ndarray,
        hand_features: np.ndarray,
        values: np.ndarray,
        action_mask: np.ndarray,
        iteration: int,
    ):
        """Add one node's worth of data."""
        node = {
            "board_ids": board_ids.copy(),
            "sit_numerical": sit_numerical.copy(),
            "combo_card_ids": combo_card_ids.copy(),
            "hand_features": hand_features.copy(),
            "values": values.copy(),
            "action_mask": action_mask.copy(),
            "num_combos": combo_card_ids.shape[0],
            "iteration": iteration,
        }

        if self.cur_id < self.buffer_size:
            self.nodes.append(node)
        else:
            idx = np.random.randint(0, self.cur_id + 1)
            if idx < self.buffer_size:
                self.nodes[idx] = node
        self.cur_id += 1

    def sample(self, batch_size: int = -1):
        """
        Sample a batch of nodes. Pads combos to max-C-in-batch.

        Returns:
            board_ids:      [B, 5] long
            sit_numerical:  [B, SITUATION_NUMERICAL_DIM] float
            combo_card_ids: [B, max_C, 2] long
            hand_features:  [B, max_C, COMBO_HAND_FEATURES_DIM] float
            values:         [B, max_C, MAX_ACTIONS] float
            action_masks:   [B, MAX_ACTIONS] float
            combo_masks:    [B, max_C] float (1 where combo is valid)
            iterations:     [B] float
        """
        data_length = len(self.nodes)
        if batch_size <= 0 or batch_size > data_length:
            idxs = list(range(data_length))
        else:
            idxs = random.sample(range(data_length), batch_size)

        batch = [self.nodes[i] for i in idxs]
        B = len(batch)
        max_C = max(n["num_combos"] for n in batch)

        board_ids = np.full((B, 5), 52, dtype=np.int64)
        sit_numerical = np.zeros((B, SITUATION_NUMERICAL_DIM), dtype=np.float32)
        combo_card_ids = np.zeros((B, max_C, 2), dtype=np.int64)
        hand_features = np.zeros((B, max_C, COMBO_HAND_FEATURES_DIM), dtype=np.float32)
        values = np.zeros((B, max_C, MAX_ACTIONS), dtype=np.float32)
        action_masks = np.zeros((B, MAX_ACTIONS), dtype=np.float32)
        combo_masks = np.zeros((B, max_C), dtype=np.float32)
        iterations = np.zeros(B, dtype=np.float32)

        for i, node in enumerate(batch):
            C = node["num_combos"]
            board_ids[i] = node["board_ids"]
            sit_numerical[i] = node["sit_numerical"]
            combo_card_ids[i, :C] = node["combo_card_ids"]
            hand_features[i, :C] = node["hand_features"]
            values[i, :C] = node["values"]
            action_masks[i] = node["action_mask"]
            combo_masks[i, :C] = 1.0
            iterations[i] = node["iteration"]

        dev = self.device
        return (
            _to_device(board_ids, torch.long, dev),
            _to_device(sit_numerical, torch.float32, dev),
            _to_device(combo_card_ids, torch.long, dev),
            _to_device(hand_features, torch.float32, dev),
            _to_device(values, torch.float32, dev),
            _to_device(action_masks, torch.float32, dev),
            _to_device(combo_masks, torch.float32, dev),
            _to_device(iterations, torch.float32, dev),
        )

    def __len__(self):
        return len(self.nodes)


# ---------------------------------------------------------------------------
# Masked MSE loss
# ---------------------------------------------------------------------------


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    combo_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """
    MSE loss that ignores padded combos and illegal actions.

    pred:        [B, C, A]
    target:      [B, C, A]
    combo_mask:  [B, C]    (1 where combo valid)
    action_mask: [B, A]    (1 where action legal)
    """
    # Full mask: [B, C, A]
    full_mask = combo_mask.unsqueeze(-1) * action_mask.unsqueeze(1)
    diff = (pred - target) ** 2
    masked_diff = diff * full_mask
    num_valid = full_mask.sum().clamp(min=1.0)
    return masked_diff.sum() / num_valid


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------


class BulkRegretTrainer:
    """
    Trains BulkRegretModelV2 following VR-PDCFR+ pattern.
    Two models: cumulative regret + immediate regret (predictive CFR).
    Uses node-level buffer with combo padding.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        train_steps: int = 500,
        alpha: float = 2.3,
        use_regret_matching_argmax: bool = True,
        reinitialize_imm: bool = True,
        device: str = "cpu",
        **model_kwargs,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.alpha = alpha
        self.use_regret_matching_argmax = use_regret_matching_argmax
        self.reinitialize_imm = reinitialize_imm
        self.device = device
        self.model_kwargs = model_kwargs

        # Cumulative regret model + target
        self.model = _try_compile(BulkRegretModelV2(**model_kwargs).to(device), device)
        self.target_model = _try_compile(BulkRegretModelV2(**model_kwargs).to(device), device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Immediate regret model (predictive)
        self.imm_model = _try_compile(BulkRegretModelV2(**model_kwargs).to(device), device)
        self.imm_optimizer = optim.Adam(self.imm_model.parameters(), lr=learning_rate)

        # Mixed-precision support
        self._autocast, self._scaler = _make_amp(device)

        # Node-level buffer
        self.buffer = NodeReservoirBuffer(buffer_size, device=device)

    def reset(self):
        self.model.reset_parameters()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.reinitialize_imm:
            self.imm_model.reset_parameters()
            self.imm_model.to(self.device)
            self.imm_optimizer = optim.Adam(self.imm_model.parameters(), lr=self.learning_rate)

    def reset_buffer(self):
        self.buffer.reset()

    def add_node_data(
        self,
        board_ids: np.ndarray,
        sit_numerical: np.ndarray,
        combo_card_ids: np.ndarray,
        hand_features: np.ndarray,
        cf_regrets: np.ndarray,
        action_mask: np.ndarray,
        iteration: int,
    ):
        """Add all combos for one node."""
        self.buffer.add(
            board_ids, sit_numerical, combo_card_ids,
            hand_features, cf_regrets, action_mask, iteration,
        )

    def train_model(self, T: int, logger=None) -> float:
        """Train both cumulative and immediate regret models."""
        if len(self.buffer) == 0:
            return 0.0

        scaler = self._scaler
        last_loss = None
        log_every = 1
        for step in range(self.train_steps):
            (board_ids, sit_numerical, combo_card_ids, hand_features,
             cf_regrets, action_masks, combo_masks, _iterations) = self.buffer.sample(self.batch_size)

            board_mask = (board_ids != 52).float()

            # Cumulative regret target (inference only — no grad)
            with torch.no_grad(), self._autocast():
                target_out = self.target_model(
                    board_ids, board_mask, sit_numerical,
                    combo_card_ids, hand_features, action_masks,
                )  # [B, C, A]
            target_pos = torch.clamp(target_out, min=0)
            weight = math.pow(T - 1, self.alpha) / (math.pow(T - 1, self.alpha) + 1) if T > 1 else 0
            regret_targets = target_pos * weight + cf_regrets  # [B, C, A]

            # --- Cumulative regret forward + backward ---
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred = self.model(
                    board_ids, board_mask, sit_numerical,
                    combo_card_ids, hand_features, action_masks,
                )
                loss = masked_mse_loss(pred, regret_targets, combo_masks, action_masks)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            # --- Immediate regret forward + backward ---
            self.imm_optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                imm_pred = self.imm_model(
                    board_ids, board_mask, sit_numerical,
                    combo_card_ids, hand_features, action_masks,
                )
                imm_loss = masked_mse_loss(imm_pred, cf_regrets, combo_masks, action_masks)

            if scaler is not None:
                scaler.scale(imm_loss).backward()
                scaler.step(self.imm_optimizer)
                scaler.update()
            else:
                imm_loss.backward()
                self.imm_optimizer.step()

            last_loss = loss.detach()
            if logger and (step % log_every == 0 or step == self.train_steps - 1):
                logger.info(
                    f"[{step}/{self.train_steps}] regret loss: {loss.detach().item():.6f}, "
                    f"imm: {imm_loss.detach().item():.6f}"
                )

        self.target_model.load_state_dict(self.model.state_dict())
        return float(last_loss.detach().item()) if last_loss is not None else 0.0

    def get_policy(
        self,
        board_ids: np.ndarray,
        board_mask: np.ndarray,
        sit_numerical: np.ndarray,
        combo_card_ids: np.ndarray,
        hand_features: np.ndarray,
        action_mask: np.ndarray,
        T: int,
    ) -> np.ndarray:
        """
        Get strategy for all combos at a node via predictive regret matching.
        Returns [C, MAX_ACTIONS] strategy.
        """
        with torch.no_grad(), self._autocast():
            bid = torch.tensor(board_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            bmask = torch.tensor(board_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            sit = torch.tensor(sit_numerical, dtype=torch.float32, device=self.device).unsqueeze(0)
            cids = torch.tensor(combo_card_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            hf = torch.tensor(hand_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            amask = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            regrets = self.model(bid, bmask, sit, cids, hf, amask).squeeze(0).float().cpu().numpy()
            imm_regrets = self.imm_model(bid, bmask, sit, cids, hf, amask).squeeze(0).float().cpu().numpy()

        # Predictive regret matching
        weight = math.pow(T - 1, self.alpha) / (math.pow(T - 1, self.alpha) + 1) if T > 1 else 0
        pred_regrets = np.maximum(
            np.maximum(regrets, 0) * weight + imm_regrets, 0
        )

        return self._batch_regret_matching(pred_regrets, action_mask)

    def _batch_regret_matching(
        self, regrets: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Regret matching for [C, A] regrets. Returns [C, A] strategy."""
        mask_arr = np.array(mask) if not isinstance(mask, np.ndarray) else mask
        legal_actions = np.where(mask_arr > 0)[0]

        C, A = regrets.shape
        strategy = np.zeros_like(regrets)

        legal_regrets = regrets * mask_arr[np.newaxis, :]
        pos_regrets = np.maximum(legal_regrets, 0)
        pos_sum = pos_regrets.sum(axis=1, keepdims=True)

        has_pos = (pos_sum > 0).squeeze(-1)
        if has_pos.ndim == 0:
            has_pos = np.array([has_pos])
        strategy[has_pos] = pos_regrets[has_pos] / pos_sum[has_pos]

        no_pos = ~has_pos
        if np.any(no_pos):
            if self.use_regret_matching_argmax:
                no_pos_regrets = legal_regrets[no_pos]
                best = np.argmax(no_pos_regrets, axis=1)
                for i, b in enumerate(best):
                    idx = np.where(no_pos)[0][i]
                    strategy[idx, b] = 1.0
            else:
                strategy[no_pos] = mask_arr / len(legal_actions)

        return strategy


class BulkPolicyTrainer:
    """Trains BulkPolicyModelV2 for average strategy."""

    def __init__(
        self,
        learning_rate: float = 3e-4,
        buffer_size: int = 20_000,
        batch_size: int = 64,
        train_steps: int = 2000,
        gamma: float = 2.0,
        device: str = "cpu",
        **model_kwargs,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.gamma = gamma
        self.device = device

        self.model = _try_compile(BulkPolicyModelV2(**model_kwargs).to(device), device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Mixed-precision support
        self._autocast, self._scaler = _make_amp(device)

        self.buffer = NodeReservoirBuffer(buffer_size, device=device)

    def reset(self):
        self.model.reset_parameters()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def reset_buffer(self):
        self.buffer.reset()

    def add_node_data(
        self,
        board_ids: np.ndarray,
        sit_numerical: np.ndarray,
        combo_card_ids: np.ndarray,
        hand_features: np.ndarray,
        policy: np.ndarray,
        action_mask: np.ndarray,
        iteration: int,
    ):
        """Add all combos for one node."""
        self.buffer.add(
            board_ids, sit_numerical, combo_card_ids,
            hand_features, policy, action_mask, iteration,
        )

    def train_model(self, T: int, logger=None) -> float:
        if len(self.buffer) == 0:
            return 0.0

        scaler = self._scaler
        last_loss = 0.0
        for step in range(self.train_steps):
            (board_ids, sit_numerical, combo_card_ids, hand_features,
             policies, action_masks, combo_masks, iterations) = self.buffer.sample(self.batch_size)

            board_mask = (board_ids != 52).float()

            # Iteration weighting for recency
            weights = torch.pow(iterations / max(T, 1) * 2, self.gamma / 2)
            # [B] → [B, 1, 1] for broadcasting
            weights = weights.unsqueeze(-1).unsqueeze(-1)

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred = self.model(
                    board_ids, board_mask, sit_numerical,
                    combo_card_ids, hand_features, action_masks,
                )

                # Weighted masked MSE
                full_mask = combo_masks.unsqueeze(-1) * action_masks.unsqueeze(1)
                diff = (pred - policies) ** 2
                weighted_diff = diff * full_mask * weights
                num_valid = full_mask.sum().clamp(min=1.0)
                loss = weighted_diff.sum() / num_valid

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            last_loss = loss.item()
            if logger and (step % 1 == 0 or step == self.train_steps - 1):
                logger.info(f"[{step}/{self.train_steps}] policy loss: {loss.item():.6f}")

        return last_loss

    def get_policy(
        self,
        board_ids: np.ndarray,
        board_mask: np.ndarray,
        sit_numerical: np.ndarray,
        combo_card_ids: np.ndarray,
        hand_features: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Get average strategy for all combos. Returns [C, MAX_ACTIONS]."""
        with torch.no_grad(), self._autocast():
            bid = torch.tensor(board_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            bmask = torch.tensor(board_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            sit = torch.tensor(sit_numerical, dtype=torch.float32, device=self.device).unsqueeze(0)
            cids = torch.tensor(combo_card_ids, dtype=torch.long, device=self.device).unsqueeze(0)
            hf = torch.tensor(hand_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            amask = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.model(bid, bmask, sit, cids, hf, amask).squeeze(0).float().cpu().numpy()
