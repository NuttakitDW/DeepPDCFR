"""Parallel DFS traversal for DeepCumuAdv training.

Workers run independent batches of DFS traversals with local buffers.
Models are passed as state_dicts and reconstructed in each worker (CPU-only).
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from deeppdcfr.game import read_game_config
from deeppdcfr.os_deep_cumu_adv import MLP, ReservoirBuffer


# ---------------------------------------------------------------------------
# Standalone helpers (mirrors class methods, picklable for workers)
# ---------------------------------------------------------------------------

def _regret_matching(regrets, legal_actions, use_regret_matching_argmax):
    legal_regrets = np.zeros_like(regrets)
    legal_regrets[legal_actions] = regrets[legal_actions]
    pos_sum = np.sum(np.maximum(legal_regrets, 0))
    if pos_sum > 0:
        return np.maximum(legal_regrets, 0) / pos_sum
    else:
        policy = np.zeros_like(regrets)
        if use_regret_matching_argmax:
            max_action_id = legal_actions[np.argmax(regrets[legal_actions])]
            policy[max_action_id] = 1
        else:
            policy[legal_actions] = 1 / len(legal_actions)
        return policy


def _predictive_regret_matching(regrets, imm_regrets, legal_actions, T, alpha, use_regret_matching_argmax):
    predictive_regrets = np.maximum(
        np.maximum(regrets, 0)
        * np.power(T - 1, alpha)
        / (np.power(T - 1, alpha) + 1)
        + imm_regrets,
        0,
    )
    return _regret_matching(predictive_regrets, legal_actions, use_regret_matching_argmax)


def _model_forward(model, x, mask):
    """Run model inference and return numpy array of masked outputs."""
    x = torch.as_tensor(x, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.float32)
    with torch.no_grad():
        return (model(x) * mask).numpy()


def _build_model(input_size, network_layers, output_size, state_dict):
    """Rebuild an MLP from state_dict on CPU."""
    model = MLP(input_size, network_layers, output_size)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Buffer extraction / merge
# ---------------------------------------------------------------------------

def _extract_buffer(buf):
    """Extract filled portion of a ReservoirBuffer as numpy arrays."""
    n = min(buf.cur_id, buf.buffer_size)
    return {
        "infostates": buf.infostate_buf[:n].copy(),
        "q_values": buf.q_value_buf[:n].copy(),
        "q_value_masks": buf.q_value_mask_buf[:n].copy(),
        "iterations": buf.iteration_buf[:n].copy(),
        "total_added": buf.cur_id,
    }


def _merge_buffer_data(target_buffer, worker_data_list):
    """Merge extracted buffer data from workers into target_buffer."""
    for data in worker_data_list:
        n = len(data["infostates"])
        for i in range(n):
            target_buffer.add(
                data["infostates"][i],
                data["q_values"][i],
                data["q_value_masks"][i],
                data["iterations"][i].item(),
            )


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _dfs_worker(args):
    """Top-level worker: runs a batch of DFS traversals with local buffers.

    Args is a single dict to work with ProcessPoolExecutor.map().
    """
    game_name = args["game_name"]
    player = args["player"]
    num_traversals = args["num_traversals"]
    num_iteration = args["num_iteration"]
    epsilon = args["epsilon"]
    fit_advantage = args["fit_advantage"]
    max_utility = args["max_utility"]
    use_regret_matching_argmax = args["use_regret_matching_argmax"]
    infostate_size = args["infostate_size"]
    action_size = args["action_size"]
    network_layers = args["network_layers"]
    seed = args["seed"]

    # Model state dicts (one per player)
    model_states = args["model_states"]
    target_model_states = args["target_model_states"]
    imm_model_states = args.get("imm_model_states")  # None for non-PDCFR
    alpha = args.get("alpha")

    # Ave policy model state dict
    ave_model_state = args["ave_model_state"]

    # Worker buffers: sized to actual workload, not full config size.
    # Each traversal produces ~100 entries max (generous upper bound for FHP).
    worker_buffer_size = num_traversals * 100

    # Seed this worker's RNG
    np.random.seed(seed)

    # Reconstruct game
    game_config = read_game_config(game_name)
    game = game_config.load_game()

    # Rebuild models on CPU
    num_players = game.num_players()
    models = []
    target_models = []
    imm_models = [] if imm_model_states else None
    for p in range(num_players):
        m = _build_model(infostate_size, network_layers, action_size, model_states[p])
        models.append(m)
        tm = _build_model(infostate_size, network_layers, action_size, target_model_states[p])
        target_models.append(tm)
        if imm_model_states:
            im = _build_model(infostate_size, network_layers, action_size, imm_model_states[p])
            imm_models.append(im)

    # Local buffers (small, sized to worker's actual workload)
    regret_buffer = ReservoirBuffer(worker_buffer_size, infostate_size, action_size, device="cpu")
    ave_policy_buffer = ReservoirBuffer(worker_buffer_size, infostate_size, action_size, device="cpu")

    nodes_touched = 0

    def get_policy(s, p):
        infostate = s.information_state_tensor()
        mask = s.legal_actions_mask()
        regrets = _model_forward(models[p], infostate, mask)
        legal_actions = s.legal_actions()
        if imm_models:
            imm_regrets = _model_forward(imm_models[p], infostate, mask)
            return _predictive_regret_matching(regrets, imm_regrets, legal_actions, num_iteration, alpha, use_regret_matching_argmax)
        else:
            return _regret_matching(regrets, legal_actions, use_regret_matching_argmax)

    def skip_chance_state(s):
        nonlocal nodes_touched
        while s.current_player() == -1:
            nodes_touched += 1
            actions, probs = zip(*s.chance_outcomes())
            aid = np.random.choice(range(len(actions)), p=probs)
            s.apply_action(actions[aid])
        return s

    def dfs(s, traverser, my_reach=1.0, opp_reach=1.0, opp_sample_reach=1.0, sample_reach=1.0):
        nonlocal nodes_touched
        nodes_touched += 1
        cur_player = s.current_player()
        if cur_player == -4:
            return s.returns()[traverser] / max_utility

        legal_actions = s.legal_actions()
        policy = get_policy(s, cur_player)
        num_actions = s.num_distinct_actions()
        uniform_policy = np.array(s.legal_actions_mask()) / len(legal_actions)
        sample_policy = (
            uniform_policy * epsilon + policy * (1 - epsilon)
            if cur_player == traverser
            else policy
        )
        sample_policy /= sample_policy.sum()
        action = np.random.choice(range(num_actions), p=sample_policy)
        sample_prob = sample_policy[action].item()
        prob = policy[action].item()
        ns = skip_chance_state(s.child(action))

        if cur_player == 1 - traverser:
            # Opponent node: add to ave policy buffer
            ave_policy_buffer.add(
                s.information_state_tensor(),
                policy,
                s.legal_actions_mask(),
                num_iteration,
            )
            q_values = np.zeros_like(policy)
            action_value = dfs(
                ns, traverser, my_reach, opp_reach * prob,
                opp_sample_reach * sample_prob, sample_reach * sample_prob,
            )
            q_values[action] += (action_value - q_values[action]) / sample_prob
            value = np.dot(q_values, policy)
        else:
            # Traverser node
            q_values = np.zeros_like(policy)
            action_value = dfs(
                ns, traverser, my_reach * prob, opp_reach,
                opp_sample_reach, sample_reach * sample_prob,
            )
            q_values[action] += (action_value - q_values[action]) / sample_prob
            value = np.dot(q_values, policy)

            cf_regrets = np.zeros_like(policy)
            im_weight = 1 if fit_advantage else opp_reach / sample_reach
            cf_regrets[legal_actions] = -value * im_weight
            cf_regrets[legal_actions] += q_values[legal_actions] * im_weight
            regret_buffer.add(
                s.information_state_tensor(),
                cf_regrets,
                s.legal_actions_mask(),
                num_iteration,
            )
        return value

    # Run traversals
    for _ in range(num_traversals):
        root = skip_chance_state(game.new_initial_state())
        dfs(root, player)

    return {
        "regret_data": _extract_buffer(regret_buffer),
        "ave_policy_data": _extract_buffer(ave_policy_buffer),
        "nodes_touched": nodes_touched,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_parallel_dfs(
    game_name,
    player,
    num_traversals,
    num_workers,
    num_iteration,
    epsilon,
    fit_advantage,
    max_utility,
    use_regret_matching_argmax,
    infostate_size,
    action_size,
    network_layers,
    advantage_buffer_size,
    ave_policy_buffer_size,
    regret_trainers,
    ave_policy_trainer,
    base_seed,
):
    """Divide DFS traversals among workers and collect results."""
    num_players = len(regret_trainers)

    # Serialize model state_dicts (move to CPU)
    model_states = []
    target_model_states = []
    imm_model_states = None
    alpha = None
    for p in range(num_players):
        rt = regret_trainers[p]
        model_states.append({k: v.cpu() for k, v in rt.model.state_dict().items()})
        target_model_states.append({k: v.cpu() for k, v in rt.target_model.state_dict().items()})
        if hasattr(rt, "imm_model"):
            if imm_model_states is None:
                imm_model_states = []
            imm_model_states.append({k: v.cpu() for k, v in rt.imm_model.state_dict().items()})
            alpha = rt.alpha

    ave_model_state = {k: v.cpu() for k, v in ave_policy_trainer.model.state_dict().items()}

    # Divide traversals among workers
    base_count = num_traversals // num_workers
    remainder = num_traversals % num_workers
    worker_args = []
    for w in range(num_workers):
        count = base_count + (1 if w < remainder else 0)
        worker_args.append({
            "game_name": game_name,
            "player": player,
            "num_traversals": count,
            "num_iteration": num_iteration,
            "epsilon": epsilon,
            "fit_advantage": fit_advantage,
            "max_utility": max_utility,
            "use_regret_matching_argmax": use_regret_matching_argmax,
            "infostate_size": infostate_size,
            "action_size": action_size,
            "network_layers": network_layers,
            "model_states": model_states,
            "target_model_states": target_model_states,
            "imm_model_states": imm_model_states,
            "alpha": alpha,
            "ave_model_state": ave_model_state,
            "seed": base_seed + w,
        })

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        results = list(pool.map(_dfs_worker, worker_args))

    # Merge results
    total_nodes = 0
    regret_data_list = []
    ave_policy_data_list = []
    for r in results:
        total_nodes += r["nodes_touched"]
        regret_data_list.append(r["regret_data"])
        ave_policy_data_list.append(r["ave_policy_data"])

    return {
        "regret_data_list": regret_data_list,
        "ave_policy_data_list": ave_policy_data_list,
        "nodes_touched": total_nodes,
    }


# ---------------------------------------------------------------------------
# Parallel LBR
# ---------------------------------------------------------------------------

def _lbr_worker(args):
    """Worker for parallel LBR computation."""
    game_name = args["game_name"]
    num_samples = args["num_samples"]
    exploiter = args["exploiter"]
    seed = args["seed"]
    infostate_size = args["infostate_size"]
    action_size = args["action_size"]
    network_layers = args["network_layers"]
    ave_model_state = args["ave_model_state"]

    np.random.seed(seed)

    game_config = read_game_config(game_name)
    game = game_config.load_game()

    ave_model = _build_model(infostate_size, network_layers, action_size, ave_model_state)

    def policy_fn(s, probs_as_dict=True):
        x = torch.as_tensor(s.information_state_tensor(), dtype=torch.float32)
        mask = torch.as_tensor(s.legal_actions_mask(), dtype=torch.float32)
        with torch.no_grad():
            logits = ave_model(x)
            legal_logits = torch.where(mask == 1, logits, torch.tensor(-10e20))
            policy = torch.softmax(legal_logits, dim=-1).numpy()
        if probs_as_dict:
            return {action: policy[action] for action in s.legal_actions()}
        return policy

    from deeppdcfr.utils import rescale_func, _lbr_traverse
    rescaled_fn = rescale_func(policy_fn)

    total = 0.0
    for _ in range(num_samples):
        state = game.new_initial_state()
        while state.is_chance_node():
            outcomes, probs = zip(*state.chance_outcomes())
            aidx = np.random.choice(len(outcomes), p=probs)
            state.apply_action(outcomes[aidx])
        total += _lbr_traverse(state, exploiter, rescaled_fn)
    return total / num_samples


def compute_lbr_parallel(
    game_name,
    num_lbr_samples,
    num_workers,
    infostate_size,
    action_size,
    network_layers,
    ave_policy_trainer,
    base_seed=42,
):
    """Parallel LBR: split samples across workers for each exploiter."""
    ave_model_state = {k: v.cpu() for k, v in ave_policy_trainer.model.state_dict().items()}

    samples_per_worker = num_lbr_samples // num_workers
    remainder = num_lbr_samples % num_workers

    all_args = []
    for exploiter in range(2):
        for w in range(num_workers):
            count = samples_per_worker + (1 if w < remainder else 0)
            all_args.append({
                "game_name": game_name,
                "num_samples": count,
                "exploiter": exploiter,
                "seed": base_seed + exploiter * num_workers + w,
                "infostate_size": infostate_size,
                "action_size": action_size,
                "network_layers": network_layers,
                "ave_model_state": ave_model_state,
            })

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as pool:
        results = list(pool.map(_lbr_worker, all_args))

    # Average results per exploiter
    lbr_values = []
    for exploiter in range(2):
        start = exploiter * num_workers
        worker_results = results[start:start + num_workers]
        # Weighted average by sample count
        total_samples = 0
        weighted_sum = 0.0
        for w in range(num_workers):
            count = samples_per_worker + (1 if w < remainder else 0)
            weighted_sum += worker_results[w] * count
            total_samples += count
        lbr_values.append(weighted_sum / total_samples)

    return (lbr_values[0] + lbr_values[1]) / 2
