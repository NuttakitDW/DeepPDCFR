import datetime
import importlib
import inspect
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Type
from difflib import SequenceMatcher
import numpy as np
import torch

from typing import Any, Callable, Union

import numpy as np
import pyspiel

SpielState = Union[pyspiel.State, Any]
SpielGame = Union[pyspiel.Game, Any]

from open_spiel.python.algorithms import exploitability
from open_spiel.python.policy import tabular_policy_from_callable
from deeppdcfr.poker_agent import (
    CandidStatistician,
    LooseAggressive,
    LoosePassive,
    TightPassive,
    TightAggressive,
)

def rescale_func(callable_func):
    def rescaled_callable_func(state: SpielState) -> dict:
        probs = callable_func(state)
        probs_sum = sum(probs.values())
        return {a: p / probs_sum for a, p in probs.items()}

    return rescaled_callable_func


def evalute_explotability(game: SpielGame, callable_func: Callable) -> float:
    policy = tabular_policy_from_callable(game, rescale_func(callable_func))
    exp = exploitability.exploitability(game, policy)
    return exp


def compute_lbr(game: SpielGame, policy_fn: Callable, num_samples: int = 10000) -> float:
    """Approximate exploitability via Monte Carlo Local Best Response.

    For each player as the exploiter, sample ``num_samples`` deals and compute the
    best-response value against the opponent's ``policy_fn``. Returns the average
    of the two LBR values (an approximation of exploitability).
    """
    rescaled_fn = rescale_func(policy_fn)
    lbr_values = []
    for exploiter in range(2):
        total = 0.0
        for _ in range(num_samples):
            state = game.new_initial_state()
            # Sample all initial chance nodes (the deal)
            while state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                aidx = np.random.choice(len(outcomes), p=probs)
                state.apply_action(outcomes[aidx])
            total += _lbr_traverse(state, exploiter, rescaled_fn)
        lbr_values.append(total / num_samples)
    return (lbr_values[0] + lbr_values[1]) / 2


def _lbr_traverse(state: SpielState, exploiter: int, policy_fn: Callable) -> float:
    """Recursive tree walk for LBR.

    - Exploiter nodes: try all legal actions, pick the max value (best response).
    - Opponent nodes: weight all actions by policy probabilities (exact EV).
    - Chance nodes (mid-game, e.g. board cards): sample one outcome randomly.
    - Terminal: return payoff for the exploiter.
    """
    if state.is_terminal():
        return state.returns()[exploiter]

    if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        aidx = np.random.choice(len(outcomes), p=probs)
        state.apply_action(outcomes[aidx])
        return _lbr_traverse(state, exploiter, policy_fn)

    current_player = state.current_player()
    legal_actions = state.legal_actions()

    if current_player == exploiter:
        # Best response: try all actions, pick max
        best_value = -float('inf')
        for action in legal_actions:
            child = state.child(action)
            value = _lbr_traverse(child, exploiter, policy_fn)
            if value > best_value:
                best_value = value
        return best_value
    else:
        # Opponent: weight by policy (exact expected value)
        probs = policy_fn(state)
        ev = 0.0
        for action in legal_actions:
            prob = probs.get(action, 0.0) if isinstance(probs, dict) else probs[action]
            child = state.child(action)
            ev += prob * _lbr_traverse(child, exploiter, policy_fn)
        return ev


def play_n_games_against_random(
    game: SpielGame, callable_func: Callable, num_random_games: int
) -> float:
    total_reward = 0
    for i in range(num_random_games // 2):
        reward = play_game_against_random(game, rescale_func(callable_func))
        total_reward += reward
    return total_reward / num_random_games


def play_game_against_random(game: SpielGame, callable_func: Callable) -> float:
    # play one game per player
    reward = 0
    for player in [0, 1]:
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes, probs = zip(*state.chance_outcomes())
                aidx = np.random.choice(range(len(outcomes)), p=probs)
                action = outcomes[aidx]
            else:
                cur_player = state.current_player()
                if cur_player == player:
                    probs = callable_func(state)
                    action = np.random.choice(
                        list(probs.keys()), p=list(probs.values())
                    )
                elif cur_player == 1 - player:
                    action = random.choice(state.legal_actions())
                else:
                    print("Got player ", str(cur_player))
                    break
            state.apply_action(action)
        reward += state.returns()[player]
    return reward


def play_n_poker_games_against_random(
    game: SpielGame, callable_func: Callable, num_random_games: int
) -> float:
    opponents = [
        CandidStatistician(),
        LooseAggressive(),
        LoosePassive(),
        TightPassive(),
        TightAggressive(),
    ]
    total_reward = 0
    for i in range(num_random_games // 2 // len(opponents)):
        reward = play_poker_game_against_random(
            game, rescale_func(callable_func), opponents
        )
        total_reward += reward
    return total_reward / num_random_games


def play_poker_game_against_random(
    game: SpielGame, callable_func: Callable, opponents
) -> float:
    reward = 0
    for opponent in opponents:
        for player in [0, 1]:
            state = game.new_initial_state()
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes, probs = zip(*state.chance_outcomes())
                    aidx = np.random.choice(range(len(outcomes)), p=probs)
                    action = outcomes[aidx]
                else:
                    cur_player = state.current_player()
                    if cur_player == player:
                        probs = callable_func(state)
                        action = np.random.choice(
                            list(probs.keys()), p=list(probs.values())
                        )
                    elif cur_player == 1 - player:
                        probs = opponent(state)
                        action = np.random.choice(
                            list(probs.keys()), p=list(probs.values())
                        )
                    else:
                        print("Got player ", str(cur_player))
                        break
                state.apply_action(action)
            reward += state.returns()[player]
    return reward


def load_module(name):
    if ":" in name:
        mod_name, attr_name = name.split(":")
    else:
        li = name.split(".")
        mod_name, attr_name = ".".join(li[:-1]), li[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_pickle(data, file):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file):
    file = Path(file)
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def init_object(init_class: Type[object], possible_args: Dict[str, Any], **kwargs):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(init_class.__init__).args
    for possible_key in possible_args_copy:
        if possible_key in args:
            continue
        for key in args:
            if 0.8 <= string_similarity(possible_key, key) < 1:
                print(possible_key, key, string_similarity(possible_key, key))
                text = "Possible spelling error between passed {} and true {}".format(
                    possible_key, key
                )
                raise ValueError(text)
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    new_object = init_class(**params)
    return new_object


def run_method(
    method: Callable,
    possible_args: Dict[str, Any],
    **kwargs,
):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(method).args
    for possible_key in possible_args_copy:
        if possible_key in args:
            continue
        for key in args:
            if 0.8 <= string_similarity(possible_key, key) < 1:
                print(possible_key, key, string_similarity(possible_key, key))
                text = "Possible spelling error between passed {} and true {}".format(
                    possible_key, key
                )
                raise ValueError(text)
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    result = method(**params)
    return result

def get_host_ip():
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

def get_server_id():
    ip = get_host_ip()
    server_id = int(ip.split(".")[-1])
    return server_id
