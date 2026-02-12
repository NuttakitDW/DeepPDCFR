import math
import os
import random
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import stats
from deeppdcfr.logger import Logger

from deeppdcfr.game import read_game_config
from deeppdcfr.utils import (
    set_seed,
    SpielState,
    evalute_explotability,
    compute_lbr,
    play_n_games_against_random,
)
from deeppdcfr.parallel import compute_lbr_parallel


class DeepCumuAdv:
    def __init__(
        self,
        game_name,
        num_episodes=400,
        advantage_buffer_size=100_0000,
        ave_policy_buffer_size=100_0000,
        learning_rate=1e-4,
        num_traversals=20,
        advantage_network_train_steps=1,
        ave_policy_network_train_steps=1,
        advantage_batch_size=-1,
        ave_policy_batch_size=-1,
        num_layers=2,
        num_hiddens=128,
        evaluation_frequency=10,
        reinitialize_advantage_networks=True,
        use_regret_matching_argmax=True,
        epsilon=0.6,
        logger=None,
        fit_advantage=True,
        use_baseline=False,
        baseline_buffer_size=100_0000,
        baseline_batch_size=-1,
        baseline_network_train_steps=1,
        gamma=0,
        play_against_random=False,
        num_random_games=20000,
        num_lbr_samples=10000,
        lbr_eval_infer_device="same",
        device="cpu",
        seed=0,
        num_workers=1,
        evaluate_at_start=True,
    ):
        self.game_name = game_name
        self.num_workers = num_workers
        self.play_against_random = play_against_random
        self.evaluate_at_start = evaluate_at_start
        self.logger = logger or Logger(writer_strings=[])
        self.game = self.load_game()
        self.num_players = self.game.num_players()
        self.infostate_size = self.game.information_state_tensor_size()
        self.action_size = self.game.num_distinct_actions()
        self.advantage_buffer_size = advantage_buffer_size
        self.ave_policy_buffer_size = ave_policy_buffer_size
        self.num_episodes = num_episodes
        self.num_traversals = num_traversals
        self.num_iterations = self.num_episodes // (self.num_traversals * 2)
        self.advantage_network_train_steps = advantage_network_train_steps
        self.ave_policy_network_train_steps = ave_policy_network_train_steps
        self.ave_policy_batch_size = ave_policy_batch_size
        self.advantage_batch_size = advantage_batch_size
        self.reinitialize_advantage_networks = reinitialize_advantage_networks
        self.learning_rate = learning_rate
        self.device = device
        self.gamma = gamma
        self.evaluation_frequency = evaluation_frequency
        self.use_regret_matching_argmax = use_regret_matching_argmax
        self.max_utility = max(self.game.max_utility(), abs(self.game.min_utility()))
        self.network_layers = [num_hiddens for _ in range(num_layers)]
        self.epsilon = epsilon
        self.num_random_games = num_random_games
        self.num_lbr_samples = num_lbr_samples
        self.lbr_eval_infer_device = lbr_eval_infer_device
        self.fit_advantage = fit_advantage
        self.use_baseline = use_baseline
        self.baseline_buffer_size = baseline_buffer_size
        self.baseline_network_train_steps = baseline_network_train_steps
        self.baseline_batch_size = baseline_batch_size
        self.num_iteration = 0
        self.nodes_touched = 0
        self.episode = 0
        set_seed(seed)
        self.init_ave_policy_trainer()
        self.init_regret_trainers()

        if self.use_baseline:
            self.init_q_value_trainer()

    def init_ave_policy_trainer(self):
        self.ave_policy_trainer = AvePolicyTrainer(
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.ave_policy_buffer_size,
            self.ave_policy_batch_size,
            self.ave_policy_network_train_steps,
            self.logger,
            self.device,
            self.gamma,
        )

    def init_regret_trainers(self):
        self.regret_trainers = [
            RegretTrainer(
                self.infostate_size,
                self.action_size,
                self.network_layers,
                self.learning_rate,
                self.advantage_buffer_size,
                self.advantage_batch_size,
                self.advantage_network_train_steps,
                self.logger,
                self.use_regret_matching_argmax,
                self.device,
            )
            for _ in range(self.num_players)
        ]
    
    def init_q_value_trainer(self):
        root_state = self.game.new_initial_state()
        history_tensor = np.append(
            root_state.information_state_tensor(0),
            root_state.information_state_tensor(1),
        )
        history_size = len(history_tensor)

        self.q_value_trainer = QValueTrainer(
            history_size,
            self.infostate_size,
            self.action_size,
            self.network_layers,
            self.learning_rate,
            self.baseline_buffer_size,
            self.baseline_batch_size,
            self.baseline_network_train_steps,
            self.logger,
            self.regret_trainers,
            self.device,
        )

    def solve(self):
        resumed = getattr(self, '_resumed', False)
        remaining = self.num_iterations - self.num_iteration
        if resumed:
            self.logger.info(
                f"Resumed: skipping {self.num_iteration} completed iterations, "
                f"{remaining} remaining"
            )
        else:
            if self.evaluate_at_start:
                self.logger.info("initial evaluate start")
                eval_t0 = time.perf_counter()
                self.evaluate()
                eval_dt_ms = (time.perf_counter() - eval_t0) * 1000.0
                self.logger.info(
                    "initial evaluate done | dt_ms={:.1f}".format(eval_dt_ms)
                )
            else:
                self.logger.info("skip initial evaluate (evaluate_at_start=false)")
            remaining = self.num_iterations
        for _ in range(remaining):
            self.iteration()

    def iteration(self):
        self.num_iteration += 1
        for player in range(self.num_players):
            self.collect_training_data(player)
            self.train_regret(player)
            if self.use_baseline:
                self.train_baseline(player)
        if (
            self.num_iteration % self.evaluation_frequency == 0
            or self.num_iteration < self.evaluation_frequency
            and self.num_iteration % (self.evaluation_frequency // 3) == 0
        ):
            self.train_average_policy()
            self.evaluate()

    def collect_training_data(self, player):
        self.regret_trainers[player].reset_buffer()
        if self.num_workers > 1:
            self._collect_training_data_parallel(player)
        else:
            self._collect_training_data_sequential(player)

    def _collect_training_data_sequential(self, player):
        progress_interval = 1000
        self.logger.info(
            "collect sequential start | it={} | player={} | traversals={}".format(
                self.num_iteration, player, self.num_traversals
            )
        )
        for traversal_id in range(1, self.num_traversals + 1):
            self.episode += 1
            root_state = self.skip_chance_state(self.game.new_initial_state())
            self.dfs(root_state, player)
            if (
                traversal_id % progress_interval == 0
                or traversal_id == self.num_traversals
            ):
                self.logger.info(
                    "collect progress | it={} | player={} | traversals={}/{}".format(
                        self.num_iteration,
                        player,
                        traversal_id,
                        self.num_traversals,
                    )
                )

    def _collect_training_data_parallel(self, player):
        self._set_dfs_threads()
        from deeppdcfr.parallel import (
            run_parallel_dfs,
            _merge_buffer_data,
            _merge_circular_buffer_data,
        )

        self.logger.info(f"Launching {self.num_workers} parallel DFS workers for player {player}...")
        result = run_parallel_dfs(
            game_name=self.game_name,
            player=player,
            num_traversals=self.num_traversals,
            num_workers=self.num_workers,
            num_iteration=self.num_iteration,
            epsilon=self.epsilon,
            fit_advantage=self.fit_advantage,
            max_utility=self.max_utility,
            use_regret_matching_argmax=self.use_regret_matching_argmax,
            infostate_size=self.infostate_size,
            action_size=self.action_size,
            network_layers=self.network_layers,
            advantage_buffer_size=self.advantage_buffer_size,
            ave_policy_buffer_size=self.ave_policy_buffer_size,
            regret_trainers=self.regret_trainers,
            ave_policy_trainer=self.ave_policy_trainer,
            use_baseline=self.use_baseline,
            q_value_trainer=self.q_value_trainer if self.use_baseline else None,
            base_seed=self.episode,
            log_fn=self.logger.info,
            log_iteration=self.num_iteration,
            log_player=player,
        )

        # Merge regret data into player's buffer
        _merge_buffer_data(
            self.regret_trainers[player].buffer,
            result["regret_data_list"],
        )
        # Merge ave policy data
        _merge_buffer_data(
            self.ave_policy_trainer.buffer,
            result["ave_policy_data_list"],
        )
        if self.use_baseline:
            _merge_circular_buffer_data(
                self.q_value_trainer.buffer,
                result["baseline_data_list"],
            )
        self.nodes_touched += result["nodes_touched"]
        self.episode += self.num_traversals
        self.logger.info(f"Parallel DFS done: {result['nodes_touched']} nodes touched")

    def train_regret(self, player):
        self._set_training_threads()
        if self.reinitialize_advantage_networks:
            self.regret_trainers[player].reset()
        regret_loss = self.regret_trainers[player].train_model(self.num_iteration)
        self.logger.record("regret_loss_{}".format(player), regret_loss)

    def train_baseline(self, player):
        self._set_training_threads()
        baseline_loss = self.q_value_trainer.train_model(self.num_iteration)
        if baseline_loss is not None:
            self.logger.record("baseline_loss_{}".format(player), baseline_loss)

    def train_average_policy(self):
        self._set_training_threads()
        self.ave_policy_trainer.reset()
        ave_policy_loss = self.ave_policy_trainer.train_model(self.num_iteration)
        self.logger.info("average policy loss: {}".format(ave_policy_loss))

    def evaluate(self):
        self.logger.record("nodes_touched", self.nodes_touched)
        self.logger.record("iteration", self.num_iteration)
        self.logger.record("episode", self.episode)
        if self.play_against_random:
            if self.poker_game:
                policy_fn = self.ave_policy_trainer.action_probabilities
                if (
                    self.lbr_eval_infer_device == "cpu"
                    and self.device != "cpu"
                ):
                    cpu_model = MLP(
                        self.ave_policy_trainer.input_size,
                        self.ave_policy_trainer.network_layers,
                        self.ave_policy_trainer.output_size,
                    ).to("cpu")
                    cpu_model.load_state_dict(self.ave_policy_trainer.model.state_dict())
                    cpu_model.eval()

                    def policy_fn(s, probs_as_dict=True):
                        x = torch.as_tensor(
                            self.ave_policy_trainer.get_infostate_tensor(s),
                            dtype=torch.float32,
                            device="cpu",
                        )
                        mask = torch.as_tensor(
                            s.legal_actions_mask(), dtype=torch.float32, device="cpu"
                        )
                        with torch.no_grad():
                            logits = cpu_model(x)
                            legal_logits = torch.where(mask == 1, logits, -10e20)
                            policy = self.ave_policy_trainer.softmax_fn(legal_logits)
                            policy = policy.cpu().numpy()
                        if probs_as_dict:
                            return {action: policy[action] for action in s.legal_actions()}
                        return policy

                if self.num_workers > 1:
                    lbr_exp = compute_lbr_parallel(
                        game_name=self.game_name,
                        num_lbr_samples=self.num_lbr_samples,
                        num_workers=self.num_workers,
                        infostate_size=self.infostate_size,
                        action_size=self.action_size,
                        network_layers=self.network_layers,
                        ave_policy_trainer=self.ave_policy_trainer,
                    )
                else:
                    lbr_exp = compute_lbr(
                        self.game,
                        policy_fn,
                        self.num_lbr_samples,
                    )
                self.logger.record("lbr_exp", lbr_exp)
            else:
                reward = play_n_games_against_random(
                    self.game,
                    self.ave_policy_trainer.action_probabilities,
                    self.num_random_games,
                )
                self.logger.record("reward", reward)
            self.logger.dump(step=self.episode)
        else:
            exp = evalute_explotability(
                self.game, self.ave_policy_trainer.action_probabilities
            )
            self.logger.record("exp", exp)
            self.logger.dump(step=self.episode)
        self.save_checkpoint()

    def save_checkpoint(self):
        if not self.logger.folder:
            return
        ckpt_dir = self.logger.folder / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'num_iteration': self.num_iteration,
            'episode': self.episode,
            'nodes_touched': self.nodes_touched,
            'ave_policy_model': self.ave_policy_trainer.model.state_dict(),
        }
        for i, rt in enumerate(self.regret_trainers):
            checkpoint[f'regret_model_{i}'] = rt.model.state_dict()
            checkpoint[f'regret_target_model_{i}'] = rt.target_model.state_dict()
            checkpoint[f'regret_optimizer_{i}'] = rt.optimizer.state_dict()
            if hasattr(rt, 'imm_model'):
                checkpoint[f'regret_imm_model_{i}'] = rt.imm_model.state_dict()
            if hasattr(rt, 'imm_optimizer'):
                checkpoint[f'regret_imm_optimizer_{i}'] = rt.imm_optimizer.state_dict()
        if self.use_baseline:
            checkpoint['baseline_model'] = self.q_value_trainer.model.state_dict()

        path = ckpt_dir / f"checkpoint_{self.episode}.pt"
        torch.save(checkpoint, path)

        latest = ckpt_dir / "latest.pt"
        tmp = ckpt_dir / "latest.pt.tmp"
        torch.save(checkpoint, tmp)
        os.replace(tmp, latest)

        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore counters
        self.num_iteration = checkpoint['num_iteration']
        self.episode = checkpoint['episode']
        self.nodes_touched = checkpoint['nodes_touched']
        self.logger.info(
            f"Resuming from {path} | iteration={self.num_iteration} "
            f"episode={self.episode} nodes_touched={self.nodes_touched}"
        )

        # Restore ave_policy model
        self.ave_policy_trainer.model.load_state_dict(checkpoint['ave_policy_model'])

        # Restore regret models + optimizers
        for i, rt in enumerate(self.regret_trainers):
            rt.model.load_state_dict(checkpoint[f'regret_model_{i}'])
            rt.target_model.load_state_dict(checkpoint[f'regret_target_model_{i}'])
            if hasattr(rt, 'imm_model') and f'regret_imm_model_{i}' in checkpoint:
                rt.imm_model.load_state_dict(checkpoint[f'regret_imm_model_{i}'])
            # Optimizer states (may be absent in old checkpoints)
            try:
                if f'regret_optimizer_{i}' in checkpoint:
                    rt.optimizer.load_state_dict(checkpoint[f'regret_optimizer_{i}'])
                if hasattr(rt, 'imm_optimizer') and f'regret_imm_optimizer_{i}' in checkpoint:
                    rt.imm_optimizer.load_state_dict(checkpoint[f'regret_imm_optimizer_{i}'])
            except Exception as e:
                self.logger.warn(f"Could not restore optimizer state for player {i}: {e}")

        # Restore baseline model
        if self.use_baseline and 'baseline_model' in checkpoint:
            self.q_value_trainer.model.load_state_dict(checkpoint['baseline_model'])

        self._resumed = True

    def dfs(
        self,
        s,
        traverser,
        my_reach=1.0,
        opp_reach=1.0,
        opp_sample_reach=1.0,
        sample_reach=1.0,
    ):
        self.nodes_touched += 1
        player = s.current_player()
        if player == -4:
            return s.returns()[traverser] / self.max_utility
        legal_actions = s.legal_actions()
        policy = self.regret_trainers[player].get_policy(s, self.num_iteration)
        num_actions = s.num_distinct_actions()
        uniform_policy = np.array(s.legal_actions_mask()) / len(s.legal_actions())
        sample_policy = (
            uniform_policy * self.epsilon + policy * (1 - self.epsilon)
            if player == traverser
            else policy
        )
        sample_policy /= sample_policy.sum()
        action = np.random.choice(range(num_actions), p=sample_policy)
        sample_prob, prob = (
            sample_policy[action].item(),
            policy[action].item(),
        )
        ns = self.skip_chance_state(s.child(action))
        if player == 1 - traverser:
            self.ave_policy_trainer.add_data(
                self.get_infostate_tensor(s),
                policy,
                s.legal_actions_mask(),
                self.num_iteration,
            )
            if self.use_baseline:
                q_values = self.q_value_trainer.get_baseline(s, traverser)
            else:
                q_values = np.zeros_like(policy)
            action_value = self.dfs(
                ns,
                traverser,
                my_reach,
                opp_reach * prob,
                opp_sample_reach * sample_prob,
                sample_reach * sample_prob,
            )
            q_values[action] += (action_value - q_values[action]) / sample_prob
            value = np.dot(q_values, policy)
        else:
            if self.use_baseline:
                q_values = self.q_value_trainer.get_baseline(s, traverser)
            else:
                q_values = np.zeros_like(policy)
            action_value = self.dfs(
                ns,
                traverser,
                my_reach * prob,
                opp_reach,
                opp_sample_reach,
                sample_reach * sample_prob,
            )
            q_values[action] += (action_value - q_values[action]) / sample_prob
            value = np.dot(q_values, policy)

            cf_regrets = np.zeros_like(policy)
            im_weight = 1 if self.fit_advantage else opp_reach / sample_reach
            cf_regrets[legal_actions] = -value * im_weight
            cf_regrets[legal_actions] += q_values[legal_actions] * im_weight
            self.regret_trainers[player].add_data(
                self.get_infostate_tensor(s),
                cf_regrets,
                s.legal_actions_mask(),
                self.num_iteration,
            )

        if self.use_baseline:
            self.q_value_trainer.add_data(
                self.get_history_tensor(s),
                action,
                self.get_history_tensor(ns),
                self.get_infostate_tensor(ns) if not ns.is_terminal() else None,
                ns.legal_actions_mask() if not ns.is_terminal() else None,
                ns.current_player(),
                int(ns.is_terminal()),
                ns.returns()[0] / self.max_utility, 
            )
        return value

    def get_infostate_tensor(self, s):
        return s.information_state_tensor()

    def get_history_tensor(self, s):        
        return np.append(
            s.information_state_tensor(0), s.information_state_tensor(1)
        )
    def load_game(self):
        game_config = read_game_config(self.game_name)
        self.poker_game = game_config.poker
        if game_config.large_game:
            if not self.play_against_random:
                self.logger.warn("Large game detected, using LBR for exploitability estimation.")
            self.play_against_random = True

        game = game_config.load_game()
        return game

    def _set_training_threads(self):
        """Use all cores for training (one thread per worker core)."""
        torch.set_num_threads(max(self.num_workers, 1))

    def _set_dfs_threads(self):
        """Use 1 thread in main process during DFS (workers handle parallelism)."""
        torch.set_num_threads(1)

    def skip_chance_state(self, s):
        while s.current_player() == -1:
            self.nodes_touched += 1
            actions, probs = zip(*s.chance_outcomes())
            aid = np.random.choice(range(len(actions)), p=probs)
            s.apply_action(actions[aid])
        return s

class Trainer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.network_layers = network_layers
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.logger = logger
        self.device = device

        self.model = self.init_model()
        self.buffer = ReservoirBuffer(
            self.buffer_size, self.input_size, self.output_size, device=self.device
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.softmax_fn = nn.Softmax(dim=-1)

    def init_model(self):
        model = MLP(self.input_size, self.network_layers, self.output_size).to(
            self.device
        )
        return model
    def reset(self):
        self.model.reset_parameters()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def reset_buffer(self):
        self.buffer.reset()

    def add_data(self, infostate, q_value, q_value_mask, iteration):
        self.buffer.add(infostate, q_value, q_value_mask, iteration)

    def get_infostate_tensor(self, s):
        return s.information_state_tensor()

    def get_history_tensor(self, s):
        return np.append(
            s.information_state_tensor(0), s.information_state_tensor(1)
        )

class RegretTrainer(Trainer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        use_regret_matching_argmax: bool,
        device: str = "cpu",
    ):
        super().__init__(
            input_size,
            output_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            device,
        )
        self.use_regret_matching_argmax = use_regret_matching_argmax
        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())

    def forward(self, model, x, mask):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            with torch.no_grad():
                legal_regrets = self.predict(model, x, mask)
                return legal_regrets.cpu().numpy()

    def predict(self, model, x, mask):
        legal_regrets = model(x) * mask
        return legal_regrets

    def train_model(self, T):
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] regret loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        self.target_model.load_state_dict(self.model.state_dict())
        return loss.item()

    def compute_loss(self, samples, T):
        infostates, cf_regrets, legal_actions_mask, iterations = samples
        with torch.no_grad():
            target_outputs = self.predict(
                self.target_model, infostates, legal_actions_mask
            )
        regrets = target_outputs + cf_regrets
        outputs = self.predict(self.model, infostates, legal_actions_mask)
        loss = self.loss_fn(outputs, regrets)
        return loss

    def get_policy(self, s: SpielState, T: int) -> np.ndarray:
        regrets = self.get_regrets(s)
        legal_actions = s.legal_actions()
        return self.regret_matching(regrets, legal_actions)

    def get_regrets(self, s: SpielState) -> np.ndarray:
        regrets = self.forward(
            self.model, self.get_infostate_tensor(s), s.legal_actions_mask()
        )
        return regrets

    def regret_matching(self, regrets, legal_actions):
        legal_regrets = np.zeros_like(regrets)
        legal_regrets[legal_actions] = regrets[legal_actions]
        pos_sum = np.sum(np.maximum(legal_regrets, 0))
        if pos_sum > 0:
            return np.maximum(legal_regrets, 0) / pos_sum
        else:
            policy = np.zeros_like(regrets)
            if self.use_regret_matching_argmax:
                max_action_id = legal_actions[np.argmax(regrets[legal_actions])]
                policy[max_action_id] = 1
            else:
                policy[legal_actions] = 1 / len(legal_actions)
            return policy


class AvePolicyTrainer(Trainer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        device: str = "cpu",
        gamma: int = 0,
    ):
        super().__init__(
            input_size,
            output_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            device=device,
        )
        self.gamma = gamma

    def forward(self, x, mask):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            mask = torch.as_tensor(mask, dtype=torch.float32)
            with torch.no_grad():
                policy = self.predict(x, mask)
                return policy.cpu().numpy()

    def predict(self, x, mask):
        logits = self.model(x)
        legal_logits = torch.where(mask == 1, logits, -10e20)
        policy = self.softmax_fn(legal_logits)
        return policy

    def train_model(self, T):
        for train_step in range(self.train_steps):
            samples = self.buffer.sample(self.batch_size)
            loss = self.compute_loss(samples, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] policy loss: {}".format(
                        train_step, self.train_steps, loss.item()
                    )
                )
        return loss.item()

    def compute_loss(self, samples, T):
        infostates, policy, legal_actions_mask, iterations = samples
        iterations = torch.pow(iterations / T * 2, self.gamma / 2)
        outputs = self.predict(infostates, legal_actions_mask)
        loss = self.loss_fn(outputs * iterations, policy * iterations)
        return loss

    def action_probabilities(self, s, probs_as_dict=True):
        policy = self.forward(self.get_infostate_tensor(s), s.legal_actions_mask())
        if probs_as_dict:
            probs_dict = {action: policy[action] for action in s.legal_actions()}
            return probs_dict
        else:
            return policy


class ReservoirBuffer:
    def __init__(self, buffer_size, infostate_size, action_size, device="cpu"):
        self.buffer_size = buffer_size
        self.infostate_size = infostate_size
        self.action_size = action_size
        self.device = device
        self.reset()

    def reset(self):
        if hasattr(self, "cur_id"):
            self.cur_id = 0
            return

        self.infostate_buf = np.empty(
            [self.buffer_size, self.infostate_size], dtype=np.float32
        )
        self.q_value_buf = np.empty(
            [self.buffer_size, self.action_size], dtype=np.float32
        )
        self.q_value_mask_buf = np.empty(
            [self.buffer_size, self.action_size], dtype=np.uint8
        )
        self.iteration_buf = np.empty([self.buffer_size, 1], dtype=np.int32)
        self.cur_id = 0

    def add(self, infostate, q_value, q_value_mask, iteration):
        if self.cur_id < self.buffer_size:
            self.add_data(self.cur_id, infostate, q_value, q_value_mask, iteration)
        else:
            idx = np.random.randint(low=0, high=self.cur_id + 1)
            if idx < self.buffer_size:
                self.add_data(idx, infostate, q_value, q_value_mask, iteration)
        self.cur_id += 1

    def add_data(self, idx, infostate, q_value, q_value_mask, iteration):
        self.infostate_buf[idx] = infostate
        self.q_value_buf[idx] = q_value
        self.q_value_mask_buf[idx] = q_value_mask
        self.iteration_buf[idx] = iteration

    def sample(self, num_samples=-1):
        self.data_length = min(self.cur_id, self.buffer_size)
        if num_samples > self.data_length:
            num_samples = -1
        if num_samples == -1:
            idxs = list(range(self.data_length))
        else:
            idxs = random.sample(range(self.data_length), num_samples)
        data = (
            self.infostate_buf[idxs],
            self.q_value_buf[idxs],
            self.q_value_mask_buf[idxs],
            self.iteration_buf[idxs],
        )
        data_tensor = tuple(map(self.numpy_to_tensor, data))
        return data_tensor

    def numpy_to_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def __len__(self):
        return self.cur_id


class SonnetLinear(nn.Module):
    def __init__(self, in_features, out_features, activation=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stddev = 1 / math.sqrt(self.in_features)
        self.weight = nn.Parameter(
            torch.Tensor(
                stats.truncnorm.rvs(
                    -2,
                    2,
                    loc=0,
                    scale=stddev,
                    size=[self.out_features, self.in_features],
                ),
            )
        )
        self.bias = nn.Parameter(torch.zeros([self.out_features]))

    def forward(self, input):
        y = F.linear(input, self.weight, self.bias)
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

    def reset_parameters(self) -> None:
        self.weight = nn.Parameter(torch.zeros([self.out_features, self.in_features]))
        self.bias = nn.Parameter(torch.zeros([self.out_features]))

    def forward(self, input):
        y = F.linear(input, self.weight, self.bias)
        if self.activation:
            y = F.relu(y)
        return y


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        input_sizes = [input_size, *hidden_size[:-1]]
        output_sizes = [*hidden_size]
        activations = [True] * (len(hidden_size))
        layer_list = [
            SonnetLinear(in_size, out_size, activation)
            for in_size, out_size, activation in zip(
                input_sizes, output_sizes, activations
            )
        ]
        layer_list.append(ZeroInitLinear(hidden_size[-1], output_size, False))
        self.layers = nn.Sequential(*layer_list)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        return self.layers(x)
class QValueTrainer(Trainer):
    def __init__(
        self,
        history_size: int,
        state_size: int,
        action_size: int,
        network_layers: list[int],
        learning_rate: float,
        buffer_size: int,
        batch_size: int,
        train_steps: int,
        logger: Logger,
        regret_trainers: list[RegretTrainer],
        device: str = "cpu",
    ):
        super().__init__(
            history_size,
            action_size,
            network_layers,
            learning_rate,
            buffer_size,
            batch_size,
            train_steps,
            logger,
            device,
        )
        self.state_size = state_size
        self.buffer = CircularBuffer(
            self.buffer_size,
            self.input_size,
            state_size,
            self.output_size,
            device=self.device,
        )
        self.model = self.init_model()
        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.regret_trainers = regret_trainers

    def init_model(self):
        model = MLP(self.input_size, self.network_layers, self.output_size).to(
            self.device
        )
        return model

    def get_baseline(self, s: SpielState, player: int) -> np.ndarray:
        history_tensor = self.get_history_tensor(s)
        coef = 1 if player == 0 else -1
        baseline = self.forward(history_tensor) * coef
        baseline = baseline * np.array(s.legal_actions_mask(), dtype=np.float32)
        # baseline =  np.zeros(s.num_distinct_actions(), dtype=float)
        return baseline

    def add_data(
        self,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        if done:
            next_state = np.zeros([self.state_size], dtype=np.float32)
            next_legal_actions_mask = np.zeros([self.output_size], dtype=np.uint8)
            next_player = 0
        self.buffer.add(
            history,
            action,
            next_history,
            next_state,
            next_legal_actions_mask,
            next_player,
            done,
            reward,
        )

    def _precompute_next_strategies(self, T):
        """Pre-compute next strategies for all buffer items using regret models.

        Since regret models don't change during baseline training, we compute
        once and reuse via index lookup — avoids 2+ forward passes per step.
        """
        n = len(self.buffer)
        chunk = max(self.batch_size, 4096)
        all_strategies = torch.zeros(n, self.output_size, device=self.device)

        with torch.no_grad():
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                next_states = torch.as_tensor(
                    self.buffer.next_state_buf[start:end],
                    dtype=torch.float32, device=self.device,
                )
                next_mask = torch.as_tensor(
                    self.buffer.next_legal_actions_mask_buf[start:end],
                    dtype=torch.float32, device=self.device,
                )
                next_players = torch.as_tensor(
                    self.buffer.next_player_buf[start:end],
                    dtype=torch.int64, device=self.device,
                )
                strategies = self._compute_strategies_batch(
                    next_states, next_mask, next_players, T,
                )
                all_strategies[start:end] = strategies
        return all_strategies

    def _compute_strategies_batch(self, next_states, next_mask, next_players, T):
        """Compute per-item strategy from regret models. Override in subclass."""
        players_next_stragies = []
        for player in [0, 1]:
            p0_regrets = self.regret_trainers[player].predict(
                self.regret_trainers[player].model, next_states, next_mask,
            )
            p0_legal_regrets = p0_regrets * next_mask
            p0_regrets_pos = torch.clamp(p0_regrets, min=0)
            p0_regrets_pos_sum = torch.sum(p0_regrets_pos, dim=1, keepdim=True)
            p0_rm_next_strategies = p0_regrets_pos / p0_regrets_pos_sum
            _, p0_max_legal_action_id = torch.max(
                torch.where(
                    next_mask == 1,
                    p0_legal_regrets,
                    torch.tensor(float("-inf"), device=self.device),
                ),
                dim=1,
            )
            p0_max_next_strategies = F.one_hot(
                p0_max_legal_action_id, self.output_size
            )
            p0_next_strategies = torch.where(
                p0_regrets_pos_sum == 0,
                p0_max_next_strategies,
                p0_rm_next_strategies,
            )
            players_next_stragies.append(p0_next_strategies)
        return torch.where(
            next_players.unsqueeze(1) == 0,
            players_next_stragies[0],
            players_next_stragies[1],
        )

    def train_model(self, T):
        self.model = self.init_model()
        self.target_model = self.init_model()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        if self.batch_size > 0 and len(self.buffer) < self.batch_size:
            return
        # Pre-compute strategies once (regret models are frozen during baseline training)
        all_next_strategies = self._precompute_next_strategies(T)

        best_ema_loss = float("inf")
        ema_loss = None
        ema_alpha = 0.05
        min_delta = 1e-4
        patience = 500
        steps_without_improvement = 0
        self.best_model = self.init_model()
        for train_step in range(self.train_steps + 1):
            samples, idxs = self.buffer.sample_with_indices(self.batch_size)
            (
                histories,
                next_histories,
                next_states,
                rewards,
                next_legal_actions_mask,
                next_players,
                actions,
                dones,
            ) = samples

            q_values = self.model(histories)  # [B, A]
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

            with torch.no_grad():
                next_q_values = self.target_model(next_histories)  # [B, A]
                next_strategies = all_next_strategies[idxs]  # [B, A]

            target = rewards + (1 - dones) * torch.sum(
                next_q_values * next_strategies, dim=1, keepdim=False
            )

            loss = self.loss_fn(q_value, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if train_step % 50 == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            current_loss = loss.item()
            if ema_loss is None:
                ema_loss = current_loss
            else:
                ema_loss = ema_alpha * current_loss + (1 - ema_alpha) * ema_loss

            if ema_loss < best_ema_loss - min_delta:
                best_ema_loss = ema_loss
                self.best_model.load_state_dict(self.model.state_dict())
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if train_step % 100 == 0:
                self.logger.info(
                    "[{}/{}] baseline loss: {} (ema: {:.6f}, best_ema: {:.6f})".format(
                        train_step, self.train_steps, current_loss, ema_loss, best_ema_loss
                    )
                )

            if steps_without_improvement >= patience:
                self.logger.info(
                    "early stop baseline at step {}/{}: best_ema_loss {:.6f}".format(
                        train_step, self.train_steps, best_ema_loss
                    )
                )
                break

        self.model.load_state_dict(self.best_model.state_dict())
        return best_ema_loss

    def forward(self, x):
        with torch.device(self.device):
            x = torch.as_tensor(x, dtype=torch.float32)
            with torch.no_grad():
                return self.model(x).cpu().numpy()


class CircularBuffer:
    def __init__(
        self, buffer_size, history_size, state_size, action_size, device="cpu"
    ):
        self.buffer_size = buffer_size
        self.history_size = history_size
        self.action_size = action_size
        self.state_size = state_size
        self.device = device
        self.reset()

    def reset(self):
        self.history_buf = np.empty(
            [self.buffer_size, self.history_size], dtype=np.float32
        )
        self.action_buf = np.empty([self.buffer_size], dtype=np.int32)
        self.next_history_buf = np.empty(
            [self.buffer_size, self.history_size], dtype=np.float32
        )
        self.next_state_buf = np.empty(
            [self.buffer_size, self.state_size], dtype=np.float32
        )
        self.next_legal_actions_mask_buf = np.empty(
            [self.buffer_size, self.action_size], dtype=np.uint8
        )
        self.next_player_buf = np.empty([self.buffer_size], dtype=np.int8)
        self.done_buf = np.empty([self.buffer_size], dtype=np.int8)
        self.reward_buf = np.empty([self.buffer_size], dtype=np.float32)
        self.cur_id = 0

    def add(
        self,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        self.add_data(
            self.cur_id,
            history,
            action,
            next_history,
            next_state,
            next_legal_actions_mask,
            next_player,
            done,
            reward,
        )
        self.cur_id = (self.cur_id + 1) % self.buffer_size

    def add_data(
        self,
        idx,
        history,
        action,
        next_history,
        next_state,
        next_legal_actions_mask,
        next_player,
        done,
        reward,
    ):
        self.history_buf[idx] = history
        self.action_buf[idx] = action
        self.next_history_buf[idx] = next_history
        self.next_state_buf[idx] = next_state
        self.next_legal_actions_mask_buf[idx] = next_legal_actions_mask
        self.next_player_buf[idx] = next_player
        self.done_buf[idx] = done
        self.reward_buf[idx] = reward

    def sample(self, num_samples=-1):
        data_length = len(self)
        if num_samples == -1:
            idxs = list(range(data_length))
        else:
            idxs = random.sample(range(data_length), num_samples)
        float_data = (
            self.history_buf[idxs],
            self.next_history_buf[idxs],
            self.next_state_buf[idxs],
            self.reward_buf[idxs],
        )
        int_data = (
            self.next_legal_actions_mask_buf[idxs],
            self.next_player_buf[idxs],
            self.action_buf[idxs],
            self.done_buf[idxs],
        )
        data_tensor = (
            *map(self.numpy_to_float_tensor, float_data),
            *map(self.numpy_to_int_tensor, int_data),
        )
        return data_tensor

    def sample_with_indices(self, num_samples=-1):
        data_length = len(self)
        if num_samples == -1:
            idxs = list(range(data_length))
        else:
            idxs = random.sample(range(data_length), num_samples)
        float_data = (
            self.history_buf[idxs],
            self.next_history_buf[idxs],
            self.next_state_buf[idxs],
            self.reward_buf[idxs],
        )
        int_data = (
            self.next_legal_actions_mask_buf[idxs],
            self.next_player_buf[idxs],
            self.action_buf[idxs],
            self.done_buf[idxs],
        )
        data_tensor = (
            *map(self.numpy_to_float_tensor, float_data),
            *map(self.numpy_to_int_tensor, int_data),
        )
        return data_tensor, idxs

    def numpy_to_float_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.float32, device=self.device)

    def numpy_to_int_tensor(self, data):
        return torch.as_tensor(data, dtype=torch.int64, device=self.device)

    def __len__(self):
        return min(self.cur_id, self.buffer_size)
