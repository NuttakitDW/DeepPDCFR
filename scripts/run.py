import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path

from deeppdcfr.exp import ServerFileStorageObserver, ex
from deeppdcfr.logger import Logger
from deeppdcfr.utils import init_object, load_module, run_method


@ex.config
def config():
    seed = 0
    algo_name = "CFR"
    game_name = "KuhnPoker"
    log_folder = "logs"

    # Scenario
    stack_range = [20, 200]
    pot_range = [3, 60]
    fixed_oop_bet_config = None
    fixed_ip_bet_config = None

    # Training
    num_iterations = 0
    num_traversals = 500
    evaluation_frequency = 3
    epsilon = 0.6

    # Regret networks
    advantage_buffer_size = 10000
    advantage_batch_size = 512
    advantage_network_train_steps = 500
    reinitialize_advantage_networks = False
    reinitialize_imm_regret_networks = True
    use_regret_matching_argmax = True
    alpha = 2.3

    # Policy network
    ave_policy_buffer_size = 20000
    ave_policy_batch_size = 512
    ave_policy_network_train_steps = 2000
    gamma = 2

    # Shared
    learning_rate = 3.0e-4

    # V2 model architecture
    card_embed_dim = 64
    situation_hidden = 384
    combo_hidden = 192
    fusion_hidden = 384
    fusion_residual_blocks = 2

    # System
    device = "cpu"
    max_time = 0
    save_interval = 600
    resume = False
    traversal_workers = 1
    traversal_device = None  # default: same as `device`
    traversal_mp_context = "spawn"
    traversal_chunk_size = 0  # 0 = auto

    # logger
    writer_strings = ["stdout"]
    save_log = False
    if save_log:
        folder = Path(__file__).parents[1] / log_folder / algo_name / game_name
        writer_strings += ["csv", "sacred", "tensorboard"]
        ex.observers.append(ServerFileStorageObserver(folder))


@ex.main
def main(algo_name, _config, _run):
    configs = dict(_config)
    if configs["save_log"]:
        configs["folder"] = configs["folder"] / str(_run._id)
    logger = init_object(Logger, configs)
    solver_class = load_module("deeppdcfr:{}".format(algo_name))

    solver = init_object(solver_class, configs, logger=logger)
    solver.solve()


if __name__ == "__main__":
    # Needed for multiprocessing (spawn) safety; Sacred's @ex.automain would run on import.
    ex.run_commandline()
