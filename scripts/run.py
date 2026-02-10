import os

# Allow PyTorch to use multiple threads during training.
# Workers set DEEPPDCFR_WORKER=1 and run single-threaded numpy,
# so this only affects the main process training loop.
os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS", "8")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "8")
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "8")

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

    # training
    num_episodes = 10000000
    advantage_buffer_size = 1000000
    ave_policy_buffer_size = 1000000
    learning_rate = 1e-3
    num_traversals = 10000
    advantage_network_train_steps = 750
    ave_policy_network_train_steps = 5000
    advantage_batch_size = 2048
    ave_policy_batch_size = 2048
    num_layers = 3
    num_hiddens = 64
    evaluation_frequency = 10
    reinitialize_advantage_networks = True
    reinitialize_imm_regret_networks = True
    use_regret_matching_argmax = True
    epsilon = 0.6
    alpha = 2.3
    gamma = 2
    fit_advantage = True
    play_against_random = False
    num_random_games = 20000
    num_lbr_samples = 10000
    device = "cpu"
    num_workers = 1

    # baseline
    use_baseline = False
    baseline_buffer_size = 1000000
    baseline_batch_size = 2048
    baseline_network_train_steps = 1000

    # logger
    writer_strings = ["stdout"]
    save_log = False
    folder = Path(__file__).parents[1] / log_folder / algo_name / game_name
    if save_log:
        writer_strings += ["csv", "sacred", "tensorboard"]
        ex.observers.append(ServerFileStorageObserver(folder))


@ex.automain
def main(algo_name, _config, _run):
    configs = dict(_config)
    if configs["save_log"]:
        configs["folder"] = configs["folder"] / str(_run._id)
    logger = init_object(Logger, configs)
    solver_class = load_module("deeppdcfr:{}".format(algo_name))

    solver = init_object(solver_class, configs, logger=logger)
    solver.solve()
