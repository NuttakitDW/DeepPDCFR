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
    resume = False

    # logger
    writer_strings = ["stdout"]
    save_log = False
    if save_log:
        folder = Path(__file__).parents[1] / log_folder / algo_name / game_name
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
