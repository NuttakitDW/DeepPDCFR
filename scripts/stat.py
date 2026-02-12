#!/usr/bin/env python3
"""Show training status with a pretty formatted display."""

import json
import glob
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deeppdcfr.game import read_game_config

LOGS_ROOT = Path(__file__).resolve().parent.parent / "logs"


def get_big_blind(game_name):
    """Get the big blind size in chips for a poker game."""
    try:
        game_config = read_game_config(game_name)
        blind_str = game_config.params.get("blind")
        if blind_str:
            # Big blind is the last (largest) value
            return max(int(b) for b in blind_str.split())
    except Exception:
        pass
    return None


def find_latest_run(algo=None, game=None):
    """Find the most recently modified metrics.json under logs/."""
    if algo and game:
        pattern = str(LOGS_ROOT / algo / game / "*/metrics.json")
    else:
        pattern = str(LOGS_ROOT / "**/metrics.json")
    files = glob.glob(pattern, recursive=True)
    if not files:
        print("No metrics.json found")
        sys.exit(1)
    return max(files, key=os.path.getmtime)


def load_run(metrics_path):
    run_dir = Path(metrics_path).parent
    with open(metrics_path) as f:
        metrics = json.load(f)

    run_json = run_dir / "run.json"
    config_json = run_dir / "config.json"
    run_info = json.load(open(run_json)) if run_json.exists() else {}
    config = json.load(open(config_json)) if config_json.exists() else {}

    return metrics, run_info, config, run_dir


def fmt_duration(td):
    total_secs = int(td.total_seconds())
    if total_secs < 0:
        return "0m"
    days, rem = divmod(total_secs, 86400)
    hours, rem = divmod(rem, 3600)
    mins, _ = divmod(rem, 60)
    if days > 0:
        return f"{days}d {hours}h {mins}m"
    if hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def fmt_number(n):
    return f"{n:,}"


def get_exp_key(metrics):
    for key in ["lbr_exp", "exp", "reward"]:
        if key in metrics:
            return key
    return None


def get_exp_label(key):
    labels = {"lbr_exp": "LBR", "exp": "exact", "reward": "reward"}
    return labels.get(key, key)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Show training status")
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--game", type=str, default=None)
    parser.add_argument("path", nargs="?", default=None, help="Direct path to run dir or metrics.json")
    args = parser.parse_args()

    if args.path:
        p = Path(args.path)
        if p.is_dir():
            metrics_path = str(p / "metrics.json")
        else:
            metrics_path = str(p)
    else:
        metrics_path = find_latest_run(args.algo, args.game)

    metrics, run_info, config, run_dir = load_run(metrics_path)

    # Extract game/algo from directory structure: logs/<algo>/<game>/<run_id>
    parts = run_dir.relative_to(LOGS_ROOT).parts
    if len(parts) >= 3:
        algo_name, game_name, run_id = parts[0], parts[1], parts[2]
    else:
        algo_name = config.get("algo_name", "?")
        game_name = config.get("game_name", "?")
        run_id = run_dir.name

    # Core metrics
    iterations = metrics.get("iteration", {}).get("values", [])
    episodes = metrics.get("episode", {}).get("values", [])

    cur_iter = iterations[-1] if iterations else 0
    cur_episode = episodes[-1] if episodes else 0

    num_episodes = config.get("num_episodes", 0)
    num_traversals = config.get("num_traversals", 1)
    total_iters = num_episodes // (num_traversals * 2) if num_traversals > 0 else 0

    progress = (cur_iter / total_iters * 100) if total_iters > 0 else 0

    # Exploitability
    exp_key = get_exp_key(metrics)
    exp_values = metrics.get(exp_key, {}).get("values", []) if exp_key else []
    exp_timestamps = metrics.get(exp_key, {}).get("timestamps", []) if exp_key else []
    cur_exp = exp_values[-1] if exp_values else None
    best_exp = min(exp_values) if exp_values else None
    best_exp_iter_idx = exp_values.index(best_exp) if best_exp is not None else 0
    best_exp_iter = iterations[best_exp_iter_idx] if best_exp_iter_idx < len(iterations) else 0

    # Recent trend (last 5 values)
    trend_n = min(5, len(exp_values))
    trend_values = exp_values[-trend_n:] if trend_n > 0 else []

    # Time calculations
    start_time_str = run_info.get("start_time")
    start_time = datetime.fromisoformat(start_time_str) if start_time_str else None

    now = datetime.now()
    elapsed = now - start_time if start_time else None

    # ETA from recent iteration pace
    eta_str = "?"
    finish_str = "?"
    if len(exp_timestamps) >= 2 and len(iterations) >= 2:
        recent_n = min(6, len(exp_timestamps))
        t0 = datetime.fromisoformat(exp_timestamps[-recent_n])
        t1 = datetime.fromisoformat(exp_timestamps[-1])
        iter_t0 = iterations[-recent_n]
        iter_t1 = iterations[-1]
        dt = (t1 - t0).total_seconds()
        di = iter_t1 - iter_t0
        if di > 0:
            secs_per_iter = dt / di
            remaining_iters = max(0, total_iters - cur_iter)
            remaining_secs = remaining_iters * secs_per_iter
            eta_td = timedelta(seconds=remaining_secs)
            finish_time = now + eta_td
            eta_str = fmt_duration(eta_td)
            finish_str = finish_time.strftime("%Y-%m-%d %H:%M")

    # Print
    W = 48
    title = f"{game_name} Training Status  ({run_id})"
    print()
    print(f"  {title}")
    print(f"  {'=' * W}")
    print(f"  {'Iteration:':<18}{fmt_number(cur_iter):>10} / {fmt_number(total_iters)}")
    print(f"  {'Episode:':<18}{fmt_number(cur_episode):>10} / {fmt_number(num_episodes)}")
    print(f"  {'Progress:':<18}{progress:>9.1f}%")
    print(f"  {chr(0x2500) * W}")

    big_blind = get_big_blind(game_name)

    if cur_exp is not None:
        if big_blind:
            cur_bb100 = cur_exp / big_blind * 100
            best_bb100 = best_exp / big_blind * 100
            print(f"  {'Exploitability:':<18}{cur_bb100:>9.2f} bb/100  ({get_exp_label(exp_key)})")
            print(f"  {'Best exploit:':<18}{best_bb100:>9.2f} bb/100  (iter {best_exp_iter})")
            if len(trend_values) >= 2:
                trend_str = " -> ".join(f"{v/big_blind*100:.2f}" for v in trend_values)
                print(f"  {'Recent trend:':<18}{trend_str}")
        else:
            print(f"  {'Exploitability:':<18}{cur_exp:>10.2f}  ({get_exp_label(exp_key)})")
            print(f"  {'Best exploit:':<18}{best_exp:>10.2f}  (iter {best_exp_iter})")
            if len(trend_values) >= 2:
                trend_str = " -> ".join(f"{v:.1f}" for v in trend_values)
                print(f"  {'Recent trend:':<18}{trend_str}")
        print(f"  {chr(0x2500) * W}")

    if elapsed:
        print(f"  {'Elapsed:':<18}{fmt_duration(elapsed):>10}")
    print(f"  {'ETA:':<18}{eta_str:>10}")
    print(f"  {'Est. finish:':<18}{finish_str:>10}")
    print()


if __name__ == "__main__":
    main()
