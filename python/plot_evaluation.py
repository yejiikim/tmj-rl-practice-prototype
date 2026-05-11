import argparse
import csv
import os
import tempfile
from collections import defaultdict

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tmj_rl_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_episode_errors(csv_path):
    """Read step-level tracking errors written by evaluate_policy.py."""
    episodes = defaultdict(list)
    with open(csv_path, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            episode = int(row["episode"])
            step = int(row["step"])
            error = float(row["tracking_error"])
            episodes[episode].append((step, error))
    return episodes


def plot_tracking_errors(episodes, output_path, success_threshold):
    """Plot tracking error over steps for each evaluation episode."""
    plt.figure(figsize=(8, 5))

    for episode in sorted(episodes):
        episode_rows = sorted(episodes[episode])
        steps = [step for step, _ in episode_rows]
        errors = [error for _, error in episode_rows]
        plt.plot(
            steps,
            errors,
            marker="o",
            linewidth=1.4,
            alpha=0.55,
        )

    plt.axhline(
        success_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"success threshold = {success_threshold:g}",
    )
    plt.xlabel("Step")
    plt.ylabel("Tracking error")
    plt.title("SAC Evaluation: Tracking Error Over Steps")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)


def main():
    parser = argparse.ArgumentParser(
        description="Plot tracking error curves from an evaluation CSV."
    )
    parser.add_argument("--input", default="runs/evaluation_sac.csv")
    parser.add_argument("--output", default="runs/tracking_error_sac.png")
    parser.add_argument("--success-threshold", type=float, default=0.01)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    episodes = read_episode_errors(args.input)
    plot_tracking_errors(episodes, args.output, args.success_threshold)
    print(f"Wrote tracking error plot to {args.output}")


if __name__ == "__main__":
    main()
