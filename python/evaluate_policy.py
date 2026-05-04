import argparse
import csv
import os
import statistics
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tmj_rl_matplotlib"),
)

from stable_baselines3 import PPO, SAC

from tmj_practice_env import TmjPracticeEnv


ALGORITHMS = {
    "ppo": PPO,
    "sac": SAC,
}


def make_env(args):
    return TmjPracticeEnv(
        base_url=args.base_url,
        wait_action=args.wait_action,
        reset_wait=args.reset_wait,
        goal_threshold=args.goal_threshold,
        max_episode_steps=args.max_episode_steps,
        goal_reward=args.goal_reward,
        progress_reward_scale=args.progress_reward_scale,
        distance_penalty_scale=args.distance_penalty_scale,
        effort_penalty_scale=args.effort_penalty_scale,
        velocity_penalty_scale=args.velocity_penalty_scale,
        action_change_penalty_scale=args.action_change_penalty_scale,
        normalize_actions=not args.raw_actions,
        use_simulation_time=not args.wall_clock_wait,
        launch_command=args.launch_command,
        launch_timeout=args.launch_timeout,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a saved policy and write step-level CSV logs."
    )
    parser.add_argument("--algo", choices=sorted(ALGORITHMS), default="sac")
    parser.add_argument("--base-url", default="http://localhost:8081")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--wait-action", type=float, default=0.05)
    parser.add_argument("--reset-wait", type=float, default=0.05)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    parser.add_argument("--goal-threshold", type=float, default=0.01)
    parser.add_argument("--goal-reward", type=float, default=5.0)
    parser.add_argument("--progress-reward-scale", type=float, default=20.0)
    parser.add_argument("--distance-penalty-scale", type=float, default=1.0)
    parser.add_argument("--effort-penalty-scale", type=float, default=0.01)
    parser.add_argument("--velocity-penalty-scale", type=float, default=0.02)
    parser.add_argument("--action-change-penalty-scale", type=float, default=0.005)
    parser.add_argument("--output", default="runs/evaluation.csv")
    parser.add_argument(
        "--raw-actions",
        action="store_true",
        help=(
            "Use [0, 1] policy actions. Default assumes the saved policy uses "
            "[-1, 1] actions mapped internally to [0, 1] excitations."
        ),
    )
    parser.add_argument(
        "--wall-clock-wait",
        action="store_true",
        help="Use wall-clock sleep instead of ArtiSynth simulation time.",
    )
    parser.add_argument(
        "--launch-command",
        default=None,
        help="Optional command used to launch ArtiSynth if REST is not alive.",
    )
    parser.add_argument("--launch-timeout", type=float, default=60.0)
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"runs/{args.algo}_tmj_practice.zip"

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    env = make_env(args)
    model = ALGORITHMS[args.algo].load(args.model_path)

    rows = []
    episode_returns = []
    episode_summaries = []
    successes = 0

    for episode in range(args.episodes):
        observation, info = env.reset()
        episode_return = 0.0
        initial_error = float(info["trackingError"])
        target_x = float(info["targetPosition"][0])
        min_error = initial_error

        for step in range(args.max_episode_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            min_error = min(min_error, float(info["trackingError"]))

            marker = info["markerPosition"]
            target = info["targetPosition"]
            forces = info["muscleForces"]

            row = {
                "episode": episode,
                "step": step,
                "time": info.get("time", 0.0),
                "marker_x": float(marker[0]),
                "marker_y": float(marker[1]),
                "marker_z": float(marker[2]),
                "target_x": float(target[0]),
                "target_y": float(target[1]),
                "target_z": float(target[2]),
                "tracking_error": float(info["trackingError"]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }

            for idx, value in enumerate(action):
                row[f"policy_action_{idx}"] = float(value)
            for idx, value in enumerate(info["commandedExcitations"]):
                row[f"excitation_{idx}"] = float(value)
            for idx, value in enumerate(forces):
                row[f"force_{idx}"] = float(value)

            rows.append(row)

            if terminated or truncated:
                successes += int(terminated)
                break

        episode_returns.append(episode_return)
        episode_summaries.append(
            {
                "episode": episode,
                "success": bool(terminated),
                "truncated": bool(truncated),
                "steps": step + 1,
                "target_x": target_x,
                "initial_error": initial_error,
                "final_error": float(info["trackingError"]),
                "min_error": min_error,
                "return": episode_return,
            }
        )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(args.output, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mean_return = sum(episode_returns) / max(len(episode_returns), 1)
    final_errors = [item["final_error"] for item in episode_summaries]
    min_errors = [item["min_error"] for item in episode_summaries]
    truncated_count = sum(item["truncated"] for item in episode_summaries)
    positive_targets = [item for item in episode_summaries if item["target_x"] >= 0.0]
    negative_targets = [item for item in episode_summaries if item["target_x"] < 0.0]

    def success_rate(items):
        if not items:
            return 0.0
        return 100.0 * sum(item["success"] for item in items) / len(items)

    print(f"Wrote evaluation log to {args.output}")
    print(f"Episodes: {args.episodes}")
    print(f"Successes: {successes}/{args.episodes}")
    print(f"Success rate: {success_rate(episode_summaries):.1f}%")
    print(f"Truncated episodes: {truncated_count}/{args.episodes}")
    print(f"Mean return: {mean_return:.4f}")
    print(f"Mean final error: {statistics.mean(final_errors):.4f}")
    print(f"Median final error: {statistics.median(final_errors):.4f}")
    print(f"Mean min error: {statistics.mean(min_errors):.4f}")
    print(
        "Positive target success rate:",
        f"{success_rate(positive_targets):.1f}% ({len(positive_targets)} episodes)",
    )
    print(
        "Negative target success rate:",
        f"{success_rate(negative_targets):.1f}% ({len(negative_targets)} episodes)",
    )

    env.close()


if __name__ == "__main__":
    main()
