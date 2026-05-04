import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tmj_rl_matplotlib"),
)

from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from tmj_practice_env import TmjPracticeEnv


def make_env(args):
    env = TmjPracticeEnv(
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
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(
        description="Train a SAC baseline on the simple TMJ practice env."
    )
    parser.add_argument("--algo", choices=["sac"], default="sac", help=argparse.SUPPRESS)
    parser.add_argument("--base-url", default="http://localhost:8081")
    parser.add_argument("--timesteps", type=int, default=5000)
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
    parser.add_argument("--save-path", default=None)
    parser.add_argument(
        "--raw-actions",
        action="store_true",
        help=(
            "Expose [0, 1] actions directly. By default the policy sees "
            "[-1, 1] and actions are mapped to [0, 1] excitations."
        ),
    )
    parser.add_argument(
        "--launch-command",
        default=None,
        help="Optional command used to launch ArtiSynth if REST is not alive.",
    )
    parser.add_argument("--launch-timeout", type=float, default=60.0)
    parser.add_argument(
        "--wall-clock-wait",
        action="store_true",
        help="Use wall-clock sleep instead of ArtiSynth simulation time.",
    )
    parser.add_argument(
        "--skip-check-env",
        action="store_true",
        help="Skip Stable-Baselines3 environment validation.",
    )
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = "runs/sac_tmj_practice"

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    env = make_env(args)

    if not args.skip_check_env:
        print("Checking Stable-Baselines3 environment compatibility...")
        check_env(env.unwrapped, warn=True)

    print("Training SAC baseline...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        buffer_size=20000,
        learning_starts=200,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.98,
        tau=0.02,
        train_freq=1,
        gradient_steps=1,
        device="cpu",
    )
    model.learn(total_timesteps=args.timesteps)

    model.save(args.save_path)
    print(f"Saved model to {args.save_path}.zip")

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=3,
        deterministic=True,
    )
    print(f"Evaluation mean reward: {mean_reward:.3f} +/- {std_reward:.3f}")

    observation, info = env.reset()
    print(
        "Post-training reset:",
        "marker x =", round(info["markerPosition"][0], 4),
        "target x =", round(info["targetPosition"][0], 4),
        "error =", round(info["trackingError"], 4),
    )

    for step in range(args.max_episode_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            step,
            "action =", [round(float(x), 3) for x in action],
            "excitations =",
            [round(float(x), 3) for x in info["commandedExcitations"]],
            "reward =", round(float(reward), 4),
            "terminated =", terminated,
            "truncated =", truncated,
            "marker x =", round(info["markerPosition"][0], 4),
            "target x =", round(info["targetPosition"][0], 4),
            "error =", round(info["trackingError"], 4),
            "forces =", [round(float(x), 4) for x in info["muscleForces"]],
        )
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
