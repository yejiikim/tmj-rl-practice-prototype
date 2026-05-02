import argparse
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "tmj_rl_matplotlib"),
)

from stable_baselines3 import PPO
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
        use_simulation_time=not args.wall_clock_wait,
    )
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO baseline on the simple TMJ practice env."
    )
    parser.add_argument("--base-url", default="http://localhost:8081")
    parser.add_argument("--timesteps", type=int, default=512)
    parser.add_argument("--wait-action", type=float, default=0.05)
    parser.add_argument("--reset-wait", type=float, default=0.05)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    parser.add_argument("--goal-threshold", type=float, default=0.01)
    parser.add_argument("--goal-reward", type=float, default=5.0)
    parser.add_argument("--save-path", default="runs/ppo_tmj_practice")
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

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    env = make_env(args)

    if not args.skip_check_env:
        print("Checking Stable-Baselines3 environment compatibility...")
        check_env(env.unwrapped, warn=True)

    print("Training PPO baseline...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=64,
        batch_size=32,
        learning_rate=3e-4,
        gamma=0.95,
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
            "reward =", round(float(reward), 4),
            "terminated =", terminated,
            "truncated =", truncated,
            "marker x =", round(info["markerPosition"][0], 4),
            "target x =", round(info["targetPosition"][0], 4),
            "error =", round(info["trackingError"], 4),
            "force =", round(info["muscleForce"], 4),
        )
        if terminated or truncated:
            break

    env.close()


if __name__ == "__main__":
    main()
