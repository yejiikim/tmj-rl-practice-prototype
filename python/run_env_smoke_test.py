import math

import numpy as np

from tmj_practice_env import TmjPracticeEnv


def main():
    env = TmjPracticeEnv(wait_action=0.05, max_episode_steps=40)

    observation, info = env.reset()
    print("Reset")
    print("observation size =", observation.shape[0])
    print("action size =", env.action_size)
    print(
        "marker x =", round(info["markerPosition"][0], 4),
        "target x =", round(info["targetPosition"][0], 4),
        "error =", round(info["trackingError"], 4),
    )

    for k in range(40):
        action_value = 0.5 + 0.5 * math.sin(0.15 * k)
        action = np.array([action_value], dtype=np.float32)

        observation, reward, terminated, truncated, info = env.step(action)

        print(
            k,
            "action =", round(action_value, 3),
            "reward =", round(reward, 4),
            "terminated =", terminated,
            "truncated =", truncated,
            "marker x =", round(info["markerPosition"][0], 4),
            "target x =", round(info["targetPosition"][0], 4),
            "error =", round(info["trackingError"], 4),
            "force =", round(info["muscleForce"], 4),
        )

        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
