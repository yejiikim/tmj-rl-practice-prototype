import numpy as np

from antagonist_env import AntagonistMuscleEnv


def heuristic_action(info, gain=4.0):
    """Simple hand-coded controller used before RL training.

    If the target is left of the marker, activate the left muscle. If the target
    is right of the marker, activate the right muscle. This is only a sanity
    check that REST actions actually move the ArtiSynth model.
    """

    marker_x = info["markerPosition"][0]
    target_x = info["targetPosition"][0]
    direction_error = target_x - marker_x
    excitation = min(abs(direction_error) * gain, 1.0)

    if direction_error < 0:
        return np.array([excitation, 0.0], dtype=np.float32)
    return np.array([0.0, excitation], dtype=np.float32)


def main():
    env = AntagonistMuscleEnv(wait_action=0.05, max_episode_steps=40)

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
        action = heuristic_action(info)
        observation, reward, terminated, truncated, info = env.step(action)

        print(
            k,
            "action =", [round(float(x), 3) for x in action],
            "reward =", round(reward, 4),
            "terminated =", terminated,
            "truncated =", truncated,
            "marker x =", round(info["markerPosition"][0], 4),
            "target x =", round(info["targetPosition"][0], 4),
            "error =", round(info["trackingError"], 4),
            "forces =", [round(float(x), 4) for x in info["muscleForces"]],
        )

        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
