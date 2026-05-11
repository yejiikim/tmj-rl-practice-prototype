import time

import requests

BASE_URL = "http://localhost:8081"


def get_state():
    """Read the current Java/ArtiSynth simulation state."""

    response = requests.get(BASE_URL + "/state", timeout=5.0)
    response.raise_for_status()
    return response.json()


def reset():
    """Start a clean Java-side episode."""

    response = requests.post(BASE_URL + "/reset", timeout=5.0)
    response.raise_for_status()
    return response.json()


def send_action(action):
    """Send a raw excitation vector to the Java REST API."""

    response = requests.post(
        BASE_URL + "/excitations",
        json={"excitations": action},
        timeout=5.0,
    )
    response.raise_for_status()
    return response.json()


def main():
    print("Checking REST server...")
    response = requests.get(BASE_URL + "/", timeout=5.0)
    response.raise_for_status()
    print(response.json())

    print("Reset state:")
    state = reset()
    print(state)

    # Manual REST-level test: choose left/right excitation from the current
    # marker and target positions, send it to Java, then print the returned
    # state. This bypasses the Gym-style environment wrapper.
    for k in range(100):
        marker_x = state["markerPosition"][0]
        target_x = state["targetPosition"][0]
        direction_error = target_x - marker_x
        excitation = min(abs(direction_error) * 4.0, 1.0)

        if direction_error < 0:
            action = [excitation, 0.0]
        else:
            action = [0.0, excitation]

        state = send_action(action)

        print(
            k,
            "action =", [round(x, 3) for x in action],
            "marker x =", round(state["markerPosition"][0], 4),
            "target x =", round(state["targetPosition"][0], 4),
            "error =", round(state["trackingError"], 4),
            "reward =", round(state["rewardLike"], 4),
            "forces =", [round(x, 4) for x in state["muscleForces"]],
        )

        time.sleep(0.05)


if __name__ == "__main__":
    main()
