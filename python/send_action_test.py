import time
import math
import requests

BASE_URL = "http://localhost:8081"


def get_state():
    response = requests.get(BASE_URL + "/state")
    response.raise_for_status()
    return response.json()


def reset():
    response = requests.post(BASE_URL + "/reset")
    response.raise_for_status()
    return response.json()


def send_action(action):
    response = requests.post(
        BASE_URL + "/excitations",
        json={"excitations": [action]},
    )
    response.raise_for_status()
    return response.json()


print("Checking REST server...")
print(requests.get(BASE_URL + "/").json())

print("Reset state:")
print(reset())

for k in range(100):
    action = 0.5 + 0.5 * math.sin(0.15 * k)

    state = send_action(action)

    print(
        k,
        "action =", round(action, 3),
        "marker x =", round(state["markerPosition"][0], 4),
        "target x =", round(state["targetPosition"][0], 4),
        "error =", round(state["trackingError"], 4),
        "reward =", round(state["rewardLike"], 4),
        "force =", round(state["muscleForce"], 4),
    )

    time.sleep(0.05)
