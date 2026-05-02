import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # Allows REST smoke tests before Gymnasium is installed.
    gym = None
    spaces = None


class _BaseEnv:
    pass


class TmjPracticeEnv(gym.Env if gym is not None else _BaseEnv):
    """Gym-style environment for the simple ArtiSynth muscle prototype.

    Java/ArtiSynth owns the physics. Python sends excitation actions and turns
    the returned state into observation, reward, and episode flags.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        wait_action: float = 0.05,
        reset_wait: float = 0.05,
        goal_threshold: float = 0.01,
        max_episode_steps: int = 100,
        goal_reward: float = 5.0,
        progress_reward_scale: float = 20.0,
        distance_penalty_scale: float = 1.0,
        effort_penalty_scale: float = 0.01,
        use_simulation_time: bool = True,
        simulation_time_timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.wait_action = wait_action
        self.reset_wait = reset_wait
        self.goal_threshold = goal_threshold
        self.max_episode_steps = max_episode_steps
        self.goal_reward = goal_reward
        self.progress_reward_scale = progress_reward_scale
        self.distance_penalty_scale = distance_penalty_scale
        self.effort_penalty_scale = effort_penalty_scale
        self.use_simulation_time = use_simulation_time
        self.simulation_time_timeout = simulation_time_timeout

        self.elapsed_steps = 0
        self.prev_distance: Optional[float] = None

        self.action_size = int(self._get("actionSize"))
        self.obs_size = int(self._get("obsSize"))

        if spaces is not None:
            self.action_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.action_size,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_size,),
                dtype=np.float32,
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if gym is not None:
            super().reset(seed=seed)

        self._post("reset")
        self._wait(self.reset_wait)
        state = self._get("state")
        self.elapsed_steps = 0
        self.prev_distance = float(state["trackingError"])

        observation = self._state_to_observation(state)
        info = self._state_to_info(state)
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape[0] != self.action_size:
            raise ValueError(
                f"Expected action with size {self.action_size}, "
                f"got {action_array.shape[0]}"
            )

        action_array = np.clip(action_array, 0.0, 1.0)
        self._post("excitations", {"excitations": action_array.tolist()})

        # This matches Amir's high-level pattern: action is sent, ArtiSynth
        # simulation time advances, then Python reads the next state.
        self._wait(self.wait_action)

        state = self._get("state")
        self.elapsed_steps += 1

        distance = float(state["trackingError"])
        reward, terminated = self._calculate_reward(distance, action_array)
        truncated = self.elapsed_steps >= self.max_episode_steps

        observation = self._state_to_observation(state)
        info = self._state_to_info(state)
        info["action"] = action_array.copy()
        info["previousDistance"] = self.prev_distance

        self.prev_distance = distance
        return observation, reward, terminated, truncated, info

    def server_is_alive(self) -> bool:
        try:
            response = requests.get(self.base_url + "/", timeout=2.0)
            return response.ok
        except requests.RequestException:
            return False

    def _calculate_reward(
        self,
        distance: float,
        action: np.ndarray,
    ) -> Tuple[float, bool]:
        if distance <= self.goal_threshold:
            return self.goal_reward, True

        if self.prev_distance is None:
            return 0.0, False

        progress = self.prev_distance - distance
        effort = float(np.sum(np.square(action)))
        reward = (
            self.progress_reward_scale * progress
            - self.distance_penalty_scale * distance
            - self.effort_penalty_scale * effort
        )
        return float(reward), False

    def _wait(self, duration: float) -> None:
        if duration <= 0.0:
            return

        if not self.use_simulation_time:
            time.sleep(duration)
            return

        start_time = float(self._get("time"))
        wall_deadline = time.time() + self.simulation_time_timeout

        while float(self._get("time")) - start_time < duration:
            if time.time() > wall_deadline:
                raise TimeoutError(
                    "ArtiSynth simulation time did not advance. "
                    "Make sure the model is playing."
                )
            time.sleep(0.002)

    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        action_excitations = self._array_value(
            state, "actionExcitations", "actionExcitation"
        )
        exciter_excitations = self._array_value(
            state, "exciterExcitations", "exciterExcitation"
        )
        muscle_excitations = self._array_value(
            state, "muscleExcitations", "muscleExcitation"
        )
        muscle_forces = self._array_value(state, "muscleForces", "muscleForce")

        values = []
        values.extend(state["markerPosition"])
        values.extend(state["markerVelocity"])
        values.extend(state["targetPosition"])
        values.append(state["trackingError"])
        values.extend(action_excitations)
        values.extend(exciter_excitations)
        values.extend(muscle_excitations)
        values.extend(muscle_forces)
        return np.asarray(values, dtype=np.float32)

    def _state_to_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        action_excitations = self._array_value(
            state, "actionExcitations", "actionExcitation"
        )
        exciter_excitations = self._array_value(
            state, "exciterExcitations", "exciterExcitation"
        )
        muscle_excitations = self._array_value(
            state, "muscleExcitations", "muscleExcitation"
        )
        muscle_forces = self._array_value(state, "muscleForces", "muscleForce")

        return {
            "markerPosition": state["markerPosition"],
            "markerVelocity": state["markerVelocity"],
            "targetPosition": state["targetPosition"],
            "trackingError": float(state["trackingError"]),
            "actionExcitations": action_excitations,
            "exciterExcitations": exciter_excitations,
            "muscleExcitations": muscle_excitations,
            "muscleForces": muscle_forces,
            "muscleForce": float(np.sum(muscle_forces)),
            "rewardLike": float(state["rewardLike"]),
            "elapsedSteps": self.elapsed_steps,
        }

    def _array_value(
        self,
        state: Dict[str, Any],
        array_key: str,
        scalar_key: str,
    ) -> np.ndarray:
        if array_key in state:
            return np.asarray(state[array_key], dtype=np.float32)
        return np.asarray([state[scalar_key]], dtype=np.float32)

    def _get(self, endpoint: str) -> Any:
        response = requests.get(f"{self.base_url}/{endpoint}", timeout=5.0)
        response.raise_for_status()
        return response.json()

    def _post(self, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        response = requests.post(
            f"{self.base_url}/{endpoint}",
            json=payload if payload is not None else {},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()
