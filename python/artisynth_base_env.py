import shlex
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


class _BaseEnv:
    pass


class ArtiSynthBaseEnv(gym.Env if gym is not None else _BaseEnv):
    """Generic Gym-style wrapper around an ArtiSynth REST controller.

    Subclasses provide task-specific observation parsing and reward logic.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        wait_action: float = 0.05,
        reset_wait: float = 0.05,
        max_episode_steps: int = 100,
        normalize_actions: bool = False,
        use_simulation_time: bool = True,
        simulation_time_timeout: float = 5.0,
        launch_command: Optional[str] = None,
        launch_timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")

        self.wait_action = wait_action
        self.reset_wait = reset_wait
        self.max_episode_steps = max_episode_steps

        self.normalize_actions = normalize_actions

        self.use_simulation_time = use_simulation_time
        self.simulation_time_timeout = simulation_time_timeout

        self.launch_command = launch_command
        self.launch_timeout = launch_timeout
        self.launch_process: Optional[subprocess.Popen] = None

        self.elapsed_steps = 0

        if self.launch_command is not None and not self.server_is_alive():
            self._launch_artisynth()

        self.action_size = int(self._get("actionSize"))
        self.obs_size = int(self._get("obsSize"))

        if spaces is not None:
            action_low = -1.0 if self.normalize_actions else 0.0
            action_high = 1.0
            self.action_space = spaces.Box(
                low=action_low,
                high=action_high,
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
        self._on_reset_state(state)

        observation = self._state_to_observation(state)
        info = self._state_to_info(state)
        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        policy_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if policy_action.shape[0] != self.action_size:
            raise ValueError(
                f"Expected action with size {self.action_size}, "
                f"got {policy_action.shape[0]}"
            )

        if self.normalize_actions:
            policy_action = np.clip(policy_action, -1.0, 1.0)
        else:
            policy_action = np.clip(policy_action, 0.0, 1.0)

        excitation_action = self.policy_action_to_excitations(policy_action)

        self._post("excitations", {"excitations": excitation_action.tolist()})

        self._wait(self.wait_action)

        state = self._get("state")
        self.elapsed_steps += 1

        reward, terminated = self._compute_reward(
            state, excitation_action, policy_action
        )

        truncated = self.elapsed_steps >= self.max_episode_steps

        observation = self._state_to_observation(state)
        info = self._state_to_info(state)
        info["policyAction"] = policy_action.copy()
        info["commandedExcitations"] = excitation_action.copy()
        info["action"] = excitation_action.copy()

        previous_distance = self._previous_distance_for_info()
        if previous_distance is not None:
            info["previousDistance"] = previous_distance

        self._on_step_end(state, excitation_action, policy_action)
        return observation, reward, terminated, truncated, info

    def _on_reset_state(self, state: Dict[str, Any]) -> None:
        """Hook for task-specific episode bookkeeping."""

    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert Java /state JSON into the observation vector for a task."""
        raise NotImplementedError(
            "Subclasses must implement _state_to_observation()."
        )

    def _state_to_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Java /state JSON into readable diagnostics for a task."""
        raise NotImplementedError("Subclasses must implement _state_to_info().")

    def _compute_reward(
        self,
        state: Dict[str, Any],
        excitation_action: np.ndarray,
        policy_action: np.ndarray,
    ) -> Tuple[float, bool]:
        """Return reward and terminated flag for one task-specific step."""
        raise NotImplementedError("Subclasses must implement _compute_reward().")

    def _previous_distance_for_info(self) -> Optional[float]:
        """Optional hook used by tracking tasks for readable diagnostics."""
        return None

    def _on_step_end(
        self,
        state: Dict[str, Any],
        excitation_action: np.ndarray,
        policy_action: np.ndarray,
    ) -> None:
        """Hook for updating task-specific state after info is created."""

    def policy_action_to_excitations(self, action: np.ndarray) -> np.ndarray:
        """Map policy action space to ArtiSynth excitation values."""
        action = np.asarray(action, dtype=np.float32)
        if self.normalize_actions:
            return np.asarray(0.5 * (action + 1.0), dtype=np.float32)
        return np.asarray(np.clip(action, 0.0, 1.0), dtype=np.float32)

    def excitations_to_policy_action(self, excitations: np.ndarray) -> np.ndarray:
        """Map ArtiSynth excitation values to the policy action space."""
        excitations = np.asarray(excitations, dtype=np.float32)
        excitations = np.clip(excitations, 0.0, 1.0)
        if self.normalize_actions:
            return np.asarray(2.0 * excitations - 1.0, dtype=np.float32)
        return excitations

    def server_is_alive(self) -> bool:
        try:
            response = requests.get(self.base_url + "/", timeout=2.0)
            return response.ok
        except requests.RequestException:
            return False

    def _wait(self, duration: float) -> None:
        """Wait until the ArtiSynth simulation has advanced by duration."""
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

    def close(self) -> None:
        if self.launch_process is not None and self.launch_process.poll() is None:
            self.launch_process.terminate()

    def _launch_artisynth(self) -> None:
        """Optionally launch ArtiSynth if the REST server is not already up."""
        self.launch_process = subprocess.Popen(
            shlex.split(self.launch_command),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + self.launch_timeout
        while time.time() < deadline:
            if self.server_is_alive():
                return
            time.sleep(0.5)
        raise TimeoutError(
            "ArtiSynth REST server did not start before launch_timeout."
        )

    def _array_value(
        self,
        state: Dict[str, Any],
        array_keys: Tuple[str, ...],
        scalar_key: str,
    ) -> np.ndarray:
        for key in array_keys:
            if key in state:
                return np.asarray(state[key], dtype=np.float32)
        return np.asarray([state[scalar_key]], dtype=np.float32)

    def _component_vector(
        self,
        state: Dict[str, Any],
        component_names: Tuple[str, ...],
        prop: str,
        flat_key: str,
    ) -> np.ndarray:
        observation = state.get("observation", {})
        if isinstance(observation, dict):
            for name in component_names:
                component = observation.get(name)
                if component is not None and prop in component:
                    return np.asarray(component[prop], dtype=np.float32)

        return np.asarray(state[flat_key], dtype=np.float32)

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
