from typing import Any, Dict, Optional, Tuple

import numpy as np

from artisynth_base_env import ArtiSynthBaseEnv
from reward import TrackingRewardConfig, compute_tracking_reward


class AntagonistMuscleEnv(ArtiSynthBaseEnv):
    """Task-specific environment for the simple two-muscle tracking prototype.

    The base class handles REST communication; this class defines the
    observation, reward, and episode bookkeeping for marker-to-target tracking.
    """

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
        velocity_penalty_scale: float = 0.0,
        action_change_penalty_scale: float = 0.0,
        normalize_actions: bool = False,
        use_simulation_time: bool = True,
        simulation_time_timeout: float = 5.0,
        launch_command: Optional[str] = None,
        launch_timeout: float = 60.0,
    ):
        self.reward_config = TrackingRewardConfig(
            goal_threshold=goal_threshold,
            goal_reward=goal_reward,
            progress_reward_scale=progress_reward_scale,
            distance_penalty_scale=distance_penalty_scale,
            effort_penalty_scale=effort_penalty_scale,
            velocity_penalty_scale=velocity_penalty_scale,
            action_change_penalty_scale=action_change_penalty_scale,
        )

        self.prev_distance: Optional[float] = None
        self.previous_action_excitation: Optional[np.ndarray] = None

        super().__init__(
            base_url=base_url,
            wait_action=wait_action,
            reset_wait=reset_wait,
            max_episode_steps=max_episode_steps,
            normalize_actions=normalize_actions,
            use_simulation_time=use_simulation_time,
            simulation_time_timeout=simulation_time_timeout,
            launch_command=launch_command,
            launch_timeout=launch_timeout,
        )

    def _state_to_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Convert the /state JSON into the policy observation."""
        marker_position = self._component_vector(
            state, ("marker",), "position", "markerPosition"
        )
        marker_velocity = self._component_vector(
            state, ("marker",), "velocity", "markerVelocity"
        )
        target_position = self._component_vector(
            state, ("target",), "position", "targetPosition"
        )
        action_excitations = self._array_value(
            state, ("actionExcitations", "excitations"), "actionExcitation"
        )
        exciter_excitations = self._array_value(
            state, ("exciterExcitations", "excitations"), "exciterExcitation"
        )
        muscle_excitations = self._array_value(
            state, ("muscleExcitations", "excitations"), "muscleExcitation"
        )
        muscle_forces = self._array_value(
            state, ("muscleForces",), "muscleForce"
        )

        values = []
        values.extend(marker_position)
        values.extend(marker_velocity)
        values.extend(target_position)
        values.append(self._tracking_error(state))
        values.extend(action_excitations)
        values.extend(exciter_excitations)
        values.extend(muscle_excitations)
        values.extend(muscle_forces)
        return np.asarray(values, dtype=np.float32)

    def _state_to_info(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return readable diagnostics that are not part of the policy input."""
        marker_position = self._component_vector(
            state, ("marker",), "position", "markerPosition"
        )
        marker_velocity = self._component_vector(
            state, ("marker",), "velocity", "markerVelocity"
        )
        target_position = self._component_vector(
            state, ("target",), "position", "targetPosition"
        )
        action_excitations = self._array_value(
            state, ("actionExcitations", "excitations"), "actionExcitation"
        )
        exciter_excitations = self._array_value(
            state, ("exciterExcitations", "excitations"), "exciterExcitation"
        )
        muscle_excitations = self._array_value(
            state, ("muscleExcitations", "excitations"), "muscleExcitation"
        )
        muscle_forces = self._array_value(
            state, ("muscleForces",), "muscleForce"
        )
        tracking_error = self._tracking_error(state)

        return {
            "time": float(state.get("time", 0.0)),
            "markerPosition": marker_position,
            "markerVelocity": marker_velocity,
            "targetPosition": target_position,
            "trackingError": tracking_error,
            "distance": tracking_error,
            "actionExcitations": action_excitations,
            "exciterExcitations": exciter_excitations,
            "muscleExcitations": muscle_excitations,
            "muscleForces": muscle_forces,
            "muscleForce": float(np.sum(muscle_forces)),
            "rewardLike": float(state["rewardLike"]),
            "elapsedSteps": self.elapsed_steps,
        }

    def _tracking_error(self, state: Dict[str, Any]) -> float:
        if "trackingError" in state:
            return float(state["trackingError"])

        marker_position = self._component_vector(
            state, ("marker",), "position", "markerPosition"
        )
        target_position = self._component_vector(
            state, ("target",), "position", "targetPosition"
        )
        return float(np.linalg.norm(marker_position - target_position))

    def _on_reset_state(self, state: Dict[str, Any]) -> None:
        self.prev_distance = self._tracking_error(state)
        self.previous_action_excitation = np.zeros(self.action_size, dtype=np.float32)

    def _compute_reward(
        self,
        state: Dict[str, Any],
        excitation_action: np.ndarray,
        policy_action: np.ndarray,
    ) -> Tuple[float, bool]:
        distance = self._tracking_error(state)
        marker_velocity = self._component_vector(
            state, ("marker",), "velocity", "markerVelocity"
        )
        return compute_tracking_reward(
            distance,
            self.prev_distance,
            excitation_action,
            self.reward_config,
            velocity_norm=float(np.linalg.norm(marker_velocity)),
            previous_action=self.previous_action_excitation,
        )

    def _previous_distance_for_info(self) -> Optional[float]:
        return self.prev_distance

    def _on_step_end(
        self,
        state: Dict[str, Any],
        excitation_action: np.ndarray,
        policy_action: np.ndarray,
    ) -> None:
        self.prev_distance = self._tracking_error(state)
        self.previous_action_excitation = excitation_action.copy()
