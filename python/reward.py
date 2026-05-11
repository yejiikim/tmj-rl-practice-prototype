from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TrackingRewardConfig:
    """Weights for the simple marker-to-target reaching reward.

    These values are hand-tuned for the prototype.
    """

    goal_threshold: float = 0.01
    goal_reward: float = 5.0
    progress_reward_scale: float = 20.0
    distance_penalty_scale: float = 1.0
    effort_penalty_scale: float = 0.01
    velocity_penalty_scale: float = 0.0
    action_change_penalty_scale: float = 0.0


def compute_tracking_reward(
    distance: float,
    previous_distance: Optional[float],
    action: np.ndarray,
    config: TrackingRewardConfig,
    velocity_norm: float = 0.0,
    previous_action: Optional[np.ndarray] = None,
) -> Tuple[float, bool]:
    """Compute the tracking reward and success flag.

    The reward favors progress toward the target and penalizes distance,
    excitation effort, velocity, and abrupt action changes.
    """

    if distance <= config.goal_threshold:
        return config.goal_reward, True

    if previous_distance is None:
        return 0.0, False

    progress = previous_distance - distance

    effort = float(np.sum(np.square(action)))

    action_change = 0.0
    if previous_action is not None:
        action_change = float(np.sum(np.square(action - previous_action)))

    reward = (
        config.progress_reward_scale * progress
        - config.distance_penalty_scale * distance
        - config.effort_penalty_scale * effort
        - config.velocity_penalty_scale * velocity_norm
        - config.action_change_penalty_scale * action_change
    )
    return float(reward), False
