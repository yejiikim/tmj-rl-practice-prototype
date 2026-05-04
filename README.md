# TMJ RL Practice Prototype

This repository contains a simple ArtiSynth + Python reinforcement-learning
prototype inspired by the action/state loop used in
[`amir-abdi/artisynth-rl`](https://github.com/amir-abdi/artisynth-rl).

The goal is not to copy Amir's code directly. The goal is to rebuild the same
core idea in a small, controllable practice model:

```text
Python policy
-> sends muscle excitation action through REST
-> Java ArtiSynth controller applies it to MuscleExciters
-> ArtiSynth advances the physics
-> Python reads state, computes reward, and trains/evaluates a policy
```

## Current Prototype

The current model is a two-muscle antagonist point-to-point task.

- A rigid box carries a marker.
- A fixed left anchor and fixed right anchor define two opposing muscles.
- The left muscle pulls the marker left.
- The right muscle pulls the marker right.
- A target point is randomized on reset.
- Python sends a two-dimensional action:

```text
[left_muscle_excitation, right_muscle_excitation]
```

Each action value is clipped to `[0, 1]`.

## What This Demonstrates

This prototype demonstrates the full RL communication pipeline:

- Java/ArtiSynth owns the physical model.
- Python owns the RL environment wrapper and reward function.
- Python sends action values through REST.
- `SimpleRlController` applies those values to ArtiSynth `MuscleExciter`s.
- Python reads the updated marker/target state after simulation advances.
- Stable-Baselines3 PPO or SAC can train against the environment.

This is still a practice model, not the final TMJ biomechanics model. It is the
minimal working version of the ArtiSynth-Python RL interface.

## Repository Layout

```text
src/artisynth/models/tmj/practice/
  MuscleExcitationPrototype.java
  SimpleRlController.java

python/
  reward.py
  tmj_practice_env.py
  evaluate_policy.py
  run_env_smoke_test.py
  train_baseline.py
  send_action_test.py
  requirements.txt
```

## Java / ArtiSynth Side

`MuscleExcitationPrototype.java` builds the ArtiSynth model:

- `leftAnchor`: fixed left muscle attachment point
- `rightAnchor`: fixed right muscle attachment point
- `box`: rigid body being moved
- `marker`: point attached to the box and used for tracking
- `target`: randomized goal point
- `leftMuscle`, `rightMuscle`: opposing axial muscles
- `leftExciter`, `rightExciter`: muscle excitation inputs
- REST server on `http://localhost:8081`

`SimpleRlController.java` stores the latest Python action vector and applies it
during the ArtiSynth simulation step. Internally, it is now written as a
generic action-to-exciter loop, so the same pattern can be extended from two
toy muscles to more muscle exciters later:

```java
for (int i = 0; i < exciters.length; i++) {
   exciters[i].setExcitation(actions[i]);
}
```

The muscle material is configured to avoid passive spring forces dominating the
task:

```java
new SimpleAxialMuscle(/*stiffness=*/0, /*damping=*/2, /*maxf=*/20)
```

This makes the task depend mainly on externally supplied muscle excitation.

## REST API

The ArtiSynth model exposes these endpoints:

```text
GET  /                 server status
GET  /actionSize        number of action values
GET  /obsSize           observation size
GET  /stateSize         state size including reward-like value
GET  /time              ArtiSynth simulation time
GET  /state             current model state
GET  /excitations       current action vector
POST /excitations       set action vector
POST /reset             reset box/action and randomize target
POST /setSeed           set Java target randomization seed
POST /setTest           store test/evaluation mode flag
```

Expected sizes for the current model:

```text
actionSize = 2
obsSize    = 18
stateSize  = 19
```

Example state fields:

```json
{
  "time": 12.34,
  "observation": {
    "marker": {
      "position": [0.0, 0.0, 0.0],
      "velocity": [0.0, 0.0, 0.0]
    },
    "target": {
      "position": [-0.112647, 0.0, 0.0],
      "velocity": [0.0, 0.0, 0.0]
    }
  },
  "excitations": [0.0, 0.0],
  "muscleForces": [0.0, 0.0],
  "properties": [],
  "markerPosition": [0.0, 0.0, 0.0],
  "markerVelocity": [0.0, 0.0, 0.0],
  "targetPosition": [-0.112647, 0.0, 0.0],
  "trackingError": 0.112647,
  "actionExcitations": [0.0, 0.0],
  "exciterExcitations": [0.0, 0.0],
  "muscleExcitations": [0.0, 0.0],
  "muscleForces": [0.0, 0.0],
  "rewardLike": -0.012689
}
```

The nested `observation` / `excitations` / `muscleForces` fields are included
to make the response closer to Amir's REST state format. The flat fields are
kept for readability and backward compatibility with the early prototype
scripts.

## Python Side

`python/tmj_practice_env.py` implements a Gymnasium-style environment:

```python
observation, info = env.reset()
observation, reward, terminated, truncated, info = env.step(action)
```

It can optionally launch ArtiSynth if the REST server is not already alive by
passing `launch_command`.

`python/reward.py` contains the reward configuration and reward function. This
keeps the reward separate from the REST/environment wrapper so it can later be
replaced with a TMJ-specific reward.

`python/run_env_smoke_test.py` uses a simple hand-coded policy:

- if the target is left of the marker, activate the left muscle
- if the target is right of the marker, activate the right muscle

`python/train_baseline.py` trains a Stable-Baselines3 PPO or SAC baseline. By
default it exposes normalized `[-1, 1]` actions to the RL policy and maps them
internally to ArtiSynth excitation values in `[0, 1]`.

`python/evaluate_policy.py` loads a saved PPO/SAC model and writes step-level CSV
logs with marker position, target position, tracking error, actions, forces,
reward, and episode flags.

## Observation

The current observation vector contains:

```text
marker position xyz        3
marker velocity xyz        3
target position xyz        3
tracking error             1
left/right action          2
left/right exciter value   2
left/right muscle value    2
left/right muscle force    2
-----------------------------
total                     18
```

## Reward Function

The actual training reward is computed in Python in `python/reward.py`.

Current reward:

```python
progress = previous_distance - current_distance
effort = left_action**2 + right_action**2
velocity = norm(marker_velocity)
action_change = norm(current_action - previous_action)**2

reward = (
    progress_reward_scale * progress
    - distance_penalty_scale * current_distance
    - effort_penalty_scale * effort
    - velocity_penalty_scale * velocity
    - action_change_penalty_scale * action_change
)
```

If the marker reaches the target:

```python
if current_distance <= goal_threshold:
    reward = goal_reward
    terminated = True
```

Current default weights:

```text
progress_reward_scale = 20.0
distance_penalty_scale = 1.0
effort_penalty_scale = 0.01
velocity_penalty_scale = 0.02
action_change_penalty_scale = 0.005
goal_threshold = 0.01
goal_reward = 5.0
```

Meaning:

- reward increases when the marker moves closer to the target
- reward decreases when the marker remains far from the target
- large muscle excitation is mildly penalized
- high marker velocity and abrupt action changes can be mildly penalized
- reaching the target gives a terminal reward

## How To Run

Start the Java model from Eclipse/ArtiSynth:

```text
artisynth.models.tmj.practice.MuscleExcitationPrototype
```

Confirm the REST server:

```bash
curl http://localhost:8081/actionSize
curl http://localhost:8081/obsSize
curl http://localhost:8081/stateSize
curl -X POST http://localhost:8081/reset
```

Expected output:

```text
2
18
19
```

Run the smoke test:

```bash
python python/run_env_smoke_test.py
```

Run SAC training, recommended next baseline:

```bash
python python/train_baseline.py --algo sac --timesteps 20000 --skip-check-env
```

Run PPO training, for comparison with the earlier baseline:

```bash
python python/train_baseline.py --algo ppo --timesteps 20000 --skip-check-env
```

The trained model is saved under `runs/`, which is ignored by git.

Evaluate a saved policy and write a CSV log:

```bash
python python/evaluate_policy.py \
  --algo sac \
  --model-path runs/sac_tmj_practice.zip \
  --episodes 20 \
  --output runs/evaluation.csv
```

Optional auto-launch pattern:

```bash
python python/train_baseline.py \
  --algo sac \
  --launch-command "artisynth -model artisynth.models.tmj.practice.MuscleExcitationPrototype -play" \
  --timesteps 20000 \
  --skip-check-env
```

## Current Results

The current smoke test succeeds. Example:

```text
target x = -0.1137
marker x = 0.0000 -> -0.0184 -> -0.0590 -> -0.1071
error    = 0.1137 ->  0.0952 ->  0.0547 ->  0.0066
result   = terminated=True
```

This verifies that:

- Python reads the target state
- Python sends a left/right excitation action
- Java applies the action through `MuscleExciter`s
- ArtiSynth moves the marker toward the target
- Python receives the updated state and termination condition

The current PPO baseline also begins to learn meaningful directional actions,
but remains an early, weak baseline.
Example after 4096 timesteps:

```text
target x = 0.069
action   = [0.0, 0.12]
marker x = 0.0000 -> 0.0496
error    = 0.0690 -> 0.0194
result   = truncated=True
```

Interpretation:

- the policy selected the right muscle for a positive target
- tracking error decreased substantially
- the baseline did not always reach the stricter `0.01` threshold yet

So the current status is:

```text
ArtiSynth/Python RL pipeline: working
heuristic control: working
PPO baseline: running and directionally meaningful
SAC baseline: added as the recommended next performance-improvement baseline
Amir-like state/action REST shape: partially implemented
generic action-to-exciter controller loop: implemented
policy quality: needs more training/reward tuning
```

## Next Steps

- Train longer and compare learning curves.
- Tune reward weights and threshold.
- Move from this point-to-point practice model toward a real TMJ/jaw model.
- Extend the action vector from two toy muscles to actual jaw muscle groups.
