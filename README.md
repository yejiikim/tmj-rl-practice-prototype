# Simple Antagonist Muscle RL Prototype

This repository contains a minimal ArtiSynth-Python reinforcement learning
prototype for controlling muscle excitations through a REST interface.

The current model is intentionally simple: a marker attached to a dynamic rigid
body is pulled by two opposing muscles, and the task is to move the marker to a
target point. The main purpose is to validate the closed-loop connection between
ArtiSynth and Python before moving toward a more complex TMJ/jaw model.

This is not a complete generic ArtiSynth RL framework yet. It is a simple
working demo with a first refactor toward a reusable REST/controller/Python
environment structure.

## Current Status

The prototype currently supports:

- a simple two-muscle antagonist model in ArtiSynth
- REST communication between Java/ArtiSynth and Python
- action commands applied through ArtiSynth `MuscleExciter`s
- a Gym-style Python environment wrapper
- SAC training with Stable-Baselines3
- smoke testing, evaluation, CSV logging, and plotting

The latest run used:

```text
SAC training timesteps: 50000
Evaluation episodes:   50
```

Result:

```text
Successes:                    50/50
Success rate:                 100.0%
Truncated episodes:           0/50
Mean return:                  7.5422
Mean final tracking error:    0.0041
Median final tracking error:  0.0034
Mean minimum tracking error:  0.0041
Positive target success rate: 100.0% (26 episodes)
Negative target success rate: 100.0% (24 episodes)
```

The success threshold is:

```text
tracking error <= 0.01
```

These results should be interpreted as a proof-of-concept for the
ArtiSynth-Python RL loop, not as a final biomechanics result.

## Model Overview

The ArtiSynth model consists of:

- two fixed anchor points
- one dynamic rigid body
- one `FrameMarker` attached to the rigid body
- one target point
- two opposing muscles
- two `MuscleExciter`s

The structure is:

<img width="480" height="160" alt="image" src="https://github.com/user-attachments/assets/a5d03215-a796-4171-a31d-803770486478" />


```text
left anchor -- left muscle -- marker/box -- right muscle -- right anchor
                                |
                              target
```

The tracked point is the `FrameMarker` attached to the dynamic rigid body. The
target is a fixed `Particle`; its x-position is randomized at reset.

The two muscles form a simple antagonist pair:

- the left muscle pulls the marker/body to the left
- the right muscle pulls the marker/body to the right

Gravity is set to zero so the task focuses on muscle-driven point-to-point
tracking.

## Repository Structure

The repository is split into Java, Python, and result files:

- `java/` contains the ArtiSynth model, REST API, and controller code.
- `python/` contains the REST-backed environment, reward, training, and evaluation scripts.
- `results/` contains the representative 50-episode SAC evaluation output.

```text
java/
  src/artisynth/models/tmj/practice/
    AntagonistMuscleDemo.java
    RlRestApi.java
    RlController.java
    MuscleExcitationRlController.java
    AntagonistMuscleRlController.java
    MuscleExcitationPrototype.java
    SimpleRlController.java

python/
  artisynth_base_env.py
  antagonist_env.py
  tmj_practice_env.py
  reward.py
  train_baseline.py
  evaluate_policy.py
  run_env_smoke_test.py
  send_action_test.py
  plot_evaluation.py
  requirements.txt

results/
  evaluation_sac_50.csv
  tracking_error_sac_50.png
```

## Java Components

### `AntagonistMuscleDemo.java`

Main ArtiSynth model. It builds the simple physical system:

- anchors
- dynamic box
- tracked frame marker
- target point
- left/right muscles
- left/right `MuscleExciter`s

It also defines the current simple reset behavior:

- clear commanded excitations
- clear exciter and muscle excitations
- reset the box pose and velocity
- sample a new target x-position

### `RlRestApi.java`

REST API layer used by the Python environment.

Main endpoints:

```text
GET  /
GET  /actionSize
GET  /obsSize
GET  /stateSize
GET  /state
GET  /time
GET  /excitations
POST /excitations
POST /reset
POST /setSeed
POST /setTest
```

Python uses these endpoints to reset the episode, send muscle excitation
actions, and read simulation state.

### `RlController.java`

Small interface between `RlRestApi` and a controller implementation.

### `MuscleExcitationRlController.java`

Reusable muscle-excitation controller. It stores the latest action vector from
Python and applies it to the registered `MuscleExciter[]` during the ArtiSynth
simulation step.

In the current model:

```text
action = [left excitation, right excitation]
```

The array-based structure is intended to make later extension to more muscle
exciters easier.

### `AntagonistMuscleRlController.java`

Small wrapper around `MuscleExcitationRlController` for the current two-muscle
demo. It exposes left/right action values in the ArtiSynth control panel.

### Compatibility Wrappers

`MuscleExcitationPrototype.java` and `SimpleRlController.java` are kept for
compatibility with the original prototype names and older launch settings.

The current main model implementation is:

```text
artisynth.models.tmj.practice.AntagonistMuscleDemo
```

## Python Components

The Python side is organized around the standard RL concepts: environment,
action, observation, reward, training, and evaluation.

### `artisynth_base_env.py`

Generic REST/Gym-style base environment.

It handles:

- `reset()`
- `step(action)`
- REST communication
- action normalization
- waiting for ArtiSynth simulation time to advance
- reading `/state`

In one environment step, Python sends an excitation action to Java, waits for
the ArtiSynth simulation time to advance, reads the new state, and returns the
observation/reward/termination values expected by Stable-Baselines3.

Task-specific environments subclass this base class.

### `antagonist_env.py`

Task-specific environment for the simple antagonist model.

It defines:

- parsing of the Java `/state` JSON
- the observation vector
- diagnostic `info` values
- reward calculation
- episode bookkeeping

### `reward.py`

Reward function for the simple point-to-point tracking task.

The reward:

- gives a success reward when the marker reaches the target
- rewards progress toward the target
- penalizes remaining distance
- penalizes excitation effort
- optionally penalizes velocity and abrupt action changes

The reward is computed in Python. The Java `rewardLike` value is only a
diagnostic value for quick state inspection.

### `train_baseline.py`

Trains a SAC policy using Stable-Baselines3.

SAC is used because the action space is continuous: the policy outputs muscle
excitation commands.

For the representative result in this repository, SAC was trained for 50000
timesteps.

### `evaluate_policy.py`

Evaluates a saved SAC policy and writes a step-level CSV log.

It reports:

- success rate
- truncated episodes
- mean return
- mean final error
- median final error
- mean minimum error
- positive-target success rate
- negative-target success rate

### `run_env_smoke_test.py`

Runs a hand-coded controller before training. This checks that:

- Python can send actions
- Java receives them
- `MuscleExciter` values change
- muscle forces appear
- the marker moves toward the target
- the episode terminates when the target is reached

## RL Loop

The current closed-loop flow is:

```text
Python policy/environment
→ POST /excitations
→ Java REST API
→ ArtiSynth controller stores latest action
→ controller applies action to MuscleExciters during simulation
→ ArtiSynth model advances
→ Python waits for simulation time to advance
→ GET /state
→ Python computes reward and the next action
```

## Action Space

The Java model expects excitation values in:

```text
[0, 1]
```

For SAC training, the Python policy uses normalized actions in:

```text
[-1, 1]
```

These are mapped internally to ArtiSynth excitation values:

```text
excitation = 0.5 * (action + 1.0)
```

Current action size:

```text
2
```

corresponding to:

```text
[left muscle excitation, right muscle excitation]
```

## Reward Function

The current reward is a simple tracking reward for the prototype. It is not
intended as the final TMJ/jaw objective.

The success condition is:

```text
tracking error <= 0.01
```

If the marker reaches this threshold, the episode terminates and receives:

```text
goal_reward = 5.0
```

Otherwise, the step reward is:

```text
reward =
  progress_reward_scale * progress
  - distance_penalty_scale * distance
  - effort_penalty_scale * effort
  - velocity_penalty_scale * velocity_norm
  - action_change_penalty_scale * action_change
```

where:

```text
progress = previous_distance - current_distance
effort = sum(action^2)
action_change = sum((action - previous_action)^2)
```

The training run reported here used:

```text
progress_reward_scale = 20.0
distance_penalty_scale = 1.0
effort_penalty_scale = 0.01
velocity_penalty_scale = 0.02
action_change_penalty_scale = 0.005
```

The progress term rewards movement toward the target. The distance term
penalizes remaining error. The effort term discourages unnecessarily large
muscle excitation commands.

## Observation

Current observation size:

```text
18
```

The observation includes:

- marker position
- marker velocity
- target position
- tracking error
- commanded action excitations
- actual exciter excitations
- muscle excitations
- muscle forces

## Reset

Python initiates reset through:

```text
POST /reset
```

The Java model then:

- clears commanded excitations
- clears exciter and muscle excitations
- resets the box pose and velocity
- samples a new target x-position

This reset is sufficient for the current simple prototype, but more robust
reset handling and timestep synchronization should be checked before scaling to
a more complex jaw/TMJ model.

## Running the Demo

### 1. Start ArtiSynth

Load this model in ArtiSynth/Eclipse:

```text
artisynth.models.tmj.practice.AntagonistMuscleDemo
```

The old compatibility entry point also works:

```text
artisynth.models.tmj.practice.MuscleExcitationPrototype
```

Press play in ArtiSynth so simulation time advances.

### 2. Check REST

```bash
curl http://localhost:8081/actionSize
curl http://localhost:8081/obsSize
curl http://localhost:8081/stateSize
curl http://localhost:8081/state
```

Expected sizes:

```text
actionSize = 2
obsSize = 18
stateSize = 19
```
<img width="577" height="231" alt="Last login Mon May 11 115 41 on ttys815" src="https://github.com/user-attachments/assets/1fe175c0-d105-4c24-a229-34e6cb809d1d" />

### 3. Install Python dependencies

```bash
cd python
pip install -r requirements.txt
```

### 4. Run a smoke test

```bash
python run_env_smoke_test.py
```
<img width="617" height="326" alt="(base) gim-ye1101p149-148-255-283 tm)_python_client  optanaconds3binpythonrun_en" src="https://github.com/user-attachments/assets/51c3cc73-714e-4882-8682-8c24966ae69b" />


### 5. Train SAC

```bash
python train_baseline.py --timesteps 50000 --skip-check-env
```

This saves the model to:

```text
runs/sac_tmj_practice.zip
```

Model `.zip` files are ignored by git.

<img width="3022" height="1726" alt="Screen Recording 2026-05-11 at 2 41 16 PM" src="https://github.com/user-attachments/assets/4f8c734c-806f-4275-b9ab-f04601af9357" />


### 6. Evaluate

```bash
python evaluate_policy.py --episodes 50 --output runs/evaluation_sac_50.csv
```
<img width="611" height="201" alt="base) 91m-ye1p149-248-255-203 (a) python Client  optanaconda3binpythonevaiua" src="https://github.com/user-attachments/assets/cc974e2d-a086-4061-9615-cc2adf6510dc" />


### 7. Plot

```bash
python plot_evaluation.py \
  --input runs/evaluation_sac_50.csv \
  --output runs/tracking_error_sac_50.png
```

The representative 50-episode evaluation files are included in `results/`.

## Current Limitations

- The physical model is intentionally simple.
- The task is point-to-point marker tracking, not full TMJ/jaw control.
- Reset is still manually implemented for the simple model.
- ArtiSynth timestep synchronization should be checked more carefully.
- OutputProbe visualization can be improved, especially for multiple muscle
  activations/forces in one plot.
- The reward is a simple tracking reward, not a final TMJ cost function.

## Next Steps

Planned next steps:

- add reset consistency tests
- improve OutputProbe visualization
- check the best ArtiSynth timestep/advance hook 
- make the state interface more component-based
- extend the same REST/controller structure toward a jaw/TMJ model
- design a task-specific jaw reward/cost function
