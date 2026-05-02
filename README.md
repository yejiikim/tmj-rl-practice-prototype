# TMJ RL Practice Prototype

This repository contains a small ArtiSynth/Python prototype for testing an
RL-style action/state interface.

The current model is intentionally simple: a single muscle pulls a rigid body
marker, while Python sends excitation actions through REST and reads back the
simulation state.

## Current Goal

This prototype tests the same general communication pattern used in
ArtiSynth RL setups:

```text
Python action
-> REST API
-> Java controller
-> MuscleExciter
-> ArtiSynth simulation
-> state/error/reward returned to Python
```

It is not yet a full RL training environment. It is the first step toward a
Gym-style environment with `reset()` and `step(action)`.

## Repository Layout

```text
src/artisynth/models/tmj/practice/
  MuscleExcitationPrototype.java
  SimpleRlController.java

python/
  send_action_test.py
  requirements.txt
```

## Java / ArtiSynth Side

`MuscleExcitationPrototype.java` builds the ArtiSynth model:

- fixed anchor particle
- rigid box
- frame marker attached to the box
- fixed target particle
- one axial muscle
- one `MuscleExciter`
- REST endpoints
- output probes

REST endpoints:

```text
GET  /
GET  /state
GET  /actionSize
GET  /excitations
POST /excitations
POST /reset
```

`SimpleRlController.java` stores Python actions and applies them during the
ArtiSynth simulation step:

```java
exciter.setExcitation(actionExcitation);
```

It also reports:

- marker position
- marker velocity
- target position
- tracking error
- action excitation
- exciter excitation
- muscle excitation
- muscle force
- reward-like value

## Python Side

`python/send_action_test.py` is a simple REST client. It:

1. checks the Java REST server,
2. calls `/reset`,
3. sends sinusoidal excitation actions to `/excitations`,
4. prints marker position, target position, tracking error, reward, and force.

## How To Run

1. Open the Java files in the Eclipse/ArtiSynth project.
2. Clean/rebuild the Eclipse project.
3. Run:

```text
artisynth.models.tmj.practice.MuscleExcitationPrototype
```

4. Confirm the console shows:

```text
MuscleExcitationPrototype REST server started at http://localhost:8081
```

5. Test reset:

```bash
curl -X POST http://localhost:8081/reset
```

6. Run the Python client:

```bash
python python/send_action_test.py
```

Expected evidence of success:

- `targetPosition` remains fixed at x = 0.2
- `actionExcitation` changes according to Python actions
- `exciterExcitation` follows the action after simulation steps
- `muscleForce` changes with excitation/state
- `trackingError` and `rewardLike` are returned to Python

## Current Status

Implemented:

- external muscle excitation action
- Java REST API
- target point
- tracking error
- reward-like value
- reset endpoint
- Python client test

Next steps:

- convert the Python client into a Gym-style environment class
- implement `step(action)` and episode termination
- connect an RL baseline
- extend from the simple muscle prototype toward a TMJ/jaw model
