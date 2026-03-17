# `pose_ai_core` Integration Guide
### AI Fitness Trainer – Developer Reference

> **Library source:** `liftr-critique-service/pose_ai_core`  
> **Guide version:** 1.0 · March 2026

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Core Concepts](#4-core-concepts)
   - 4.1 [Pose & KEYPOINTS](#41-pose--keypoints)
   - 4.2 [PoseEstimator](#42-poseestimator)
   - 4.3 [PoseHeuristics & HEURISTICS](#43-poseheuristics--heuristics)
   - 4.4 [MV_DIRECTIONS & MovementVector](#44-mv_directions--movementvector)
5. [Exercise Framework](#5-exercise-framework)
   - 5.1 [ExerciseState](#51-exercisestate)
   - 5.2 [Critique](#52-critique)
   - 5.3 [Progress](#53-progress)
   - 5.4 [Exercise base class](#54-exercise-base-class)
   - 5.5 [The update loop](#55-the-update-loop)
6. [Built-in Exercises](#6-built-in-exercises)
   - 6.1 [ShoulderPress](#61-shoulderpress)
   - 6.2 [BicepCurl](#62-bicepcurl)
7. [Adding a New Exercise](#7-adding-a-new-exercise)
   - 7.1 [Step-by-step skeleton](#71-step-by-step-skeleton)
   - 7.2 [Worked example – Squat](#72-worked-example--squat)
   - 7.3 [Registering the exercise](#73-registering-the-exercise)
8. [Integrating with a Mobile App](#8-integrating-with-a-mobile-app)
   - 8.1 [Recommended stack](#81-recommended-stack)
   - 8.2 [Per-frame processing pipeline](#82-per-frame-processing-pipeline)
   - 8.3 [Minimal end-to-end snippet](#83-minimal-end-to-end-snippet)
9. [Available Heuristics Reference](#9-available-heuristics-reference)
10. [Available Keypoints Reference](#10-available-keypoints-reference)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Overview

`pose_ai_core` is a **standalone Python library** extracted from the `liftr-critique-service` backend. It provides everything needed to:

- **Detect human poses** in images and video frames using a MobileNet-based neural network.
- **Measure joint angles and movement directions** from detected poses.
- **Model exercises** as state machines that automatically count reps and fire real-time form critiques.
- **Track progress** of a joint through its range of motion (useful for progress-bar UI in the mobile app).

The library has **no dependency** on the rest of `liftr-critique-service` — it is fully self-contained and can be dropped into any Python project.

---

## 2. Architecture

```
pose_ai_core/
├── __init__.py            ← top-level public API
│
├── pose.py                ← Pose dataclass + KEYPOINTS enum
├── estimator.py           ← PoseEstimator (camera / video → [Pose])
├── measurements.py        ← PoseHeuristics, HEURISTICS, MV_DIRECTIONS
├── exercise.py            ← Exercise base class, Critique, Progress, ExerciseState
├── utils.py               ← midpoint(), COLORS
├── preprocessing.py       ← image normalisation helpers (internal)
│
├── models/                ← MobileNet model definition (internal)
│   └── with_mobilenet.py
│
├── modules/               ← keypoint extraction, PAF grouping (internal)
│   ├── keypoints.py
│   ├── pose.py
│   └── …
│
└── exercises/             ← built-in exercise implementations
    ├── __init__.py
    ├── shoulder_press.py
    └── bicep_curl.py
```

**Data flow per frame:**

```
Camera frame (BGR numpy array)
        │
        ▼
  PoseEstimator.estimate(frame)
        │  returns List[Pose]
        ▼
  PoseHeuristics.update(pose)
        │  computes all joint angles + movement vectors
        ▼
  Exercise.update(pose, heuristics)
        │  returns (state, [Critique], [progress_tuple])
        ▼
  Mobile UI  ←  rep count, critique messages, progress values
```

---

## 3. Installation

### Option A – pip install from source (recommended for mobile backend)

```bash
pip install /path/to/liftr-critique-service/pose_ai_core
```

### Option B – editable install (development, changes sync live)

```bash
pip install -e /path/to/liftr-critique-service/pose_ai_core
```

### Option C – copy the folder

Copy the entire `pose_ai_core/` directory into your project root and import it directly.

### Dependencies

```
numpy>=1.14.0
opencv-python>=3.4.0.14
torch>=0.4.1
torchvision>=0.2.1
```

Install them with:

```bash
pip install numpy opencv-python torch torchvision
```

> **Model weights:** You must supply the pre-trained checkpoint file  
> `checkpoint_iter_370000.pth` (from `liftr-critique-service/critique/pose/`).  
> Copy it to a stable path and pass that path to `PoseEstimator`.

---

## 4. Core Concepts

### 4.1 Pose & KEYPOINTS

A `Pose` object represents one detected person in a single frame.

```python
from pose_ai_core import Pose, KEYPOINTS

# keypoints is a (18, 2) numpy array of (x, y) pixel coords.
# A value of -1 means the keypoint was not detected.
pose.keypoints[KEYPOINTS.L_ELB]   # → [x, y] of left elbow
pose.keypoints[KEYPOINTS.R_SHO]   # → [x, y] of right shoulder
pose.confidence                    # overall detection confidence (0–1)
pose.draw(frame)                   # draw skeleton onto a BGR image in-place
```

The 18 COCO keypoints available via `KEYPOINTS`:

| Constant | Index | Body part |
|---|---|---|
| `NOSE` | 0 | Nose |
| `NECK` | 1 | Neck |
| `R_SHO` | 2 | Right shoulder |
| `R_ELB` | 3 | Right elbow |
| `R_WRI` | 4 | Right wrist |
| `L_SHO` | 5 | Left shoulder |
| `L_ELB` | 6 | Left elbow |
| `L_WRI` | 7 | Left wrist |
| `R_HIP` | 8 | Right hip |
| `R_KNEE` | 9 | Right knee |
| `R_ANK` | 10 | Right ankle |
| `L_HIP` | 11 | Left hip |
| `L_KNEE` | 12 | Left knee |
| `L_ANK` | 13 | Left ankle |
| `R_EYE` | 14 | Right eye |
| `L_EYE` | 15 | Left eye |
| `R_EAR` | 16 | Right ear |
| `L_EAR` | 17 | Left ear |

### 4.2 PoseEstimator

```python
from pose_ai_core import PoseEstimator

estimator = PoseEstimator(
    checkpoint_path="path/to/checkpoint_iter_370000.pth",
    height_size=256,   # network input height (default 256)
    use_gpu=True       # set False for CPU-only devices
)

# Single image
import cv2
frame = cv2.imread("frame.jpg")
poses = estimator.estimate(frame, conf_thresh=0.2)  # → List[Pose]

# Video / live camera
for frame, poses in estimator.estimate_video("workout.mp4"):
    ...  # process each frame
```

### 4.3 PoseHeuristics & HEURISTICS

`PoseHeuristics` computes and caches joint angles (in degrees or radians) for every frame. Create **one instance per session** and call `update()` every frame.

```python
from pose_ai_core import PoseHeuristics, HEURISTICS

heuristics = PoseHeuristics(degrees=True)   # True → angles in degrees

heuristics.update(pose)                     # call once per frame

angle = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)   # float | None
```

`get_angle()` returns `None` if any required keypoint is missing from the frame.

### 4.4 MV_DIRECTIONS & MovementVector

`MovementVector` tracks the recent movement direction of a keypoint over a short history window.

```python
from pose_ai_core import MV_DIRECTIONS

mv = heuristics.get_movement(KEYPOINTS.L_WRI)
mv.x   # 'HOLD' | 'LEFT' | 'RIGHT'
mv.y   # 'HOLD' | 'UP'   | 'DOWN'

if mv.y == MV_DIRECTIONS.HOLD:
    print("Wrist is stationary vertically")
```

This is particularly useful for detecting when a user has **paused at the top or bottom** of a rep before the state machine advances.

---

## 5. Exercise Framework

### 5.1 ExerciseState

A lightweight named tuple:

```python
ExerciseState(id='SET_UP', label='Set up', func=<bound method>)
```

- `id` – string identifier, matched against `STATES` constants
- `label` – human-readable name for display in the mobile UI
- `func` – called every frame; returns the **next** state ID

### 5.2 Critique

Represents a single form-check rule.

```python
from pose_ai_core import Critique

Critique(
    name='lock_elbows',
    states=['UP'],                              # active in these states only
    msg='Do not lock your elbows at the top.',  # shown to the user
    func=self._critique_lock_elbows             # returns True when problem detected
)
```

When `func` returns `True`, the critique is included in the list returned by `exercise.update()`.

### 5.3 Progress

Tracks how far a joint has moved through a defined range, mapped to `[0.0, 1.0]`.

```python
from pose_ai_core import Progress, HEURISTICS, KEYPOINTS

p = Progress('raise_elbow', states=['RAISE'])
p.add_range(HEURISTICS.LEFT_ELBOW, KEYPOINTS.L_ELB, low=90, high=155)

# exercise.update() automatically calls check_progress() and returns the results.
# Each result is a tuple: (heuristic_id, keypoint_id, value_0_to_1)
```

Use the `value_0_to_1` to drive a **progress bar** or **arc indicator** in the UI.

### 5.4 Exercise base class

| Method | Purpose |
|---|---|
| `_add_state(state, initial=False)` | Register a state; mark the entry state with `initial=True` |
| `_add_critique(critique)` | Register a form-check rule |
| `_add_progress(progress)` | Register a progress tracker |
| `_set_rep_transition(from_id, to_id)` | Declare which state transition counts as one rep |
| `_in_range(value, target, threshold)` | Helper: `True` if `abs(target - value) < threshold` |
| `update(pose, heuristics)` | Drive one frame; returns `(state, critiques, progress)` |
| `reset()` | Reset state and rep counter to initial values |

Public attributes:

| Attribute | Type | Description |
|---|---|---|
| `exercise.name` | `str` | Display name |
| `exercise.reps` | `int` | Completed rep count |
| `exercise.state` | `ExerciseState` | Current state |

### 5.5 The update loop

```python
state, critiques, progress = exercise.update(pose, heuristics)

print(f"State : {state.label}")
print(f"Reps  : {exercise.reps}")

for critique in critiques:
    print(f"⚠️  {critique.msg}")

for heuristic_id, keypoint_id, value in progress:
    print(f"Progress ({heuristic_id}): {value:.0%}")
```

---

## 6. Built-in Exercises

### 6.1 ShoulderPress

```python
from pose_ai_core.exercises import ShoulderPress

ex = ShoulderPress()
```

| State | ID | Transition condition |
|---|---|---|
| Set up | `SET_UP` | Both shoulders ~180°, elbows ~90°, wrists held still |
| Raise | `UP` | Both shoulders < 130° (arms overhead), wrists held |
| Lower | `DOWN` | Both shoulders > 170° (arms back down), wrists held |

**Rep transition:** `DOWN` → `UP`

**Critiques:**
- `lock_elbows` – elbow angle > 170° at the top of the press
- `too_low` – shoulder angle > 220° (arms dropped too far down)

**Progress trackers:** `raise_shoulder`, `lower_shoulder`, `raise_elbow`, `lower_elbow`

### 6.2 BicepCurl

```python
from pose_ai_core.exercises import BicepCurl

ex = BicepCurl(side='left')   # or 'right'
```

| State | ID | Transition condition |
|---|---|---|
| Set up | `SET_UP` | Elbow ~270°, wrist held still |
| Raise | `RAISE` | Elbow > 295° (fully curled), wrist held |
| Lower | `LOWER` | Elbow < 220° (fully extended), wrist held |

**Rep transition:** `RAISE` → `LOWER`

**Critiques:**
- `elbow_deviation` – elbow horizontal offset from shoulder > 60 px (elbow is swinging out)

**Progress trackers:** `raise_elbow`, `lower_elbow`

---

## 7. Adding a New Exercise

### 7.1 Step-by-step skeleton

Create a new file in `pose_ai_core/exercises/your_exercise.py`:

```python
from ..pose import Pose, KEYPOINTS
from ..measurements import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from ..exercise import Exercise, ExerciseState, Critique, Progress


class YourExercise(Exercise):

    # ── 1. Declare your states ────────────────────────────────────────────────
    class STATES:
        SET_UP = 'SET_UP'
        PHASE_A = 'PHASE_A'
        PHASE_B = 'PHASE_B'
        # add as many phases as the movement needs

    def __init__(self):
        super().__init__('Your Exercise Name')

        # ── 2. Register states ────────────────────────────────────────────────
        self._add_state(
            ExerciseState(self.STATES.SET_UP, "Set up", self._state_set_up),
            initial=True
        )
        self._add_state(ExerciseState(self.STATES.PHASE_A, "Phase A", self._state_phase_a))
        self._add_state(ExerciseState(self.STATES.PHASE_B, "Phase B", self._state_phase_b))

        # ── 3. Declare the rep transition ─────────────────────────────────────
        # One rep is completed every time the machine moves from PHASE_B → PHASE_A
        self._set_rep_transition(self.STATES.PHASE_B, self.STATES.PHASE_A)

        # ── 4. Add critiques ──────────────────────────────────────────────────
        self._add_critique(Critique(
            'your_critique_name',
            [self.STATES.PHASE_A, self.STATES.PHASE_B],  # active states
            'Human-readable feedback message for the user.',
            self._critique_your_rule
        ))

        # ── 5. Add progress trackers ──────────────────────────────────────────
        phase_a_progress = Progress('phase_a_joint', [self.STATES.PHASE_A])
        phase_a_progress.add_range(HEURISTICS.RIGHT_KNEE, KEYPOINTS.R_KNEE, low=170, high=90)
        self._add_progress(phase_a_progress)

    # ── State functions ───────────────────────────────────────────────────────
    # Each must return the ID of the NEXT state (can be the same state to stay put).

    def _state_set_up(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        # Read heuristics, decide if user is in position
        angle = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        mv    = heuristics.get_movement(KEYPOINTS.R_WRI)
        if angle and self._in_range(angle, 170, 15):
            return self.STATES.PHASE_A
        return self.STATES.SET_UP

    def _state_phase_a(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        angle = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        if angle and angle < 100:
            return self.STATES.PHASE_B
        return self.STATES.PHASE_A

    def _state_phase_b(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        angle = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        if angle and angle > 160:
            return self.STATES.PHASE_A
        return self.STATES.PHASE_B

    # ── Critique functions ────────────────────────────────────────────────────
    # Return True when the problem IS detected (triggers the message).

    def _critique_your_rule(self, pose: Pose, heuristics: PoseHeuristics) -> bool:
        angle = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        return angle is not None and angle < 70   # e.g. going too deep
```

### 7.2 Worked example – Squat

```python
from ..pose import Pose, KEYPOINTS
from ..measurements import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from ..exercise import Exercise, ExerciseState, Critique, Progress


class Squat(Exercise):

    class STATES:
        SET_UP  = 'SET_UP'
        DESCEND = 'DESCEND'
        ASCEND  = 'ASCEND'

    def __init__(self):
        super().__init__('Squat')

        self._add_state(
            ExerciseState(self.STATES.SET_UP, "Stand tall", self._state_set_up),
            initial=True
        )
        self._add_state(ExerciseState(self.STATES.DESCEND, "Descend", self._state_descend))
        self._add_state(ExerciseState(self.STATES.ASCEND,  "Ascend",  self._state_ascend))

        # Rep = bottom of squat → coming back up
        self._set_rep_transition(self.STATES.DESCEND, self.STATES.ASCEND)

        self._add_critique(Critique(
            'knees_caving',
            [self.STATES.DESCEND],
            'Keep your knees in line with your toes – do not let them cave inward.',
            self._critique_knees_caving
        ))
        self._add_critique(Critique(
            'too_shallow',
            [self.STATES.DESCEND],
            'Try to reach parallel – your hips should drop to knee height.',
            self._critique_too_shallow
        ))

        descend_progress = Progress('descend', [self.STATES.DESCEND])
        descend_progress.add_range(HEURISTICS.RIGHT_KNEE, KEYPOINTS.R_KNEE, 170, 90)
        descend_progress.add_range(HEURISTICS.LEFT_KNEE,  KEYPOINTS.L_KNEE, 170, 90)
        self._add_progress(descend_progress)

        ascend_progress = Progress('ascend', [self.STATES.ASCEND])
        ascend_progress.add_range(HEURISTICS.RIGHT_KNEE, KEYPOINTS.R_KNEE, 90, 170)
        ascend_progress.add_range(HEURISTICS.LEFT_KNEE,  KEYPOINTS.L_KNEE, 90, 170)
        self._add_progress(ascend_progress)

    def _state_set_up(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        r_knee = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        l_knee = heuristics.get_angle(HEURISTICS.LEFT_KNEE)
        r_hip  = heuristics.get_movement(KEYPOINTS.R_HIP)
        if r_knee and l_knee:
            if (self._in_range(r_knee, 170, 15) and
                    self._in_range(l_knee, 170, 15) and
                    r_hip.y == MV_DIRECTIONS.HOLD):
                return self.STATES.DESCEND
        return self.STATES.SET_UP

    def _state_descend(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        r_knee = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        l_knee = heuristics.get_angle(HEURISTICS.LEFT_KNEE)
        r_hip_mv = heuristics.get_movement(KEYPOINTS.R_HIP)
        if r_knee and l_knee:
            if r_knee < 100 and l_knee < 100 and r_hip_mv.y == MV_DIRECTIONS.HOLD:
                return self.STATES.ASCEND
        return self.STATES.DESCEND

    def _state_ascend(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        r_knee = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        l_knee = heuristics.get_angle(HEURISTICS.LEFT_KNEE)
        r_hip_mv = heuristics.get_movement(KEYPOINTS.R_HIP)
        if r_knee and l_knee:
            if r_knee > 160 and l_knee > 160 and r_hip_mv.y == MV_DIRECTIONS.HOLD:
                return self.STATES.DESCEND
        return self.STATES.ASCEND

    def _critique_knees_caving(self, pose: Pose, heuristics: PoseHeuristics) -> bool:
        r_knee = pose.keypoints[KEYPOINTS.R_KNEE]
        r_ank  = pose.keypoints[KEYPOINTS.R_ANK]
        l_knee = pose.keypoints[KEYPOINTS.L_KNEE]
        l_ank  = pose.keypoints[KEYPOINTS.L_ANK]
        if r_knee[0] == -1 or l_knee[0] == -1:
            return False
        # Knee is caving if it is further inward than the ankle
        r_caving = r_knee[0] > r_ank[0] + 20
        l_caving = l_knee[0] < l_ank[0] - 20
        return r_caving or l_caving

    def _critique_too_shallow(self, pose: Pose, heuristics: PoseHeuristics) -> bool:
        r_knee = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        if r_knee is not None:
            return r_knee > 120   # hasn't reached 90° (parallel)
        return False
```

### 7.3 Registering the exercise

Open `pose_ai_core/exercises/__init__.py` and add two lines:

```python
from .squat import Squat          # ← add this import

EXERCISES = {
    'shoulder_press': ShoulderPress,
    'bicep_curl':     BicepCurl,
    'squat':          Squat,      # ← add this entry
}
```

The exercise is now accessible everywhere via:

```python
from pose_ai_core.exercises import EXERCISES
ExClass = EXERCISES['squat']
exercise = ExClass()
```

---

## 8. Integrating with a Mobile App

### 8.1 Recommended stack

| Layer | Recommendation |
|---|---|
| Mobile framework | React Native / Flutter (camera frames → Python backend via REST or WebSocket) |
| Backend | FastAPI or Flask running `pose_ai_core` as a service |
| Camera input | Device camera → JPEG/PNG frames sent to backend, or on-device via Kivy/BeeWare |
| Model inference | CPU for most phones; GPU if running on-device via ONNX export |

### 8.2 Per-frame processing pipeline

```
┌───────────────────────────────────────────────────────────┐
│  Mobile device                                            │
│                                                           │
│  Camera frame → encode as JPEG → POST /analyze           │
└───────────────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Python backend (FastAPI + pose_ai_core)                  │
│                                                           │
│  1. Decode JPEG → numpy BGR array                        │
│  2. estimator.estimate(frame) → poses                    │
│  3. heuristics.update(poses[0])                          │
│  4. exercise.update(poses[0], heuristics)                │
│     → state, critiques, progress                         │
│  5. Return JSON response                                  │
└───────────────────────────────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│  Mobile device receives JSON                              │
│                                                           │
│  {                                                        │
│    "state":    "Raise",                                   │
│    "reps":     3,                                         │
│    "critiques": ["Keep elbow inline with shoulder."],     │
│    "progress": [{"joint": "LEFT_ELBOW", "value": 0.62}]  │
│  }                                                        │
│                                                           │
│  → Update rep counter UI                                  │
│  → Show critique alert if critiques non-empty             │
│  → Animate joint progress arc                            │
└───────────────────────────────────────────────────────────┘
```

### 8.3 Minimal end-to-end snippet

```python
# backend/main.py  (FastAPI example)
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile
from pose_ai_core import PoseEstimator, PoseHeuristics
from pose_ai_core.exercises import EXERCISES

app = FastAPI()

estimator  = PoseEstimator("checkpoints/checkpoint_iter_370000.pth", use_gpu=False)
heuristics = PoseHeuristics(degrees=True)

# One exercise instance per active user session (store in session state)
sessions: dict = {}


@app.post("/session/start")
def start_session(user_id: str, exercise_name: str):
    ExClass = EXERCISES[exercise_name]
    sessions[user_id] = ExClass() if exercise_name != 'bicep_curl' else ExClass(side='left')
    return {"status": "started", "exercise": exercise_name}


@app.post("/session/frame")
async def process_frame(user_id: str, frame: UploadFile):
    exercise = sessions.get(user_id)
    if exercise is None:
        return {"error": "No active session"}

    # Decode uploaded image
    data  = await frame.read()
    arr   = np.frombuffer(data, np.uint8)
    img   = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    poses = estimator.estimate(img)
    if not poses:
        return {"error": "No person detected"}

    heuristics.update(poses[0])
    state, critiques, progress = exercise.update(poses[0], heuristics)

    return {
        "state":    state.label,
        "reps":     exercise.reps,
        "critiques": [c.msg for c in critiques],
        "progress": [
            {"joint": h_id, "value": round(v, 3)}
            for h_id, _kpt, v in progress
        ]
    }


@app.post("/session/reset")
def reset_session(user_id: str):
    if user_id in sessions:
        sessions[user_id].reset()
    return {"status": "reset"}
```

---

## 9. Available Heuristics Reference

| Constant | Joint measured | Notes |
|---|---|---|
| `HEURISTICS.RIGHT_ELBOW` | Right elbow angle | Shoulder → Elbow → Wrist |
| `HEURISTICS.LEFT_ELBOW` | Left elbow angle | |
| `HEURISTICS.AVG_ELBOWS` | Mean of both elbows | |
| `HEURISTICS.RIGHT_SHLDR` | Right shoulder angle | Neck → R_Shoulder → R_Elbow |
| `HEURISTICS.LEFT_SHLDR` | Left shoulder angle | |
| `HEURISTICS.AVG_SHLDRS` | Mean of both shoulders | |
| `HEURISTICS.RIGHT_HIP` | Right hip angle | R_Shoulder → R_Hip → R_Knee |
| `HEURISTICS.LEFT_HIP` | Left hip angle | |
| `HEURISTICS.AVG_HIPS` | Mean of both hips | |
| `HEURISTICS.RIGHT_KNEE` | Right knee angle | R_Hip → R_Knee → R_Ankle |
| `HEURISTICS.LEFT_KNEE` | Left knee angle | |
| `HEURISTICS.AVG_KNEES` | Mean of both knees | |
| `HEURISTICS.RIGHT_ANKLE` | Right ankle angle | Floor point → R_Ankle → R_Knee |
| `HEURISTICS.LEFT_ANKLE` | Left ankle angle | |
| `HEURISTICS.AVG_ANKLES` | Mean of both ankles | |
| `HEURISTICS.SIDE_NECK` | Neck tilt angle | Nose → Neck → midpoint(hips) |

> All angles are returned in **degrees** when `PoseHeuristics(degrees=True)` is used (recommended).

---

## 10. Available Keypoints Reference

See [Section 4.1](#41-pose--keypoints) for the full table.  
Quick reference for exercise authoring:

```python
# Upper body
KEYPOINTS.NECK, KEYPOINTS.NOSE
KEYPOINTS.R_SHO, KEYPOINTS.L_SHO   # shoulders
KEYPOINTS.R_ELB, KEYPOINTS.L_ELB   # elbows
KEYPOINTS.R_WRI, KEYPOINTS.L_WRI   # wrists

# Lower body
KEYPOINTS.R_HIP,  KEYPOINTS.L_HIP
KEYPOINTS.R_KNEE, KEYPOINTS.L_KNEE
KEYPOINTS.R_ANK,  KEYPOINTS.L_ANK
```

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: Checkpoint file not found` | Wrong path to `.pth` file | Double-check the `checkpoint_path` argument to `PoseEstimator` |
| `get_angle()` always returns `None` | Keypoints missing from frame | Ensure the subject is fully visible; lower `conf_thresh` slightly |
| Rep count never increments | `_set_rep_transition` not firing | Print `state.id` each frame — confirm both transition states are actually being reached |
| State machine stuck in `SET_UP` | Setup conditions too strict | Widen angle thresholds in your `_state_set_up` implementation |
| Critique fires every frame | Critique check is too aggressive | Add hysteresis: only fire after the condition is true for N consecutive frames |
| `No person detected` response | Subject out of frame or low light | Ensure subject is fully in frame; improve lighting |
| High CPU on mobile device | Running inference on-device | Move inference to a server; send frames over WebSocket for lower latency |

