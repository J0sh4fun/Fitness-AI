"""
Exercise framework – state machine, rep counting, critique & progress tracking.
"""

from typing import List, Dict, Tuple, Optional, Callable
from collections import namedtuple

from .pose import Pose, KEYPOINTS
from .measurements import PoseHeuristics, HEURISTICS, MV_DIRECTIONS


# ── Building blocks ───────────────────────────────────────────────────────────

class Critique:
    """
    A single form-check rule attached to one or more exercise states.

    Args:
        name:  Unique identifier for this critique.
        states: List of state IDs where the check is active.
        msg:   Human-readable feedback message shown when the rule fires.
        func:  Callable(pose, heuristics) → bool.
               Return True when the form problem is detected.
    """

    def __init__(self, name: str, states: List[str], msg: str, func: Callable):
        self.name   = name
        self.states = states
        self.msg    = msg
        self.func   = func

    def __call__(self, pose: Pose, heuristics: PoseHeuristics) -> bool:
        return self.func(pose, heuristics)


class Progress:
    """
    Tracks how far through a movement a joint has travelled.

    Usage::

        p = Progress('raise_elbow', [Exercise.STATES.RAISE])
        p.add_range(HEURISTICS.RIGHT_ELBOW, KEYPOINTS.R_ELB, low=90, high=155)
        results = p.check_progress(heuristics)
        # → [(heuristic_id, keypoint_id, value_0_to_1), ...]
    """

    def __init__(self, name: str, states: List[str]):
        self.name   = name
        self.states = states
        self._ranges: list = []

    def add_range(
        self,
        heuristic_id: str,
        keypoint: int,
        low: float,
        high: float
    ) -> None:
        """Register a joint range: [low°, high°] maps to progress [0, 1]."""
        self._ranges.append((heuristic_id, keypoint, low, high))

    def check_progress(
        self, heuristics: PoseHeuristics
    ) -> List[Tuple[str, int, float]]:
        """Return (heuristic_id, kpt_id, progress_0_to_1) for each range."""
        results = []
        for h_id, kpt_id, low, high in self._ranges:
            val = heuristics.get_angle(h_id)
            if val is None:
                continue
            results.append((h_id, kpt_id, (val - low) / (high - low)))
        return results


ExerciseState = namedtuple('ExerciseState', 'id label func')


# ── Base Exercise ─────────────────────────────────────────────────────────────

class Exercise:
    """
    Abstract base class for a single exercise type.

    Sub-class and:
    1. Define a nested STATES class with string constants.
    2. In __init__ call _add_state(), _add_critique(), _add_progress(),
       and _set_rep_transition().
    3. Implement state-transition functions that return the next state ID.

    Then drive the loop with::

        exercise.update(pose, heuristics)
        # → (current_state, [active_critiques], [progress_tuples])
    """

    class STATES:
        pass

    def __init__(self, name: str):
        self._states: Dict[str, ExerciseState]  = {}
        self._critiques: List[Critique]          = []
        self._progresses: List[Progress]         = []
        self.start_state: Optional[ExerciseState] = None
        self._rep_transition: Optional[Tuple[str, str]] = None

        self.name  = name
        self.state: Optional[ExerciseState] = None
        self.reps  = 0
        self._rep_had_critique = False  # tracks if any critique fired during the current rep

    # ── Registration helpers ──────────────────────────────────────────────────

    def _add_state(self, state: ExerciseState, initial: bool = False):
        if initial:
            self.start_state = state
        self._states[state.id] = state

    def _add_critique(self, critique: Critique):
        self._critiques.append(critique)

    def _add_progress(self, progress: Progress):
        self._progresses.append(progress)

    def _set_rep_transition(self, from_state: str, to_state: str):
        """Define which state transition constitutes one completed rep."""
        self._rep_transition = (from_state, to_state)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_reps(self, curr_state: ExerciseState, next_state: ExerciseState) -> bool:
        if self._rep_transition is None:
            raise RuntimeError("Repetition transition not set – call _set_rep_transition().")
        return (
            self._rep_transition[0] == curr_state.id and
            self._rep_transition[1] == next_state.id
        )

    def _in_range(self, test_val: float, target_val: float, thresh: float) -> bool:
        """True if test_val is within thresh of target_val."""
        return abs(target_val - test_val) < thresh

    # ── Main update loop ──────────────────────────────────────────────────────

    def update(
        self,
        pose: Pose,
        heuristics: PoseHeuristics
    ) -> Tuple[ExerciseState, List[Critique], List[Tuple]]:
        """
        Advance the exercise state machine by one frame.

        Args:
            pose:       Current detected Pose.
            heuristics: PoseHeuristics computed for that pose.

        Returns:
            (current_state, active_critiques, progress_items)
        """
        if self.state is None:
            self.state = self.start_state

        # Collect active critiques
        critiques = [
            c for c in self._critiques
            if self.state.id in c.states and c(pose, heuristics)
        ]

        if critiques:
            self._rep_had_critique = True

        # Collect active progress values
        progress = []
        for p in self._progresses:
            if self.state.id in p.states:
                progress.extend(p.check_progress(heuristics))

        # Advance state
        next_state_id = self._states[self.state.id].func(pose, heuristics)
        next_state    = self._states[next_state_id]
        if next_state != self.state:
            if self._check_reps(self.state, next_state):
                if not self._rep_had_critique:
                    self.reps += 1
                self._rep_had_critique = False  # reset for the next rep
            self.state = next_state

        return self.state, critiques, progress

    def reset(self):
        """Reset state and rep counter back to initial values."""
        self.state = self.start_state
        self.reps  = 0
        self._rep_had_critique = False

