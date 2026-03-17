"""
Measurements & Heuristics
Angle calculation, movement tracking, and pose analytics.
"""

import logging
import statistics
from statistics import mean
from math import degrees as deg
from typing import Union, Dict, Optional

import cv2
import numpy as np

from .pose import Pose, KEYPOINTS
from .utils import midpoint

# ── Constants ────────────────────────────────────────────────────────────────

MV_HOLD_THRESH = 15   # pixels – movement below this is considered "held"
MV_HISTORY     = 10   # number of frames kept in the MovementVector history


# ── Heuristic name constants ──────────────────────────────────────────────────

class HEURISTICS:
    AVG_HIPS    = "AVG_HIPS"
    RIGHT_HIP   = "RIGHT_HIP"
    LEFT_HIP    = "LEFT_HIP"
    AVG_ANKLES  = "AVG_ANKLES"
    RIGHT_ANKLE = "RIGHT_ANKLE"
    LEFT_ANKLE  = "LEFT_ANKLE"
    AVG_ELBOWS  = "AVG_ELBOWS"
    RIGHT_ELBOW = "RIGHT_ELBOW"
    LEFT_ELBOW  = "LEFT_ELBOW"
    AVG_KNEES   = "AVG_KNEES"
    RIGHT_KNEE  = "RIGHT_KNEE"
    LEFT_KNEE   = "LEFT_KNEE"
    AVG_SHLDRS  = "AVG_SHLDRS"
    RIGHT_SHLDR = "RIGHT_SHLDR"
    LEFT_SHLDR  = "LEFT_SHLDR"
    SIDE_NECK   = "SIDE_NECK"


class MV_DIRECTIONS:
    HOLD  = 'HOLD'
    UP    = 'UP'
    DOWN  = 'DOWN'
    LEFT  = 'LEFT'
    RIGHT = 'RIGHT'


# ── Low-level geometry ────────────────────────────────────────────────────────

def calc_floor_pt(pose: Pose) -> np.ndarray:
    """Return the midpoint between the two ankles (estimated floor contact)."""
    return np.int_(midpoint(pose.keypoints[KEYPOINTS.L_ANK], pose.keypoints[KEYPOINTS.R_ANK]))


def calc_angle(
    pose: Pose,
    kpt1: Union[int, tuple, np.ndarray],
    kpt2: Union[int, tuple, np.ndarray],
    kpt3: Union[int, tuple, np.ndarray],
    degrees: bool = False,
    flip: bool = False
) -> Optional[float]:
    """
    Calculate the angle at kpt2 formed by the vectors kpt1→kpt2 and kpt3→kpt2.

    Args:
        pose: The Pose to read keypoints from.
        kpt1/kpt2/kpt3:
            - int   → use that keypoint from the pose directly.
            - tuple → use the midpoint of the two listed keypoints.
            - ndarray → use the value as-is (e.g. a synthetic floor point).
        degrees: Return degrees instead of radians.
        flip: Mirror the angle around 2π (useful for right-side joints).

    Returns:
        Angle value, or None if a required keypoint is missing.
    """

    def _resolve(kpt):
        if isinstance(kpt, int):
            pt = pose.keypoints[kpt]
            return None if pt[0] == -1 else pt
        elif isinstance(kpt, tuple):
            if pose.keypoints[kpt[0]][0] == -1 or pose.keypoints[kpt[1]][0] == -1:
                return None
            return np.int_(midpoint(pose.keypoints[kpt[0]], pose.keypoints[kpt[1]]))
        elif isinstance(kpt, np.ndarray):
            return kpt
        else:
            raise TypeError(f"Keypoint argument must be int, tuple, or ndarray, got {type(kpt)}")

    pt1 = _resolve(kpt1)
    pt2 = _resolve(kpt2)
    pt3 = _resolve(kpt3)

    if pt1 is None or pt2 is None or pt3 is None:
        return None

    v1 = pt1 - pt2
    v2 = pt3 - pt2

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return None

    cross_product = np.cross(v1, v2)
    dot_product   = np.dot(v1, v2)
    angle = np.arctan2(abs(cross_product), dot_product)

    if cross_product < 0:
        angle = 2 * np.pi - angle

    if flip:
        angle = 2 * np.pi - angle

    return deg(angle) if degrees else angle


# ── Per-joint angle helpers ───────────────────────────────────────────────────

def _right_hip(pose, degrees=False):
    return calc_angle(pose, KEYPOINTS.R_SHO, KEYPOINTS.R_HIP, KEYPOINTS.R_KNEE, degrees, flip=True)

def _left_hip(pose, degrees=False):
    return calc_angle(pose, KEYPOINTS.L_SHO, KEYPOINTS.L_HIP, KEYPOINTS.L_KNEE, degrees)

def _right_ankle(pose, degrees=False):
    return calc_angle(pose, calc_floor_pt(pose), KEYPOINTS.R_ANK, KEYPOINTS.R_KNEE, degrees, flip=True)

def _left_ankle(pose, degrees=False):
    return calc_angle(pose, calc_floor_pt(pose), KEYPOINTS.L_ANK, KEYPOINTS.L_KNEE, degrees)

def _right_elbow(pose, degrees=False):
    return calc_angle(pose, KEYPOINTS.R_SHO, KEYPOINTS.R_ELB, KEYPOINTS.R_WRI, degrees, flip=True)

def _left_elbow(pose, degrees=False):
    return calc_angle(pose, KEYPOINTS.L_SHO, KEYPOINTS.L_ELB, KEYPOINTS.L_WRI, degrees)

def _right_knee(pose, degrees=False):
    return calc_angle(pose, KEYPOINTS.R_HIP, KEYPOINTS.R_KNEE, KEYPOINTS.R_ANK, degrees, flip=True)

def _left_knee(pose, degrees=False):
    return calc_angle(pose, KEYPOINTS.L_HIP, KEYPOINTS.L_KNEE, KEYPOINTS.L_ANK, degrees)

def _right_shldr(pose, degrees=False):
    return calc_angle(pose, KEYPOINTS.NECK, KEYPOINTS.R_SHO, KEYPOINTS.R_ELB, degrees, flip=True)

def _left_shldr(pose, degrees=False):
    return calc_angle(pose, KEYPOINTS.NECK, KEYPOINTS.L_SHO, KEYPOINTS.L_ELB, degrees)

def _side_neck(pose, degrees=False):
    return calc_angle(
        pose,
        KEYPOINTS.NOSE, KEYPOINTS.NECK,
        (KEYPOINTS.L_HIP, KEYPOINTS.R_HIP),
        degrees
    )


# ── DrawPosition ──────────────────────────────────────────────────────────────

class DrawPosition:
    """Stores a relative draw location anchored to one or more keypoints."""

    def __init__(self, kpt_ids, x_offset: int = 0, y_offset: int = 0):
        self._kpt_ids  = kpt_ids
        self._x_offset = x_offset
        self._y_offset = y_offset

    def pos(self, pose: Pose):
        x_pos = mean([pose.keypoints[k][0] for k in self._kpt_ids]) + self._x_offset
        y_pos = mean([pose.keypoints[k][1] for k in self._kpt_ids]) + self._y_offset
        return int(x_pos), int(y_pos)


# ── MovementVector ────────────────────────────────────────────────────────────

class MovementVector:
    """
    Tracks the recent movement direction of a single keypoint across frames.

    Attributes:
        x: current horizontal direction ('HOLD' | 'LEFT' | 'RIGHT')
        y: current vertical direction   ('HOLD' | 'UP'   | 'DOWN')
    """

    def __init__(
        self,
        kpt_id: int,
        hold_thresh: int = MV_HOLD_THRESH,
        len_history: int = MV_HISTORY
    ):
        self._kpt_id      = kpt_id
        self._x_history: list = []
        self._y_history: list = []
        self._hold_thresh = hold_thresh
        self._len_history = len_history

        self.x = MV_DIRECTIONS.HOLD
        self.y = MV_DIRECTIONS.HOLD

    def update(self, pose: Pose) -> bool:
        """
        Update direction state from a new pose frame.

        Returns:
            False if the keypoint was not detected; True otherwise.
        """
        kpt = pose.keypoints[self._kpt_id]
        if kpt[0] == -1:
            return False

        try:
            x_mean = mean(self._x_history)
            y_mean = mean(self._y_history)
        except statistics.StatisticsError:
            x_mean = kpt[0]
            y_mean = kpt[1]

        x_diff = x_mean - kpt[0]
        y_diff = y_mean - kpt[1]

        self.x = (MV_DIRECTIONS.RIGHT if x_diff > 0 and abs(x_diff) > self._hold_thresh
                  else MV_DIRECTIONS.LEFT  if x_diff < 0 and abs(x_diff) > self._hold_thresh
                  else MV_DIRECTIONS.HOLD)

        self.y = (MV_DIRECTIONS.UP   if y_diff > 0 and abs(y_diff) > self._hold_thresh
                  else MV_DIRECTIONS.DOWN if y_diff < 0 and abs(y_diff) > self._hold_thresh
                  else MV_DIRECTIONS.HOLD)

        self._x_history.append(kpt[0])
        self._y_history.append(kpt[1])

        if len(self._x_history) > self._len_history:
            self._x_history.pop(0)
        if len(self._y_history) > self._len_history:
            self._y_history.pop(0)

        return True


# ── PoseHeuristics ────────────────────────────────────────────────────────────

class PoseHeuristics:
    """
    Computes and caches all joint angles and movement vectors for a pose.

    Usage::

        heuristics = PoseHeuristics(degrees=True)
        heuristics.update(pose)

        knee_angle = heuristics.get_angle(HEURISTICS.AVG_KNEES)
        wrist_mv   = heuristics.get_movement(KEYPOINTS.L_WRI)
    """

    # Map of heuristic-name → angle-function
    heuristic_funcs = {
        HEURISTICS.RIGHT_HIP:   _right_hip,
        HEURISTICS.LEFT_HIP:    _left_hip,
        HEURISTICS.RIGHT_ANKLE: _right_ankle,
        HEURISTICS.LEFT_ANKLE:  _left_ankle,
        HEURISTICS.RIGHT_ELBOW: _right_elbow,
        HEURISTICS.LEFT_ELBOW:  _left_elbow,
        HEURISTICS.RIGHT_KNEE:  _right_knee,
        HEURISTICS.LEFT_KNEE:   _left_knee,
        HEURISTICS.RIGHT_SHLDR: _right_shldr,
        HEURISTICS.LEFT_SHLDR:  _left_shldr,
        HEURISTICS.SIDE_NECK:   _side_neck,
    }

    # Map of average-heuristic-name → (left-key, right-key)
    avg_heuristics = {
        HEURISTICS.AVG_ANKLES: (HEURISTICS.RIGHT_ANKLE, HEURISTICS.LEFT_ANKLE),
        HEURISTICS.AVG_ELBOWS: (HEURISTICS.RIGHT_ELBOW, HEURISTICS.LEFT_ELBOW),
        HEURISTICS.AVG_KNEES:  (HEURISTICS.RIGHT_KNEE,  HEURISTICS.LEFT_KNEE),
        HEURISTICS.AVG_HIPS:   (HEURISTICS.RIGHT_HIP,   HEURISTICS.LEFT_HIP),
        HEURISTICS.AVG_SHLDRS: (HEURISTICS.RIGHT_SHLDR, HEURISTICS.LEFT_SHLDR),
    }

    # Visual draw positions for overlays
    heuristics_draw_positions = {
        HEURISTICS.AVG_HIPS:    DrawPosition([KEYPOINTS.R_HIP,  KEYPOINTS.L_HIP]),
        HEURISTICS.RIGHT_HIP:   DrawPosition([KEYPOINTS.R_HIP],  y_offset=-20),
        HEURISTICS.LEFT_HIP:    DrawPosition([KEYPOINTS.L_HIP],  y_offset=-20),
        HEURISTICS.AVG_ANKLES:  DrawPosition([KEYPOINTS.L_ANK,  KEYPOINTS.R_ANK]),
        HEURISTICS.RIGHT_ANKLE: DrawPosition([KEYPOINTS.R_ANK]),
        HEURISTICS.LEFT_ANKLE:  DrawPosition([KEYPOINTS.L_ANK]),
        HEURISTICS.AVG_ELBOWS:  DrawPosition([KEYPOINTS.L_ELB,  KEYPOINTS.R_ELB]),
        HEURISTICS.RIGHT_ELBOW: DrawPosition([KEYPOINTS.R_ELB],  y_offset=-20),
        HEURISTICS.LEFT_ELBOW:  DrawPosition([KEYPOINTS.L_ELB],  y_offset=20),
        HEURISTICS.AVG_KNEES:   DrawPosition([KEYPOINTS.L_KNEE, KEYPOINTS.R_KNEE]),
        HEURISTICS.RIGHT_KNEE:  DrawPosition([KEYPOINTS.R_KNEE]),
        HEURISTICS.LEFT_KNEE:   DrawPosition([KEYPOINTS.L_KNEE]),
        HEURISTICS.AVG_SHLDRS:  DrawPosition([KEYPOINTS.L_SHO,  KEYPOINTS.R_SHO], y_offset=-30),
        HEURISTICS.RIGHT_SHLDR: DrawPosition([KEYPOINTS.R_SHO],  x_offset=-50, y_offset=-20),
        HEURISTICS.LEFT_SHLDR:  DrawPosition([KEYPOINTS.L_SHO],  x_offset=50,  y_offset=20),
        HEURISTICS.SIDE_NECK:   DrawPosition([KEYPOINTS.NECK]),
    }

    def __init__(self, pose: Optional[Pose] = None, degrees: bool = False):
        self._curr_pose = pose
        self.degrees = degrees

        self.heuristics: Dict[str, Optional[float]] = {}
        self.movement_vectors: Dict[int, MovementVector] = {
            kpt: MovementVector(kpt) for kpt in KEYPOINTS.all()
        }

        if pose is not None:
            self.update(pose)

    # ── Internal update methods ───────────────────────────────────────────────

    def _update_heuristics(self, pose: Optional[Pose] = None):
        if pose is None:
            pose = self._curr_pose

        for key, func in self.heuristic_funcs.items():
            val = func(self._curr_pose, self.degrees)
            self.heuristics[key] = val if val is not np.nan else None

        for key, (k1, k2) in self.avg_heuristics.items():
            h1 = self.heuristics.get(k1)
            h2 = self.heuristics.get(k2)
            self.heuristics[key] = mean([h1, h2]) if h1 is not None and h2 is not None else None

    def _update_movement_vectors(self, pose: Optional[Pose] = None):
        if pose is None:
            pose = self._curr_pose
        for mv in self.movement_vectors.values():
            mv.update(pose)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, pose: Pose):
        """Feed a new Pose frame; recomputes all angles and movement vectors."""
        self._curr_pose = pose
        if self._curr_pose is not None:
            self._update_heuristics()
            self._update_movement_vectors()

    def get_angle(self, heuristic_id: str) -> Optional[float]:
        """Return the cached angle for the given heuristic key."""
        return self.heuristics.get(heuristic_id)

    def get_movement(self, kpt_id: int) -> MovementVector:
        """Return the MovementVector for a specific keypoint."""
        return self.movement_vectors[kpt_id]

    def draw(self, img):
        """Overlay all heuristic angles and movement arrows onto an image."""
        for key, val in self.heuristics.items():
            if val is not None:
                dp = self.heuristics_draw_positions.get(key)
                if dp is not None:
                    draw_pos = dp.pos(self._curr_pose)
                    cv2.putText(
                        img, f"{key} {val:.1f}",
                        draw_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255)
                    )

        for kpt_id, mv in self.movement_vectors.items():
            dp = self._curr_pose.keypoints[kpt_id]
            if dp[0] != -1:
                cv2.putText(img, f"{mv.x}", (dp[0], dp[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255))
                cv2.putText(img, f"{mv.y}", (dp[0], dp[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255))
    def draw_specific(self, img, heuristic_keys):
        """Overlay only specific heuristic angles onto the image."""
        for key in heuristic_keys:
            val = self.heuristics.get(key)
            if val is not None:
                dp = self.heuristics_draw_positions.get(key)
                if dp is not None:
                    draw_pos = dp.pos(self._curr_pose)
                    cv2.putText(
                        img, f"{key} {val:.1f}",
                        draw_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

