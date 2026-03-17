"""
Liftr Pose AI Core - Extracted AI Module
A standalone pose estimation library for fitness applications
"""

from .estimator import PoseEstimator
from .pose import Pose, KEYPOINTS
from .measurements import (
    calc_angle,
    calc_floor_pt,
    HEURISTICS,
    MV_DIRECTIONS,
    MovementVector,
    PoseHeuristics,
    DrawPosition
)
from .exercise import Exercise, ExerciseState, Critique, Progress
from .exercises import ShoulderPress, BicepCurl, EXERCISES

__version__ = "1.0.0"
__all__ = [
    'PoseEstimator',
    'Pose',
    'KEYPOINTS',
    'calc_angle',
    'calc_floor_pt',
    'HEURISTICS',
    'MV_DIRECTIONS',
    'MovementVector',
    'PoseHeuristics',
    'DrawPosition',
    'Exercise',
    'ExerciseState',
    'Critique',
    'Progress',
    'ShoulderPress',
    'BicepCurl',
    'EXERCISES',
]

