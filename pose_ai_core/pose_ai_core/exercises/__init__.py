"""
pose_ai_core.exercises – built-in exercise implementations.
"""

from ..exercise import Exercise, ExerciseState, Critique, Progress
from .shoulder_press import ShoulderPress
from .bicep_curl import BicepCurl

EXERCISES = {
    'shoulder_press': ShoulderPress,
    'bicep_curl': BicepCurl,
}

__all__ = [
    'Exercise',
    'ExerciseState',
    'Critique',
    'Progress',
    'ShoulderPress',
    'BicepCurl',
    'EXERCISES',
]

