"""
pose_ai_core.exercises – built-in exercise implementations.
"""

from ..exercise import Exercise, ExerciseState, Critique, Progress
from .shoulder_press import ShoulderPress
from .bicep_curl import BicepCurl
from .squat import Squat
EXERCISES = {
    'shoulder_press': ShoulderPress,
    'bicep_curl': BicepCurl,
    'squat': Squat
}

__all__ = [
    'Exercise',
    'ExerciseState',
    'Critique',
    'Progress',
    'ShoulderPress',
    'BicepCurl',
    'Squat'
    'EXERCISES',
]

