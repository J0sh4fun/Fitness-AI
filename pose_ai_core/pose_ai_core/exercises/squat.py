"""
Squat exercise – state machine, rep counting, critique & progress.
"""

from ..pose import Pose, KEYPOINTS
from ..measurements import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from ..exercise import Exercise, ExerciseState, Critique, Progress


class Squat(Exercise):
    """
    Squat exercise (bodyweight or barbell).

    States
    ------
    SET_UP → STAND → SQUAT → STAND → …

    Rep is counted on each SQUAT → STAND transition (full descent + ascent completed).

    Critiques
    ---------
    - asymmetrical_squat : Left and right knees bend at significantly different angles 
                           (> 25 degrees difference).

    Progress
    --------
    Tracks the average knee angle through the lowering and raising phases.
    """

    class STATES:
        SET_UP = 'SET_UP'
        DESCEND = 'DESCEND' # Lowering phase
        ASCEND = 'ASCEND'   # Raising phase
    def __init__(self):
        super().__init__('Squat')

        # 1. Define the states clearly
        self._add_state(ExerciseState(self.STATES.SET_UP, "Set up", self._state_set_up), initial=True)
        self._add_state(ExerciseState(self.STATES.DESCEND, "Lowering", self._state_descend))
        self._add_state(ExerciseState(self.STATES.ASCEND, "Raising", self._state_ascend))

        # 2. Update the rep transition
        # A rep is officially "done" when you finish the ascent and return to a standing/set-up position
        self._set_rep_transition(self.STATES.ASCEND, self.STATES.SET_UP)

        # ── Critiques ─────────────────────────────────────────────────────────
        self._add_critique(Critique(
            'asymmetrical_squat',
            [self.STATES.DESCEND, self.STATES.ASCEND],
            'Ensure your weight is evenly distributed. Your knees are bending unevenly.',
            self._critique_asymmetrical
        ))

        # ── Progress ──────────────────────────────────────────────────────────
        # Tracking progress using the average knee angle. 
        # Standing straight is ~175°, bottom of squat is ~90° or lower.
        
        lower_progress = Progress('lower_squat', [self.STATES.DESCEND])
        lower_progress.add_range(HEURISTICS.AVG_KNEES, KEYPOINTS.L_KNEE, 175, 90)
        self._add_progress(lower_progress)

        raise_progress = Progress('raise_squat', [self.STATES.ASCEND])
        raise_progress.add_range(HEURISTICS.AVG_KNEES, KEYPOINTS.L_KNEE, 90, 175)
        self._add_progress(raise_progress)

# ── Helpers ───────────────────────────────────────────────────────────────

    def _get_active_knee_angle(self, heuristics: PoseHeuristics):
        """
        Safely gets the knee angle. Uses average if both are visible, 
        otherwise falls back to whichever single knee is currently visible.
        Normalizes reflex angles (>180) to interior angles.
        """
        l_knee = heuristics.get_angle(HEURISTICS.LEFT_KNEE)
        r_knee = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)
        
        # Helper to normalize reflex angles (outside of the knee) 
        # into interior angles (inside of the knee)
        def normalize(angle):
            if angle is None:
                return None
            return 360.0 - angle if angle > 180 else angle
            
        l_knee = normalize(l_knee)
        r_knee = normalize(r_knee)
        
        if l_knee is not None and r_knee is not None:
            return (l_knee + r_knee) / 2.0
        elif l_knee is not None:
            return l_knee
        elif r_knee is not None:
            return r_knee
        
        return None

# squat.py

    # ── State functions ───────────────────────────────────────────────────────

    def _state_set_up(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        active_knee = self._get_active_knee_angle(heuristics)
        if active_knee is None: return self.STATES.SET_UP

        # Transition to DESCEND when knees bend past 160°
        if active_knee < 160:
            return self.STATES.DESCEND
        return self.STATES.SET_UP

    def _state_descend(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        active_knee = self._get_active_knee_angle(heuristics)
        if active_knee is None: return self.STATES.DESCEND

        # Transition to ASCEND only after reaching depth (e.g., <= 115°)
        if active_knee <= 115:
            return self.STATES.ASCEND
        
        # Reset to SET_UP if they stand back up without hitting depth
        if active_knee > 165:
            return self.STATES.SET_UP
            
        return self.STATES.DESCEND

    def _state_ascend(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        active_knee = self._get_active_knee_angle(heuristics)
        if active_knee is None: return self.STATES.ASCEND

        # Complete the rep only when standing back up to 165°
        if active_knee >= 165:
            return self.STATES.SET_UP
        
        return self.STATES.ASCEND

    # ── Critique functions ────────────────────────────────────────────────────

    def _critique_asymmetrical(self, pose: Pose, heuristics: PoseHeuristics) -> bool:
        """
        Checks if one leg is taking significantly more load/bending differently 
        than the other.
        """
        l_knee = heuristics.get_angle(HEURISTICS.LEFT_KNEE)
        r_knee = heuristics.get_angle(HEURISTICS.RIGHT_KNEE)

        # If both knees are visible and the difference in their angles is over 25 degrees
        if l_knee and r_knee:
            if abs(l_knee - r_knee) > 25:
                return True
        return False