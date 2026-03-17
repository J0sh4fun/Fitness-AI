"""
Shoulder Press exercise – state machine, rep counting, critique & progress.
"""

from ..pose import Pose, KEYPOINTS
from ..measurements import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from ..exercise import Exercise, ExerciseState, Critique, Progress


class ShoulderPress(Exercise):
    """
    Overhead Shoulder Press exercise.

    States
    ------
    SET_UP → RAISE → LOWER → RAISE → …

    Rep is counted on each LOWER → RAISE transition.

    Critiques
    ---------
    - lock_elbows : elbow angle > 170° at the top of the press.
    - too_low     : shoulder angle > 220° (arms have dropped too far down).

    Progress
    --------
    Tracks shoulder and elbow angles through the raise and lower phases.
    """

    class STATES:
        START    = 'START'   # New initial/reset state
        SET_UP = 'SET_UP'
        RAISE  = 'UP'
        LOWER  = 'DOWN'

    def __init__(self):
        super().__init__('Shoulder Press')

# ── States ────────────────────────────────────────────────────────────
        # 'START' is now the initial state.
        self._add_state(ExerciseState(self.STATES.START, "Return to Start", self._state_start), initial=True)
        self._add_state(ExerciseState(self.STATES.SET_UP, "Ready", self._state_set_up))
        self._add_state(ExerciseState(self.STATES.RAISE, "Raise", self._state_up))
        self._add_state(ExerciseState(self.STATES.LOWER, "Lower", self._state_down))

        # Rep counts ONLY when transitioning from LOWER back to START
        # This forces the user to complete the full range and hold the bottom.
        self._set_rep_transition(self.STATES.LOWER, self.STATES.START)

        # ── Critiques ─────────────────────────────────────────────────────────
        self._add_critique(Critique(
            'lock_elbows',
            [self.STATES.RAISE],
            'Make sure not to lock your elbows at the top of your press.',
            self._critique_lock_elbows
        ))
        self._add_critique(Critique(
            'too_low',
            [self.STATES.LOWER, self.STATES.RAISE],
            'Your arms should make about a 90 degree angle with your body at the bottom.',
            self._critique_too_low
        ))

        # ── Progress ──────────────────────────────────────────────────────────
        raise_shoulder_progress = Progress('raise_shoulder', [self.STATES.RAISE])
        raise_shoulder_progress.add_range(HEURISTICS.RIGHT_SHLDR, KEYPOINTS.R_SHO, 180, 130)
        raise_shoulder_progress.add_range(HEURISTICS.LEFT_SHLDR,  KEYPOINTS.L_SHO, 180, 130)
        self._add_progress(raise_shoulder_progress)

        lower_shoulder_progress = Progress('lower_shoulder', [self.STATES.LOWER])
        lower_shoulder_progress.add_range(HEURISTICS.RIGHT_SHLDR, KEYPOINTS.R_SHO, 130, 180)
        lower_shoulder_progress.add_range(HEURISTICS.LEFT_SHLDR,  KEYPOINTS.L_SHO, 130, 180)
        self._add_progress(lower_shoulder_progress)

        raise_elbow_progress = Progress('raise_elbow', [self.STATES.RAISE])
        raise_elbow_progress.add_range(HEURISTICS.RIGHT_ELBOW, KEYPOINTS.R_ELB, 90, 155)
        raise_elbow_progress.add_range(HEURISTICS.LEFT_ELBOW,  KEYPOINTS.L_ELB, 90, 155)
        self._add_progress(raise_elbow_progress)

        lower_elbow_progress = Progress('lower_elbow', [self.STATES.LOWER])
        lower_elbow_progress.add_range(HEURISTICS.RIGHT_ELBOW, KEYPOINTS.R_ELB, 155, 90)
        lower_elbow_progress.add_range(HEURISTICS.LEFT_ELBOW,  KEYPOINTS.L_ELB, 155, 90)
        self._add_progress(lower_elbow_progress)

    # ── State functions ───────────────────────────────────────────────────────

    def _state_start(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        """
        Initial state: User must hold the weights at shoulder level 
        to trigger the 'SET_UP' phase.
        """
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        left_shldr  = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        
        # Check if the user is in the rough starting position (arms at 180 degrees to torso)
        if left_shldr and right_shldr:
            if self._in_range(left_shldr, 180, 25) and self._in_range(right_shldr, 180, 25):
                return self.STATES.SET_UP
        return self.STATES.START

    def _state_set_up(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        """
        Verification: User must be COMPLETELY still in the start position
        before they are allowed to 'RAISE'.
        """
        r_wri_mv = heuristics.get_movement(KEYPOINTS.R_WRI)
        l_wri_mv = heuristics.get_movement(KEYPOINTS.L_WRI)
        
        # Must be holding still to proceed to RAISE
        if (r_wri_mv.x == MV_DIRECTIONS.HOLD and r_wri_mv.y == MV_DIRECTIONS.HOLD and
            l_wri_mv.x == MV_DIRECTIONS.HOLD and l_wri_mv.y == MV_DIRECTIONS.HOLD):
            return self.STATES.RAISE
        
        # If they move out of position, drop back to START
        return self.STATES.START
    def _state_up(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        left_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        if left_shldr and right_shldr:
            if right_shldr < 130 and left_shldr < 130:
                r_wri_mv = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_mv = heuristics.get_movement(KEYPOINTS.L_WRI)
                if (r_wri_mv.y == MV_DIRECTIONS.HOLD and
                l_wri_mv.y == MV_DIRECTIONS.HOLD):
                    return self.STATES.LOWER
        return self.STATES.RAISE 
    def _state_down(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        """
        Modified LOWER state: Instead of going straight to RAISE, 
        force a transition to START to finish the rep.
        """
        left_shldr  = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        right_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)

        if left_shldr and right_shldr:
            # If they reached the bottom (arms reset to ~180 degrees)
            if right_shldr > 170 and left_shldr > 170:
                r_wri_mv = heuristics.get_movement(KEYPOINTS.R_WRI)
                l_wri_mv = heuristics.get_movement(KEYPOINTS.L_WRI)
                
                # If they stop moving at the bottom, the rep is done.
                if r_wri_mv.y == MV_DIRECTIONS.HOLD and l_wri_mv.y == MV_DIRECTIONS.HOLD:
                    return self.STATES.START
        return self.STATES.LOWER
    # ── Critique functions ────────────────────────────────────────────────────

    def _critique_lock_elbows(self, pose: Pose, heuristics: PoseHeuristics) -> bool:
        r_elb = heuristics.get_angle(HEURISTICS.RIGHT_ELBOW)
        l_elb = heuristics.get_angle(HEURISTICS.LEFT_ELBOW)
        if r_elb is not None and r_elb > 170:
            return True
        if l_elb is not None and l_elb > 170:
            return True
        return False

    def _critique_too_low(self, pose: Pose, heuristics: PoseHeuristics) -> bool:
        r_shldr = heuristics.get_angle(HEURISTICS.RIGHT_SHLDR)
        l_shldr = heuristics.get_angle(HEURISTICS.LEFT_SHLDR)
        if r_shldr is not None and r_shldr > 220:
            return True
        if l_shldr is not None and l_shldr > 220:
            return True
        return False

