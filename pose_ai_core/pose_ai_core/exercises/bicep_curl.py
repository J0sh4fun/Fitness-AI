"""
Bicep Curl exercise – state machine, rep counting, critique & progress.
"""

from ..pose import Pose, KEYPOINTS
from ..measurements import PoseHeuristics, HEURISTICS, MV_DIRECTIONS
from ..exercise import Exercise, ExerciseState, Critique, Progress


class BicepCurl(Exercise):
    """
    Bicep Curl exercise (single arm – left or right).

    States
    ------
    SET_UP → RAISE → LOWER → RAISE → …

    Rep is counted on each LOWER → RAISE transition (full raise + lower completed).

    Critiques
    ---------
    - elbow_deviation : elbow drifts away from the shoulder (> 60 px horizontal offset).

    Progress
    --------
    Tracks elbow angle through the raise and lower phases.

    Args:
        side: 'left' (default) or 'right' – which arm to track.
    """

    class STATES:
        SET_UP = 'SET_UP'
        RAISE  = 'RAISE'
        LOWER  = 'LOWER'

    def __init__(self, side: str = 'left'):
        super().__init__('Bicep Curl')

        if side not in ('left', 'right', 'both'):
            raise ValueError("`side` must be 'left', 'right', or 'both'")
        self.side = side

        # ── States ────────────────────────────────────────────────────────────
        self._add_state(
            ExerciseState(self.STATES.SET_UP, "Set up", self._state_set_up),
            initial=True
        )
        self._add_state(ExerciseState(self.STATES.RAISE, "Raise", self._state_raise))
        self._add_state(ExerciseState(self.STATES.LOWER, "Lower", self._state_lower))

        # Rep counts on LOWER → RAISE (arm has completed the full raise + lower cycle)
        self._set_rep_transition(self.STATES.LOWER, self.STATES.RAISE)

        # ── Critiques ─────────────────────────────────────────────────────────
        self._add_critique(Critique(
            'elbow_deviation',
            [self.STATES.RAISE, self.STATES.LOWER],
            'Ensure that you keep your elbow stationary and inline with your shoulder.',
            self._critique_elbow_deviation
        ))

        # ── Progress ──────────────────────────────────────────────────────────
        # Both sides share the same heuristic keys; choose based on self.side
        # at runtime, but we register both directions unconditionally so the
        # correct one is always present for whichever side is selected.
        raise_elbow_progress = Progress('raise_elbow', [self.STATES.RAISE])
        raise_elbow_progress.add_range(HEURISTICS.LEFT_ELBOW,  KEYPOINTS.L_ELB, 175, 15)
        raise_elbow_progress.add_range(HEURISTICS.RIGHT_ELBOW, KEYPOINTS.R_ELB, 175, 15)
        self._add_progress(raise_elbow_progress)

        lower_elbow_progress = Progress('lower_elbow', [self.STATES.LOWER])
        lower_elbow_progress.add_range(HEURISTICS.LEFT_ELBOW,  KEYPOINTS.L_ELB, 15, 175)
        lower_elbow_progress.add_range(HEURISTICS.RIGHT_ELBOW, KEYPOINTS.R_ELB, 15, 175)
        self._add_progress(lower_elbow_progress)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_arms(self, heuristics: PoseHeuristics):
        """Return a list of (shoulder_angle, elbow_angle, wrist_movement) for
        each tracked arm. When side='both', both entries must satisfy conditions."""
        arms = []
        if self.side in ('left', 'both'):
            arms.append((
                heuristics.get_angle(HEURISTICS.LEFT_SHLDR),
                heuristics.get_angle(HEURISTICS.LEFT_ELBOW),
                heuristics.get_movement(KEYPOINTS.L_WRI),
            ))
        if self.side in ('right', 'both'):
            arms.append((
                heuristics.get_angle(HEURISTICS.RIGHT_SHLDR),
                heuristics.get_angle(HEURISTICS.RIGHT_ELBOW),
                heuristics.get_movement(KEYPOINTS.R_WRI),
            ))
        return arms

    # ── State functions ───────────────────────────────────────────────────────

    def _state_set_up(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        arms = self._get_arms(heuristics)
        # Rest position: arm hanging at side, elbow angle ~175° (observed 170–200°)
        if all(
            shldr and elbow and
            self._in_range(elbow, 175, 25) and
            wri_mv.x == MV_DIRECTIONS.HOLD and
            wri_mv.y == MV_DIRECTIONS.HOLD
            for shldr, elbow, wri_mv in arms
        ):
            return self.STATES.RAISE   # arm is at bottom, enter the "ready to curl" state
        return self.STATES.SET_UP

    def _state_raise(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        arms = self._get_arms(heuristics)
        # Fully curled: left elbow ~2–6°; right elbow ~10° OR ~342° (flip wraps near 0°)
        if all(
            elbow and (elbow < 30 or elbow > 330) and
            wri_mv.x == MV_DIRECTIONS.HOLD and
            wri_mv.y == MV_DIRECTIONS.HOLD
            for _, elbow, wri_mv in arms
        ):
            return self.STATES.LOWER
        return self.STATES.RAISE

    def _state_lower(self, pose: Pose, heuristics: PoseHeuristics) -> str:
        arms = self._get_arms(heuristics)
        # Arm returned to the rest/extended position (~175°) after lowering
        if all(
            elbow and self._in_range(elbow, 175, 30) and
            wri_mv.x == MV_DIRECTIONS.HOLD and
            wri_mv.y == MV_DIRECTIONS.HOLD
            for _, elbow, wri_mv in arms
        ):
            return self.STATES.RAISE
        return self.STATES.LOWER

    # ── Critique functions ────────────────────────────────────────────────────

    def _critique_elbow_deviation(self, pose: Pose, heuristics: PoseHeuristics) -> bool:
        pairs = []
        if self.side in ('left', 'both'):
            pairs.append((KEYPOINTS.L_ELB, KEYPOINTS.L_SHO))
        if self.side in ('right', 'both'):
            pairs.append((KEYPOINTS.R_ELB, KEYPOINTS.R_SHO))

        for elb_kpt, shldr_kpt in pairs:
            elb_pos   = pose.keypoints[elb_kpt]
            shldr_pos = pose.keypoints[shldr_kpt]
            if elb_pos[0] == -1 or shldr_pos[0] == -1:
                continue
            if not self._in_range(abs(elb_pos[0] - shldr_pos[0]), 0, 60):
                return True
        return False

