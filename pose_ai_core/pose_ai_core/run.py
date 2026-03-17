import cv2
import click
import logging
from pose_ai_core.estimator import PoseEstimator
from pose_ai_core.utils import VideoReader
from pose_ai_core.measurements import PoseHeuristics, HEURISTICS
from pose_ai_core.exercises import EXERCISES
# Global state for buttons
show_angles = {
    "ELBOW": False,
    "KNEE": False,
    "HIP": False
}

def handle_clicks(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check Elbow Button
        if 10 <= x <= 110 and 10 <= y <= 50:
            show_angles["ELBOW"] = not show_angles["ELBOW"]
        # Check Knee Button
        elif 120 <= x <= 220 and 10 <= y <= 50:
            show_angles["KNEE"] = not show_angles["KNEE"]
        # Check Hip Button
        elif 230 <= x <= 330 and 10 <= y <= 50:
            show_angles["HIP"] = not show_angles["HIP"]

def draw_ui(frame):
    for i, (label, active) in enumerate(show_angles.items()):
        x_start = 10 + (i * 110)
        color = (0, 255, 0) if active else (0, 0, 255)
        cv2.rectangle(frame, (x_start, 10), (x_start + 100, 50), color, -1)
        cv2.putText(frame, label, (x_start + 10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def draw_sidebar(frame, heuristics):
    """Draw a semi-transparent sidebar with real-time angle data."""
    h, w, _ = frame.shape
    sidebar_w = 250
    
    # 1. Create a semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - sidebar_w, 0), (w, h), (0, 0, 0), -1)
    
    # Apply the overlay with 60% transparency
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # 2. Render Header
    cv2.putText(frame, "ANGLES", (w - sidebar_w + 10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    
    # 3. Render all active heuristics
    y_pos = 70
    for key, value in heuristics.heuristics.items():
        # Clean up key names (e.g., RIGHT_ELBOW -> Right Elbow)
        label = key.replace("_", " ").title()
        if value is not None:
            # --- FIX: Handle None values gracefully ---
            display_val = f"{value:.1f}" if value is not None else "N/A"
            color = (0, 255, 0) if value is not None else (100, 100, 100)
            
            cv2.putText(frame, f"{label}:", (w - sidebar_w + 10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, display_val, (w - 70, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 30
@click.group()
def cli():
    pass

@cli.command(name='live_pose')
@click.argument('camera', default="0")
@click.option('--checkpoint', required=True, help='Path to .pth checkpoint')
def live_pose(camera, checkpoint):
    # Initialize Multi-threaded reader
    frame_provider = VideoReader(camera)
    estimator = PoseEstimator(checkpoint_path=checkpoint)

    cv2.namedWindow("Pose AI")
    cv2.setMouseCallback("Pose AI", handle_clicks)

    try:
        # Iterate through the threaded reader
        for frame in frame_provider:
            poses = estimator.estimate(frame)
            draw_ui(frame)

            for pose in poses:
                pose.draw(frame)
                ph = PoseHeuristics(pose, degrees=True)
                
                # Logic to show specific angles
                targets = []
                if show_angles["ELBOW"]: targets.extend([HEURISTICS.RIGHT_ELBOW, HEURISTICS.LEFT_ELBOW])
                if show_angles["KNEE"]: targets.extend([HEURISTICS.RIGHT_KNEE, HEURISTICS.LEFT_KNEE])
                if show_angles["HIP"]: targets.extend([HEURISTICS.RIGHT_HIP, HEURISTICS.LEFT_HIP])
                
                if hasattr(ph, 'draw_specific'):
                    ph.draw_specific(frame, targets)
                else:
                    # Fallback if draw_specific isn't added yet
                    ph.draw(frame)

            cv2.imshow("Pose AI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        frame_provider.stop()
        cv2.destroyAllWindows()

@cli.command(name='live_critique', context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument('exercise_name')
@click.argument('camera', default="0")
@click.option('--checkpoint', required=True, help='Path to .pth checkpoint')
@click.pass_context
def live_critique(ctx, exercise_name, camera, checkpoint):
    """Run live exercise tracking with optional parameters."""
    
    # 1. Map extra arguments (like --side right) into a dictionary
    # ctx.args contains leftover arguments not consumed by Click
    extra_args = {}
    it = iter(ctx.args)
    for arg in it:
        if arg.startswith('--'):
            key = arg.lstrip('-').replace('-', '_')
            try:
                extra_args[key] = next(it)
            except StopIteration:
                extra_args[key] = True  # Handle flags without values

    exercises = EXERCISES
    if exercise_name not in exercises:
        click.echo(f"Error: Exercise '{exercise_name}' not supported.")
        return

    # Initialize components
    frame_provider = VideoReader(camera)
    estimator = PoseEstimator(checkpoint_path=checkpoint)
    ph = PoseHeuristics(degrees=True)   # created once so MovementVector history persists

    # 2. Pass dynamic extra_args to the constructor (e.g., side='right')
    try:
        exercise = exercises[exercise_name](**extra_args)
    except TypeError as e:
        click.echo(f"Error initializing {exercise_name}: {e}")
        return

    click.echo(f"Starting {exercise_name} with parameters: {extra_args}")

    try:
        for frame in frame_provider:
            poses = estimator.estimate(frame)
            if poses:
                pose = poses[0]
                pose.draw(frame)

                ph.update(pose)           # update the persistent instance each frame
                draw_sidebar(frame, ph)

                # Advance state and check form
                states, critiques, items = exercise.update(pose, ph)
                
                # Overlay results
                cv2.putText(frame, f"REPS: {exercise.reps}", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                cv2.putText(frame, f"PHASE: {states.label}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 3. Show Critiques (if any)
                y_offset = 160
                if critiques:
                    cv2.putText(frame, f"CRITIQUE: {len(critiques)}", (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # for critique in list(critiques)[-3:]: # Show last 3
                    #     y_offset += 30
                    #     cv2.putText(frame, f"- {critique.msg}", (20, y_offset), 
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Pose Critique AI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        frame_provider.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    cli()