"""
Pose Estimator - Core AI for human pose detection
Uses MobileNet-based architecture for real-time pose estimation
"""

import os
from typing import List, Optional

import cv2
import numpy as np
import torch

from .models.with_mobilenet import PoseEstimationWithMobileNet
from .modules import keypoints, pose as pose_module
from .preprocessing import normalize, pad_width
from .pose import Pose
from .utils import COLORS

HEIGHT_SIZE = 256
STRIDE = 8
UPSAMPLE_RATIO = 4


class PoseEstimator:
    """
    Main class for human pose estimation.

    Features:
    - Real-time pose detection using MobileNet backbone
    - 18 keypoint detection (COCO format)
    - Multi-person detection support
    - GPU acceleration support

    Args:
        checkpoint_path: Path to pre-trained model weights (.pth file)
        height_size: Input height for the network (default: 256)
        use_gpu: Whether to use GPU if available (default: True)
    """

    def __init__(self, checkpoint_path: str, height_size: int = 256, use_gpu: bool = True):
        self.height_size = height_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.net = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._init_net(checkpoint_path)
        else:
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    def _init_net(self, checkpoint_path: str):
        """Initialize the neural network with pre-trained weights."""
        self.net = PoseEstimationWithMobileNet()
        self.net.eval()

        if self.use_gpu:
            self.net = self.net.cuda()

        checkpoint = torch.load(checkpoint_path, map_location='cuda' if self.use_gpu else 'cpu')
        self.net.load_checkpoint(checkpoint)

    def _infer(
        self,
        img: np.ndarray,
        net_input_height_size: int = HEIGHT_SIZE,
        stride: int = STRIDE,
        upsample_ratio: int = UPSAMPLE_RATIO,
        pad_value: tuple = (0, 0, 0),
        img_mean: tuple = (128, 128, 128),
        img_scale: float = 1/256
    ):
        """
        Internal inference method.

        Args:
            img: Input image (BGR format)

        Returns:
            heatmaps: Keypoint confidence maps
            pafs: Part Affinity Fields for limb detection
            scale: Scaling factor used
            pad: Padding applied
        """
        height, width, _ = img.shape
        scale = net_input_height_size / height

        # Resize and normalize image
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, img_mean, img_scale)

        # Pad image to be divisible by stride
        min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

# Convert to tensor and move to GPU
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if self.use_gpu:
            tensor_img = tensor_img.cuda()

        # Run inference with memory optimizations
        with torch.no_grad():
            # Use autocast if on GPU for half-precision (FP16) speedup
            if self.use_gpu:
                with torch.amp.autocast('cuda'):
                    stages_output = self.net(tensor_img)
            else:
                stages_output = self.net(tensor_img)

        # Move only the final stage results back to CPU for OpenCV processing
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().float().numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                             interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().float().numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio,
                         interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad

    def estimate(self, img: np.ndarray, conf_thresh: float = 0.2) -> List[Pose]:
        """
        Estimate poses in an image.

        Args:
            img: Input image (BGR format from OpenCV)
            conf_thresh: Minimum confidence threshold for poses (0-1)

        Returns:
            List of Pose objects detected in the image

        Example:
            >>> estimator = PoseEstimator("model.pth")
            >>> img = cv2.imread("person.jpg")
            >>> poses = estimator.estimate(img)
            >>> for pose in poses:
            ...     pose.draw(img)
        """
        if self.net is None:
            return []

        # Run inference
        heatmaps, pafs, scale, pad = self._infer(img)

        # Extract keypoints from heatmaps
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(pose_module.Pose.num_kpts):
            total_keypoints_num += keypoints.extract_keypoints(
                heatmaps[:, :, kpt_idx],
                all_keypoints_by_type,
                total_keypoints_num
            )

        # Group keypoints into poses
        pose_entries, all_keypoints = keypoints.group_keypoints(
            all_keypoints_by_type,
            pafs,
            demo=True
        )

        # Convert coordinates back to original image space
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * STRIDE / UPSAMPLE_RATIO - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * STRIDE / UPSAMPLE_RATIO - pad[0]) / scale

        # Create Pose objects
        current_poses = []
        color_num = 0

        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue

            pose_keypoints = np.ones((pose_module.Pose.num_kpts, 2), dtype=np.int32) * -1

            for kpt_id in range(pose_module.Pose.num_kpts):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])

            pose = pose_module.Pose(pose_keypoints, pose_entries[n][18], COLORS[color_num])
            color_num = (color_num + 1) % len(COLORS)

            if pose.confidence > conf_thresh:
                current_poses.append(pose)

        return current_poses

    def estimate_video(self, video_path: str, conf_thresh: float = 0.2):
        """
        Generator for estimating poses in a video file.

        Args:
            video_path: Path to video file or camera index (0 for webcam)
            conf_thresh: Minimum confidence threshold

        Yields:
            (frame, poses) tuples for each video frame

        Example:
            >>> for frame, poses in estimator.estimate_video("workout.mp4"):
            ...     for pose in poses:
            ...         pose.draw(frame)
            ...     cv2.imshow("Poses", frame)
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f'Video {video_path} cannot be opened')

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                poses = self.estimate(frame, conf_thresh)
                yield frame, poses
        finally:
            cap.release()

