"""
3D Pose Tracking Engine using MediaPipe BlazePose GHUM.
Upgraded from 2D pose estimation to provide depth information for better punch analysis.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Optional, Dict, List, Tuple
from collections import deque

from .utils import normalize_landmarks, smooth_positions
from .heavy_bag_optimizer import HeavyBagOptimizer, TrainingMode


class PoseTracker:
    """
    Advanced 3D pose tracking using MediaPipe BlazePose GHUM model.
    Provides real-time landmark detection with depth information.
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 min_tracking_confidence: float = 0.5,
                 model_complexity: int = 2,
                 enable_segmentation: bool = False,
                 smooth_landmarks: bool = True):
        """
        Initialize the pose tracker with 3D capabilities.
        
        Args:
            min_detection_confidence: Minimum confidence for pose detection
            min_tracking_confidence: Minimum confidence for pose tracking
            model_complexity: 0=Lite, 1=Full, 2=Heavy (more accurate)
            enable_segmentation: Whether to enable pose segmentation
            smooth_landmarks: Whether to apply smoothing to landmarks
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose model with 3D support
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.smooth_landmarks = smooth_landmarks
        self.landmark_buffer = deque(maxlen=5)  # For smoothing
        self.last_valid_landmarks = None
        
        # Heavy bag optimization
        self.heavy_bag_optimizer = HeavyBagOptimizer()
        self.current_training_mode = TrainingMode.UNKNOWN
        
        # Key landmark indices for boxing analysis
        self.key_landmarks = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
            'nose': self.mp_pose.PoseLandmark.NOSE
        }
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Process a video frame and extract 3D pose landmarks.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, landmarks_dict)
        """
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process pose detection
        results = self.pose.process(rgb_frame)
        
        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks if detected
        landmarks_dict = None
        if results.pose_landmarks:
            landmarks_dict = self._extract_landmarks_3d(
                results.pose_landmarks, frame_width, frame_height
            )
            
            # Detect training mode (shadowboxing vs heavy bag)
            self.current_training_mode = self.heavy_bag_optimizer.detect_training_mode(landmarks_dict)
            
            # Filter landmarks by confidence based on training mode
            landmarks_dict = self.heavy_bag_optimizer.filter_landmarks_by_confidence(
                landmarks_dict, self.current_training_mode
            )
            
            # Apply heavy bag optimizations if in heavy bag mode
            if self.current_training_mode == TrainingMode.HEAVY_BAG:
                primary_side = self.heavy_bag_optimizer.get_primary_side(landmarks_dict)
                landmarks_dict = self.heavy_bag_optimizer.adjust_classification_for_heavy_bag(
                    landmarks_dict, primary_side
                )
            
            # Apply smoothing if enabled
            if self.smooth_landmarks and landmarks_dict:
                landmarks_dict = self._apply_smoothing(landmarks_dict)
            
            # Draw pose landmarks on frame
            self._draw_landmarks(processed_frame, results.pose_landmarks)
            
            self.last_valid_landmarks = landmarks_dict
        
        return processed_frame, landmarks_dict
    
    def _extract_landmarks_3d(self, pose_landmarks, frame_width: int, frame_height: int) -> Dict:
        """
        Extract 3D coordinates for key landmarks needed for boxing analysis.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            frame_width: Width of the video frame
            frame_height: Height of the video frame
            
        Returns:
            Dictionary with landmark names and 3D coordinates
        """
        landmarks_3d = {}
        
        try:
            for name, landmark_idx in self.key_landmarks.items():
                try:
                    landmark = pose_landmarks.landmark[landmark_idx.value]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x = landmark.x * frame_width
                    y = landmark.y * frame_height
                    z = landmark.z * frame_width  # Z is relative to hip depth
                    
                    # Only include landmark if it has reasonable visibility
                    if landmark.visibility > 0.1:  # Very permissive threshold
                        landmarks_3d[name] = {
                            'x': float(x),
                            'y': float(y), 
                            'z': float(z),
                            'visibility': float(landmark.visibility)
                        }
                except (IndexError, AttributeError) as e:
                    # Skip this landmark if extraction fails
                    continue
                    
        except Exception as e:
            # Return empty dict if extraction completely fails
            return {}
        
        return landmarks_3d
    
    def _apply_smoothing(self, landmarks_dict: Dict) -> Dict:
        """
        Apply temporal smoothing to reduce landmark jitter.
        
        Args:
            landmarks_dict: Current frame landmarks
            
        Returns:
            Smoothed landmarks dictionary
        """
        self.landmark_buffer.append(landmarks_dict)
        
        if len(self.landmark_buffer) < 3:
            return landmarks_dict
        
        smoothed_landmarks = {}
        
        for landmark_name in landmarks_dict.keys():
            positions = []
            for frame_landmarks in self.landmark_buffer:
                landmark = frame_landmarks[landmark_name]
                positions.append([landmark['x'], landmark['y'], landmark['z']])
            
            # Apply smoothing to positions
            smoothed_positions = smooth_positions(positions, window_size=len(positions))
            smoothed_pos = smoothed_positions[-1]  # Get most recent smoothed position
            
            smoothed_landmarks[landmark_name] = {
                'x': smoothed_pos[0],
                'y': smoothed_pos[1],
                'z': smoothed_pos[2],
                'visibility': landmarks_dict[landmark_name]['visibility']
            }
        
        return smoothed_landmarks
    
    def _draw_landmarks(self, frame: np.ndarray, pose_landmarks) -> None:
        """
        Draw pose landmarks and connections on the frame.
        
        Args:
            frame: Video frame to draw on
            pose_landmarks: MediaPipe pose landmarks
        """
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    def get_arm_landmarks(self, landmarks_dict: Dict, arm: str) -> Dict:
        """
        Extract landmarks for a specific arm (left or right).
        
        Args:
            landmarks_dict: Full landmarks dictionary
            arm: 'left' or 'right'
            
        Returns:
            Dictionary with shoulder, elbow, wrist, hip coordinates (may be empty)
        """
        if not landmarks_dict:
            return {}
        
        arm_landmarks = {}
        
        # Only include landmarks that exist and have reasonable confidence
        required_keys = ['shoulder', 'elbow', 'wrist', 'hip']
        for key in required_keys:
            landmark_key = f'{arm}_{key}'
            if landmark_key in landmarks_dict:
                landmark = landmarks_dict[landmark_key]
                if landmark.get('visibility', 0) > 0.3:  # Reasonable threshold
                    arm_landmarks[key] = landmark
        
        return arm_landmarks
    
    def is_pose_detected(self, landmarks_dict: Dict) -> bool:
        """
        Check if a valid pose is currently detected.
        
        Args:
            landmarks_dict: Landmarks dictionary
            
        Returns:
            True if pose is detected with sufficient confidence
        """
        if not landmarks_dict:
            return False
        
        # Check if key landmarks have good visibility
        key_points = ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']
        
        for point in key_points:
            if point not in landmarks_dict:
                return False
            if landmarks_dict[point]['visibility'] < 0.5:
                return False
        
        return True
    
    def get_setup_guidance(self, landmarks_dict: Dict) -> List[str]:
        """
        Get camera setup guidance for optimal pose tracking.
        
        Args:
            landmarks_dict: Current landmarks
            
        Returns:
            List of setup suggestions
        """
        return self.heavy_bag_optimizer.get_setup_guidance(landmarks_dict)
    
    def get_training_mode(self) -> TrainingMode:
        """Get the current detected training mode."""
        return self.current_training_mode
    
    def release(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
