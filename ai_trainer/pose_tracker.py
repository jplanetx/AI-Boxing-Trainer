"""
3D Pose Tracking Engine using MediaPipe BlazePose GHUM.
Upgraded from 2D pose estimation to provide depth information for better punch analysis.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Dict, List, Tuple
from collections import deque

# Explicit imports to resolve Pylance errors
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles

from .utils import smooth_positions
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
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=enable_segmentation,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.smooth_landmarks = smooth_landmarks
        self.landmark_buffer = deque(maxlen=5)
        self.last_valid_landmarks = None
        self.heavy_bag_optimizer = HeavyBagOptimizer()
        self.current_training_mode = TrainingMode.UNKNOWN
        
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
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        landmarks_dict = None
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            landmarks_dict = self._extract_landmarks_3d(
                results.pose_landmarks, frame_width, frame_height
            )
            
            self.current_training_mode = self.heavy_bag_optimizer.detect_training_mode(landmarks_dict)
            landmarks_dict = self.heavy_bag_optimizer.filter_landmarks_by_confidence(
                landmarks_dict, self.current_training_mode
            )
            
            if self.current_training_mode == TrainingMode.HEAVY_BAG:
                primary_side = self.heavy_bag_optimizer.get_primary_side(landmarks_dict)
                landmarks_dict = self.heavy_bag_optimizer.adjust_classification_for_heavy_bag(
                    landmarks_dict, primary_side
                )
            
            if self.smooth_landmarks and landmarks_dict:
                landmarks_dict = self._apply_smoothing(landmarks_dict)
            
            self._draw_landmarks(processed_frame, results.pose_landmarks)
            self.last_valid_landmarks = landmarks_dict
        
        return processed_frame, landmarks_dict
    
    def _extract_landmarks_3d(self, pose_landmarks, frame_width: int, frame_height: int) -> Dict:
        landmarks_3d = {}
        try:
            for name, landmark_idx in self.key_landmarks.items():
                try:
                    landmark = pose_landmarks.landmark[landmark_idx.value]
                    if landmark.visibility > 0.1:
                        landmarks_3d[name] = {
                            'x': float(landmark.x * frame_width),
                            'y': float(landmark.y * frame_height), 
                            'z': float(landmark.z * frame_width),
                            'visibility': float(landmark.visibility)
                        }
                except (IndexError, AttributeError):
                    continue
        except Exception:
            return {}
        return landmarks_3d
    
    def _apply_smoothing(self, landmarks_dict: Dict) -> Dict:
        self.landmark_buffer.append(landmarks_dict)
        if len(self.landmark_buffer) < 3:
            return landmarks_dict
        
        smoothed_landmarks = {}
        for landmark_name in landmarks_dict.keys():
            positions = []
            for frame_landmarks in self.landmark_buffer:
                if landmark_name in frame_landmarks:
                    landmark = frame_landmarks[landmark_name]
                    positions.append([landmark['x'], landmark['y'], landmark['z']])
            
            if positions:
                smoothed_pos_list = smooth_positions(positions, window_size=len(positions))
                smoothed_pos = smoothed_pos_list[-1]
                
                smoothed_landmarks[landmark_name] = {
                    'x': smoothed_pos[0],
                    'y': smoothed_pos[1],
                    'z': smoothed_pos[2],
                    'visibility': landmarks_dict[landmark_name]['visibility']
                }
        return smoothed_landmarks
    
    def _draw_landmarks(self, frame: np.ndarray, pose_landmarks) -> None:
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            list(self.mp_pose.POSE_CONNECTIONS),
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
        )
    
    def get_arm_landmarks(self, landmarks_dict: Dict, arm: str) -> Dict:
        if not landmarks_dict:
            return {}
        
        arm_landmarks = {}
        required_keys = ['shoulder', 'elbow', 'wrist', 'hip']
        for key in required_keys:
            landmark_key = f'{arm}_{key}'
            if landmark_key in landmarks_dict:
                landmark = landmarks_dict[landmark_key]
                if landmark.get('visibility', 0) > 0.3:
                    arm_landmarks[key] = landmark
        return arm_landmarks
    
    def is_pose_detected(self, landmarks_dict: Dict) -> bool:
        if not landmarks_dict:
            return False
        
        key_points = ['left_shoulder', 'right_shoulder', 'left_wrist', 'right_wrist']
        for point in key_points:
            if point not in landmarks_dict or landmarks_dict[point]['visibility'] < 0.5:
                return False
        return True
    
    def get_setup_guidance(self, landmarks_dict: Dict) -> List[str]:
        return self.heavy_bag_optimizer.get_setup_guidance(landmarks_dict)
    
    def get_training_mode(self) -> TrainingMode:
        return self.current_training_mode
    
    def release(self) -> None:
        if hasattr(self, 'pose'):
            self.pose.close()