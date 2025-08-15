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
        """
        Extract 3D landmarks with enhanced depth processing for BlazePose GHUM model.
        Z-coordinate represents relative depth from the camera plane.
        """
        landmarks_3d = {}
        try:
            # First pass: collect all valid landmarks
            valid_landmarks = {}
            for name, landmark_idx in self.key_landmarks.items():
                try:
                    landmark = pose_landmarks.landmark[landmark_idx.value]
                    if landmark.visibility > 0.1:
                        valid_landmarks[name] = landmark
                except (IndexError, AttributeError):
                    continue
            
            # Calculate depth reference from torso center for normalization
            torso_landmarks = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            torso_z_values = []
            for torso_name in torso_landmarks:
                if torso_name in valid_landmarks:
                    torso_z_values.append(valid_landmarks[torso_name].z)
            
            # Use median torso depth as reference point
            reference_z = np.median(torso_z_values) if torso_z_values else 0.0
            
            # Second pass: normalize and store landmarks with enhanced 3D data
            for name, landmark in valid_landmarks.items():
                # Normalize Z relative to torso center for better depth analysis
                normalized_z = (landmark.z - reference_z) * frame_width
                
                landmarks_3d[name] = {
                    'x': float(landmark.x * frame_width),
                    'y': float(landmark.y * frame_height), 
                    'z': float(normalized_z),  # Relative depth from torso plane
                    'raw_z': float(landmark.z * frame_width),  # Original depth
                    'visibility': float(landmark.visibility),
                    'world_x': float(getattr(landmark, 'world_x', landmark.x)),
                    'world_y': float(getattr(landmark, 'world_y', landmark.y)),
                    'world_z': float(getattr(landmark, 'world_z', landmark.z))
                }
                
        except Exception as e:
            print(f"Error extracting 3D landmarks: {e}")
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
    
    def calculate_3d_velocity(self, landmarks_dict: Dict, arm: str) -> Optional[float]:
        """
        Calculate 3D velocity of wrist movement using enhanced depth data.
        Returns velocity in pixels/frame considering x, y, and z components.
        """
        if len(self.landmark_buffer) < 2:
            return None
            
        wrist_key = f'{arm}_wrist'
        if wrist_key not in landmarks_dict:
            return None
            
        # Get current and previous wrist positions
        current_pos = landmarks_dict[wrist_key]
        
        # Find previous frame with valid wrist data
        prev_pos = None
        for i in range(len(self.landmark_buffer) - 2, -1, -1):
            if wrist_key in self.landmark_buffer[i]:
                prev_pos = self.landmark_buffer[i][wrist_key]
                break
                
        if prev_pos is None:
            return None
            
        # Calculate 3D displacement
        dx = current_pos['x'] - prev_pos['x']
        dy = current_pos['y'] - prev_pos['y']
        dz = current_pos['z'] - prev_pos['z']
        
        # 3D velocity magnitude
        velocity_3d = np.sqrt(dx**2 + dy**2 + dz**2)
        return float(velocity_3d)
    
    def get_punch_trajectory_3d(self, landmarks_dict: Dict, arm: str) -> Optional[Dict]:
        """
        Analyze 3D punch trajectory for improved classification.
        Returns trajectory analysis including forward extension and depth change.
        """
        if len(self.landmark_buffer) < 3:
            return None
            
        wrist_key = f'{arm}_wrist'
        shoulder_key = f'{arm}_shoulder'
        
        if wrist_key not in landmarks_dict or shoulder_key not in landmarks_dict:
            return None
            
        # Collect trajectory points
        trajectory_points = []
        for frame_landmarks in self.landmark_buffer:
            if wrist_key in frame_landmarks and shoulder_key in frame_landmarks:
                wrist = frame_landmarks[wrist_key]
                shoulder = frame_landmarks[shoulder_key]
                
                # Calculate relative position from shoulder
                rel_x = wrist['x'] - shoulder['x']
                rel_y = wrist['y'] - shoulder['y']
                rel_z = wrist['z'] - shoulder['z']
                
                trajectory_points.append({
                    'x': rel_x, 'y': rel_y, 'z': rel_z,
                    'timestamp': len(trajectory_points)
                })
        
        if len(trajectory_points) < 3:
            return None
            
        # Analyze trajectory characteristics
        start_point = trajectory_points[0]
        end_point = trajectory_points[-1]
        
        # Forward extension (positive Z indicates forward movement)
        forward_extension = end_point['z'] - start_point['z']
        
        # Lateral movement
        lateral_movement = abs(end_point['x'] - start_point['x'])
        
        # Vertical movement
        vertical_movement = end_point['y'] - start_point['y']
        
        # Calculate trajectory smoothness (less jitter = smoother punch)
        smoothness = self._calculate_trajectory_smoothness(trajectory_points)
        
        return {
            'forward_extension': forward_extension,
            'lateral_movement': lateral_movement,
            'vertical_movement': vertical_movement,
            'smoothness': smoothness,
            'trajectory_length': len(trajectory_points),
            'total_distance_3d': self._calculate_total_3d_distance(trajectory_points)
        }
    
    def _calculate_trajectory_smoothness(self, trajectory_points: List[Dict]) -> float:
        """Calculate trajectory smoothness based on acceleration changes."""
        if len(trajectory_points) < 3:
            return 0.0
            
        accelerations = []
        for i in range(1, len(trajectory_points) - 1):
            prev_point = trajectory_points[i-1]
            curr_point = trajectory_points[i]
            next_point = trajectory_points[i+1]
            
            # Calculate acceleration in 3D
            vel1_x = curr_point['x'] - prev_point['x']
            vel1_y = curr_point['y'] - prev_point['y']
            vel1_z = curr_point['z'] - prev_point['z']
            
            vel2_x = next_point['x'] - curr_point['x']
            vel2_y = next_point['y'] - curr_point['y']
            vel2_z = next_point['z'] - curr_point['z']
            
            acc_x = vel2_x - vel1_x
            acc_y = vel2_y - vel1_y
            acc_z = vel2_z - vel1_z
            
            acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            accelerations.append(acc_magnitude)
        
        # Lower standard deviation = smoother trajectory
        if accelerations:
            smoothness = 1.0 / (1.0 + np.std(accelerations))
            return float(smoothness)
        return 0.0
    
    def _calculate_total_3d_distance(self, trajectory_points: List[Dict]) -> float:
        """Calculate total 3D distance traveled by the trajectory."""
        if len(trajectory_points) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(1, len(trajectory_points)):
            prev = trajectory_points[i-1]
            curr = trajectory_points[i]
            
            dx = curr['x'] - prev['x']
            dy = curr['y'] - prev['y']
            dz = curr['z'] - prev['z']
            
            distance = np.sqrt(dx**2 + dy**2 + dz**2)
            total_distance += distance
            
        return float(total_distance)
    
    def release(self) -> None:
        if hasattr(self, 'pose'):
            self.pose.close()
