"""
Heavy Bag Mode Enhancement for AI Boxing Trainer
Implements angled positioning detection and asymmetric tracking for real-world heavy bag training.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .utils import calculate_angle, calculate_distance


class TrainingMode(Enum):
    """Training mode detection."""
    SHADOWBOXING = "shadowboxing"
    HEAVY_BAG = "heavy_bag"
    UNKNOWN = "unknown"


class HeavyBagOptimizer:
    """
    Specialized algorithms for heavy bag training with angled camera positioning.
    Handles partial occlusion and asymmetric pose visibility.
    """
    
    def __init__(self):
        """Initialize heavy bag optimization parameters."""
        
        # Confidence thresholds for different training modes
        self.confidence_thresholds = {
            TrainingMode.SHADOWBOXING: 0.5,  # Standard threshold
            TrainingMode.HEAVY_BAG: 0.7,     # Higher threshold for reliability
        }
        
        # Minimum landmarks required for each mode
        self.min_landmarks_required = {
            TrainingMode.SHADOWBOXING: {
                'left': ['left_shoulder', 'left_elbow', 'left_wrist'],
                'right': ['right_shoulder', 'right_elbow', 'right_wrist']
            },
            TrainingMode.HEAVY_BAG: {
                'left': ['left_shoulder', 'left_elbow', 'left_wrist'],  # Primary side
                'right': ['right_wrist']  # Minimum for secondary side
            }
        }
        
        # Asymmetric weights for heavy bag mode (favor visible side)
        self.asymmetric_weights = {
            'primary_side': 0.8,    # Heavily weight the visible side
            'secondary_side': 0.2   # Lower weight for partially occluded side
        }
        
        # Training mode detection parameters
        self.mode_detection_buffer = []
        self.mode_buffer_size = 30  # 1 second at 30fps
        self.current_mode = TrainingMode.UNKNOWN
        
        # Landmark interpolation for missing data
        self.landmark_history = {arm: {} for arm in ['left', 'right']}
        self.history_length = 10
    
    def detect_training_mode(self, landmarks_dict: Dict) -> TrainingMode:
        """
        Automatically detect if user is shadowboxing or doing heavy bag work.
        
        Args:
            landmarks_dict: Current frame landmarks
            
        Returns:
            Detected training mode
        """
        if not landmarks_dict:
            return TrainingMode.UNKNOWN
        
        # Calculate visibility score for each side
        left_visibility = self._calculate_side_visibility(landmarks_dict, 'left')
        right_visibility = self._calculate_side_visibility(landmarks_dict, 'right')
        
        # Detect asymmetric visibility (indicates angled positioning)
        visibility_ratio = abs(left_visibility - right_visibility)
        
        # Mode detection logic
        if visibility_ratio > 0.3:  # Significant asymmetry
            mode_vote = TrainingMode.HEAVY_BAG
        elif left_visibility > 0.8 and right_visibility > 0.8:  # Both sides clearly visible
            mode_vote = TrainingMode.SHADOWBOXING
        else:
            mode_vote = TrainingMode.UNKNOWN
        
        # Add to buffer for temporal smoothing
        self.mode_detection_buffer.append(mode_vote)
        if len(self.mode_detection_buffer) > self.mode_buffer_size:
            self.mode_detection_buffer.pop(0)
        
        # Determine mode based on majority vote
        if len(self.mode_detection_buffer) >= 10:
            mode_counts = {}
            for mode in self.mode_detection_buffer:
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
            
            # Update current mode if confident
            most_common_mode = max(mode_counts, key=mode_counts.get)
            if mode_counts[most_common_mode] > len(self.mode_detection_buffer) * 0.6:
                self.current_mode = most_common_mode
        
        return self.current_mode
    
    def _calculate_side_visibility(self, landmarks_dict: Dict, side: str) -> float:
        """Calculate visibility score for one side of the body."""
        if not landmarks_dict:
            return 0.0
            
        key_landmarks = [f'{side}_shoulder', f'{side}_elbow', f'{side}_wrist', f'{side}_hip']
        
        visible_count = 0
        total_confidence = 0
        
        for landmark_name in key_landmarks:
            landmark = landmarks_dict.get(landmark_name)
            if landmark:
                confidence = landmark.get('visibility', 0)
                if confidence > 0.5:
                    visible_count += 1
                    total_confidence += confidence
        
        if visible_count == 0:
            return 0.0
        
        return (visible_count / len(key_landmarks)) * (total_confidence / visible_count)
    
    def filter_landmarks_by_confidence(self, landmarks_dict: Dict, 
                                     training_mode: TrainingMode) -> Dict:
        """
        Filter landmarks based on confidence thresholds for the current training mode.
        
        Args:
            landmarks_dict: Raw landmarks from pose tracker
            training_mode: Current training mode
            
        Returns:
            Filtered landmarks dictionary
        """
        if not landmarks_dict:
            return {}
        
        threshold = self.confidence_thresholds.get(training_mode, 0.5)
        filtered_landmarks = {}
        
        for landmark_name, landmark_data in landmarks_dict.items():
            confidence = landmark_data.get('visibility', 0)
            
            if confidence >= threshold:
                filtered_landmarks[landmark_name] = landmark_data
            else:
                # Try to interpolate from history if available
                interpolated = self._interpolate_landmark(landmark_name, landmark_data)
                if interpolated:
                    filtered_landmarks[landmark_name] = interpolated
        
        # Update landmark history
        self._update_landmark_history(filtered_landmarks)
        
        return filtered_landmarks
    
    def _interpolate_landmark(self, landmark_name: str, current_data: Dict) -> Optional[Dict]:
        """
        Interpolate missing or low-confidence landmarks from recent history.
        
        Args:
            landmark_name: Name of the landmark to interpolate
            current_data: Current (low-confidence) landmark data
            
        Returns:
            Interpolated landmark data or None
        """
        # Determine which side this landmark belongs to
        side = 'left' if 'left' in landmark_name else 'right'
        
        if landmark_name not in self.landmark_history[side]:
            return None
        
        history = self.landmark_history[side][landmark_name]
        if len(history) < 3:  # Need at least 3 points for interpolation
            return None
        
        # Simple linear interpolation from recent positions
        recent_positions = history[-3:]
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(recent_positions)):
            prev_pos = recent_positions[i-1]
            curr_pos = recent_positions[i]
            velocity = [
                curr_pos['x'] - prev_pos['x'],
                curr_pos['y'] - prev_pos['y'],
                curr_pos['z'] - prev_pos['z']
            ]
            velocities.append(velocity)
        
        if velocities:
            avg_velocity = np.mean(velocities, axis=0)
            last_pos = recent_positions[-1]
            
            # Predict next position
            predicted_pos = {
                'x': last_pos['x'] + avg_velocity[0],
                'y': last_pos['y'] + avg_velocity[1], 
                'z': last_pos['z'] + avg_velocity[2],
                'visibility': 0.6  # Medium confidence for interpolated data
            }
            
            return predicted_pos
        
        return None
    
    def _update_landmark_history(self, landmarks_dict: Dict) -> None:
        """Update landmark history for interpolation."""
        for landmark_name, landmark_data in landmarks_dict.items():
            side = 'left' if 'left' in landmark_name else 'right'
            
            if landmark_name not in self.landmark_history[side]:
                self.landmark_history[side][landmark_name] = []
            
            history = self.landmark_history[side][landmark_name]
            history.append({
                'x': landmark_data['x'],
                'y': landmark_data['y'],
                'z': landmark_data['z'],
                'visibility': landmark_data['visibility']
            })
            
            # Keep history within limits
            if len(history) > self.history_length:
                history.pop(0)
    
    def get_primary_side(self, landmarks_dict: Dict) -> str:
        """
        Determine which side has better visibility (primary side for heavy bag mode).
        
        Args:
            landmarks_dict: Current landmarks
            
        Returns:
            'left' or 'right' indicating primary side
        """
        left_visibility = self._calculate_side_visibility(landmarks_dict, 'left')
        right_visibility = self._calculate_side_visibility(landmarks_dict, 'right')
        
        return 'left' if left_visibility > right_visibility else 'right'
    
    def adjust_classification_for_heavy_bag(self, landmarks_dict: Dict, 
                                          primary_side: str) -> Dict:
        """
        Adjust landmark weights for heavy bag mode classification.
        
        Args:
            landmarks_dict: Filtered landmarks
            primary_side: Side with better visibility
            
        Returns:
            Adjusted landmarks with weighted confidence
        """
        if not landmarks_dict:
            return landmarks_dict
        
        adjusted_landmarks = landmarks_dict.copy()
        
        for landmark_name, landmark_data in adjusted_landmarks.items():
            if primary_side in landmark_name:
                # Boost confidence for primary side
                landmark_data = landmark_data.copy()
                landmark_data['visibility'] = min(1.0, 
                    landmark_data['visibility'] * (1 + self.asymmetric_weights['primary_side']))
            else:
                # Reduce confidence for secondary side  
                landmark_data = landmark_data.copy()
                landmark_data['visibility'] *= self.asymmetric_weights['secondary_side']
            
            adjusted_landmarks[landmark_name] = landmark_data
        
        return adjusted_landmarks
    
    def get_setup_guidance(self, landmarks_dict: Dict) -> List[str]:
        """
        Provide setup guidance based on current pose visibility.
        
        Args:
            landmarks_dict: Current landmarks
            
        Returns:
            List of setup suggestions
        """
        guidance = []
        
        if not landmarks_dict:
            guidance.append("⚠️ No pose detected - check camera positioning")
            return guidance
        
        left_vis = self._calculate_side_visibility(landmarks_dict, 'left')
        right_vis = self._calculate_side_visibility(landmarks_dict, 'right')
        
        # Overall visibility check
        if left_vis < 0.4 and right_vis < 0.4:
            guidance.append("📹 Move closer to camera or improve lighting")
        
        # Heavy bag positioning guidance
        if self.current_mode == TrainingMode.HEAVY_BAG:
            if left_vis > right_vis:
                guidance.append("✅ Good positioning for right-handed heavy bag training")
            else:
                guidance.append("🔄 For better accuracy, position so your left side faces the camera")
        
        # Asymmetry detection
        visibility_diff = abs(left_vis - right_vis)
        if visibility_diff > 0.5:
            guidance.append("⚖️ Very asymmetric view - consider adjusting camera angle")
        
        return guidance
