import numpy as np
from collections import deque
from typing import Dict, Optional
from enum import Enum


class PunchType(Enum):
    """Enumeration of punch types."""
    JAB = "jab"
    CROSS = "cross" 
    HOOK = "hook"
    UPPERCUT = "uppercut"
    UNKNOWN = "unknown"


class PunchStage(Enum):
    """Enumeration of punch execution stages."""
    GUARD = "guard"
    PUNCHING = "punching"
    RETURNING = "returning"


def normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector of v. If zero-length, returns original v."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute the angle (in degrees) between vectors v1 and v2."""
    v1_u = normalize(v1)
    v2_u = normalize(v2)
    cosine = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


class PunchClassifier:
    """
    Robust punch classifier with dynamic hand assignment,
    torso-normalized trajectory, and adaptive thresholds.
    """
    def __init__(self, thresholds: Optional[Dict[str, float]] = None, 
                 trajectory_buffer_size: int = 30, fps: int = 30):
        # Initialize per-arm trajectory buffers
        self.trajectory_buffers = {
            'left': deque(maxlen=5),
            'right': deque(maxlen=5)
        }
        
        # Punch tracking state
        self.punch_stages = {
            'left': PunchStage.GUARD,
            'right': PunchStage.GUARD
        }
        
        # Statistics tracking
        self.punch_counts = {'left': 0, 'right': 0}
        self.punch_scores = {'left': 0, 'right': 0}
        self.last_punch_types = {'left': PunchType.UNKNOWN, 'right': PunchType.UNKNOWN}
        
        # Default thresholds (can be tuned or passed in)
        if thresholds is None:
            thresholds = {
                'min_velocity_factor': 1.5,
                'min_distance': 0.05,       # meters
                'max_return_angle': 100.0,  # degrees
                'punch_cone_angle': 35.0    # degrees
            }
        self.thresholds: Dict[str, float] = thresholds
        # Dynamic baseline velocity (updated per session)
        self.dynamic_baseline_velocity = None

    def get_active_arm(self, landmarks: Dict[str, np.ndarray]) -> Optional[str]:
        """Determine which arm is currently extending most forward."""
        # Check that all required landmarks are present
        required_keys = ['left_wrist', 'right_wrist', 'left_shoulder', 'right_shoulder', 'nose']
        if not all(key in landmarks for key in required_keys):
            return None
            
        try:
            lw = landmarks['left_wrist']
            rw = landmarks['right_wrist']
            ls = landmarks['left_shoulder']
            rs = landmarks['right_shoulder']
            nose = landmarks['nose']
        except KeyError:
            return None

        # Torso forward vector
        torso_vec = normalize(nose - 0.5 * (ls + rs))
        # Arm extension vectors
        left_vec = normalize(lw - ls)
        right_vec = normalize(rw - rs)
        # Compare projections
        left_score = np.dot(left_vec, torso_vec)
        right_score = np.dot(right_vec, torso_vec)
        return 'left' if left_score > right_score else 'right'

    def get_torso_axis(self, landmarks: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Compute the main torso axis from hips and shoulders."""
        # Check that all required landmarks are present
        required_keys = ['left_hip', 'right_hip', 'left_shoulder', 'right_shoulder']
        if not all(key in landmarks for key in required_keys):
            return None
            
        try:
            lh = landmarks['left_hip']
            rh = landmarks['right_hip']
            ls = landmarks['left_shoulder']
            rs = landmarks['right_shoulder']
        except KeyError:
            return None
        return normalize((rh - lh) + (rs - ls))

    def update_baselines(self):
        """Update dynamic baseline velocity from recent motion."""
        velocities = []
        for arm, buf in self.trajectory_buffers.items():
            if len(buf) >= 2:
                # Euclidean speed between last two points
                v = np.linalg.norm(buf[-1] - buf[-2])
                velocities.append(v)
        if velocities:
            self.dynamic_baseline_velocity = float(np.mean(velocities))

    def is_punch_motion(self, arm: str) -> bool:
        """Check if recent trajectory constitutes a punch."""
        buf = self.trajectory_buffers[arm]
        if len(buf) < 2:
            return False
        start = buf[0]
        end = buf[-1]
        displacement = np.linalg.norm(end - start)
        # Velocity spike
        raw_speed = np.linalg.norm(buf[-1] - buf[-2])
        # Compute dynamic threshold
        if self.dynamic_baseline_velocity:
            speed_thresh = self.thresholds['min_velocity_factor'] * self.dynamic_baseline_velocity
        else:
            speed_thresh = 0.5  # fallback
        # Angle relative to torso
        wrist_vec = end - start
        torso_axis = self.get_torso_axis(self.current_landmarks)
        if torso_axis is None:
            return False
        punch_angle = angle_between(wrist_vec, torso_axis)

        return bool(
            raw_speed > speed_thresh and
            displacement > self.thresholds['min_distance'] and
            punch_angle < self.thresholds['punch_cone_angle']
        )

    def process_frame(self, landmarks: Dict[str, np.ndarray]) -> Optional[Dict[str, any]]:
        """Call this for each frame; returns punch info if detected."""
        # Store for current baseline/angle
        self.current_landmarks = landmarks
        # Determine active arm
        active = self.get_active_arm(landmarks)
        if active is None:
            return None
        # Append new wrist position
        wrist = landmarks.get(f'{active}_wrist')
        if wrist is None:
            return None
        self.trajectory_buffers[active].append(wrist)
        # Update baselines
        self.update_baselines()

        # If punch motion detected
        if self.is_punch_motion(active):
            # Reset buffer to avoid duplicate detection
            self.trajectory_buffers[active].clear()
            return { 'arm': active, 'type': 'punch' }
        return None

    def classify_punch(self, landmarks: Dict[str, np.ndarray], arm: str, training_mode=None) -> tuple:
        """
        Classify punch type for compatibility with main.py.
        Returns (punch_type, count, score) tuple.
        """
        result = self.process_frame(landmarks)
        
        if result and result['arm'] == arm:
            # Increment count and update stage
            self.punch_counts[arm] += 1
            self.punch_stages[arm] = PunchStage.PUNCHING
            self.last_punch_types[arm] = PunchType.UNKNOWN  # Simplified - could add type detection
            
            # Simple scoring based on motion
            self.punch_scores[arm] = min(100, self.punch_scores[arm] + 10)
            
            return (PunchType.UNKNOWN, self.punch_counts[arm], self.punch_scores[arm])
        else:
            # Reset stage to guard if no punch detected
            self.punch_stages[arm] = PunchStage.GUARD
            return (PunchType.UNKNOWN, self.punch_counts[arm], self.punch_scores[arm])

    def get_punch_statistics(self, arm: str) -> Dict[str, any]:
        """Get statistics for an arm."""
        return {
            'count': self.punch_counts[arm],
            'score': int(self.punch_scores[arm]),
            'last_type': self.last_punch_types[arm].value
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.punch_counts = {'left': 0, 'right': 0}
        self.punch_scores = {'left': 0, 'right': 0}
        self.last_punch_types = {'left': PunchType.UNKNOWN, 'right': PunchType.UNKNOWN}
        self.punch_stages = {'left': PunchStage.GUARD, 'right': PunchStage.GUARD}
