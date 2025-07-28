"""
Utility functions for pose analysis and mathematical calculations.
Extracted from main.py to make functions reusable across modules.
"""

import numpy as np
from typing import List, Tuple, Union


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """
    Calculate the angle between three points.
    
    Args:
        a: First point [x, y] or [x, y, z]
        b: Vertex point [x, y] or [x, y, z]  
        c: Third point [x, y] or [x, y, z]
    
    Returns:
        Angle in degrees (0-180)
    """
    a = np.array(a)
    b = np.array(b) 
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle


def calculate_distance(point1: List[float], point2: List[float]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point [x, y] or [x, y, z]
        point2: Second point [x, y] or [x, y, z]
    
    Returns:
        Distance between points
    """
    p1 = np.array(point1)
    p2 = np.array(point2)
    return np.linalg.norm(p2 - p1)


def calculate_velocity(positions: List[List[float]], timestamps: List[float]) -> float:
    """
    Calculate average velocity from a series of positions and timestamps.
    
    Args:
        positions: List of [x, y] or [x, y, z] coordinates
        timestamps: Corresponding timestamp for each position
    
    Returns:
        Average velocity in units/second
    """
    if len(positions) < 2 or len(timestamps) < 2:
        return 0.0
    
    total_distance = 0.0
    total_time = timestamps[-1] - timestamps[0]
    
    for i in range(1, len(positions)):
        total_distance += calculate_distance(positions[i-1], positions[i])
    
    return total_distance / total_time if total_time > 0 else 0.0


def normalize_landmarks(landmarks, frame_width: int, frame_height: int) -> List[List[float]]:
    """
    Convert MediaPipe normalized coordinates to pixel coordinates.
    
    Args:
        landmarks: MediaPipe landmark results
        frame_width: Width of the video frame
        frame_height: Height of the video frame
    
    Returns:
        List of [x, y, z] coordinates in pixels
    """
    normalized_points = []
    
    for landmark in landmarks.landmark:
        x = landmark.x * frame_width
        y = landmark.y * frame_height
        z = landmark.z * frame_width  # Z is relative to hip depth
        normalized_points.append([x, y, z])
    
    return normalized_points


def extract_trajectory_vector(positions: List[List[float]]) -> np.ndarray:
    """
    Extract the primary trajectory vector from a series of positions.
    
    Args:
        positions: List of [x, y, z] coordinates
    
    Returns:
        Normalized trajectory vector [dx, dy, dz]
    """
    if len(positions) < 2:
        return np.array([0, 0, 0])
    
    start_pos = np.array(positions[0])
    end_pos = np.array(positions[-1])
    
    trajectory = end_pos - start_pos
    magnitude = np.linalg.norm(trajectory)
    
    return trajectory / magnitude if magnitude > 0 else np.array([0, 0, 0])


def smooth_positions(positions: List[List[float]], window_size: int = 5) -> List[List[float]]:
    """
    Apply moving average smoothing to reduce noise in position data.
    
    Args:
        positions: List of [x, y, z] coordinates  
        window_size: Size of the smoothing window
    
    Returns:
        Smoothed positions
    """
    if len(positions) < window_size:
        return positions
    
    smoothed = []
    positions_array = np.array(positions)
    
    for i in range(len(positions)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(positions), i + window_size // 2 + 1)
        
        smoothed_pos = np.mean(positions_array[start_idx:end_idx], axis=0)
        smoothed.append(smoothed_pos.tolist())
    
    return smoothed


def is_guard_position(elbow_angle: float, shoulder_angle: float, 
                     elbow_threshold: float = 60, shoulder_threshold: float = 45) -> bool:
    """
    Determine if the arm is in a guard position based on joint angles.
    
    Args:
        elbow_angle: Angle at the elbow joint
        shoulder_angle: Angle at the shoulder joint  
        elbow_threshold: Maximum elbow angle for guard position
        shoulder_threshold: Maximum shoulder angle for guard position
    
    Returns:
        True if in guard position
    """
    return elbow_angle < elbow_threshold and shoulder_angle < shoulder_threshold


def is_extended_position(elbow_angle: float, extension_threshold: float = 160) -> bool:
    """
    Determine if the arm is in an extended position.
    
    Args:
        elbow_angle: Angle at the elbow joint
        extension_threshold: Minimum angle for extended position
    
    Returns:
        True if arm is extended
    """
    return elbow_angle > extension_threshold
