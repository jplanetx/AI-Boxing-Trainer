#!/usr/bin/env python3
"""
MoveNet Boxing Trainer - Using Google's MoveNet Lightning for superior punch detection
Based on research showing MoveNet outperforms MediaPipe for rapid movements.
"""

import cv2
import time
import numpy as np
import sys
import os

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_AVAILABLE = True
    print("TensorFlow available - MoveNet can be used")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available - will use fallback detection")


class MoveNetBoxingTrainer:
    """Boxing trainer using Google's MoveNet Lightning model."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # Initialize MoveNet model
        self.model = None
        self.use_movenet = False
        self._initialize_movenet()
        
        # Simplified punch tracking
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        self.last_positions = {'left': None, 'right': None}
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.session_start = time.time()
        
        # Recent punches for UI
        self.recent_punches = []
        
        # MoveNet keypoint indices (COCO format)
        self.KEYPOINT_DICT = {
            'nose': 0,
            'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
        print("MoveNet Boxing Trainer initialized")
        if self.use_movenet:
            print("Using MoveNet Lightning model for pose detection")
        else:
            print("Using fallback optical flow detection")
    
    def _initialize_movenet(self):
        """Initialize MoveNet model."""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available - using fallback detection")
            return
            
        try:
            print("Loading MoveNet Lightning model...")
            model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            self.model = hub.load(model_url)
            self.use_movenet = True
            print("MoveNet Lightning loaded successfully!")
        except Exception as e:
            print(f"Failed to load MoveNet: {e}")
            print("Will use fallback optical flow detection")
            self.use_movenet = False
    
    def initialize_camera(self) -> bool:
        """Initialize camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False
                
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def preprocess_frame_for_movenet(self, frame: np.ndarray):
        """Preprocess frame for MoveNet input."""
        # Resize to 192x192 (MoveNet Lightning input size)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = tf.image.resize_with_pad(
            tf.expand_dims(frame_rgb, axis=0), 192, 192
        )
        frame_int32 = tf.cast(frame_resized, dtype=tf.int32)
        return frame_int32
    
    def process_frame_movenet(self, frame: np.ndarray) -> dict:
        """Process frame using MoveNet."""
        if not self.use_movenet or self.model is None:
            return {}
        
        try:
            # Preprocess
            input_image = self.preprocess_frame_for_movenet(frame)
            
            # Run inference
            outputs = self.model.signatures['serving_default'](input_image)
            keypoints = outputs['output_0'].numpy()
            
            # Convert to landmarks dictionary
            landmarks_dict = self._movenet_to_landmarks(keypoints[0, 0, :, :], frame.shape)
            return landmarks_dict
            
        except Exception as e:
            print(f"Error in MoveNet processing: {e}")
            return {}
    
    def _movenet_to_landmarks(self, keypoints: np.ndarray, frame_shape: tuple) -> dict:
        """Convert MoveNet keypoints to landmarks dictionary."""
        height, width = frame_shape[:2]
        landmarks_dict = {}
        
        for name, idx in self.KEYPOINT_DICT.items():
            y, x, confidence = keypoints[idx]
            if confidence > 0.2:  # Lower threshold for better detection
                landmarks_dict[name] = {
                    'x': float(x * width),
                    'y': float(y * height),
                    'z': 0.0,  # MoveNet doesn't provide depth
                    'visibility': float(confidence)
                }
        
        return landmarks_dict
    
    def calculate_elbow_angle(self, landmarks_dict: dict, arm: str) -> float:
        """Calculate elbow angle for punch detection."""
        try:
            shoulder_key = f'{arm}_shoulder'
            elbow_key = f'{arm}_elbow'
            wrist_key = f'{arm}_wrist'
            
            if not all(key in landmarks_dict for key in [shoulder_key, elbow_key, wrist_key]):
                return 0
            
            shoulder = landmarks_dict[shoulder_key]
            elbow = landmarks_dict[elbow_key]
            wrist = landmarks_dict[wrist_key]
            
            # Calculate angle using vectors
            vec1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
            vec2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
            
            # Calculate angle
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            
            return float(angle)
            
        except Exception as e:
            return 0
    
    def detect_punch_simple_angles(self, landmarks_dict: dict, arm: str) -> bool:
        """Simple, reliable punch detection using elbow angles."""
        if not landmarks_dict:
            return False
        
        # Check required landmarks exist
        required_keys = [f'{arm}_shoulder', f'{arm}_elbow', f'{arm}_wrist']
        if not all(key in landmarks_dict for key in required_keys):
            return False
        
        # Check confidence
        for key in required_keys:
            if landmarks_dict[key].get('visibility', 0) < 0.3:
                return False
        
        # Calculate current elbow angle
        current_angle = self.calculate_elbow_angle(landmarks_dict, arm)
        if current_angle == 0:
            return False
        
        # Simple state machine: bent -> extended = punch
        if self.arm_states[arm] == 'unknown':
            self.arm_states[arm] = 'bent' if current_angle < 130 else 'extended'
            return False
        
        # Detect punch: transition from bent to extended
        if self.arm_states[arm] == 'bent' and current_angle > 140:
            # Punch detected!
            self.arm_states[arm] = 'extended'
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            
            # Add to recent punches
            punch_info = {
                'arm': arm,
                'time': time.time(),
                'angle': current_angle,
                'type': 'jab' if arm == 'left' else 'cross'
            }
            self.recent_punches.append(punch_info)
            
            # Keep only last 5 punches
            if len(self.recent_punches) > 5:
                self.recent_punches.pop(0)
            
            print(f"PUNCH DETECTED: {arm.upper()} {punch_info['type']} (angle: {current_angle:.1f}°)")
            return True
        
        # Return to bent position
        elif self.arm_states[arm] == 'extended' and current_angle < 130:
            self.arm_states[arm] = 'bent'
        
        return False
    
    def fallback_optical_flow_detection(self, frame: np.ndarray) -> dict:
        """Fallback detection using optical flow if MoveNet fails."""
        # Simple optical flow-based detection
        # This is a placeholder - would implement Lucas-Kanade or similar
        print("Using optical flow fallback (not implemented)")
        return {}
    
    def update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Draw UI with pose keypoints and punch counts."""
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Header
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        model_name = "MoveNet Lightning" if self.use_movenet else "Optical Flow"
        cv2.putText(frame, f'BOXING TRAINER - {model_name}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Pose status
        pose_status = f"POSE: {len(landmarks_dict)} points" if landmarks_dict else "POSE: NO DETECTION"
        pose_color = (0, 255, 0) if landmarks_dict else (0, 0, 255)
        cv2.putText(frame, pose_status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        # Draw keypoints if available
        if landmarks_dict and self.use_movenet:
            self.draw_keypoints(frame, landmarks_dict)
        
        # Punch counters
        cv2.rectangle(frame, (20, 100), (width-20, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 100), (width-20, 220), (0, 255, 0), 3)
        
        cv2.putText(frame, f'LEFT: {self.punch_counts["left"]}', (40, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 3)
        cv2.putText(frame, f'RIGHT: {self.punch_counts["right"]}', (300, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 3)
        cv2.putText(frame, f'TOTAL: {self.punch_counts["total"]}', (150, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Recent punches
        if self.recent_punches:
            cv2.rectangle(frame, (20, height-100), (width-20, height-20), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, height-100), (width-20, height-20), (255, 255, 0), 2)
            
            cv2.putText(frame, 'RECENT PUNCHES:', (30, height-75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for i, punch in enumerate(self.recent_punches[-3:]):
                age = current_time - punch['time']
                if age < 3:  # Show for 3 seconds
                    text = f"{punch['arm'].upper()} {punch['type'].upper()} ({punch['angle']:.0f}°)"
                    cv2.putText(frame, text, (30, height - 50 + i * 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_keypoints(self, frame: np.ndarray, landmarks_dict: dict):
        """Draw MoveNet keypoints on frame."""
        # Define connections for skeleton
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
        ]
        
        # Draw connections
        for start, end in connections:
            if start in landmarks_dict and end in landmarks_dict:
                start_point = (int(landmarks_dict[start]['x']), int(landmarks_dict[start]['y']))
                end_point = (int(landmarks_dict[end]['x']), int(landmarks_dict[end]['y']))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw keypoints
        for name, landmark in landmarks_dict.items():
            if landmark.get('visibility', 0) > 0.3:
                x, y = int(landmark['x']), int(landmark['y'])
                color = (0, 0, 255) if 'wrist' in name else (0, 255, 0)
                cv2.circle(frame, (x, y), 5, color, -1)
    
    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            return False
        elif key == ord('r'):
            self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
            self.recent_punches = []
            self.arm_states = {'left': 'unknown', 'right': 'unknown'}
            self.session_start = time.time()
            print("RESET: All counts cleared!")
        
        return True
    
    def run(self):
        """Main application loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*60)
        print("MOVENET BOXING TRAINER")
        print("="*60)
        print("Using Google's MoveNet Lightning for pose detection")
        print("Simple elbow angle-based punch detection")
        print("R = Reset counts | Q = Quit")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Process pose using MoveNet or fallback
                if self.use_movenet:
                    landmarks_dict = self.process_frame_movenet(frame)
                else:
                    landmarks_dict = self.fallback_optical_flow_detection(frame)
                
                # Detect punches using simple angle method
                if landmarks_dict:
                    self.detect_punch_simple_angles(landmarks_dict, 'left')
                    self.detect_punch_simple_angles(landmarks_dict, 'right')
                
                # Draw UI
                final_frame = self.draw_ui(frame, landmarks_dict)
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('MoveNet Boxing Trainer', final_frame)
                
                # Handle input
                if not self.handle_keyboard_input():
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        duration = int(time.time() - self.session_start)
        minutes, seconds = divmod(duration, 60)
        
        print(f"\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Duration: {minutes:02d}:{seconds:02d}")
        print(f"Left Punches: {self.punch_counts['left']}")
        print(f"Right Punches: {self.punch_counts['right']}")
        print(f"Total Punches: {self.punch_counts['total']}")
        print("="*50)


def main():
    """Main entry point."""
    try:
        trainer = MoveNetBoxingTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())