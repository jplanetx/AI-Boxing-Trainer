#!/usr/bin/env python3
"""
Stable Boxing Trainer - Fixed camera settings and intelligent punch detection
Addresses false positives and camera image quality issues
"""

import cv2
import time
import numpy as np
import sys
import os
from collections import deque

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


class StableTrainer:
    """Stable trainer with fixed camera settings and smart detection."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # MediaPipe setup - balanced settings
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Balanced performance
            enable_segmentation=False,
            min_detection_confidence=0.5,  # Reasonable threshold
            min_tracking_confidence=0.3    # Allow some tracking wobble
        )
        
        # Punch tracking
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        
        # Intelligent detection - track velocity and acceleration
        self.hand_history = {
            'left': deque(maxlen=15),   # 15 frames of history (0.5 seconds)
            'right': deque(maxlen=15)
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.session_start = time.time()
        
        # Recent punches
        self.recent_punches = []
        
        # Proper setup time
        self.setup_phase = True
        self.frames_processed = 0
        self.min_frames_before_detection = 90  # 3 seconds
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_cooldown = 1.0  # 1 second cooldown
        
        print("Stable Boxing Trainer initialized")
        print("Fixed camera settings and intelligent punch detection")
        
    def initialize_camera(self) -> bool:
        """Initialize camera with proper settings to fix image quality."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # FIX CAMERA SETTINGS - remove auto adjustments that cause heat imagery
            try:
                # Reset to defaults/manual control
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
                self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)         # Moderate exposure
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)        # Default brightness
                self.cap.set(cv2.CAP_PROP_CONTRAST, 32)         # Default contrast  
                self.cap.set(cv2.CAP_PROP_SATURATION, 64)       # Default saturation
                self.cap.set(cv2.CAP_PROP_GAIN, 0)              # No gain boost
                
                print("Camera settings optimized for normal image quality")
            except Exception as e:
                print(f"Could not adjust camera settings: {e}")
                print("Using default camera settings")
            
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False
                
            # Test capture to verify settings
            ret, test_frame = self.cap.read()
            if ret:
                print(f"Camera test successful: {test_frame.shape}")
            else:
                print("Camera test failed")
                
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points."""
        a = np.array(a)
        b = np.array(b) 
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def calculate_velocity(self, positions):
        """Calculate velocity from position history."""
        if len(positions) < 3:
            return np.array([0.0, 0.0])
        
        # Use last 3 positions for velocity
        recent = list(positions)[-3:]
        velocities = []
        
        for i in range(1, len(recent)):
            vel = np.array(recent[i]) - np.array(recent[i-1])
            velocities.append(vel)
        
        # Average velocity
        if velocities:
            return np.mean(velocities, axis=0)
        return np.array([0.0, 0.0])
    
    def is_punch_motion(self, arm: str, current_pos: tuple, current_angle: float) -> dict:
        """Intelligent punch detection using motion analysis."""
        # Add current position to history
        self.hand_history[arm].append(current_pos)
        
        if len(self.hand_history[arm]) < 10:
            return {'is_punch': False, 'confidence': 0.0, 'reason': 'insufficient_history'}
        
        # Calculate velocity and acceleration
        velocity = self.calculate_velocity(self.hand_history[arm])
        speed = np.linalg.norm(velocity)
        
        # Check for rapid forward motion (punch characteristics)
        forward_motion = velocity[0] if len(velocity) > 0 else 0  # Assuming camera faces user
        
        # Punch criteria
        criteria = {
            'speed': speed > 15,          # Minimum speed threshold
            'forward': abs(forward_motion) > 8,  # Forward motion component
            'angle_extended': current_angle > 160,  # Arm extended
            'rapid_change': False
        }
        
        # Check for rapid angle change (punch characteristic)
        if len(self.hand_history[arm]) >= 5:
            # Look at angle changes over recent history
            positions = list(self.hand_history[arm])[-5:]
            distances = [np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1])) 
                        for i in range(1, len(positions))]
            max_distance = max(distances) if distances else 0
            criteria['rapid_change'] = max_distance > 25
        
        # Calculate confidence based on criteria
        confidence = sum(criteria.values()) / len(criteria)
        
        # Require multiple criteria for punch detection
        is_punch = sum([criteria['speed'], criteria['forward'], criteria['angle_extended']]) >= 2
        
        return {
            'is_punch': is_punch,
            'confidence': confidence,
            'speed': speed,
            'forward_motion': forward_motion,
            'criteria': criteria,
            'reason': 'motion_analysis'
        }
    
    def extract_landmarks(self, pose_landmarks, frame_width: int, frame_height: int) -> dict:
        """Extract landmarks with reasonable threshold."""
        landmarks_dict = {}
        
        if not pose_landmarks:
            return landmarks_dict
            
        key_landmarks = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
        }
        
        for name, landmark_idx in key_landmarks.items():
            try:
                landmark = pose_landmarks.landmark[landmark_idx.value]
                if landmark.visibility > 0.5:  # Reasonable visibility threshold
                    landmarks_dict[name] = {
                        'x': float(landmark.x * frame_width),
                        'y': float(landmark.y * frame_height),
                        'visibility': float(landmark.visibility)
                    }
            except (IndexError, AttributeError):
                continue
                
        return landmarks_dict
    
    def detect_intelligent_punch(self, landmarks_dict: dict, arm: str) -> bool:
        """Intelligent punch detection with motion analysis."""
        if not landmarks_dict or self.setup_phase:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_punch_time[arm] < self.punch_cooldown:
            return False
        
        # Check required landmarks
        shoulder_key = f'{arm}_shoulder'
        elbow_key = f'{arm}_elbow'
        wrist_key = f'{arm}_wrist'
        
        if not all(key in landmarks_dict for key in [shoulder_key, elbow_key, wrist_key]):
            return False
        
        shoulder = landmarks_dict[shoulder_key]
        elbow = landmarks_dict[elbow_key]
        wrist = landmarks_dict[wrist_key]
        
        # Check visibility
        if any(lm.get('visibility', 0) < 0.5 for lm in [shoulder, elbow, wrist]):
            return False
        
        # Calculate elbow angle
        current_angle = self.calculate_angle(
            [shoulder['x'], shoulder['y']],
            [elbow['x'], elbow['y']],
            [wrist['x'], wrist['y']]
        )
        
        current_pos = (wrist['x'], wrist['y'])
        
        # Use intelligent motion analysis
        motion_result = self.is_punch_motion(arm, current_pos, current_angle)
        
        # Traditional state machine with motion validation
        bent_threshold = 130
        extended_threshold = 160
        
        # State machine
        if self.arm_states[arm] == 'unknown':
            self.arm_states[arm] = 'bent' if current_angle < bent_threshold else 'extended'
            return False
        
        # Detect punch: bent -> extended WITH motion validation
        if (self.arm_states[arm] == 'bent' and 
            current_angle > extended_threshold and 
            motion_result['is_punch'] and 
            motion_result['confidence'] > 0.4):
            
            # Punch detected with motion validation!
            self.arm_states[arm] = 'extended'
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            self.last_punch_time[arm] = current_time
            
            # Simple classification
            punch_type = 'jab' if arm == 'left' else 'straight'
            punch_number = 1 if arm == 'left' else 2
            
            # Check for hook based on motion pattern
            if motion_result['forward_motion'] < 5:  # Less forward motion
                punch_type = 'hook'
                punch_number = 3 if arm == 'left' else 4
            
            # Add to recent punches
            punch_info = {
                'arm': arm,
                'time': current_time,
                'angle': current_angle,
                'type': punch_type,
                'number': punch_number,
                'confidence': motion_result['confidence'],
                'speed': motion_result['speed']
            }
            self.recent_punches.append(punch_info)
            
            if len(self.recent_punches) > 5:
                self.recent_punches.pop(0)
            
            print(f"INTELLIGENT: #{punch_number} {arm.upper()} {punch_type.upper()} "
                  f"(angle:{current_angle:.0f}° | speed:{motion_result['speed']:.1f} | conf:{motion_result['confidence']:.2f})")
            return True
        
        # Return to bent
        elif self.arm_states[arm] == 'extended' and current_angle < bent_threshold:
            self.arm_states[arm] = 'bent'
        
        return False
    
    def update_setup_phase(self):
        """Update setup phase."""
        self.frames_processed += 1
        if self.frames_processed >= self.min_frames_before_detection:
            if self.setup_phase:
                self.setup_phase = False
                print("SETUP COMPLETE - Intelligent detection active!")
    
    def update_fps(self):
        """Update FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_camera_view(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Clean camera view with minimal overlay."""
        height, width = frame.shape[:2]
        
        # Minimal header
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(frame, 'STABLE TRAINER - INTELLIGENT DETECTION', (20, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Setup countdown
        if self.setup_phase:
            countdown = max(0, self.min_frames_before_detection - self.frames_processed)
            setup_seconds = countdown // 30 + 1
            cv2.putText(frame, f'Setup: {setup_seconds}s', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'DETECTION ACTIVE', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Frame quality indicator
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (width - 120, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Simple hand indicators
        if landmarks_dict:
            for arm in ['left', 'right']:
                wrist_key = f'{arm}_wrist'
                if wrist_key in landmarks_dict:
                    wrist = landmarks_dict[wrist_key]
                    hand_pos = (int(wrist['x']), int(wrist['y']))
                    
                    # Green circles for detected hands
                    cv2.circle(frame, hand_pos, 10, (0, 255, 0), 2)
                    cv2.putText(frame, arm[0].upper(), (hand_pos[0] + 15, hand_pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def create_data_window(self) -> np.ndarray:
        """Clean data display."""
        data_window = np.zeros((600, 800, 3), dtype=np.uint8)
        current_time = time.time()
        
        # Title
        cv2.putText(data_window, 'STABLE BOXING TRAINER', (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Session info
        session_duration = int(current_time - self.session_start)
        minutes, seconds = divmod(session_duration, 60)
        cv2.putText(data_window, f'Session: {minutes:02d}:{seconds:02d} | FPS: {self.current_fps:.1f}', 
                   (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Detection method
        cv2.putText(data_window, 'METHOD: Intelligent Motion Analysis + Angle Detection', (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Punch counters
        counter_y = 140
        
        # Left counter
        cv2.rectangle(data_window, (50, counter_y), (350, counter_y + 100), (255, 0, 255), 3)
        cv2.putText(data_window, 'LEFT', (70, counter_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['left']), (150, counter_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 100, 255), 6)
        
        # Right counter
        cv2.rectangle(data_window, (400, counter_y), (700, counter_y + 100), (255, 0, 255), 3)
        cv2.putText(data_window, 'RIGHT', (420, counter_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['right']), (500, counter_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 100), 6)
        
        # Total counter
        total_y = 270
        cv2.rectangle(data_window, (200, total_y), (550, total_y + 120), (255, 255, 255), 4)
        cv2.putText(data_window, 'TOTAL', (220, total_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(data_window, str(self.punch_counts['total']), (320, total_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 8)
        
        # Recent punches with intelligence data
        recent_y = 420
        cv2.putText(data_window, 'RECENT INTELLIGENT DETECTIONS:', (50, recent_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, punch in enumerate(self.recent_punches[-4:]):
            age = current_time - punch['time']
            if age < 10:
                y_pos = recent_y + 30 + (i * 25)
                speed = punch.get('speed', 0)
                confidence = punch.get('confidence', 0)
                text = f"#{punch['number']} {punch['arm'].upper()} {punch['type'].upper()} (spd:{speed:.1f} conf:{confidence:.2f})"
                cv2.putText(data_window, text, (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Features
        cv2.putText(data_window, 'FEATURES: Motion Analysis | Velocity Tracking | False Positive Reduction', 
                   (50, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(data_window, 'R=Reset | Q=Quit', (50, 590), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return data_window
    
    def process_frame(self, frame: np.ndarray):
        """Process frame with stable settings."""
        frame_height, frame_width = frame.shape[:2]
        
        # Pose detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        landmarks_dict = {}
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            landmarks_dict = self.extract_landmarks(
                results.pose_landmarks, frame_width, frame_height
            )
            self.mp_drawing.draw_landmarks(
                processed_frame, results.pose_landmarks,
                list(self.mp_pose.POSE_CONNECTIONS)
            )
        
        return processed_frame, landmarks_dict
    
    def handle_keyboard_input(self) -> bool:
        """Handle input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
            self.recent_punches = []
            self.arm_states = {'left': 'unknown', 'right': 'unknown'}
            self.hand_history = {'left': deque(maxlen=15), 'right': deque(maxlen=15)}
            self.session_start = time.time()
            print("RESET: All counts and tracking history cleared!")
        
        return True
    
    def run(self):
        """Main loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*70)
        print("STABLE BOXING TRAINER")
        print("="*70)
        print("FIXES APPLIED:")
        print("• Fixed camera settings to prevent 'heat imagery' effect")
        print("• Intelligent motion analysis to reduce false positives")
        print("• Velocity and acceleration tracking")
        print("• Higher confidence thresholds for punch detection")
        print("• Motion pattern validation")
        print("")
        print("DETECTION METHOD:")
        print("• Requires rapid forward motion + arm extension")
        print("• Validates motion patterns before counting")
        print("• 1-second cooldown between punches")
        print("• Confidence scoring for each detection")
        print("")
        print("CONTROLS:")
        print("• R = Reset all counts and history")
        print("• Q = Quit")
        print("="*70 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.update_setup_phase()
                processed_frame, landmarks_dict = self.process_frame(frame)
                
                if landmarks_dict:
                    self.detect_intelligent_punch(landmarks_dict, 'left')
                    self.detect_intelligent_punch(landmarks_dict, 'right')
                
                camera_display = self.draw_camera_view(processed_frame, landmarks_dict)
                data_display = self.create_data_window()
                
                self.update_fps()
                
                cv2.imshow('Stable Camera', camera_display)
                cv2.imshow('Stable Data', data_display)
                cv2.moveWindow('Stable Camera', 50, 50)
                cv2.moveWindow('Stable Data', 700, 50)
                
                if not self.handle_keyboard_input():
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """Main entry point."""
    try:
        trainer = StableTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())