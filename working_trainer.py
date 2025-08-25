#!/usr/bin/env python3
"""
Working Boxing Trainer - Simple, reliable punch detection
Focus on functionality over complexity
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


class WorkingTrainer:
    """Simple, working trainer focused on reliable punch detection."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # MediaPipe setup - simple settings
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3
        )
        
        # Punch tracking
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        self.last_punch_time = {'left': 0, 'right': 0}
        self.cooldown_period = 0.8  # 800ms between punches per hand
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Frame dimensions
        self.frame_width = 640
        self.frame_height = 480
        
        print("Working Boxing Trainer initialized")
        print("Simple and reliable punch detection")
        
    def init_camera(self) -> bool:
        """Initialize camera with minimal settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False
                
            # Set basic resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Use automatic exposure - let camera handle it
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Automatic
                print("Camera initialized with automatic settings")
            except Exception as e:
                print(f"Using default camera settings: {e}")
            
            # Test capture
            ret, test_frame = self.cap.read()
            if ret:
                print(f"Camera test successful: {test_frame.shape}")
                return True
            else:
                print("Camera test failed")
                return False
                
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle between three points"""
        try:
            # Convert to numpy arrays
            a = np.array([p1.x, p1.y])
            b = np.array([p2.x, p2.y])  # vertex
            c = np.array([p3.x, p3.y])
            
            # Calculate vectors
            ba = a - b
            bc = c - b
            
            # Calculate angle
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except:
            return None
    
    def detect_punch(self, landmarks):
        """Simple punch detection based on elbow angles"""
        try:
            current_time = time.time()
            
            # Get key landmarks
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Check visibility
            if (left_shoulder.visibility < 0.5 or left_elbow.visibility < 0.5 or 
                right_shoulder.visibility < 0.5 or right_elbow.visibility < 0.5):
                return
            
            # Calculate elbow angles
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            if left_angle is None or right_angle is None:
                return
            
            # Simple state machine for each arm
            # Bent arm: angle < 140 degrees
            # Extended arm: angle > 160 degrees
            
            # Left arm
            if (current_time - self.last_punch_time['left']) > self.cooldown_period:
                if self.arm_states['left'] == 'bent' and left_angle > 160:
                    # Punch detected
                    self.punch_counts['left'] += 1
                    self.punch_counts['total'] += 1
                    self.last_punch_time['left'] = current_time
                    self.arm_states['left'] = 'extended'
                    print(f"LEFT PUNCH! Total: {self.punch_counts['total']}")
                elif left_angle < 140:
                    self.arm_states['left'] = 'bent'
                elif left_angle > 160:
                    self.arm_states['left'] = 'extended'
            
            # Right arm
            if (current_time - self.last_punch_time['right']) > self.cooldown_period:
                if self.arm_states['right'] == 'bent' and right_angle > 160:
                    # Punch detected
                    self.punch_counts['right'] += 1
                    self.punch_counts['total'] += 1
                    self.last_punch_time['right'] = current_time
                    self.arm_states['right'] = 'extended'
                    print(f"RIGHT PUNCH! Total: {self.punch_counts['total']}")
                elif right_angle < 140:
                    self.arm_states['right'] = 'bent'
                elif right_angle > 160:
                    self.arm_states['right'] = 'extended'
                    
        except Exception as e:
            print(f"Error in punch detection: {e}")
    
    def draw_info(self, frame):
        """Draw information on frame"""
        # Background rectangle for text
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "WORKING BOXING TRAINER", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Punch counts
        cv2.putText(frame, f"LEFT PUNCHES: {self.punch_counts['left']}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"RIGHT PUNCHES: {self.punch_counts['right']}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"TOTAL: {self.punch_counts['total']}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", 
                   (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main training loop"""
        if not self.init_camera():
            return
        
        print("\\n" + "="*70)
        print("WORKING BOXING TRAINER")
        print("="*70)
        print("INSTRUCTIONS:")
        print("• Stand 3-4 feet from camera")
        print("• Make sure your full upper body is visible")
        print("• Throw punches with clear arm extension")
        print("• Press 'q' to quit, 'r' to reset counts")
        print("="*70)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                
                # Process pose
                results = self.pose.process(rgb_frame)
                
                # Convert back to BGR
                rgb_frame.flags.writeable = True
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Detect punches if pose found
                if results.pose_landmarks:
                    # Draw pose landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    # Detect punches
                    self.detect_punch(results.pose_landmarks)
                
                # Draw info
                self.draw_info(frame)
                
                # Calculate and show FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    fps = 30 / (time.time() - self.start_time)
                    print(f"FPS: {fps:.1f}")
                    self.start_time = time.time()
                
                # Show frame
                cv2.imshow('Working Boxing Trainer', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Reset counts
                    self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
                    self.arm_states = {'left': 'unknown', 'right': 'unknown'}
                    print("Counts reset!")
                    
        except KeyboardInterrupt:
            print("\\nTraining interrupted by user")
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Final stats
        elapsed = time.time() - self.start_time
        print(f"\\nTraining session complete!")
        print(f"Left punches: {self.punch_counts['left']}")
        print(f"Right punches: {self.punch_counts['right']}")
        print(f"Total punches: {self.punch_counts['total']}")
        if elapsed > 0:
            print(f"Average punches per minute: {(self.punch_counts['total'] * 60 / elapsed):.1f}")


if __name__ == "__main__":
    trainer = WorkingTrainer()
    trainer.run()