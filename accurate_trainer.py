#!/usr/bin/env python3
"""
Accurate Boxing Trainer - Better camera settings and spatial punch detection
Fixes left/right confusion during fast combinations
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


class AccurateTrainer:
    """Trainer focused on accuracy and proper left/right detection."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # MediaPipe setup
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4
        )
        
        # Punch tracking with spatial awareness
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.last_punch_time = {'left': 0, 'right': 0}
        self.cooldown_period = 0.6  # Reduced for faster combinations
        
        # Spatial tracking for left/right accuracy
        self.wrist_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10)
        }
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        print("Accurate Boxing Trainer initialized")
        print("Enhanced left/right detection and camera brightness")
        
    def init_camera(self) -> bool:
        """Initialize camera with better brightness settings"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)  # Use DirectShow on Windows
            
            if not self.cap.isOpened():
                # Fallback to default
                self.cap = cv2.VideoCapture(self.camera_id)
                
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False
                
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try multiple brightness approaches
            try:
                # Method 1: Auto exposure with brightness boost
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)       # Increase brightness
                self.cap.set(cv2.CAP_PROP_CONTRAST, 50)         # Increase contrast
                print("Applied brightness boost settings")
                
                # Test and see if we need manual exposure
                time.sleep(1)  # Give camera time to adjust
                ret, test_frame = self.cap.read()
                if ret:
                    avg_brightness = np.mean(cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY))
                    print(f"Current brightness level: {avg_brightness:.1f}")
                    
                    # If still too dark, try manual exposure
                    if avg_brightness < 80:
                        print("Frame too dark, trying manual exposure...")
                        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual
                        self.cap.set(cv2.CAP_PROP_EXPOSURE, -4)         # Brighter exposure
                        time.sleep(0.5)
                        
                        ret, test_frame = self.cap.read()
                        if ret:
                            avg_brightness = np.mean(cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY))
                            print(f"New brightness level: {avg_brightness:.1f}")
                            
            except Exception as e:
                print(f"Camera setting adjustment failed: {e}")
                print("Using default settings")
            
            # Final test
            ret, test_frame = self.cap.read()
            if ret:
                print(f"Camera initialized successfully: {test_frame.shape}")
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
            a = np.array([p1.x, p1.y])
            b = np.array([p2.x, p2.y])
            c = np.array([p3.x, p3.y])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            angle = np.arccos(cosine_angle)
            
            return np.degrees(angle)
        except:
            return None
    
    def update_wrist_history(self, landmarks):
        """Track wrist positions for spatial accuracy"""
        try:
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            if left_wrist.visibility > 0.3:
                self.wrist_history['left'].append((left_wrist.x, left_wrist.y, time.time()))
            
            if right_wrist.visibility > 0.3:
                self.wrist_history['right'].append((right_wrist.x, right_wrist.y, time.time()))
                
        except Exception as e:
            pass
    
    def detect_forward_movement(self, side):
        """Detect if wrist moved forward significantly (punch motion)"""
        if len(self.wrist_history[side]) < 5:
            return False
            
        try:
            # Get recent positions
            recent = list(self.wrist_history[side])[-5:]
            
            # Check for forward movement (decreasing y values in camera space)
            y_positions = [pos[1] for pos in recent]
            
            # Forward movement = y coordinate decreases (toward camera)
            if y_positions[0] - y_positions[-1] > 0.03:  # Moved forward
                return True
                
            return False
        except:
            return False
    
    def detect_punch(self, landmarks):
        """Enhanced punch detection with spatial verification"""
        try:
            current_time = time.time()
            
            # Update wrist tracking
            self.update_wrist_history(landmarks)
            
            # Get landmarks
            left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            
            right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Calculate angles
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            if left_angle is None or right_angle is None:
                return
            
            # Left punch detection
            if ((current_time - self.last_punch_time['left']) > self.cooldown_period and 
                left_angle > 155 and  # Arm extended
                left_wrist.visibility > 0.4 and
                self.detect_forward_movement('left')):
                
                # Spatial verification - left wrist should be on left side of body
                nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                if left_wrist.x > nose.x:  # Left wrist is actually on the left side
                    self.punch_counts['left'] += 1
                    self.punch_counts['total'] += 1
                    self.last_punch_time['left'] = current_time
                    print(f"LEFT PUNCH! (angle: {left_angle:.0f}°) Total: {self.punch_counts['total']}")
            
            # Right punch detection  
            if ((current_time - self.last_punch_time['right']) > self.cooldown_period and
                right_angle > 155 and  # Arm extended
                right_wrist.visibility > 0.4 and
                self.detect_forward_movement('right')):
                
                # Spatial verification - right wrist should be on right side of body
                nose = landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
                if right_wrist.x < nose.x:  # Right wrist is actually on the right side
                    self.punch_counts['right'] += 1
                    self.punch_counts['total'] += 1
                    self.last_punch_time['right'] = current_time
                    print(f"RIGHT PUNCH! (angle: {right_angle:.0f}°) Total: {self.punch_counts['total']}")
                    
        except Exception as e:
            print(f"Error in punch detection: {e}")
    
    def draw_info(self, frame):
        """Draw enhanced information on frame"""
        h, w = frame.shape[:2]
        
        # Background rectangle
        cv2.rectangle(frame, (10, 10), (450, 140), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "ACCURATE BOXING TRAINER", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Punch counts with percentage
        total = self.punch_counts['total']
        if total > 0:
            left_pct = (self.punch_counts['left'] / total) * 100
            right_pct = (self.punch_counts['right'] / total) * 100
        else:
            left_pct = right_pct = 0
            
        cv2.putText(frame, f"LEFT: {self.punch_counts['left']} ({left_pct:.0f}%)", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"RIGHT: {self.punch_counts['right']} ({right_pct:.0f}%)", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"TOTAL: {self.punch_counts['total']}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Spatial tracking: L/R detection improved", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, "Press 'q' to quit, 'r' to reset", 
                   (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """Main training loop"""
        if not self.init_camera():
            return
        
        print("\n" + "="*70)
        print("ACCURATE BOXING TRAINER")
        print("="*70)
        print("IMPROVEMENTS:")
        print("• Enhanced camera brightness settings")
        print("• Spatial left/right verification")
        print("• Forward movement detection")
        print("• Reduced cooldown for fast combinations")
        print("="*70)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Process pose
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = self.pose.process(rgb_frame)
                rgb_frame.flags.writeable = True
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                # Detect punches
                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    self.detect_punch(results.pose_landmarks)
                
                # Draw info
                self.draw_info(frame)
                
                # Show frame
                cv2.imshow('Accurate Boxing Trainer', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
                    print("Counts reset!")
                    
        except KeyboardInterrupt:
            print("\nTraining interrupted")
        except Exception as e:
            print(f"Error during training: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nSession complete!")
        print(f"Left: {self.punch_counts['left']}, Right: {self.punch_counts['right']}")
        print(f"Total: {self.punch_counts['total']}")


if __name__ == "__main__":
    trainer = AccurateTrainer()
    trainer.run()