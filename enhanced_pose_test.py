#!/usr/bin/env python3
"""
Enhanced MediaPipe Boxing Pose Test
Features punch history with visual decay and improved visibility
"""

import cv2
import numpy as np
import time
import sys
import os
from collections import deque

# MediaPipe imports
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

class EnhancedBoxingPoseTest:
    """Enhanced MediaPipe boxing pose test with punch history"""
    
    def __init__(self):
        print("Initializing enhanced boxing pose test...")
        
        # MediaPipe setup
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.mediapipe_pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2
        )
        
        # Camera setup
        self.cap = None
        self.frame_count = 0
        
        # Punch history tracking
        self.punch_history = deque(maxlen=5)  # Last 5 punches
        self.last_punch_type = "guard"
        self.last_punch_time = 0
        
        # Detection results storage
        self.results_log = []
        
    def init_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                
            # Set resolution and frame rate
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, frame = self.cap.read()
            if ret:
                print(f"Camera initialized: {frame.shape}")
                return True
            return False
        except Exception as e:
            print(f"Camera init failed: {e}")
            return False
    
    def process_mediapipe_frame(self, frame):
        """Process frame with MediaPipe"""
        start_time = time.time()
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.mediapipe_pose.process(rgb_frame)
            
            processing_time = (time.time() - start_time) * 1000
            return results, processing_time
            
        except Exception as e:
            print(f"MediaPipe processing error: {e}")
            return None, 0
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """Calculate arm angle for punch detection"""
        try:
            # Convert to vectors
            v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
            v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            return np.degrees(angle)
        except:
            return None
    
    def analyze_mediapipe_results(self, results):
        """Analyze MediaPipe results for boxing accuracy"""
        if not results.pose_landmarks:
            return {}
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get key points
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Calculate angles
            left_angle = self.calculate_arm_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y], 
                [left_wrist.x, left_wrist.y]
            )
            right_angle = self.calculate_arm_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            )
            
            return {
                'left_angle': left_angle,
                'right_angle': right_angle,
                'left_confidence': left_wrist.visibility,
                'right_confidence': right_wrist.visibility,
                'left_wrist': [left_wrist.x, left_wrist.y],
                'right_wrist': [right_wrist.x, right_wrist.y]
            }
        except Exception as e:
            return {}
    
    def detect_punch_and_update_history(self, analysis):
        """Detect punch and update history with timing"""
        if not analysis:
            return "No detection", "none"
        
        left_angle = analysis.get('left_angle', 180)
        right_angle = analysis.get('right_angle', 180)
        left_conf = analysis.get('left_confidence', 0)
        right_conf = analysis.get('right_confidence', 0)
        
        # Punch detection thresholds
        punch_threshold = 140  # degrees
        conf_threshold = 0.7
        current_time = time.time()
        
        left_extended = left_angle and left_angle > punch_threshold and left_conf > conf_threshold
        right_extended = right_angle and right_angle > punch_threshold and right_conf > conf_threshold
        
        # Determine current action
        if left_extended and right_extended:
            current_action = "COMBO"
            action_type = "combo"
        elif left_extended:
            current_action = "LEFT PUNCH"
            action_type = "left"
        elif right_extended:
            current_action = "RIGHT PUNCH"
            action_type = "right"
        else:
            current_action = "GUARD"
            action_type = "guard"
        
        # Add to history only if action changed and it's a punch
        if (action_type != self.last_punch_type and 
            action_type in ["left", "right", "combo"] and
            current_time - self.last_punch_time > 0.3):  # Minimum 300ms between punch logs
            
            self.punch_history.append({
                'action': current_action,
                'time': current_time,
                'type': action_type
            })
            self.last_punch_time = current_time
        
        self.last_punch_type = action_type
        return current_action, action_type
    
    def draw_punch_history(self, frame):
        """Draw punch history sidebar with fading effect"""
        h, w = frame.shape[:2]
        
        # History sidebar background (right side)
        sidebar_width = 200
        cv2.rectangle(frame, (w - sidebar_width, 0), (w, h), (20, 20, 20), -1)
        
        # Title
        cv2.putText(frame, "PUNCH HISTORY", 
                   (w - sidebar_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw history with decay effect
        y_start = 70
        for i, punch_data in enumerate(reversed(list(self.punch_history))):
            age = time.time() - punch_data['time']
            
            # Calculate fade/size based on position (0 = most recent)
            scale = 1.0 - (i * 0.15)  # Scale: 1.0, 0.85, 0.7, 0.55, 0.4
            alpha = 1.0 - (i * 0.2)   # Fade: 1.0, 0.8, 0.6, 0.4, 0.2
            
            if scale < 0.4:  # Skip if too small
                continue
            
            # Color based on punch type
            if punch_data['type'] == 'left':
                color = (0, 255, 255)  # Yellow
            elif punch_data['type'] == 'right':
                color = (255, 0, 255)  # Magenta
            else:  # combo
                color = (0, 255, 0)    # Green
            
            # Apply alpha fade
            faded_color = tuple(int(c * alpha) for c in color)
            
            # Draw punch with size scaling
            font_scale = 0.5 * scale
            thickness = max(1, int(2 * scale))
            
            y_pos = y_start + (i * 40)
            
            # Punch text
            cv2.putText(frame, punch_data['action'], 
                       (w - sidebar_width + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, faded_color, thickness)
            
            # Time ago
            time_ago = f"{age:.1f}s ago"
            cv2.putText(frame, time_ago, 
                       (w - sidebar_width + 15, y_pos + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, 
                       (150, 150, 150), 1)
    
    def draw_current_status(self, frame, current_action, action_type, analysis, processing_time):
        """Draw large, visible current status"""
        h, w = frame.shape[:2]
        sidebar_width = 200
        
        # Main status area background (top left)
        cv2.rectangle(frame, (10, 10), (w - sidebar_width - 20, 150), (0, 0, 0), -1)
        
        # Current action - LARGE and prominent
        color = (0, 255, 0) if action_type in ["left", "right", "combo"] else (255, 255, 255)
        cv2.putText(frame, f"CURRENT: {current_action}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Technical details - smaller
        if analysis:
            cv2.putText(frame, f"Left: {analysis.get('left_angle', 0):.0f}° ({analysis.get('left_confidence', 0):.2f})", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Right: {analysis.get('right_angle', 0):.0f}° ({analysis.get('right_confidence', 0):.2f})", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Processing: {processing_time:.1f}ms", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw wrist markers
        if analysis:
            left_wrist = analysis.get('left_wrist')
            right_wrist = analysis.get('right_wrist')
            if left_wrist:
                x, y = int(left_wrist[0] * w), int(left_wrist[1] * h)
                cv2.circle(frame, (x, y), 12, (0, 255, 255), -1)
                cv2.putText(frame, "L", (x-8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            if right_wrist:
                x, y = int(right_wrist[0] * w), int(right_wrist[1] * h)
                cv2.circle(frame, (x, y), 12, (255, 0, 255), -1)
                cv2.putText(frame, "R", (x-8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    def draw_instructions(self, frame):
        """Draw instructions at bottom"""
        h, w = frame.shape[:2]
        sidebar_width = 200
        
        # Instructions background
        cv2.rectangle(frame, (10, h-60), (w - sidebar_width - 20, h-10), (0, 0, 0), -1)
        
        cv2.putText(frame, "Controls: 's' = save analysis  |  'q' = quit", 
                   (20, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        cv2.putText(frame, "Punch to see history tracking with visual decay", 
                   (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_test(self):
        """Run the enhanced boxing test"""
        if not self.init_camera():
            print("Failed to initialize camera")
            return
        
        print("\n" + "="*70)
        print("ENHANCED BOXING POSE TEST WITH PUNCH HISTORY")
        print("="*70)
        print("FEATURES:")
        print("• Real-time pose detection with large, visible status")
        print("• Punch history sidebar with visual decay")
        print("• Improved wrist markers and visibility")
        print("• No interference between current status and history")
        print("="*70)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process with MediaPipe
                mp_results, mp_time = self.process_mediapipe_frame(frame)
                
                # Analyze results and update history
                analysis = self.analyze_mediapipe_results(mp_results)
                current_action, action_type = self.detect_punch_and_update_history(analysis)
                
                # Draw all components
                self.draw_current_status(frame, current_action, action_type, analysis, mp_time)
                self.draw_punch_history(frame)
                self.draw_instructions(frame)
                
                # Show frame
                cv2.imshow('Enhanced Boxing Pose Test', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save analysis snapshot
                    timestamp = time.strftime("%H:%M:%S")
                    self.results_log.append({
                        'time': timestamp,
                        'current_action': current_action,
                        'analysis': analysis,
                        'processing_time': mp_time,
                        'punch_history': list(self.punch_history)
                    })
                    print(f"Analysis snapshot saved at {timestamp}")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nTest interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*50)
        print("ENHANCED BOXING TEST COMPLETE")
        print("="*50)
        print(f"Frames processed: {self.frame_count}")
        print(f"Total punches logged: {len([p for p in self.punch_history if p['type'] != 'guard'])}")
        print(f"Analysis snapshots saved: {len(self.results_log)}")
        
        if self.results_log:
            processing_times = [r['processing_time'] for r in self.results_log if r['processing_time'] > 0]
            if processing_times:
                print(f"Average processing time: {np.mean(processing_times):.1f}ms")


if __name__ == "__main__":
    test = EnhancedBoxingPoseTest()
    test.run_test()