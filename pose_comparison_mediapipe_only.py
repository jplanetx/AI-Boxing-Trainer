#!/usr/bin/env python3
"""
MediaPipe Pose Detection Test
Test MediaPipe accuracy for boxing movements (TensorFlow-free version)
"""

import cv2
import numpy as np
import time
import sys
import os

# MediaPipe imports
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

class MediaPipePoseTest:
    """Test MediaPipe pose detection for boxing accuracy"""
    
    def __init__(self):
        print("Initializing MediaPipe pose detection test...")
        
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
            
            # Calculate wrist speeds (simple approach)
            wrist_positions = {
                'left_wrist': [left_wrist.x, left_wrist.y],
                'right_wrist': [right_wrist.x, right_wrist.y]
            }
            
            return {
                'left_angle': left_angle,
                'right_angle': right_angle,
                'left_confidence': left_wrist.visibility,
                'right_confidence': right_wrist.visibility,
                'left_wrist': [left_wrist.x, left_wrist.y],
                'right_wrist': [right_wrist.x, right_wrist.y],
                'wrist_positions': wrist_positions
            }
        except Exception as e:
            return {}
    
    def detect_punch_potential(self, analysis):
        """Simple punch detection logic"""
        if not analysis:
            return "No detection", "none"
        
        left_angle = analysis.get('left_angle', 180)
        right_angle = analysis.get('right_angle', 180)
        left_conf = analysis.get('left_confidence', 0)
        right_conf = analysis.get('right_confidence', 0)
        
        # Simple punch detection based on arm extension
        punch_threshold = 140  # degrees - more extended = potential punch
        conf_threshold = 0.7
        
        left_extended = left_angle and left_angle > punch_threshold and left_conf > conf_threshold
        right_extended = right_angle and right_angle > punch_threshold and right_conf > conf_threshold
        
        if left_extended and right_extended:
            return "Both arms extended", "combo"
        elif left_extended:
            return "Left punch detected", "left"
        elif right_extended:
            return "Right punch detected", "right"
        else:
            return "Guard position", "guard"
    
    def draw_analysis(self, frame, analysis, processing_time):
        """Draw analysis results on frame"""
        h, w = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (10, 10), (w-10, 250), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "MEDIAPIPE BOXING POSE TEST", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_pos = 60
        
        # Processing time
        cv2.putText(frame, f"Processing: {processing_time:.1f}ms", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 30
        
        if analysis:
            # Arm angles
            cv2.putText(frame, f"Left arm: {analysis.get('left_angle', 0):.0f}° (conf: {analysis.get('left_confidence', 0):.2f})", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 25
            cv2.putText(frame, f"Right arm: {analysis.get('right_angle', 0):.0f}° (conf: {analysis.get('right_confidence', 0):.2f})", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 35
            
            # Punch detection
            punch_status, punch_type = self.detect_punch_potential(analysis)
            color = (0, 255, 0) if punch_type != "none" else (255, 255, 255)
            cv2.putText(frame, f"Status: {punch_status}", 
                       (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw wrist positions
            left_wrist = analysis.get('left_wrist')
            right_wrist = analysis.get('right_wrist')
            if left_wrist:
                x, y = int(left_wrist[0] * w), int(left_wrist[1] * h)
                cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
                cv2.putText(frame, "L", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            if right_wrist:
                x, y = int(right_wrist[0] * w), int(right_wrist[1] * h)
                cv2.circle(frame, (x, y), 8, (255, 0, 255), -1)
                cv2.putText(frame, "R", (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        else:
            cv2.putText(frame, "No pose detected", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Instructions
        y_pos = h - 80
        cv2.putText(frame, "Instructions:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_pos += 20
        cv2.putText(frame, "• Perform boxing movements", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 15
        cv2.putText(frame, "• Press 's' to save analysis", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_pos += 15
        cv2.putText(frame, "• Press 'q' to quit", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run_test(self):
        """Run the MediaPipe test"""
        if not self.init_camera():
            print("Failed to initialize camera")
            return
        
        print("\n" + "="*60)
        print("MEDIAPIPE BOXING POSE DETECTION TEST")
        print("="*60)
        print("INSTRUCTIONS:")
        print("• Perform various boxing movements")
        print("• Observe pose detection accuracy")
        print("• Press 's' to save current analysis")
        print("• Press 'q' to quit")
        print("="*60)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process with MediaPipe
                mp_results, mp_time = self.process_mediapipe_frame(frame)
                
                # Analyze results
                analysis = self.analyze_mediapipe_results(mp_results)
                
                # Draw analysis
                self.draw_analysis(frame, analysis, mp_time)
                
                # Show frame
                cv2.imshow('MediaPipe Boxing Pose Test', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save analysis snapshot
                    timestamp = time.strftime("%H:%M:%S")
                    self.results_log.append({
                        'time': timestamp,
                        'analysis': analysis,
                        'processing_time': mp_time
                    })
                    print(f"Analysis saved at {timestamp}")
                
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
        print("MEDIAPIPE TEST COMPLETE")
        print("="*50)
        print(f"Frames processed: {self.frame_count}")
        print(f"Analysis snapshots saved: {len(self.results_log)}")
        
        if self.results_log:
            # Calculate averages
            processing_times = [r['processing_time'] for r in self.results_log if r['processing_time'] > 0]
            
            if processing_times:
                print(f"Average processing time: {np.mean(processing_times):.1f}ms")
                print(f"Max processing time: {np.max(processing_times):.1f}ms")
                print(f"Min processing time: {np.min(processing_times):.1f}ms")


if __name__ == "__main__":
    test = MediaPipePoseTest()
    test.run_test()