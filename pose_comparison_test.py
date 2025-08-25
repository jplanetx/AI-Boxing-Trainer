#!/usr/bin/env python3
"""
Pose Detection Comparison Test
Compare MediaPipe vs MoveNet accuracy for boxing movements
"""

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import time
import sys
import os

# MediaPipe imports
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

class PoseComparisonTest:
    """Test different pose detection models for boxing accuracy"""
    
    def __init__(self):
        print("Initializing pose detection comparison test...")
        
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
        
        # MoveNet setup
        try:
            print("Loading MoveNet Lightning...")
            model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            self.movenet_model = hub.load(model_url)
            self.movenet = self.movenet_model.signatures['serving_default']
            print("MoveNet Lightning loaded successfully!")
            self.movenet_available = True
        except Exception as e:
            print(f"MoveNet failed to load: {e}")
            print("Will test MediaPipe only")
            self.movenet_available = False
        
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
    
    def process_movenet_frame(self, frame):
        """Process frame with MoveNet Lightning"""
        if not self.movenet_available:
            return None, 0
            
        start_time = time.time()
        
        try:
            # Resize and normalize for MoveNet (192x192)
            input_frame = cv2.resize(frame, (192, 192))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_frame = tf.cast(input_frame, dtype=tf.int32)
            input_frame = tf.expand_dims(input_frame, axis=0)
            
            # Run inference
            outputs = self.movenet(input_frame)
            keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # Shape: [17, 3]
            
            processing_time = (time.time() - start_time) * 1000
            return keypoints, processing_time
            
        except Exception as e:
            print(f"MoveNet processing error: {e}")
            return None, 0
    
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
    
    def analyze_movenet_results(self, keypoints):
        """Analyze MoveNet keypoints for boxing accuracy"""
        if keypoints is None:
            return {}
        
        try:
            # MoveNet keypoint indices
            # 0: nose, 5: left_shoulder, 6: right_shoulder
            # 7: left_elbow, 8: right_elbow, 9: left_wrist, 10: right_wrist
            
            left_shoulder = keypoints[5][:2]  # [y, x] format
            left_elbow = keypoints[7][:2]
            left_wrist = keypoints[9][:2]
            
            right_shoulder = keypoints[6][:2]
            right_elbow = keypoints[8][:2]
            right_wrist = keypoints[10][:2]
            
            # Calculate confidence scores
            left_confidence = min(keypoints[5][2], keypoints[7][2], keypoints[9][2])
            right_confidence = min(keypoints[6][2], keypoints[8][2], keypoints[10][2])
            
            # Calculate arm angles
            left_angle = self.calculate_arm_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = self.calculate_arm_angle(right_shoulder, right_elbow, right_wrist)
            
            return {
                'left_angle': left_angle,
                'right_angle': right_angle,
                'left_confidence': left_confidence,
                'right_confidence': right_confidence,
                'left_wrist': left_wrist,
                'right_wrist': right_wrist
            }
        except Exception as e:
            return {}
    
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
    
    def draw_comparison(self, frame, mp_analysis, mn_analysis, mp_time, mn_time):
        """Draw comparison results on frame"""
        h, w = frame.shape[:2]
        
        # Background
        cv2.rectangle(frame, (10, 10), (w-10, 200), (0, 0, 0), -1)
        
        # Title
        cv2.putText(frame, "POSE DETECTION COMPARISON TEST", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        y_pos = 60
        
        # MediaPipe results
        cv2.putText(frame, "MEDIAPIPE BLAZEPOSE:", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_pos += 25
        
        if mp_analysis:
            cv2.putText(frame, f"Left: {mp_analysis.get('left_angle', 0):.0f}° conf:{mp_analysis.get('left_confidence', 0):.2f}", 
                       (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_pos += 20
            cv2.putText(frame, f"Right: {mp_analysis.get('right_angle', 0):.0f}° conf:{mp_analysis.get('right_confidence', 0):.2f}", 
                       (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No detection", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        y_pos += 25
        cv2.putText(frame, f"Processing: {mp_time:.1f}ms", (30, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 35
        
        # MoveNet results
        if self.movenet_available:
            cv2.putText(frame, "MOVENET LIGHTNING:", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_pos += 25
            
            if mn_analysis:
                cv2.putText(frame, f"Left: {mn_analysis.get('left_angle', 0):.0f}° conf:{mn_analysis.get('left_confidence', 0):.2f}", 
                           (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20
                cv2.putText(frame, f"Right: {mn_analysis.get('right_angle', 0):.0f}° conf:{mn_analysis.get('right_confidence', 0):.2f}", 
                           (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                cv2.putText(frame, "No detection", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            y_pos += 25
            cv2.putText(frame, f"Processing: {mn_time:.1f}ms", (30, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "MOVENET: Not available", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    def run_comparison(self):
        """Run the comparison test"""
        if not self.init_camera():
            print("Failed to initialize camera")
            return
        
        print("\n" + "="*70)
        print("POSE DETECTION COMPARISON TEST")
        print("="*70)
        print("INSTRUCTIONS:")
        print("• Perform various boxing movements")
        print("• Observe detection accuracy differences")
        print("• Press 's' to save current analysis")
        print("• Press 'q' to quit")
        print("="*70)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process with both models
                mp_results, mp_time = self.process_mediapipe_frame(frame)
                mn_keypoints, mn_time = self.process_movenet_frame(frame)
                
                # Analyze results
                mp_analysis = self.analyze_mediapipe_results(mp_results)
                mn_analysis = self.analyze_movenet_results(mn_keypoints)
                
                # Draw comparison
                self.draw_comparison(frame, mp_analysis, mn_analysis, mp_time, mn_time)
                
                # Show frame
                cv2.imshow('Pose Detection Comparison', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save analysis snapshot
                    timestamp = time.strftime("%H:%M:%S")
                    self.results_log.append({
                        'time': timestamp,
                        'mediapipe': mp_analysis,
                        'movenet': mn_analysis,
                        'mp_time': mp_time,
                        'mn_time': mn_time
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
        print("COMPARISON TEST COMPLETE")
        print("="*50)
        print(f"Frames processed: {self.frame_count}")
        print(f"Analysis snapshots saved: {len(self.results_log)}")
        
        if self.results_log:
            # Calculate averages
            mp_times = [r['mp_time'] for r in self.results_log if r['mp_time'] > 0]
            mn_times = [r['mn_time'] for r in self.results_log if r['mn_time'] > 0]
            
            if mp_times:
                print(f"MediaPipe avg processing: {np.mean(mp_times):.1f}ms")
            if mn_times:
                print(f"MoveNet avg processing: {np.mean(mn_times):.1f}ms")


if __name__ == "__main__":
    test = PoseComparisonTest()
    test.run_comparison()