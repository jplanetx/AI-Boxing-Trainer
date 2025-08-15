#!/usr/bin/env python3
"""
AI Boxing Trainer - Robust Production Version
Handles real-world landmark availability patterns based on debug testing.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os
from typing import Optional, Dict

# Explicit imports to resolve Pylance errors
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

class RobustBoxingTrainer:
    def __init__(self, camera_id: int = 0):
        print("🥊 Robust AI Boxing Trainer - Production Version")
        print("=" * 60)
        
        self.mp_pose = mp_pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp_drawing
        
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.punch_counts = {'left': 0, 'right': 0}
        self.punch_scores = {'left': 0, 'right': 0}
        self.punch_stages = {'left': 'guard', 'right': 'guard'}
        self.last_punch_types = {'left': 'unknown', 'right': 'unknown'}
        self.last_punch_time = {'left': 0, 'right': 0}
        
        self.key_landmarks = {
            'left_shoulder': self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            'right_shoulder': self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            'left_elbow': self.mp_pose.PoseLandmark.LEFT_ELBOW,
            'right_elbow': self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            'left_wrist': self.mp_pose.PoseLandmark.LEFT_WRIST,
            'right_wrist': self.mp_pose.PoseLandmark.RIGHT_WRIST,
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
        }
        
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.error_count = 0
    
    def extract_landmarks(self, pose_landmarks, frame_width: int, frame_height: int) -> Dict:
        landmarks_dict = {}
        if not pose_landmarks:
            return landmarks_dict
        try:
            for name, landmark_idx in self.key_landmarks.items():
                try:
                    landmark = pose_landmarks.landmark[landmark_idx.value]
                    if landmark.visibility > 0.3:
                        landmarks_dict[name] = {
                            'x': float(landmark.x * frame_width),
                            'y': float(landmark.y * frame_height),
                            'z': float(landmark.z * frame_width),
                            'visibility': float(landmark.visibility)
                        }
                except (IndexError, AttributeError):
                    continue
        except Exception:
            return {}
        return landmarks_dict
    
    def calculate_angle_safe(self, p1: Dict, p2: Dict, p3: Dict) -> float:
        try:
            a = np.array([p1['x'], p1['y']])
            b = np.array([p2['x'], p2['y']])
            c = np.array([p3['x'], p3['y']])
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            return 360 - angle if angle > 180.0 else angle
        except:
            return 90.0
    
    def analyze_punch_simple(self, landmarks_dict: Dict, arm: str) -> None:
        try:
            required = [f'{arm}_shoulder', f'{arm}_elbow', f'{arm}_wrist']
            if not all(key in landmarks_dict for key in required):
                return
            
            shoulder = landmarks_dict[f'{arm}_shoulder']
            elbow = landmarks_dict[f'{arm}_elbow']
            wrist = landmarks_dict[f'{arm}_wrist']
            
            elbow_angle = self.calculate_angle_safe(shoulder, elbow, wrist)
            current_stage = self.punch_stages[arm]
            
            if current_stage == 'guard' and elbow_angle > 150:
                self.punch_stages[arm] = 'punching'
                self.punch_counts[arm] += 1
                self.last_punch_time[arm] = time.time()
                
                if wrist['y'] < shoulder['y']:
                    self.last_punch_types[arm] = 'uppercut'
                elif abs(wrist['x'] - shoulder['x']) > abs(wrist['y'] - shoulder['y']):
                    self.last_punch_types[arm] = 'hook'
                else:
                    self.last_punch_types[arm] = 'jab' if arm == 'left' else 'cross'
            
            elif current_stage == 'punching' and elbow_angle < 90:
                self.punch_stages[arm] = 'guard'
                duration = time.time() - self.last_punch_time[arm]
                if duration > 0:
                    self.punch_scores[arm] = int(min(100, (1 / duration) * 100))
        except Exception as e:
            self.error_count += 1
            if self.error_count % 100 == 1:
                print(f"⚠️  Analysis error for {arm}: {e}")
    
    def draw_ui(self, frame: np.ndarray, landmarks_dict: Dict) -> None:
        try:
            frame_height, frame_width = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (600, 140), (245, 117, 16), -1)
            
            # Left arm
            cv2.putText(frame, 'LEFT ARM', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Count: {self.punch_counts['left']}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Score: {int(self.punch_scores['left'])}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Type: {self.last_punch_types['left'].upper()}", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Stage: {self.punch_stages['left'].upper()}", (15, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Right arm
            cv2.putText(frame, 'RIGHT ARM', (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f"Count: {self.punch_counts['right']}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Score: {int(self.punch_scores['right'])}", (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Type: {self.last_punch_types['right'].upper()}", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Stage: {self.punch_stages['right'].upper()}", (300, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.putText(frame, f"Landmarks: {len(landmarks_dict)}/8", (frame_width - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (frame_width - 120, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception:
            pass
    
    def run(self):
        print("🎮 Controls: 'q' to quit, 'r' to reset stats")
        print("🎯 Position yourself so 5-8 landmarks are visible")
        print("=" * 50)
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                frame_height, frame_width = frame.shape[:2]
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = self.pose.process(rgb_frame)
                rgb_frame.flags.writeable = True
                processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                
                landmarks_dict = {}
                if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
                    landmarks_dict = self.extract_landmarks(results.pose_landmarks, frame_width, frame_height)
                    self.mp_drawing.draw_landmarks(processed_frame, results.pose_landmarks, list(self.mp_pose.POSE_CONNECTIONS))
                
                self.analyze_punch_simple(landmarks_dict, 'left')
                self.analyze_punch_simple(landmarks_dict, 'right')
                self.draw_ui(processed_frame, landmarks_dict)
                
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_start_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                cv2.imshow('Robust AI Boxing Trainer', processed_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('r'):
                    self.punch_counts = {'left': 0, 'right': 0}
                    self.punch_scores = {'left': 0, 'right': 0}
                    self.punch_stages = {'left': 'guard', 'right': 'guard'}
                    print("📊 Statistics reset!")
        
        except KeyboardInterrupt:
            print("\n⏹️  Training stopped by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("🧹 Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        print("✅ Cleanup complete!")

def main():
    try:
        trainer = RobustBoxingTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"❌ Failed to start trainer: {e}")

if __name__ == "__main__":
    main()