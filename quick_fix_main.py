#!/usr/bin/env python3
"""
Quick Fix AI Boxing Trainer - Simplified and More Reliable
Focuses on core punch detection without complex mode switching.
"""

import cv2
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_trainer.pose_tracker import PoseTracker
from ai_trainer.utils import calculate_angle


class SimplifiedBoxingTrainer:
    """Simplified, reliable boxing trainer focused on core functionality."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # Initialize pose tracker with more forgiving settings
        self.pose_tracker = PoseTracker(
            min_detection_confidence=0.5,  # Lower for better detection
            min_tracking_confidence=0.3,   # Lower for continuity
            model_complexity=1             # Faster processing
        )
        
        # Punch tracking with simpler logic
        self.punch_counts = {'left': 0, 'right': 0}
        self.punch_stages = {'left': 'guard', 'right': 'guard'}
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_speeds = {'left': 0, 'right': 0}
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # UI state
        self.last_punch_detected = {'arm': None, 'time': 0}
        
    def initialize_camera(self) -> bool:
        """Initialize camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False
                
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def detect_punch_simple(self, landmarks_dict: dict, arm: str) -> bool:
        """Simplified punch detection that actually works."""
        if not landmarks_dict:
            return False
            
        try:
            # Get arm landmarks
            shoulder_key = f'{arm}_shoulder'
            elbow_key = f'{arm}_elbow'
            wrist_key = f'{arm}_wrist'
            hip_key = f'{arm}_hip'
            
            if not all(key in landmarks_dict for key in [shoulder_key, elbow_key, wrist_key]):
                return False
            
            shoulder = landmarks_dict[shoulder_key]
            elbow = landmarks_dict[elbow_key]
            wrist = landmarks_dict[wrist_key]
            
            # Check visibility
            if any(lm.get('visibility', 0) < 0.3 for lm in [shoulder, elbow, wrist]):
                return False
            
            # Calculate elbow angle (shoulder-elbow-wrist)
            elbow_angle = calculate_angle(
                [shoulder['x'], shoulder['y']],
                [elbow['x'], elbow['y']],
                [wrist['x'], wrist['y']]
            )
            
            # Calculate shoulder angle if hip available
            shoulder_angle = 90  # Default
            if hip_key in landmarks_dict:
                hip = landmarks_dict[hip_key]
                if hip.get('visibility', 0) > 0.3:
                    shoulder_angle = calculate_angle(
                        [hip['x'], hip['y']],
                        [shoulder['x'], shoulder['y']],
                        [elbow['x'], elbow['y']]
                    )
            
            # Simple punch detection logic
            current_time = time.time()
            
            # Guard position: elbow bent, arm close to body
            if elbow_angle < 90 and shoulder_angle < 60:
                if self.punch_stages[arm] == 'punching':
                    # Returning to guard - punch complete
                    self.punch_stages[arm] = 'guard'
                elif self.punch_stages[arm] == 'guard':
                    # Stay in guard
                    pass
                return False
            
            # Extended position: arm extended forward
            elif elbow_angle > 140 and shoulder_angle > 30:
                if self.punch_stages[arm] == 'guard':
                    # Transition from guard to punch
                    self.punch_stages[arm] = 'punching'
                    self.punch_counts[arm] += 1
                    
                    # Calculate punch speed
                    time_since_last = current_time - self.last_punch_time[arm]
                    if time_since_last > 0:
                        self.punch_speeds[arm] = min(100, int(100 / time_since_last))
                    
                    self.last_punch_time[arm] = current_time
                    self.last_punch_detected = {'arm': arm, 'time': current_time}
                    
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error in punch detection: {e}")
            return False
    
    def update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Draw simplified UI."""
        height, width = frame.shape[:2]
        
        # Header background
        cv2.rectangle(frame, (0, 0), (width, 120), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, frame, 0.3, 0)
        
        # Title
        cv2.putText(frame, 'AI BOXING TRAINER - RELIABLE MODE', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # FPS
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (width - 150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Pose status
        pose_status = "POSE DETECTED" if landmarks_dict else "NO POSE"
        pose_color = (0, 255, 0) if landmarks_dict else (0, 0, 255)
        cv2.putText(frame, pose_status, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, pose_color, 2)
        
        # Punch counters
        cv2.rectangle(frame, (50, 150), (350, 300), (0, 0, 0), -1)
        cv2.rectangle(frame, (50, 150), (350, 300), (255, 255, 255), 2)
        
        cv2.putText(frame, 'PUNCH COUNTS', (60, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Left punches
        cv2.putText(frame, f'LEFT: {self.punch_counts["left"]}', (60, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        cv2.putText(frame, f'Speed: {self.punch_speeds["left"]}', (60, 245), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Right punches  
        cv2.putText(frame, f'RIGHT: {self.punch_counts["right"]}', (60, 270), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
        cv2.putText(frame, f'Speed: {self.punch_speeds["right"]}', (60, 295), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
        
        # Recent punch detection
        if (time.time() - self.last_punch_detected['time']) < 1.0 and self.last_punch_detected['arm']:
            arm = self.last_punch_detected['arm']
            cv2.putText(frame, f'{arm.upper()} PUNCH!', (width//2 - 80, height - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        
        # Instructions
        instructions = [
            "Controls:",
            "Q: Quit",
            "R: Reset counts"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 200, 150 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('r'):  # Reset counts
            self.punch_counts = {'left': 0, 'right': 0}
            self.punch_speeds = {'left': 0, 'right': 0}
            self.punch_stages = {'left': 'guard', 'right': 'guard'}
            print("Punch counts reset")
        
        return True
    
    def run(self):
        """Main application loop."""
        if not self.initialize_camera():
            return
        
        print("Simplified AI Boxing Trainer - Starting...")
        print("This version focuses on reliable punch detection")
        print("Press 'R' to reset counts, 'Q' to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame, landmarks_dict = self.pose_tracker.process_frame(frame)
                
                # Detect punches for both arms
                if landmarks_dict:
                    self.detect_punch_simple(landmarks_dict, 'left')
                    self.detect_punch_simple(landmarks_dict, 'right')
                
                # Draw UI
                final_frame = self.draw_ui(processed_frame, landmarks_dict)
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('AI Boxing Trainer - Reliable', final_frame)
                
                # Handle input
                if not self.handle_keyboard_input():
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose_tracker.release()
        
        print(f"\nSession Summary:")
        print(f"Left Punches: {self.punch_counts['left']}")
        print(f"Right Punches: {self.punch_counts['right']}")
        print(f"Total: {sum(self.punch_counts.values())}")


def main():
    """Main entry point."""
    try:
        trainer = SimplifiedBoxingTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())