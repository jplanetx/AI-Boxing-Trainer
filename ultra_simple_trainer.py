#!/usr/bin/env python3
"""
Ultra Simple Boxing Trainer - Just detect arm extensions
No complex algorithms, just basic movement detection that works.
"""

import cv2
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_trainer.pose_tracker import PoseTracker


class UltraSimpleTrainer:
    """Ultra simple trainer that just detects when arms extend forward."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # Use basic pose tracker
        self.pose_tracker = PoseTracker(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.3,
            model_complexity=0  # Fastest
        )
        
        # Simple tracking - just detect wrist movements
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.last_wrist_positions = {'left': None, 'right': None}
        self.movement_threshold = 50  # pixels
        self.recent_punches = []
        
        # Performance
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.session_start = time.time()
        
        print("Ultra Simple Trainer initialized - will detect any significant arm movement")
        
    def initialize_camera(self) -> bool:
        """Initialize camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Lower res for speed
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
    
    def detect_movement(self, landmarks_dict: dict, arm: str) -> bool:
        """Ultra simple movement detection - just track wrist position changes."""
        if not landmarks_dict:
            return False
            
        wrist_key = f'{arm}_wrist'
        if wrist_key not in landmarks_dict:
            return False
            
        wrist = landmarks_dict[wrist_key]
        if wrist.get('visibility', 0) < 0.3:
            return False
            
        current_pos = np.array([wrist['x'], wrist['y']])
        
        # First time seeing this wrist
        if self.last_wrist_positions[arm] is None:
            self.last_wrist_positions[arm] = current_pos
            return False
        
        # Calculate movement distance
        last_pos = self.last_wrist_positions[arm]
        movement_distance = np.linalg.norm(current_pos - last_pos)
        
        # If significant movement detected
        if movement_distance > self.movement_threshold:
            # Count as punch
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            
            # Add to recent punches
            punch_info = {
                'arm': arm,
                'time': time.time(),
                'distance': movement_distance
            }
            self.recent_punches.append(punch_info)
            
            # Keep only last 5
            if len(self.recent_punches) > 5:
                self.recent_punches.pop(0)
            
            print(f"MOVEMENT DETECTED: {arm.upper()} arm moved {movement_distance:.1f} pixels - COUNT: {self.punch_counts[arm]}")
            
            # Update last position
            self.last_wrist_positions[arm] = current_pos
            return True
        
        # Update position more gradually for small movements
        if movement_distance > 10:  # Small movements update position
            self.last_wrist_positions[arm] = current_pos
            
        return False
    
    def update_fps(self):
        """Update FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Ultra simple UI."""
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Simple header
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'ULTRA SIMPLE BOXING TRAINER', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Pose status
        pose_status = "POSE: OK" if landmarks_dict else "POSE: NO DETECTION"
        pose_color = (0, 255, 0) if landmarks_dict else (0, 0, 255)
        cv2.putText(frame, pose_status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        # HUGE punch counters - can't miss these
        cv2.rectangle(frame, (20, 100), (width-20, 250), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 100), (width-20, 250), (0, 255, 0), 3)
        
        # Giant text
        cv2.putText(frame, f'LEFT: {self.punch_counts["left"]}', (40, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 4)
        cv2.putText(frame, f'RIGHT: {self.punch_counts["right"]}', (300, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 255), 4)
        cv2.putText(frame, f'TOTAL: {self.punch_counts["total"]}', (200, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        
        # Show wrist positions if available
        if landmarks_dict:
            for arm in ['left', 'right']:
                wrist_key = f'{arm}_wrist'
                if wrist_key in landmarks_dict:
                    wrist = landmarks_dict[wrist_key]
                    if wrist.get('visibility', 0) > 0.3:
                        x, y = int(wrist['x']), int(wrist['y'])
                        color = (0, 255, 0) if arm == 'left' else (0, 0, 255)
                        cv2.circle(frame, (x, y), 10, color, -1)
                        cv2.putText(frame, f'{arm.upper()}', (x+15, y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Recent movements
        if self.recent_punches:
            cv2.rectangle(frame, (20, height-120), (width-20, height-20), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, height-120), (width-20, height-20), (255, 255, 0), 2)
            
            cv2.putText(frame, 'RECENT MOVEMENTS:', (30, height-95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            for i, punch in enumerate(self.recent_punches[-3:]):
                age = current_time - punch['time']
                if age < 5:  # Show for 5 seconds
                    text = f"{punch['arm'].upper()} - {punch['distance']:.0f}px"
                    cv2.putText(frame, text, (30, height - 65 + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, 'R: Reset | Q: Quit | Any arm movement = count', (10, height-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def handle_keyboard_input(self) -> bool:
        """Handle input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            return False
        elif key == ord('r'):
            self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
            self.recent_punches = []
            self.last_wrist_positions = {'left': None, 'right': None}
            self.session_start = time.time()
            print("RESET: All counts cleared!")
        
        return True
    
    def run(self):
        """Main loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*60)
        print("ULTRA SIMPLE BOXING TRAINER")
        print("="*60)
        print("This version counts ANY significant arm movement as a punch")
        print("Movement threshold: 50 pixels")
        print("Just move your arms forward/back and it should count!")
        print("R = Reset counts | Q = Quit")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Process pose
                processed_frame, landmarks_dict = self.pose_tracker.process_frame(frame)
                
                # Detect movements (any arm movement counts)
                if landmarks_dict:
                    self.detect_movement(landmarks_dict, 'left')
                    self.detect_movement(landmarks_dict, 'right')
                
                # Draw UI
                final_frame = self.draw_ui(processed_frame, landmarks_dict)
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('Ultra Simple Boxing Trainer', final_frame)
                
                # Handle input
                if not self.handle_keyboard_input():
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose_tracker.release()
        
        duration = int(time.time() - self.session_start)
        minutes, seconds = divmod(duration, 60)
        
        print(f"\n" + "="*40)
        print("FINAL SUMMARY")
        print("="*40)
        print(f"Duration: {minutes:02d}:{seconds:02d}")
        print(f"Left movements: {self.punch_counts['left']}")
        print(f"Right movements: {self.punch_counts['right']}")
        print(f"Total movements: {self.punch_counts['total']}")
        print("="*40)


def main():
    """Main entry point."""
    try:
        trainer = UltraSimpleTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())