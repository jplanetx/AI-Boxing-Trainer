#!/usr/bin/env python3
"""
Simple Boxing Trainer - Works without complex dependencies
Pure OpenCV + MediaPipe solution with optimal settings from research.
"""

import cv2
import time
import numpy as np
import sys
import os

# Simple MediaPipe import
try:
    import mediapipe as mp
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available")


class SimpleBoxingTrainer:
    """Simple, reliable boxing trainer - no complex dependencies."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        if not MEDIAPIPE_AVAILABLE:
            print("ERROR: MediaPipe required but not available")
            return
        
        # Initialize MediaPipe with OPTIMAL settings from research
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,          # Balanced performance
            enable_segmentation=False,
            min_detection_confidence=0.3, # Lower for fast movements
            min_tracking_confidence=0.2   # Lower for continuous tracking
        )
        
        # Simple punch tracking
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.session_start = time.time()
        
        # Recent punches
        self.recent_punches = []
        
        # Setup phase and cooldowns
        self.setup_phase = True
        self.frames_processed = 0
        self.min_frames_before_detection = 90  # 3 seconds
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_cooldown = 1.0  # 1 second cooldown
        
        print("Simple Boxing Trainer initialized")
        print("Using MediaPipe with research-optimized settings")
        
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
    
    def extract_landmarks(self, pose_landmarks, frame_width: int, frame_height: int) -> dict:
        """Extract landmarks with low confidence threshold."""
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
                if landmark.visibility > 0.2:  # Low threshold
                    landmarks_dict[name] = {
                        'x': float(landmark.x * frame_width),
                        'y': float(landmark.y * frame_height),
                        'visibility': float(landmark.visibility)
                    }
            except (IndexError, AttributeError):
                continue
                
        return landmarks_dict
    
    def detect_punch_simple(self, landmarks_dict: dict, arm: str) -> bool:
        """Simple, effective punch detection."""
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
        if any(lm.get('visibility', 0) < 0.3 for lm in [shoulder, elbow, wrist]):
            return False
        
        # Calculate elbow angle
        current_angle = self.calculate_angle(
            [shoulder['x'], shoulder['y']],
            [elbow['x'], elbow['y']],
            [wrist['x'], wrist['y']]
        )
        
        # Simple thresholds
        bent_threshold = 110
        extended_threshold = 150
        
        # State machine
        if self.arm_states[arm] == 'unknown':
            self.arm_states[arm] = 'bent' if current_angle < bent_threshold else 'extended'
            return False
        
        # Detect punch: bent -> extended
        if self.arm_states[arm] == 'bent' and current_angle > extended_threshold:
            # Punch detected!
            self.arm_states[arm] = 'extended'
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            self.last_punch_time[arm] = current_time
            
            # Add to recent punches
            punch_type = 'jab' if arm == 'left' else 'straight'
            punch_number = 1 if arm == 'left' else 2
            
            punch_info = {
                'arm': arm,
                'time': current_time,
                'angle': current_angle,
                'type': punch_type,
                'number': punch_number
            }
            self.recent_punches.append(punch_info)
            
            if len(self.recent_punches) > 5:
                self.recent_punches.pop(0)
            
            print(f"PUNCH: #{punch_number} {arm.upper()} {punch_type.upper()} ({current_angle:.0f}°)")
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
                print("SETUP COMPLETE - Punch detection active!")
    
    def update_fps(self):
        """Update FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Draw simple, clear UI."""
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Header
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'SIMPLE BOXING TRAINER', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Setup status
        if self.setup_phase:
            countdown = max(0, self.min_frames_before_detection - self.frames_processed)
            cv2.putText(frame, f'SETUP: {countdown//30 + 1}s', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'DETECTING PUNCHES', (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # HUGE punch counters
        cv2.rectangle(frame, (20, 100), (width-20, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 100), (width-20, 220), (0, 255, 0), 3)
        
        cv2.putText(frame, f'LEFT: {self.punch_counts["left"]}', (40, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 255, 100), 4)
        cv2.putText(frame, f'RIGHT: {self.punch_counts["right"]}', (320, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 255), 4)
        cv2.putText(frame, f'TOTAL: {self.punch_counts["total"]}', (150, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        
        # Recent punches
        if self.recent_punches:
            cv2.rectangle(frame, (20, height-100), (width-20, height-20), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, height-100), (width-20, height-20), (255, 255, 0), 2)
            
            cv2.putText(frame, 'RECENT PUNCHES:', (30, height-75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for i, punch in enumerate(self.recent_punches[-3:]):
                age = current_time - punch['time']
                if age < 3:
                    text = f"#{punch['number']} {punch['arm'].upper()} {punch['type'].upper()}"
                    cv2.putText(frame, text, (30, height - 50 + i * 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def process_frame(self, frame: np.ndarray):
        """Process frame with MediaPipe."""
        # Don't flip frame to preserve left/right
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process with MediaPipe
        results = self.pose.process(rgb_frame)
        
        # Convert back to BGR
        rgb_frame.flags.writeable = True
        processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        landmarks_dict = {}
        if hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            landmarks_dict = self.extract_landmarks(
                results.pose_landmarks, frame_width, frame_height
            )
            
            # Draw pose
            self.mp_drawing.draw_landmarks(
                processed_frame,
                results.pose_landmarks,
                list(self.mp_pose.POSE_CONNECTIONS)
            )
        
        return processed_frame, landmarks_dict
    
    def handle_keyboard_input(self) -> bool:
        """Handle input."""
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
        """Main loop."""
        if not MEDIAPIPE_AVAILABLE:
            print("Cannot run - MediaPipe not available")
            return
            
        if not self.initialize_camera():
            return
        
        print("\n" + "="*50)
        print("SIMPLE BOXING TRAINER")
        print("="*50)
        print("Research-optimized MediaPipe settings")
        print("No complex dependencies - just works!")
        print("R = Reset | Q = Quit")
        print("="*50 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Update setup phase
                self.update_setup_phase()
                
                # Process frame
                processed_frame, landmarks_dict = self.process_frame(frame)
                
                # Detect punches
                if landmarks_dict:
                    self.detect_punch_simple(landmarks_dict, 'left')
                    self.detect_punch_simple(landmarks_dict, 'right')
                
                # Draw UI
                final_frame = self.draw_ui(processed_frame, landmarks_dict)
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('Simple Boxing Trainer', final_frame)
                
                # Handle input
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
        
        duration = int(time.time() - self.session_start)
        minutes, seconds = divmod(duration, 60)
        
        print(f"\nSESSION SUMMARY:")
        print(f"Duration: {minutes:02d}:{seconds:02d}")
        print(f"Left: {self.punch_counts['left']} | Right: {self.punch_counts['right']} | Total: {self.punch_counts['total']}")


def main():
    """Main entry point."""
    try:
        trainer = SimpleBoxingTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())