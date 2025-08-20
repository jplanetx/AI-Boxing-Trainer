#!/usr/bin/env python3
"""
Low Sensitivity Boxing Trainer - Much stricter detection to eliminate false positives
Only counts very deliberate punching motions.
"""

import cv2
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


class LowSensitivityTrainer:
    """Ultra-conservative punch detection to eliminate false positives."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # MediaPipe with optimal settings
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.2
        )
        
        # Punch tracking
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.session_start = time.time()
        
        # Recent punches
        self.recent_punches = []
        
        # MUCH MORE CONSERVATIVE SETTINGS
        self.setup_phase = True
        self.frames_processed = 0
        self.min_frames_before_detection = 150  # 5 seconds setup time
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_cooldown = 1.5  # 1.5 second cooldown
        
        # Movement validation
        self.last_wrist_positions = {'left': None, 'right': None}
        self.min_movement_distance = 80  # Must move at least 80 pixels
        
        print("Low Sensitivity Trainer initialized")
        print("VERY strict detection - only deliberate punches count")
        print("5-second setup time, 1.5s cooldown between punches")
        
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
        """Extract landmarks."""
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
                if landmark.visibility > 0.4:  # Higher visibility requirement
                    landmarks_dict[name] = {
                        'x': float(landmark.x * frame_width),
                        'y': float(landmark.y * frame_height),
                        'visibility': float(landmark.visibility)
                    }
            except (IndexError, AttributeError):
                continue
                
        return landmarks_dict
    
    def validate_deliberate_movement(self, landmarks_dict: dict, arm: str) -> bool:
        """Validate that this is a deliberate punching movement."""
        wrist_key = f'{arm}_wrist'
        
        if wrist_key not in landmarks_dict:
            return False
        
        current_wrist = np.array([landmarks_dict[wrist_key]['x'], landmarks_dict[wrist_key]['y']])
        
        # First time seeing this wrist
        if self.last_wrist_positions[arm] is None:
            self.last_wrist_positions[arm] = current_wrist
            return False
        
        # Calculate distance moved
        movement_distance = np.linalg.norm(current_wrist - self.last_wrist_positions[arm])
        
        # Update position for next check
        self.last_wrist_positions[arm] = current_wrist
        
        # Must move significant distance to be considered deliberate
        return movement_distance >= self.min_movement_distance
    
    def detect_punch_ultra_conservative(self, landmarks_dict: dict, arm: str) -> bool:
        """Ultra-conservative punch detection with multiple validation layers."""
        if not landmarks_dict or self.setup_phase:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_punch_time[arm] < self.punch_cooldown:
            return False
        
        # Check required landmarks with HIGH visibility
        shoulder_key = f'{arm}_shoulder'
        elbow_key = f'{arm}_elbow'
        wrist_key = f'{arm}_wrist'
        
        if not all(key in landmarks_dict for key in [shoulder_key, elbow_key, wrist_key]):
            return False
        
        shoulder = landmarks_dict[shoulder_key]
        elbow = landmarks_dict[elbow_key]
        wrist = landmarks_dict[wrist_key]
        
        # STRICT visibility requirements
        if any(lm.get('visibility', 0) < 0.5 for lm in [shoulder, elbow, wrist]):
            return False
        
        # Validate deliberate movement
        if not self.validate_deliberate_movement(landmarks_dict, arm):
            return False
        
        # Calculate elbow angle
        current_angle = self.calculate_angle(
            [shoulder['x'], shoulder['y']],
            [elbow['x'], elbow['y']],
            [wrist['x'], wrist['y']]
        )
        
        # VERY STRICT thresholds
        bent_threshold = 100    # Much lower - must be clearly bent
        extended_threshold = 155  # Much higher - must be very extended
        
        # State machine
        if self.arm_states[arm] == 'unknown':
            self.arm_states[arm] = 'bent' if current_angle < bent_threshold else 'extended'
            return False
        
        # Detect punch: bent -> extended with STRICT requirements
        if (self.arm_states[arm] == 'bent' and 
            current_angle > extended_threshold and 
            current_angle > bent_threshold + 50):  # Must be SIGNIFICANT angle change
            
            # Additional validation: check forward movement
            forward_distance = abs(wrist['x'] - shoulder['x'])
            if forward_distance < 60:  # Must have good forward extension
                return False
            
            # Punch detected!
            self.arm_states[arm] = 'extended'
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            self.last_punch_time[arm] = current_time
            
            # Simple classification
            punch_type = 'jab' if arm == 'left' else 'straight'
            punch_number = 1 if arm == 'left' else 2
            
            if forward_distance < 80:  # Less forward = hook
                punch_type = 'hook'
                punch_number = 3 if arm == 'left' else 4
            
            # Add to recent punches
            punch_info = {
                'arm': arm,
                'time': current_time,
                'angle': current_angle,
                'type': punch_type,
                'number': punch_number,
                'movement': f"{forward_distance:.0f}px"
            }
            self.recent_punches.append(punch_info)
            
            if len(self.recent_punches) > 3:  # Only keep last 3
                self.recent_punches.pop(0)
            
            print(f"PUNCH: #{punch_number} {arm.upper()} {punch_type.upper()} ({current_angle:.0f}° / {forward_distance:.0f}px)")
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
                print("SETUP COMPLETE - Ultra-conservative punch detection active!")
                print("Only very deliberate punching motions will be counted")
    
    def update_fps(self):
        """Update FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_camera_view(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Draw camera view."""
        height, width = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.putText(frame, 'LOW SENSITIVITY MODE', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Setup status
        if self.setup_phase:
            countdown = max(0, self.min_frames_before_detection - self.frames_processed)
            cv2.putText(frame, f'SETUP: {countdown//30 + 1}s', (width - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'READY', (width - 100, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame
    
    def create_data_window(self) -> np.ndarray:
        """Create data display."""
        data_window = np.zeros((800, 1200, 3), dtype=np.uint8)
        current_time = time.time()
        
        # Title
        cv2.putText(data_window, 'LOW SENSITIVITY PUNCH COUNTER', (200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        # Session info
        session_duration = int(current_time - self.session_start)
        minutes, seconds = divmod(session_duration, 60)
        cv2.putText(data_window, f'Session: {minutes:02d}:{seconds:02d} | FPS: {self.current_fps:.1f}', 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Settings display
        cv2.putText(data_window, 'STRICT SETTINGS: 5s setup, 1.5s cooldown, 80px min movement', 
                   (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Punch counters
        counter_y = 200
        
        # Left
        cv2.rectangle(data_window, (50, counter_y), (550, counter_y + 150), (0, 255, 255), 3)
        cv2.putText(data_window, 'LEFT', (200, counter_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['left']), (250, counter_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (100, 255, 255), 8)
        
        # Right
        cv2.rectangle(data_window, (600, counter_y), (1100, counter_y + 150), (0, 255, 255), 3)
        cv2.putText(data_window, 'RIGHT', (750, counter_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['right']), (800, counter_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 100), 8)
        
        # Total
        total_y = 400
        cv2.rectangle(data_window, (300, total_y), (850, total_y + 180), (255, 255, 255), 4)
        cv2.putText(data_window, 'TOTAL', (500, total_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        cv2.putText(data_window, str(self.punch_counts['total']), (500, total_y + 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 10)
        
        # Recent punches
        recent_y = 620
        cv2.putText(data_window, 'RECENT PUNCHES (Only Deliberate Motions):', (50, recent_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        for i, punch in enumerate(self.recent_punches[-3:]):
            age = current_time - punch['time']
            if age < 10:  # Show longer
                y_pos = recent_y + 40 + (i * 30)
                text = f"#{punch['number']} {punch['arm'].upper()} {punch['type'].upper()} ({punch['angle']:.0f}° / {punch['movement']})"
                cv2.putText(data_window, text, (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return data_window
    
    def process_frame(self, frame: np.ndarray):
        """Process frame."""
        frame_height, frame_width = frame.shape[:2]
        
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
        
        if key == ord('q') or key == 27:
            return False
        elif key == ord('r'):
            self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
            self.recent_punches = []
            self.arm_states = {'left': 'unknown', 'right': 'unknown'}
            self.last_wrist_positions = {'left': None, 'right': None}
            self.session_start = time.time()
            print("RESET: All counts cleared!")
        
        return True
    
    def run(self):
        """Main loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*60)
        print("LOW SENSITIVITY BOXING TRAINER")
        print("="*60)
        print("ULTRA-CONSERVATIVE DETECTION:")
        print("• 5-second setup time")
        print("• 1.5-second cooldown between punches")
        print("• Must move 80+ pixels to be considered deliberate")
        print("• High visibility requirements")
        print("• Very strict angle thresholds")
        print("")
        print("This should eliminate false positives from general movement")
        print("CONTROLS: R = Reset | Q/ESC = Quit")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.update_setup_phase()
                processed_frame, landmarks_dict = self.process_frame(frame)
                
                if landmarks_dict:
                    self.detect_punch_ultra_conservative(landmarks_dict, 'left')
                    self.detect_punch_ultra_conservative(landmarks_dict, 'right')
                
                camera_display = self.draw_camera_view(processed_frame, landmarks_dict)
                data_display = self.create_data_window()
                
                self.update_fps()
                
                cv2.imshow('Camera View', camera_display)
                cv2.imshow('Data Display', data_display)
                cv2.moveWindow('Camera View', 50, 50)
                cv2.moveWindow('Data Display', 700, 50)
                
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
        trainer = LowSensitivityTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())