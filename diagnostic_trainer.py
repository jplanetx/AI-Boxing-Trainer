#!/usr/bin/env python3
"""
Diagnostic Boxing Trainer - Debug version to identify detection issues
Provides detailed logging and relaxed requirements to diagnose problems
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


class DiagnosticTrainer:
    """Diagnostic trainer to identify detection issues."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # MediaPipe setup with diagnostic-friendly settings
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.2,  # Even lower for debugging
            min_tracking_confidence=0.1    # Very low for debugging
        )
        
        # Punch tracking
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        
        # Diagnostic tracking
        self.debug_info = {
            'landmarks_detected': 0,
            'angle_calculations': 0,
            'bag_required': True,  # Toggle for debugging
            'bag_detected': False,
            'contact_detections': 0,
            'state_transitions': 0
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.session_start = time.time()
        
        # Recent punches
        self.recent_punches = []
        
        # Relaxed setup for diagnostics
        self.setup_phase = True
        self.frames_processed = 0
        self.min_frames_before_detection = 90  # 3 seconds only
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_cooldown = 0.5  # Shorter cooldown for testing
        
        # Heavy bag detection (optional for diagnostics)
        self.bag_detector = HeavyBagDetector()
        self.bag_area = None
        self.bag_detected = False
        self.contact_threshold = 80  # Larger threshold for testing
        
        print("Diagnostic Boxing Trainer initialized")
        print("Relaxed thresholds for debugging detection issues")
        
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
                
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera initialized: {actual_width}x{actual_height}")
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
        """Extract landmarks with diagnostic info."""
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
                if landmark.visibility > 0.2:  # Lower threshold for debugging
                    landmarks_dict[name] = {
                        'x': float(landmark.x * frame_width),
                        'y': float(landmark.y * frame_height),
                        'visibility': float(landmark.visibility)
                    }
            except (IndexError, AttributeError):
                continue
        
        if landmarks_dict:
            self.debug_info['landmarks_detected'] += 1
                
        return landmarks_dict
    
    def is_hand_contacting_bag(self, hand_pos: tuple) -> bool:
        """Check contact - but allow bypassing for diagnostics."""
        if not self.debug_info['bag_required']:
            return True  # Always allow if bag not required
        
        if not self.bag_detected or self.bag_area is None:
            return False
        
        hand_x, hand_y = hand_pos
        bag_x, bag_y, bag_w, bag_h = self.bag_area
        
        # Calculate distance from hand to bag edges
        closest_x = max(bag_x, min(hand_x, bag_x + bag_w))
        closest_y = max(bag_y, min(hand_y, bag_y + bag_h))
        
        distance = np.sqrt((hand_x - closest_x)**2 + (hand_y - closest_y)**2)
        
        is_contact = distance <= self.contact_threshold
        if is_contact:
            self.debug_info['contact_detections'] += 1
        
        return is_contact
    
    def detect_punch_diagnostic(self, landmarks_dict: dict, arm: str) -> bool:
        """Diagnostic punch detection with extensive logging."""
        if not landmarks_dict:
            print(f"DEBUG: No landmarks for {arm} arm")
            return False
        
        if self.setup_phase:
            print(f"DEBUG: Still in setup phase ({self.frames_processed}/{self.min_frames_before_detection})")
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_punch_time[arm] < self.punch_cooldown:
            print(f"DEBUG: {arm} arm in cooldown")
            return False
        
        # Check required landmarks
        shoulder_key = f'{arm}_shoulder'
        elbow_key = f'{arm}_elbow'
        wrist_key = f'{arm}_wrist'
        
        missing_landmarks = []
        if shoulder_key not in landmarks_dict:
            missing_landmarks.append(shoulder_key)
        if elbow_key not in landmarks_dict:
            missing_landmarks.append(elbow_key)
        if wrist_key not in landmarks_dict:
            missing_landmarks.append(wrist_key)
        
        if missing_landmarks:
            print(f"DEBUG: Missing landmarks for {arm}: {missing_landmarks}")
            return False
        
        shoulder = landmarks_dict[shoulder_key]
        elbow = landmarks_dict[elbow_key]
        wrist = landmarks_dict[wrist_key]
        
        # Check visibility with diagnostic output
        low_visibility = []
        for name, lm in [('shoulder', shoulder), ('elbow', elbow), ('wrist', wrist)]:
            if lm.get('visibility', 0) < 0.3:
                low_visibility.append(f"{name}={lm.get('visibility', 0):.2f}")
        
        if low_visibility:
            print(f"DEBUG: {arm} low visibility: {low_visibility}")
            return False
        
        # Check if hand is contacting bag (or bypass if not required)
        hand_pos = (wrist['x'], wrist['y'])
        is_contact = self.is_hand_contacting_bag(hand_pos)
        
        if self.debug_info['bag_required'] and not is_contact:
            print(f"DEBUG: {arm} hand not contacting bag")
            return False
        
        # Calculate elbow angle
        current_angle = self.calculate_angle(
            [shoulder['x'], shoulder['y']],
            [elbow['x'], elbow['y']],
            [wrist['x'], wrist['y']]
        )
        
        self.debug_info['angle_calculations'] += 1
        
        # Very relaxed thresholds for diagnostics
        bent_threshold = 140  # Much more relaxed
        extended_threshold = 160  # Much more relaxed
        
        print(f"DEBUG: {arm} angle={current_angle:.1f}° state={self.arm_states[arm]} bent_thresh={bent_threshold} ext_thresh={extended_threshold}")
        
        # State machine with logging
        if self.arm_states[arm] == 'unknown':
            new_state = 'bent' if current_angle < bent_threshold else 'extended'
            self.arm_states[arm] = new_state
            print(f"DEBUG: {arm} arm state initialized to {new_state}")
            return False
        
        # Detect punch: bent -> extended
        if self.arm_states[arm] == 'bent' and current_angle > extended_threshold:
            print(f"PUNCH DETECTED: {arm} arm {current_angle:.1f}° (bent -> extended)")
            
            # Punch detected!
            self.arm_states[arm] = 'extended'
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            self.last_punch_time[arm] = current_time
            self.debug_info['state_transitions'] += 1
            
            # Simple classification for diagnostics
            punch_type = 'jab' if arm == 'left' else 'straight'
            punch_number = 1 if arm == 'left' else 2
            
            # Add to recent punches
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
            
            print(f"SUCCESS: #{punch_number} {arm.upper()} {punch_type.upper()} ({current_angle:.0f}°)")
            return True
        
        # Return to bent
        elif self.arm_states[arm] == 'extended' and current_angle < bent_threshold:
            self.arm_states[arm] = 'bent'
            print(f"DEBUG: {arm} arm returned to bent ({current_angle:.1f}°)")
        
        return False
    
    def update_setup_phase(self):
        """Update setup phase."""
        self.frames_processed += 1
        if self.frames_processed >= self.min_frames_before_detection:
            if self.setup_phase:
                self.setup_phase = False
                print("DIAGNOSTIC: Setup phase complete - detection active!")
                print(f"Bag detection: {'ENABLED' if self.debug_info['bag_required'] else 'DISABLED'}")
    
    def update_fps(self):
        """Update FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_camera_view(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Diagnostic camera view."""
        height, width = frame.shape[:2]
        
        # Header with diagnostic info
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'DIAGNOSTIC MODE', (20, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Setup countdown
        if self.setup_phase:
            countdown = max(0, self.min_frames_before_detection - self.frames_processed)
            setup_seconds = countdown // 30 + 1
            cv2.putText(frame, f'Setup: {setup_seconds}s', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'DETECTION ACTIVE', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Bag mode toggle indicator
        bag_mode = "BAG REQUIRED" if self.debug_info['bag_required'] else "NO BAG NEEDED"
        bag_color = (255, 0, 0) if self.debug_info['bag_required'] else (0, 255, 0)
        cv2.putText(frame, f'Mode: {bag_mode}', (20, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, bag_color, 2)
        
        # Landmarks info
        if landmarks_dict:
            cv2.putText(frame, f'Landmarks: {len(landmarks_dict)}/6', (width - 200, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Arm states
            left_state = self.arm_states['left']
            right_state = self.arm_states['right']
            cv2.putText(frame, f'L:{left_state} R:{right_state}', (width - 200, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bag detection
        if self.bag_detected and self.bag_area:
            x, y, w, h = self.bag_area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, 'BAG', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hand positions with angle info
        if landmarks_dict:
            for arm in ['left', 'right']:
                wrist_key = f'{arm}_wrist'
                elbow_key = f'{arm}_elbow'
                shoulder_key = f'{arm}_shoulder'
                
                if all(key in landmarks_dict for key in [wrist_key, elbow_key, shoulder_key]):
                    wrist = landmarks_dict[wrist_key]
                    elbow = landmarks_dict[elbow_key]
                    shoulder = landmarks_dict[shoulder_key]
                    
                    # Calculate current angle
                    angle = self.calculate_angle(
                        [shoulder['x'], shoulder['y']],
                        [elbow['x'], elbow['y']],
                        [wrist['x'], wrist['y']]
                    )
                    
                    hand_pos = (int(wrist['x']), int(wrist['y']))
                    
                    # Color based on contact (if required)
                    if self.debug_info['bag_required']:
                        is_contact = self.is_hand_contacting_bag(hand_pos)
                        color = (0, 255, 0) if is_contact else (0, 0, 255)
                    else:
                        color = (255, 255, 0)  # Yellow if bag not required
                    
                    cv2.circle(frame, hand_pos, 15, color, -1)
                    cv2.putText(frame, f'{arm[0].upper()}: {angle:.0f}°', 
                               (hand_pos[0] + 20, hand_pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def create_data_window(self) -> np.ndarray:
        """Diagnostic data display."""
        data_window = np.zeros((700, 900, 3), dtype=np.uint8)
        current_time = time.time()
        
        # Title
        cv2.putText(data_window, 'DIAGNOSTIC DATA', (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        
        # Session info
        session_duration = int(current_time - self.session_start)
        minutes, seconds = divmod(session_duration, 60)
        cv2.putText(data_window, f'Time: {minutes:02d}:{seconds:02d} | FPS: {self.current_fps:.1f}', 
                   (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Diagnostic counters
        debug_y = 100
        cv2.putText(data_window, 'DIAGNOSTIC COUNTERS:', (50, debug_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        debug_info_display = [
            f"Landmarks detected: {self.debug_info['landmarks_detected']}",
            f"Angle calculations: {self.debug_info['angle_calculations']}",
            f"Contact detections: {self.debug_info['contact_detections']}",
            f"State transitions: {self.debug_info['state_transitions']}",
            f"Bag required: {'YES' if self.debug_info['bag_required'] else 'NO'}",
            f"Bag detected: {'YES' if self.bag_detected else 'NO'}"
        ]
        
        for i, info in enumerate(debug_info_display):
            cv2.putText(data_window, info, (50, debug_y + 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Punch counters
        counter_y = 300
        
        # Left counter
        cv2.rectangle(data_window, (50, counter_y), (400, counter_y + 100), (255, 0, 255), 3)
        cv2.putText(data_window, 'LEFT', (70, counter_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['left']), (200, counter_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 100, 255), 6)
        
        # Right counter
        cv2.rectangle(data_window, (450, counter_y), (800, counter_y + 100), (255, 0, 255), 3)
        cv2.putText(data_window, 'RIGHT', (470, counter_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['right']), (580, counter_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 100), 6)
        
        # Total counter
        total_y = 430
        cv2.rectangle(data_window, (250, total_y), (600, total_y + 120), (255, 255, 255), 4)
        cv2.putText(data_window, 'TOTAL', (270, total_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(data_window, str(self.punch_counts['total']), (380, total_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 8)
        
        # Recent punches
        recent_y = 580
        cv2.putText(data_window, 'RECENT DETECTIONS:', (50, recent_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, punch in enumerate(self.recent_punches[-3:]):
            age = current_time - punch['time']
            if age < 10:
                y_pos = recent_y + 30 + (i * 25)
                text = f"#{punch['number']} {punch['arm'].upper()} {punch['type'].upper()} ({punch['angle']:.0f}°)"
                cv2.putText(data_window, text, (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(data_window, 'CONTROLS: R=Reset | B=Toggle Bag Mode | Q=Quit', (50, 680), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return data_window
    
    def process_frame(self, frame: np.ndarray):
        """Process frame with diagnostics."""
        frame_height, frame_width = frame.shape[:2]
        
        # Try to detect bag during setup (but don't require it)
        if self.setup_phase:
            self.bag_area = self.bag_detector.detect_bag(frame)
            if self.bag_area is not None:
                self.bag_detected = True
        
        # Pose detection
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
        """Handle input with diagnostic toggles."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
            self.recent_punches = []
            self.arm_states = {'left': 'unknown', 'right': 'unknown'}
            self.debug_info.update({
                'landmarks_detected': 0,
                'angle_calculations': 0,
                'contact_detections': 0,
                'state_transitions': 0
            })
            self.session_start = time.time()
            print("DIAGNOSTIC RESET: All counts and debug info cleared!")
        elif key == ord('b'):  # Toggle bag requirement
            self.debug_info['bag_required'] = not self.debug_info['bag_required']
            mode = "REQUIRED" if self.debug_info['bag_required'] else "NOT REQUIRED"
            print(f"DIAGNOSTIC: Bag detection {mode}")
        
        return True
    
    def run(self):
        """Main diagnostic loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*70)
        print("DIAGNOSTIC BOXING TRAINER")
        print("="*70)
        print("DIAGNOSTIC FEATURES:")
        print("• Extensive logging to identify detection issues")
        print("• Relaxed thresholds for easier detection")
        print("• Toggle bag requirement for testing")
        print("• Real-time diagnostic counters")
        print("• Angle and state information display")
        print("")
        print("CONTROLS:")
        print("• R = Reset all counts and diagnostics")
        print("• B = Toggle bag detection requirement")
        print("• Q = Quit")
        print("="*70 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.update_setup_phase()
                processed_frame, landmarks_dict = self.process_frame(frame)
                
                if landmarks_dict:
                    self.detect_punch_diagnostic(landmarks_dict, 'left')
                    self.detect_punch_diagnostic(landmarks_dict, 'right')
                
                camera_display = self.draw_camera_view(processed_frame, landmarks_dict)
                data_display = self.create_data_window()
                
                self.update_fps()
                
                cv2.imshow('Diagnostic Camera', camera_display)
                cv2.imshow('Diagnostic Data', data_display)
                cv2.moveWindow('Diagnostic Camera', 50, 50)
                cv2.moveWindow('Diagnostic Data', 700, 50)
                
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


class HeavyBagDetector:
    """Simple heavy bag detector."""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=40
        )
        self.frame_count = 0
        
    def detect_bag(self, frame: np.ndarray) -> tuple:
        """Detect heavy bag in frame."""
        self.frame_count += 1
        
        if self.frame_count < 10:
            self.background_subtractor.apply(frame)
            return None
        
        try:
            fg_mask = self.background_subtractor.apply(frame, learningRate=0.005)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 5000:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = h / w if w > 0 else 0
                    if aspect_ratio > 1.0:
                        return (x, y, w, h)
            
            return None
            
        except Exception as e:
            return None


def main():
    """Main entry point."""
    try:
        trainer = DiagnosticTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())