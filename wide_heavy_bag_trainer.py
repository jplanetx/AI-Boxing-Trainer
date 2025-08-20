#!/usr/bin/env python3
"""
Wide-Angle Heavy Bag Boxing Trainer - Better for capturing both user and bag
Uses higher resolution and provides positioning guidance.
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


class WideHeavyBagTrainer:
    """Wide-angle heavy bag trainer optimized for full setup capture."""
    
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
        
        # Setup and cooldowns - LONGER setup for positioning
        self.setup_phase = True
        self.frames_processed = 0
        self.min_frames_before_detection = 180  # 6 seconds for positioning
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_cooldown = 1.0
        
        # Heavy bag detection
        self.bag_detector = HeavyBagDetector()
        self.bag_area = None
        self.bag_detected = False
        self.contact_threshold = 60  # Slightly larger for wider view
        
        print("Wide-Angle Heavy Bag Trainer initialized")
        print("Higher resolution for better bag + user capture")
        print("6-second setup time for positioning")
        
    def initialize_camera(self) -> bool:
        """Initialize camera with wide resolution."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            # WIDE resolution for better field of view
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try to set wide FOV if camera supports it
            try:
                self.cap.set(cv2.CAP_PROP_ZOOM, 0)  # Zoom out if possible
            except:
                pass
            
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False
                
            # Get actual resolution
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
                if landmark.visibility > 0.3:
                    landmarks_dict[name] = {
                        'x': float(landmark.x * frame_width),
                        'y': float(landmark.y * frame_height),
                        'visibility': float(landmark.visibility)
                    }
            except (IndexError, AttributeError):
                continue
                
        return landmarks_dict
    
    def is_hand_contacting_bag(self, hand_pos: tuple) -> bool:
        """Check if hand position is contacting the heavy bag."""
        if not self.bag_detected or self.bag_area is None:
            return False
        
        hand_x, hand_y = hand_pos
        
        # Check if hand is within contact threshold of bag area
        bag_x, bag_y, bag_w, bag_h = self.bag_area
        
        # Calculate distance from hand to bag edges
        closest_x = max(bag_x, min(hand_x, bag_x + bag_w))
        closest_y = max(bag_y, min(hand_y, bag_y + bag_h))
        
        distance = np.sqrt((hand_x - closest_x)**2 + (hand_y - closest_y)**2)
        
        return distance <= self.contact_threshold
    
    def detect_punch_with_contact(self, landmarks_dict: dict, arm: str) -> bool:
        """Detect punch with heavy bag contact requirement."""
        if not landmarks_dict or self.setup_phase:
            return False
        
        # Must have bag detected
        if not self.bag_detected:
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
        if any(lm.get('visibility', 0) < 0.4 for lm in [shoulder, elbow, wrist]):
            return False
        
        # Check if hand is contacting bag
        hand_pos = (wrist['x'], wrist['y'])
        if not self.is_hand_contacting_bag(hand_pos):
            return False  # No contact = no punch
        
        # Calculate elbow angle
        current_angle = self.calculate_angle(
            [shoulder['x'], shoulder['y']],
            [elbow['x'], elbow['y']],
            [wrist['x'], wrist['y']]
        )
        
        # Relaxed thresholds since we have contact validation
        bent_threshold = 120
        extended_threshold = 145
        
        # State machine
        if self.arm_states[arm] == 'unknown':
            self.arm_states[arm] = 'bent' if current_angle < bent_threshold else 'extended'
            return False
        
        # Detect punch: bent -> extended WITH bag contact
        if self.arm_states[arm] == 'bent' and current_angle > extended_threshold:
            # Punch detected with bag contact!
            self.arm_states[arm] = 'extended'
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            self.last_punch_time[arm] = current_time
            
            # Calculate distance to bag for debugging
            bag_x, bag_y, bag_w, bag_h = self.bag_area
            bag_center_x = bag_x + bag_w // 2
            bag_center_y = bag_y + bag_h // 2
            distance_to_bag = np.sqrt((wrist['x'] - bag_center_x)**2 + (wrist['y'] - bag_center_y)**2)
            
            # Simple classification
            punch_type = 'jab' if arm == 'left' else 'straight'
            punch_number = 1 if arm == 'left' else 2
            
            forward_distance = abs(wrist['x'] - shoulder['x'])
            if forward_distance < 80:
                punch_type = 'hook'
                punch_number = 3 if arm == 'left' else 4
            
            # Add to recent punches
            punch_info = {
                'arm': arm,
                'time': current_time,
                'angle': current_angle,
                'type': punch_type,
                'number': punch_number,
                'bag_distance': f"{distance_to_bag:.0f}px"
            }
            self.recent_punches.append(punch_info)
            
            if len(self.recent_punches) > 5:
                self.recent_punches.pop(0)
            
            print(f"BAG CONTACT: #{punch_number} {arm.upper()} {punch_type.upper()} ({current_angle:.0f}° / {distance_to_bag:.0f}px from bag)")
            return True
        
        # Return to bent
        elif self.arm_states[arm] == 'extended' and current_angle < bent_threshold:
            self.arm_states[arm] = 'bent'
        
        return False
    
    def update_setup_phase(self):
        """Update setup phase and bag detection."""
        self.frames_processed += 1
        if self.frames_processed >= self.min_frames_before_detection:
            if self.setup_phase:
                self.setup_phase = False
                if self.bag_detected:
                    print("SETUP COMPLETE - Heavy bag detected! Contact-based punch detection active!")
                    print(f"Bag area: {self.bag_area}")
                else:
                    print("SETUP COMPLETE - No heavy bag detected. Try pressing 'B' to re-detect.")
    
    def update_fps(self):
        """Update FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_camera_view(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Draw camera view with positioning guidance."""
        height, width = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.putText(frame, 'WIDE-ANGLE HEAVY BAG MODE', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
        
        # Resolution info
        cv2.putText(frame, f'Resolution: {width}x{height}', (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Bag detection status
        if self.bag_detected:
            cv2.putText(frame, 'BAG DETECTED', (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw bag area
            if self.bag_area:
                x, y, w, h = self.bag_area
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv2.putText(frame, 'HEAVY BAG', (x, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'SCANNING FOR BAG...', (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # Setup status with positioning guide
        if self.setup_phase:
            countdown = max(0, self.min_frames_before_detection - self.frames_processed)
            setup_seconds = countdown // 30 + 1
            cv2.putText(frame, f'SETUP: {setup_seconds}s - Position yourself and bag in view', 
                       (width - 600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw positioning guides
            center_x, center_y = width // 2, height // 2
            
            # Center line for positioning
            cv2.line(frame, (center_x, 0), (center_x, height), (255, 255, 0), 2)
            cv2.putText(frame, 'CENTER', (center_x - 40, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Suggested areas
            left_third = width // 3
            right_third = 2 * width // 3
            
            cv2.putText(frame, 'YOU', (left_third - 30, center_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.putText(frame, 'BAG', (right_third - 30, center_y + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
        
        # Draw hand contact indicators
        if landmarks_dict and self.bag_detected:
            for arm in ['left', 'right']:
                wrist_key = f'{arm}_wrist'
                if wrist_key in landmarks_dict:
                    wrist = landmarks_dict[wrist_key]
                    hand_pos = (int(wrist['x']), int(wrist['y']))
                    
                    # Check contact
                    is_contact = self.is_hand_contacting_bag(hand_pos)
                    color = (0, 255, 0) if is_contact else (0, 0, 255)
                    
                    cv2.circle(frame, hand_pos, 15, color, -1)
                    cv2.putText(frame, arm.upper(), (hand_pos[0] + 20, hand_pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def create_data_window(self) -> np.ndarray:
        """Create data display."""
        data_window = np.zeros((900, 1400, 3), dtype=np.uint8)
        current_time = time.time()
        
        # Title
        cv2.putText(data_window, 'WIDE-ANGLE HEAVY BAG TRAINER', (200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
        
        # Session info
        session_duration = int(current_time - self.session_start)
        minutes, seconds = divmod(session_duration, 60)
        cv2.putText(data_window, f'Session: {minutes:02d}:{seconds:02d} | FPS: {self.current_fps:.1f}', 
                   (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Bag status
        bag_status = "BAG DETECTED" if self.bag_detected else "NO BAG DETECTED"
        bag_color = (0, 255, 0) if self.bag_detected else (255, 255, 0)
        cv2.putText(data_window, f'Status: {bag_status}', (50, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, bag_color, 2)
        
        # Positioning tips
        if not self.bag_detected:
            tips = [
                "POSITIONING TIPS:",
                "• Stand on left side of camera view",
                "• Place heavy bag on right side",
                "• Step back until both fit in frame", 
                "• Press 'B' to re-scan for bag"
            ]
            for i, tip in enumerate(tips):
                color = (255, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(data_window, tip, (50, 170 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Contact requirement
        cv2.putText(data_window, 'Only counts punches that contact the heavy bag', 
                   (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Punch counters
        counter_y = 350
        
        # Left
        cv2.rectangle(data_window, (50, counter_y), (650, counter_y + 150), (255, 0, 255), 3)
        cv2.putText(data_window, 'LEFT CONTACTS', (200, counter_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['left']), (300, counter_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 100, 255), 8)
        
        # Right
        cv2.rectangle(data_window, (700, counter_y), (1300, counter_y + 150), (255, 0, 255), 3)
        cv2.putText(data_window, 'RIGHT CONTACTS', (830, counter_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['right']), (950, counter_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 100), 8)
        
        # Total
        total_y = 540
        cv2.rectangle(data_window, (350, total_y), (1000, total_y + 180), (255, 255, 255), 4)
        cv2.putText(data_window, 'TOTAL CONTACTS', (500, total_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
        cv2.putText(data_window, str(self.punch_counts['total']), (600, total_y + 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 10)
        
        # Recent contacts
        recent_y = 760
        cv2.putText(data_window, 'RECENT BAG CONTACTS:', (50, recent_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        for i, punch in enumerate(self.recent_punches[-3:]):
            age = current_time - punch['time']
            if age < 10:
                y_pos = recent_y + 40 + (i * 30)
                text = f"#{punch['number']} {punch['arm'].upper()} {punch['type'].upper()} ({punch['bag_distance']} from bag)"
                cv2.putText(data_window, text, (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return data_window
    
    def process_frame(self, frame: np.ndarray):
        """Process frame with pose detection and bag detection."""
        frame_height, frame_width = frame.shape[:2]
        
        # Update bag detection during setup phase
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
        """Handle input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return False
        elif key == ord('r'):
            self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
            self.recent_punches = []
            self.arm_states = {'left': 'unknown', 'right': 'unknown'}
            self.session_start = time.time()
            print("RESET: All counts cleared!")
        elif key == ord('b'):  # Re-detect bag
            self.bag_detected = False
            self.bag_area = None
            print("Re-detecting heavy bag...")
        
        return True
    
    def run(self):
        """Main loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*70)
        print("WIDE-ANGLE HEAVY BAG TRAINER")
        print("="*70)
        print("FEATURES:")
        print("• High resolution (1920x1080) for wide field of view")
        print("• 6-second setup time for positioning")
        print("• Visual positioning guides during setup")
        print("• Only counts punches that contact the bag")
        print("• ESC disabled to prevent accidental quit")
        print("")
        print("POSITIONING:")
        print("• Stand on LEFT side of camera view")
        print("• Place heavy bag on RIGHT side")
        print("• Step back until both you and bag fit in frame")
        print("")
        print("CONTROLS:")
        print("• R = Reset counts")
        print("• B = Re-detect bag")
        print("• Q = Quit (ESC disabled to prevent accidents)")
        print("="*70 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.update_setup_phase()
                processed_frame, landmarks_dict = self.process_frame(frame)
                
                if landmarks_dict:
                    self.detect_punch_with_contact(landmarks_dict, 'left')
                    self.detect_punch_with_contact(landmarks_dict, 'right')
                
                camera_display = self.draw_camera_view(processed_frame, landmarks_dict)
                data_display = self.create_data_window()
                
                self.update_fps()
                
                cv2.imshow('Wide Camera View', camera_display)
                cv2.imshow('Data Display', data_display)
                cv2.moveWindow('Wide Camera View', 50, 50)
                cv2.moveWindow('Data Display', 800, 50)
                
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
    """Simple heavy bag detector using computer vision."""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=50
        )
        self.frame_count = 0
        
    def detect_bag(self, frame: np.ndarray) -> tuple:
        """Detect heavy bag in frame. Returns (x, y, w, h) or None."""
        self.frame_count += 1
        
        # Need several frames to establish background
        if self.frame_count < 15:  # More frames for wider view
            self.background_subtractor.apply(frame)
            return None
        
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame, learningRate=0.005)
            
            # Morphological operations to clean up mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (likely the bag)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Filter by size - bag should be reasonably large
                if area > 8000:  # Larger minimum for wide view
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Filter by aspect ratio - bags are typically tall
                    aspect_ratio = h / w if w > 0 else 0
                    if aspect_ratio > 1.1:  # Slightly less strict for wide angle
                        return (x, y, w, h)
            
            return None
            
        except Exception as e:
            return None


def main():
    """Main entry point."""
    try:
        trainer = WideHeavyBagTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())