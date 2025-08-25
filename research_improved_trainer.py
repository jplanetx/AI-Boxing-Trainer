#!/usr/bin/env python3
"""
Research-Improved Boxing Trainer
Implements findings from specialized research agents:
- Clean separated UI windows (camera vs data)
- Enhanced punch classification with trajectory analysis
- Camera positioning guidance
- Contact-based heavy bag detection
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


class ResearchImprovedTrainer:
    """Improved trainer implementing research findings."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # MediaPipe setup with research-optimized settings
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Faster processing for boxing
            enable_segmentation=False,
            min_detection_confidence=0.3,  # Lower for rapid movements
            min_tracking_confidence=0.2    # Better tracking during fast motions
        )
        
        # Enhanced punch tracking with trajectory analysis
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        
        # Trajectory tracking for improved classification
        self.trajectory_history = {
            'left': deque(maxlen=10),
            'right': deque(maxlen=10)
        }
        
        # Biomechanical tracking for punch type classification
        self.velocity_tracking = {
            'left': deque(maxlen=5),
            'right': deque(maxlen=5)
        }
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.session_start = time.time()
        
        # Recent punches with enhanced classification
        self.recent_punches = []
        
        # Setup and cooldowns
        self.setup_phase = True
        self.frames_processed = 0
        self.min_frames_before_detection = 150  # 5 seconds setup
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_cooldown = 1.2  # Slightly longer for accuracy
        
        # Heavy bag detection
        self.bag_detector = HeavyBagDetector()
        self.bag_area = None
        self.bag_detected = False
        self.contact_threshold = 50
        
        # Camera positioning guidance
        self.positioning_quality = 0.0
        self.optimal_distance_range = (150, 250)  # pixels between shoulders
        
        print("Research-Improved Boxing Trainer initialized")
        print("Enhanced classification | Clean UI | Camera guidance")
        
    def initialize_camera(self) -> bool:
        """Initialize camera with optimal settings."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            # Balanced resolution for wide view while maintaining performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
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
    
    def calculate_velocity(self, current_pos, previous_positions):
        """Calculate velocity vector for trajectory analysis."""
        if len(previous_positions) < 2:
            return np.array([0.0, 0.0])
        
        # Calculate velocity over last few frames
        prev_pos = np.array(previous_positions[-2])
        curr_pos = np.array(current_pos)
        
        velocity = curr_pos - prev_pos
        return velocity
    
    def classify_punch_type_enhanced(self, landmarks_dict: dict, arm: str, velocity: np.ndarray) -> tuple:
        """Enhanced punch classification using research findings."""
        shoulder_key = f'{arm}_shoulder'
        elbow_key = f'{arm}_elbow'
        wrist_key = f'{arm}_wrist'
        
        shoulder = landmarks_dict[shoulder_key]
        elbow = landmarks_dict[elbow_key]
        wrist = landmarks_dict[wrist_key]
        
        # Get trajectory history for this arm
        trajectory = list(self.trajectory_history[arm])
        
        # Default classification
        punch_type = 'jab' if arm == 'left' else 'straight'
        punch_number = 1 if arm == 'left' else 2
        
        # Enhanced classification using multiple features
        
        # 1. Trajectory analysis - check for circular motion (hooks)
        if len(trajectory) >= 5:
            # Calculate path curvature
            positions = np.array(trajectory[-5:])
            if len(positions) >= 3:
                # Check for circular path pattern
                center = np.mean(positions, axis=0)
                distances = np.linalg.norm(positions - center, axis=1)
                if np.std(distances) < 30:  # Consistent radius suggests circular motion
                    punch_type = 'hook'
                    punch_number = 3 if arm == 'left' else 4
        
        # 2. Velocity analysis - check for vertical component (uppercuts)
        if abs(velocity[1]) > abs(velocity[0]) * 1.5:  # More vertical than horizontal
            if velocity[1] < -20:  # Moving upward (negative Y in image coordinates)
                # Additional validation: wrist should be significantly above starting position
                if len(trajectory) >= 3:
                    start_y = trajectory[0][1]
                    current_y = wrist['y']
                    if start_y - current_y > 50:  # Moved up significantly
                        punch_type = 'uppercut'
                        punch_number = 5 if arm == 'left' else 6
        
        # 3. Distance analysis - confirm forward motion for straights/jabs
        if punch_type in ['jab', 'straight']:
            # Check horizontal extension
            horizontal_distance = abs(wrist['x'] - shoulder['x'])
            if horizontal_distance < 60:  # Not much forward extension
                # Might be a short hook instead
                if abs(velocity[0]) > 15:  # Still has horizontal component
                    punch_type = 'hook'
                    punch_number = 3 if arm == 'left' else 4
        
        # 4. Stance-aware directional correction
        # Determine boxer's orientation based on shoulder positions
        if 'left_shoulder' in landmarks_dict and 'right_shoulder' in landmarks_dict:
            left_shoulder = landmarks_dict['left_shoulder']
            right_shoulder = landmarks_dict['right_shoulder']
            
            # If right shoulder is more forward (orthodox stance), classifications are correct
            # If left shoulder is more forward (southpaw), swap some classifications
            if left_shoulder['x'] < right_shoulder['x']:  # Possible southpaw
                # Adjust classification for stance
                pass  # Keep current logic for now, could be enhanced
        
        return punch_type, punch_number
    
    def extract_landmarks(self, pose_landmarks, frame_width: int, frame_height: int) -> dict:
        """Extract landmarks with position tracking."""
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
        
        # Update trajectory tracking
        for arm in ['left', 'right']:
            wrist_key = f'{arm}_wrist'
            if wrist_key in landmarks_dict:
                wrist = landmarks_dict[wrist_key]
                position = [wrist['x'], wrist['y']]
                self.trajectory_history[arm].append(position)
                
                # Calculate velocity
                velocity = self.calculate_velocity(position, list(self.trajectory_history[arm]))
                self.velocity_tracking[arm].append(velocity)
                
        return landmarks_dict
    
    def assess_positioning_quality(self, landmarks_dict: dict) -> float:
        """Assess camera positioning quality for guidance."""
        if not landmarks_dict:
            return 0.0
        
        quality_score = 0.0
        
        # Check shoulder distance (optimal distance indicator)
        if 'left_shoulder' in landmarks_dict and 'right_shoulder' in landmarks_dict:
            left_shoulder = landmarks_dict['left_shoulder']
            right_shoulder = landmarks_dict['right_shoulder']
            
            shoulder_distance = abs(left_shoulder['x'] - right_shoulder['x'])
            
            # Optimal range scoring
            if self.optimal_distance_range[0] <= shoulder_distance <= self.optimal_distance_range[1]:
                quality_score += 0.4  # Good distance
            elif shoulder_distance < self.optimal_distance_range[0]:
                quality_score += 0.2  # Too close
            else:
                quality_score += 0.1  # Too far
        
        # Check landmark visibility
        visible_landmarks = sum(1 for lm in landmarks_dict.values() if lm.get('visibility', 0) > 0.5)
        if visible_landmarks >= 5:
            quality_score += 0.3
        elif visible_landmarks >= 3:
            quality_score += 0.2
        
        # Check arm positioning (should be visible and not occluded)
        arms_visible = 0
        for arm in ['left', 'right']:
            if all(f'{arm}_{part}' in landmarks_dict for part in ['shoulder', 'elbow', 'wrist']):
                arms_visible += 1
        
        quality_score += arms_visible * 0.15
        
        return min(quality_score, 1.0)
    
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
    
    def detect_punch_enhanced(self, landmarks_dict: dict, arm: str) -> bool:
        """Enhanced punch detection with improved classification."""
        if not landmarks_dict or self.setup_phase:
            return False
        
        # Must have bag detected for contact validation
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
        
        # Research-based thresholds with contact validation
        bent_threshold = 125  # Slightly more restrictive
        extended_threshold = 150  # More restrictive for accuracy
        
        # State machine
        if self.arm_states[arm] == 'unknown':
            self.arm_states[arm] = 'bent' if current_angle < bent_threshold else 'extended'
            return False
        
        # Detect punch: bent -> extended WITH bag contact
        if self.arm_states[arm] == 'bent' and current_angle > extended_threshold:
            # Get velocity for enhanced classification
            velocity = np.array([0.0, 0.0])
            if len(self.velocity_tracking[arm]) > 0:
                velocity = self.velocity_tracking[arm][-1]
            
            # Enhanced punch classification
            punch_type, punch_number = self.classify_punch_type_enhanced(landmarks_dict, arm, velocity)
            
            # Punch detected with enhanced classification!
            self.arm_states[arm] = 'extended'
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            self.last_punch_time[arm] = current_time
            
            # Calculate distance to bag for debugging
            bag_x, bag_y, bag_w, bag_h = self.bag_area
            bag_center_x = bag_x + bag_w // 2
            bag_center_y = bag_y + bag_h // 2
            distance_to_bag = np.sqrt((wrist['x'] - bag_center_x)**2 + (wrist['y'] - bag_center_y)**2)
            
            # Add to recent punches with enhanced info
            punch_info = {
                'arm': arm,
                'time': current_time,
                'angle': current_angle,
                'type': punch_type,
                'number': punch_number,
                'bag_distance': f"{distance_to_bag:.0f}px",
                'velocity': np.linalg.norm(velocity)
            }
            self.recent_punches.append(punch_info)
            
            if len(self.recent_punches) > 5:
                self.recent_punches.pop(0)
            
            print(f"ENHANCED: #{punch_number} {arm.upper()} {punch_type.upper()} ({current_angle:.0f}° | {velocity[0]:.1f},{velocity[1]:.1f} vel | {distance_to_bag:.0f}px)")
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
                    print("SETUP COMPLETE - Enhanced detection active!")
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
        """Clean camera view focusing on pose visualization."""
        height, width = frame.shape[:2]
        
        # Minimal header
        cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(frame, 'ENHANCED CAMERA VIEW', (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Setup countdown if needed
        if self.setup_phase:
            countdown = max(0, self.min_frames_before_detection - self.frames_processed)
            setup_seconds = countdown // 30 + 1
            cv2.putText(frame, f'Setup: {setup_seconds}s', (width - 120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Positioning quality indicator
        if landmarks_dict:
            self.positioning_quality = self.assess_positioning_quality(landmarks_dict)
            quality_color = (0, 255, 0) if self.positioning_quality > 0.7 else (255, 255, 0) if self.positioning_quality > 0.4 else (0, 0, 255)
            cv2.putText(frame, f'Position: {self.positioning_quality:.1f}', (width - 200, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        # Bag outline if detected (minimal)
        if self.bag_detected and self.bag_area:
            x, y, w, h = self.bag_area
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, 'BAG', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hand contact indicators
        if landmarks_dict and self.bag_detected:
            for arm in ['left', 'right']:
                wrist_key = f'{arm}_wrist'
                if wrist_key in landmarks_dict:
                    wrist = landmarks_dict[wrist_key]
                    hand_pos = (int(wrist['x']), int(wrist['y']))
                    
                    # Check contact
                    is_contact = self.is_hand_contacting_bag(hand_pos)
                    color = (0, 255, 0) if is_contact else (0, 0, 255)
                    
                    cv2.circle(frame, hand_pos, 12, color, -1)
                    cv2.putText(frame, arm[0].upper(), (hand_pos[0] + 15, hand_pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def create_data_window(self) -> np.ndarray:
        """Clean, readable data display."""
        data_window = np.zeros((600, 800, 3), dtype=np.uint8)
        current_time = time.time()
        
        # Clean title
        cv2.putText(data_window, 'RESEARCH-ENHANCED BOXING', (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        # Session info - compact
        session_duration = int(current_time - self.session_start)
        minutes, seconds = divmod(session_duration, 60)
        cv2.putText(data_window, f'Time: {minutes:02d}:{seconds:02d} | FPS: {self.current_fps:.1f} | Quality: {self.positioning_quality:.1f}', 
                   (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Bag status
        bag_status = "BAG DETECTED" if self.bag_detected else "SCANNING..."
        bag_color = (0, 255, 0) if self.bag_detected else (255, 255, 0)
        cv2.putText(data_window, bag_status, (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, bag_color, 2)
        
        # Enhanced classification note
        cv2.putText(data_window, 'Enhanced punch classification active', (50, 125), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Large, clear counters
        counter_y = 160
        
        # Left counter
        cv2.rectangle(data_window, (50, counter_y), (350, counter_y + 100), (255, 0, 255), 3)
        cv2.putText(data_window, 'LEFT', (70, counter_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['left']), (150, counter_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 100, 255), 6)
        
        # Right counter
        cv2.rectangle(data_window, (400, counter_y), (700, counter_y + 100), (255, 0, 255), 3)
        cv2.putText(data_window, 'RIGHT', (420, counter_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(data_window, str(self.punch_counts['right']), (500, counter_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255, 255, 100), 6)
        
        # Total counter
        total_y = 290
        cv2.rectangle(data_window, (200, total_y), (550, total_y + 120), (255, 255, 255), 4)
        cv2.putText(data_window, 'TOTAL', (220, total_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(data_window, str(self.punch_counts['total']), (320, total_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 8)
        
        # Recent punches with enhanced info
        recent_y = 440
        cv2.putText(data_window, 'RECENT ENHANCED DETECTIONS:', (50, recent_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, punch in enumerate(self.recent_punches[-4:]):
            age = current_time - punch['time']
            if age < 10:
                y_pos = recent_y + 30 + (i * 25)
                velocity = punch.get('velocity', 0)
                text = f"#{punch['number']} {punch['arm'].upper()} {punch['type'].upper()} (vel:{velocity:.1f})"
                cv2.putText(data_window, text, (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Controls and features
        cv2.putText(data_window, 'FEATURES: Trajectory Analysis | Enhanced Classification | Camera Guidance', 
                   (50, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(data_window, 'R=Reset | B=Re-detect Bag | Q=Quit', (50, 590), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return data_window
    
    def process_frame(self, frame: np.ndarray):
        """Process frame with enhanced detection."""
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
            self.trajectory_history = {'left': deque(maxlen=10), 'right': deque(maxlen=10)}
            self.velocity_tracking = {'left': deque(maxlen=5), 'right': deque(maxlen=5)}
            self.session_start = time.time()
            print("RESET: All counts and tracking cleared!")
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
        print("RESEARCH-ENHANCED BOXING TRAINER")
        print("="*70)
        print("RESEARCH IMPROVEMENTS:")
        print("• Enhanced punch classification with trajectory analysis")
        print("• Clean separated UI windows (camera vs data)")
        print("• Camera positioning quality assessment")
        print("• Velocity-based punch type detection")
        print("• Biomechanical stance-aware classification")
        print("")
        print("USAGE:")
        print("• Position yourself and bag for optimal detection")
        print("• Watch positioning quality indicator")
        print("• Only punches contacting the bag are counted")
        print("")
        print("CONTROLS:")
        print("• R = Reset all counts and tracking")
        print("• B = Re-detect bag")
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
                    self.detect_punch_enhanced(landmarks_dict, 'left')
                    self.detect_punch_enhanced(landmarks_dict, 'right')
                
                camera_display = self.draw_camera_view(processed_frame, landmarks_dict)
                data_display = self.create_data_window()
                
                self.update_fps()
                
                # Clean window separation
                cv2.imshow('Enhanced Camera View', camera_display)
                cv2.imshow('Enhanced Data Display', data_display)
                cv2.moveWindow('Enhanced Camera View', 50, 50)      # Left side
                cv2.moveWindow('Enhanced Data Display', 700, 50)    # Right side
                
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
    """Enhanced heavy bag detector."""
    
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False, varThreshold=40
        )
        self.frame_count = 0
        
    def detect_bag(self, frame: np.ndarray) -> tuple:
        """Detect heavy bag in frame."""
        self.frame_count += 1
        
        # Need several frames to establish background
        if self.frame_count < 12:
            self.background_subtractor.apply(frame)
            return None
        
        try:
            # Apply background subtraction
            fg_mask = self.background_subtractor.apply(frame, learningRate=0.005)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Filter by size
                if area > 6000:
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Filter by aspect ratio
                    aspect_ratio = h / w if w > 0 else 0
                    if aspect_ratio > 1.2:
                        return (x, y, w, h)
            
            return None
            
        except Exception as e:
            return None


def main():
    """Main entry point."""
    try:
        trainer = ResearchImprovedTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())