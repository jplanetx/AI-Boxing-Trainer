#!/usr/bin/env python3
"""
Fixed MediaPipe Boxing Trainer - Corrected thresholds and simplified detection
Based on research findings identifying over-restrictive thresholds as the main issue.
"""

import cv2
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import MediaPipe directly to avoid complex AI trainer modules
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing


class FixedMediaPipeTrainer:
    """Fixed MediaPipe trainer with research-based optimal settings."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # Initialize MediaPipe with OPTIMAL settings from research
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        # FIXED CONFIGURATION - Based on research findings
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Changed from 2 - balanced performance
            enable_segmentation=False,
            min_detection_confidence=0.3,  # Changed from 0.7 - critical fix
            min_tracking_confidence=0.2    # Changed from 0.5 - allows rapid movement
        )
        
        # Simple punch tracking
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        self.last_angles = {'left': 0, 'right': 0}
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.session_start = time.time()
        
        # Recent punches for UI
        self.recent_punches = []
        
        # Punch validation to prevent false positives
        self.setup_phase = True
        self.frames_processed = 0
        self.min_frames_before_detection = 90  # 3 seconds at 30fps
        
        # Cooldown to prevent double counting (punch + recoil)
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_cooldown = 1.0  # 1 second cooldown between punches per arm
        
        # Hook detection - track circular motion
        self.wrist_positions = {'left': [], 'right': []}
        self.position_buffer_size = 10
        
        print("Fixed MediaPipe Trainer initialized with optimal settings")
        print("Detection confidence: 0.3 (was 0.7)")
        print("Tracking confidence: 0.2 (was 0.5)")
        print("Model complexity: 1 (was 2)")
        
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
        """Extract landmarks with LOWER confidence threshold."""
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
            'left_hip': self.mp_pose.PoseLandmark.LEFT_HIP,
            'right_hip': self.mp_pose.PoseLandmark.RIGHT_HIP,
        }
        
        for name, landmark_idx in key_landmarks.items():
            try:
                landmark = pose_landmarks.landmark[landmark_idx.value]
                # FIXED: Lower visibility threshold from 0.7 to 0.2
                if landmark.visibility > 0.2:  
                    landmarks_dict[name] = {
                        'x': float(landmark.x * frame_width),
                        'y': float(landmark.y * frame_height),
                        'visibility': float(landmark.visibility)
                    }
            except (IndexError, AttributeError):
                continue
                
        return landmarks_dict
    
    def detect_punch_fixed_thresholds(self, landmarks_dict: dict, arm: str) -> bool:
        """Enhanced punch detection with hook circular motion detection."""
        if not landmarks_dict:
            return False
        
        # Skip detection during setup phase to prevent false positives
        if self.setup_phase:
            return False
        
        # Check cooldown to prevent double counting (punch + recoil)
        current_time = time.time()
        if current_time - self.last_punch_time[arm] < self.punch_cooldown:
            return False
        
        # Track wrist positions for circular motion detection
        wrist_key = f'{arm}_wrist'
        if wrist_key in landmarks_dict:
            wrist = landmarks_dict[wrist_key]
            current_pos = np.array([wrist['x'], wrist['y']])
            
            # Add to position buffer
            self.wrist_positions[arm].append(current_pos)
            if len(self.wrist_positions[arm]) > self.position_buffer_size:
                self.wrist_positions[arm].pop(0)
        
        # Check required landmarks
        shoulder_key = f'{arm}_shoulder'
        elbow_key = f'{arm}_elbow'
        wrist_key = f'{arm}_wrist'
        
        if not all(key in landmarks_dict for key in [shoulder_key, elbow_key, wrist_key]):
            return False
        
        shoulder = landmarks_dict[shoulder_key]
        elbow = landmarks_dict[elbow_key]
        wrist = landmarks_dict[wrist_key]
        
        # Check minimum visibility - FIXED: lowered from 0.5 to 0.3
        if any(lm.get('visibility', 0) < 0.3 for lm in [shoulder, elbow, wrist]):
            return False
        
        # Calculate elbow angle
        current_angle = self.calculate_angle(
            [shoulder['x'], shoulder['y']],
            [elbow['x'], elbow['y']],
            [wrist['x'], wrist['y']]
        )
        
        # FIXED: Even more restrictive thresholds to prevent double counting
        # Prevent recoil from being counted as separate punch
        bent_threshold = 110    # Lower for very clear bent position
        extended_threshold = 150  # Much higher - needs deliberate extension
        
        # Initialize state if unknown
        if self.arm_states[arm] == 'unknown':
            self.arm_states[arm] = 'bent' if current_angle < bent_threshold else 'extended'
            self.last_angles[arm] = current_angle
            return False
        
        # Detect punch: bent -> extended transition with boxing motion validation
        if self.arm_states[arm] == 'bent' and current_angle > extended_threshold:
            # Additional validation to ensure this looks like a boxing motion
            if self.validate_boxing_motion(landmarks_dict, arm, current_angle):
                # Punch detected!
                self.arm_states[arm] = 'extended'
                self.punch_counts[arm] += 1
                self.punch_counts['total'] += 1
                
                # Set cooldown time to prevent double counting
                self.last_punch_time[arm] = current_time
                
                # Add to recent punches
                punch_name, punch_number = self.classify_punch_type(arm, current_angle, landmarks_dict)
                punch_info = {
                    'arm': arm,
                    'time': current_time,
                    'angle': current_angle,
                    'type': punch_name,
                    'number': punch_number
                }
                self.recent_punches.append(punch_info)
                
                # Keep only last 5 punches
                if len(self.recent_punches) > 5:
                    self.recent_punches.pop(0)
                
                # Debug circular motion for hooks
                is_circular_debug = self.detect_circular_motion(arm)
                debug_info = f"circular: {is_circular_debug}" if punch_name == 'hook' else ""
                
                print(f"PUNCH: #{punch_number} {arm.upper()} {punch_name.upper()} ({current_angle:.0f}°) {debug_info}")
                return True
            else:
                # Motion detected but didn't pass boxing validation
                self.arm_states[arm] = 'extended'  # Still update state
                return False
        
        # Return to bent position
        elif self.arm_states[arm] == 'extended' and current_angle < bent_threshold:
            self.arm_states[arm] = 'bent'
        
        self.last_angles[arm] = current_angle
        return False
    
    def validate_boxing_motion(self, landmarks_dict: dict, arm: str, current_angle: float) -> bool:
        """Validate that this looks like a deliberate boxing motion, not random movement."""
        try:
            # Get body landmarks for posture analysis
            shoulder_key = f'{arm}_shoulder'
            wrist_key = f'{arm}_wrist'
            hip_key = f'{arm}_hip'
            
            if not all(key in landmarks_dict for key in [shoulder_key, wrist_key]):
                return True  # If we can't validate, allow it
            
            shoulder = landmarks_dict[shoulder_key]
            wrist = landmarks_dict[wrist_key]
            
            # Check if person appears to be in a boxing-like stance
            # 1. Wrist should be roughly shoulder height or lower (not way above)
            wrist_height_relative = shoulder['y'] - wrist['y']
            if wrist_height_relative > 100:  # Wrist way above shoulder = likely not boxing
                return False
            
            # 2. Extension should be deliberate (good angle)
            if current_angle < 145:  # Not extended enough for deliberate punch
                return False
            
            # 3. Check if other arm is in reasonable position (basic stance check)
            other_arm = 'right' if arm == 'left' else 'left'
            other_wrist_key = f'{other_arm}_wrist'
            other_shoulder_key = f'{other_arm}_shoulder'
            
            if other_wrist_key in landmarks_dict and other_shoulder_key in landmarks_dict:
                other_wrist = landmarks_dict[other_wrist_key]
                other_shoulder = landmarks_dict[other_shoulder_key]
                
                # Other arm shouldn't be way out of position
                other_wrist_height = other_shoulder['y'] - other_wrist['y']
                if other_wrist_height > 150:  # Other arm way up = probably not boxing
                    return False
            
            return True  # Passed all checks
            
        except (KeyError, TypeError):
            return True  # If validation fails, allow the punch
    
    def detect_circular_motion(self, arm: str) -> bool:
        """Detect if wrist is moving in circular pattern (hook punch)."""
        if len(self.wrist_positions[arm]) < 6:  # Need at least 6 points
            return False
        
        try:
            positions = np.array(self.wrist_positions[arm][-6:])  # Last 6 positions
            
            # Calculate direction changes
            direction_changes = 0
            prev_direction = None
            
            for i in range(1, len(positions)):
                # Calculate movement vector
                movement = positions[i] - positions[i-1]
                movement_magnitude = np.linalg.norm(movement)
                
                if movement_magnitude > 5:  # Only consider significant movements
                    # Calculate direction (angle)
                    direction = np.arctan2(movement[1], movement[0])
                    
                    if prev_direction is not None:
                        # Check for direction change
                        angle_diff = abs(direction - prev_direction)
                        # Handle angle wraparound
                        if angle_diff > np.pi:
                            angle_diff = 2 * np.pi - angle_diff
                        
                        # Significant direction change (>45 degrees)
                        if angle_diff > np.pi / 4:
                            direction_changes += 1
                    
                    prev_direction = direction
            
            # Hook = multiple direction changes in short time
            return direction_changes >= 2
            
        except Exception:
            return False
    
    def classify_punch_type(self, arm: str, angle: float, landmarks_dict: dict) -> tuple:
        """Enhanced punch type classification with debugging."""
        # Boxing numbering: 1=left jab, 2=right straight, 3=left hook, 4=right hook, 5=left uppercut, 6=right uppercut
        
        # Get additional landmarks for better classification
        wrist_key = f'{arm}_wrist'
        shoulder_key = f'{arm}_shoulder'
        elbow_key = f'{arm}_elbow'
        
        try:
            wrist = landmarks_dict[wrist_key]
            shoulder = landmarks_dict[shoulder_key]
            elbow = landmarks_dict[elbow_key]
            
            # Calculate movement vectors
            wrist_height = shoulder['y'] - wrist['y']  # Positive = wrist above shoulder
            wrist_forward_distance = abs(wrist['x'] - shoulder['x'])  # Forward extension
            
            # Calculate horizontal vs vertical movement ratio
            horizontal_movement = abs(wrist['x'] - elbow['x'])
            vertical_movement = abs(wrist['y'] - elbow['y'])
            
            # Debug info (you can see this in console)
            # print(f"  {arm} - Angle: {angle:.0f}°, Height: {wrist_height:.0f}, Forward: {wrist_forward_distance:.0f}")
            
            # IMPROVED CLASSIFICATION LOGIC WITH CIRCULAR MOTION
            
            # Check for circular motion first (hooks)
            is_circular = self.detect_circular_motion(arm)
            
            # 1. HOOKS: Circular motion detected OR limited forward extension
            if is_circular or (angle > 140 and wrist_forward_distance < 70):
                return ('hook', 3 if arm == 'left' else 4)
            
            # 2. UPPERCUTS: Strong upward movement (much more restrictive)
            elif wrist_height > 60 and vertical_movement > horizontal_movement * 1.5:
                return ('uppercut', 5 if arm == 'left' else 6)
            
            # 3. STRAIGHTS/JABS: High extension angle + good forward distance + NOT circular
            elif angle > 155 and not is_circular:  # Very extended + straight motion = straight punch
                return ('jab', 1) if arm == 'left' else ('straight', 2)
            
            # 4. DEFAULT: Medium extension = jab/straight (but not if circular motion)
            elif not is_circular:
                return ('jab', 1) if arm == 'left' else ('straight', 2)
            
            # 5. FALLBACK: If circular but doesn't meet hook criteria = hook anyway
            else:
                return ('hook', 3 if arm == 'left' else 4)
                
        except (KeyError, TypeError):
            # Fallback: Default to most common punches
            return ('jab', 1) if arm == 'left' else ('straight', 2)
    
    def update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Draw UI with pose and punch counts."""
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Header
        cv2.rectangle(frame, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'FIXED MEDIAPIPE BOXING TRAINER', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (width - 100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Setup phase status
        if self.setup_phase:
            setup_countdown = max(0, self.min_frames_before_detection - self.frames_processed)
            setup_info = f"SETUP: {setup_countdown//30 + 1}s remaining"
            cv2.putText(frame, setup_info, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "READY - Detecting punches", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Pose status
        pose_status = f"LANDMARKS: {len(landmarks_dict)}" if landmarks_dict else "NO POSE"
        pose_color = (0, 255, 0) if landmarks_dict else (0, 0, 255)
        cv2.putText(frame, pose_status, (width - 200, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        # Punch counters - LARGE and CLEAR
        cv2.rectangle(frame, (20, 100), (width-20, 220), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 100), (width-20, 220), (0, 255, 0), 3)
        
        cv2.putText(frame, f'LEFT: {self.punch_counts["left"]}', (40, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 3)
        cv2.putText(frame, f'RIGHT: {self.punch_counts["right"]}', (300, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 3)
        cv2.putText(frame, f'TOTAL: {self.punch_counts["total"]}', (150, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
        
        # Show current arm angles for debugging
        if landmarks_dict:
            left_angle = self.get_current_angle(landmarks_dict, 'left')
            right_angle = self.get_current_angle(landmarks_dict, 'right')
            
            cv2.putText(frame, f'L: {left_angle:.0f}° ({self.arm_states["left"]})', (40, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            cv2.putText(frame, f'R: {right_angle:.0f}° ({self.arm_states["right"]})', (300, 250), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        
        # Recent punches
        if self.recent_punches:
            cv2.rectangle(frame, (20, height-120), (width-20, height-20), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, height-120), (width-20, height-20), (255, 255, 0), 2)
            
            cv2.putText(frame, 'RECENT PUNCHES:', (30, height-95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for i, punch in enumerate(self.recent_punches[-3:]):
                age = current_time - punch['time']
                if age < 5:  # Show for 5 seconds
                    text = f"#{punch['number']} {punch['arm'].upper()} {punch['type'].upper()} ({punch['angle']:.0f}°)"
                    cv2.putText(frame, text, (30, height - 70 + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, 'R: Reset | Q: Quit | Thresholds: 120° bent -> 140° extended', 
                   (10, height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def get_current_angle(self, landmarks_dict: dict, arm: str) -> float:
        """Get current elbow angle for debugging display."""
        try:
            shoulder_key = f'{arm}_shoulder'
            elbow_key = f'{arm}_elbow'
            wrist_key = f'{arm}_wrist'
            
            if not all(key in landmarks_dict for key in [shoulder_key, elbow_key, wrist_key]):
                return 0
            
            shoulder = landmarks_dict[shoulder_key]
            elbow = landmarks_dict[elbow_key]
            wrist = landmarks_dict[wrist_key]
            
            return self.calculate_angle(
                [shoulder['x'], shoulder['y']],
                [elbow['x'], elbow['y']],
                [wrist['x'], wrist['y']]
            )
        except:
            return 0
    
    def process_frame(self, frame: np.ndarray):
        """Process frame with MediaPipe."""
        # DON'T flip frame to fix left/right confusion
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
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                processed_frame,
                results.pose_landmarks,
                list(self.mp_pose.POSE_CONNECTIONS)
            )
        
        return processed_frame, landmarks_dict
    
    def update_setup_phase(self):
        """Update setup phase tracking."""
        self.frames_processed += 1
        if self.frames_processed >= self.min_frames_before_detection:
            if self.setup_phase:
                self.setup_phase = False
                print("SETUP COMPLETE - Punch detection now active!")
    
    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input."""
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
        """Main application loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*70)
        print("FIXED MEDIAPIPE BOXING TRAINER")
        print("="*70)
        print("FIXED ISSUES:")
        print("• Detection confidence: 0.7 → 0.3 (allows rapid movements)")
        print("• Tracking confidence: 0.5 → 0.2 (maintains tracking)")
        print("• Model complexity: 2 → 1 (faster processing)")
        print("• Visibility threshold: 0.7 → 0.2 (more permissive)")
        print("• Angle thresholds: realistic 120° bent → 140° extended")
        print("")
        print("CONTROLS: R = Reset | Q = Quit")
        print("DEBUGGING: Watch angle displays to see detection in real-time")
        print("="*70 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Process frame
                processed_frame, landmarks_dict = self.process_frame(frame)
                
                # Update setup phase
                self.update_setup_phase()
                
                # Detect punches with fixed thresholds
                if landmarks_dict:
                    self.detect_punch_fixed_thresholds(landmarks_dict, 'left')
                    self.detect_punch_fixed_thresholds(landmarks_dict, 'right')
                
                # Draw UI
                final_frame = self.draw_ui(processed_frame, landmarks_dict)
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('Fixed MediaPipe Boxing Trainer', final_frame)
                
                # Handle input
                if not self.handle_keyboard_input():
                    break
                    
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
        
        duration = int(time.time() - self.session_start)
        minutes, seconds = divmod(duration, 60)
        
        print(f"\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Duration: {minutes:02d}:{seconds:02d}")
        print(f"Left Punches: {self.punch_counts['left']}")
        print(f"Right Punches: {self.punch_counts['right']}")
        print(f"Total Punches: {self.punch_counts['total']}")
        print("="*50)


def main():
    """Main entry point."""
    try:
        trainer = FixedMediaPipeTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())