#!/usr/bin/env python3
"""
Shadowboxing Trainer - No bag required, optimized landmark detection
Designed to work with challenging lighting/positioning conditions
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


class ShadowboxTrainer:
    """Shadowboxing trainer optimized for landmark detection."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # MediaPipe setup with MAXIMUM detection sensitivity
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # HIGHEST complexity for better detection
            enable_segmentation=False,
            min_detection_confidence=0.1,  # VERY low - detect anything
            min_tracking_confidence=0.1    # VERY low - track anything
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
        
        # Very short setup for immediate testing
        self.setup_phase = True
        self.frames_processed = 0
        self.min_frames_before_detection = 60  # 2 seconds only
        self.last_punch_time = {'left': 0, 'right': 0}
        self.punch_cooldown = 0.3  # Very short for testing
        
        # Position tracking for visibility improvement
        self.position_guidance = {
            'too_close': False,
            'too_far': False,
            'good_position': False
        }
        
        print("Shadowboxing Trainer initialized")
        print("NO BAG REQUIRED - Pure shadowboxing mode")
        print("Optimized for challenging lighting conditions")
        
    def initialize_camera(self) -> bool:
        """Initialize camera with auto-exposure and gain."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            # Camera settings optimized for pose detection
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Try to optimize camera settings for better landmark detection
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Enable auto exposure
                self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)      # Moderate brightness
                self.cap.set(cv2.CAP_PROP_CONTRAST, 0.5)        # Moderate contrast
            except:
                pass  # Some cameras don't support these settings
            
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                return False
                
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera optimized: {actual_width}x{actual_height}")
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
        """Extract landmarks with ultra-low visibility threshold."""
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
                # ULTRA LOW visibility threshold - accept almost anything
                if landmark.visibility > 0.1:  
                    landmarks_dict[name] = {
                        'x': float(landmark.x * frame_width),
                        'y': float(landmark.y * frame_height),
                        'visibility': float(landmark.visibility)
                    }
            except (IndexError, AttributeError):
                continue
                
        return landmarks_dict
    
    def assess_positioning(self, landmarks_dict: dict) -> dict:
        """Provide positioning guidance to improve landmark detection."""
        guidance = {
            'distance': 'unknown',
            'lighting': 'unknown',
            'visibility_score': 0.0,
            'suggestions': []
        }
        
        if not landmarks_dict:
            guidance['suggestions'] = ['No pose detected - check lighting and distance']
            return guidance
        
        # Calculate visibility score
        total_visibility = sum(lm.get('visibility', 0) for lm in landmarks_dict.values())
        guidance['visibility_score'] = total_visibility / max(len(landmarks_dict), 1)
        
        # Distance assessment based on shoulder width
        if 'left_shoulder' in landmarks_dict and 'right_shoulder' in landmarks_dict:
            shoulder_distance = abs(landmarks_dict['left_shoulder']['x'] - landmarks_dict['right_shoulder']['x'])
            
            if shoulder_distance < 80:
                guidance['distance'] = 'too_far'
                guidance['suggestions'].append('Move closer to camera')
            elif shoulder_distance > 300:
                guidance['distance'] = 'too_close'
                guidance['suggestions'].append('Move back from camera')
            else:
                guidance['distance'] = 'good'
        
        # Wrist visibility assessment
        missing_wrists = []
        low_vis_wrists = []
        
        for arm in ['left', 'right']:
            wrist_key = f'{arm}_wrist'
            if wrist_key not in landmarks_dict:
                missing_wrists.append(arm)
            elif landmarks_dict[wrist_key].get('visibility', 0) < 0.3:
                low_vis_wrists.append(f"{arm}({landmarks_dict[wrist_key]['visibility']:.2f})")
        
        if missing_wrists:
            guidance['suggestions'].append(f'Missing {", ".join(missing_wrists)} wrist(s) - adjust lighting/position')
        
        if low_vis_wrists:
            guidance['suggestions'].append(f'Low visibility: {", ".join(low_vis_wrists)} - improve lighting')
        
        # Overall lighting assessment
        if guidance['visibility_score'] < 0.4:
            guidance['lighting'] = 'poor'
            guidance['suggestions'].append('Improve lighting - face a window or add lamps')
        elif guidance['visibility_score'] > 0.7:
            guidance['lighting'] = 'good'
        else:
            guidance['lighting'] = 'fair'
        
        return guidance
    
    def detect_punch_shadowbox(self, landmarks_dict: dict, arm: str) -> bool:
        """Simplified shadowboxing punch detection - no bag required."""
        if not landmarks_dict or self.setup_phase:
            return False
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_punch_time[arm] < self.punch_cooldown:
            return False
        
        # Check required landmarks with fallback
        shoulder_key = f'{arm}_shoulder'
        elbow_key = f'{arm}_elbow'
        wrist_key = f'{arm}_wrist'
        
        if not all(key in landmarks_dict for key in [shoulder_key, elbow_key, wrist_key]):
            return False
        
        shoulder = landmarks_dict[shoulder_key]
        elbow = landmarks_dict[elbow_key]
        wrist = landmarks_dict[wrist_key]
        
        # ULTRA LOW visibility requirements for challenging conditions
        if any(lm.get('visibility', 0) < 0.15 for lm in [shoulder, elbow, wrist]):
            return False
        
        # Calculate elbow angle
        current_angle = self.calculate_angle(
            [shoulder['x'], shoulder['y']],
            [elbow['x'], elbow['y']],
            [wrist['x'], wrist['y']]
        )
        
        # VERY relaxed thresholds for shadowboxing
        bent_threshold = 150   # Very relaxed
        extended_threshold = 170  # Very relaxed
        
        # State machine
        if self.arm_states[arm] == 'unknown':
            self.arm_states[arm] = 'bent' if current_angle < bent_threshold else 'extended'
            return False
        
        # Detect punch: bent -> extended (NO BAG CONTACT REQUIRED)
        if self.arm_states[arm] == 'bent' and current_angle > extended_threshold:
            # Punch detected!
            self.arm_states[arm] = 'extended'
            self.punch_counts[arm] += 1
            self.punch_counts['total'] += 1
            self.last_punch_time[arm] = current_time
            
            # Simple classification
            punch_type = 'jab' if arm == 'left' else 'straight'
            punch_number = 1 if arm == 'left' else 2
            
            # Check for hook (less forward extension)
            forward_distance = abs(wrist['x'] - shoulder['x'])
            if forward_distance < 100:  # More relaxed hook detection
                punch_type = 'hook'
                punch_number = 3 if arm == 'left' else 4
            
            # Add to recent punches
            punch_info = {
                'arm': arm,
                'time': current_time,
                'angle': current_angle,
                'type': punch_type,
                'number': punch_number,
                'visibility': wrist.get('visibility', 0)
            }
            self.recent_punches.append(punch_info)
            
            if len(self.recent_punches) > 5:
                self.recent_punches.pop(0)
            
            print(f"SHADOWBOX: #{punch_number} {arm.upper()} {punch_type.upper()} ({current_angle:.0f}° | vis:{wrist.get('visibility', 0):.2f})")
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
                print("SHADOWBOX READY - Start throwing punches!")
    
    def update_fps(self):
        """Update FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_camera_view(self, frame: np.ndarray, landmarks_dict: dict, guidance: dict) -> np.ndarray:
        """Clean camera view with positioning guidance."""
        height, width = frame.shape[:2]
        
        # Header
        cv2.rectangle(frame, (0, 0), (width, 90), (0, 0, 0), -1)
        cv2.putText(frame, 'SHADOWBOX MODE - NO BAG NEEDED', (20, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Setup countdown
        if self.setup_phase:
            countdown = max(0, self.min_frames_before_detection - self.frames_processed)
            setup_seconds = countdown // 30 + 1
            cv2.putText(frame, f'Setup: {setup_seconds}s', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, 'DETECTION ACTIVE', (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Positioning guidance
        if guidance['visibility_score'] > 0:
            score_color = (0, 255, 0) if guidance['visibility_score'] > 0.7 else (255, 255, 0) if guidance['visibility_score'] > 0.4 else (0, 0, 255)
            cv2.putText(frame, f'Visibility: {guidance["visibility_score"]:.2f}', (20, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_color, 2)
        
        # Distance guidance
        distance_text = guidance.get('distance', 'unknown')
        distance_color = (0, 255, 0) if distance_text == 'good' else (255, 255, 0)
        cv2.putText(frame, f'Distance: {distance_text}', (width - 200, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, distance_color, 2)
        
        # Lighting assessment
        lighting_text = guidance.get('lighting', 'unknown')
        lighting_color = (0, 255, 0) if lighting_text == 'good' else (255, 255, 0) if lighting_text == 'fair' else (0, 0, 255)
        cv2.putText(frame, f'Lighting: {lighting_text}', (width - 200, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, lighting_color, 2)
        
        # Hand positions with visibility info
        if landmarks_dict:
            for arm in ['left', 'right']:
                wrist_key = f'{arm}_wrist'
                if wrist_key in landmarks_dict:
                    wrist = landmarks_dict[wrist_key]
                    visibility = wrist.get('visibility', 0)
                    
                    hand_pos = (int(wrist['x']), int(wrist['y']))
                    
                    # Color based on visibility
                    if visibility > 0.7:
                        color = (0, 255, 0)  # Green - excellent
                    elif visibility > 0.4:
                        color = (255, 255, 0)  # Yellow - fair
                    else:
                        color = (0, 0, 255)  # Red - poor
                    
                    cv2.circle(frame, hand_pos, 15, color, -1)
                    cv2.putText(frame, f'{arm[0].upper()}:{visibility:.2f}', 
                               (hand_pos[0] + 20, hand_pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Suggestions at bottom
        if guidance.get('suggestions'):
            suggestion_text = guidance['suggestions'][0]  # Show first suggestion
            cv2.putText(frame, f'TIP: {suggestion_text}', (20, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def create_data_window(self, guidance: dict) -> np.ndarray:
        """Shadowboxing data display."""
        data_window = np.zeros((600, 800, 3), dtype=np.uint8)
        current_time = time.time()
        
        # Title
        cv2.putText(data_window, 'SHADOWBOXING DATA', (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Session info
        session_duration = int(current_time - self.session_start)
        minutes, seconds = divmod(session_duration, 60)
        cv2.putText(data_window, f'Time: {minutes:02d}:{seconds:02d} | FPS: {self.current_fps:.1f} | Vis: {guidance.get("visibility_score", 0):.2f}', 
                   (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Mode indicator
        cv2.putText(data_window, 'MODE: SHADOWBOXING (No Bag Required)', (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Punch counters
        counter_y = 140
        
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
        total_y = 270
        cv2.rectangle(data_window, (200, total_y), (550, total_y + 120), (255, 255, 255), 4)
        cv2.putText(data_window, 'TOTAL', (220, total_y + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.putText(data_window, str(self.punch_counts['total']), (320, total_y + 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 8)
        
        # Recent punches
        recent_y = 420
        cv2.putText(data_window, 'RECENT SHADOWBOX PUNCHES:', (50, recent_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, punch in enumerate(self.recent_punches[-4:]):
            age = current_time - punch['time']
            if age < 10:
                y_pos = recent_y + 30 + (i * 25)
                visibility = punch.get('visibility', 0)
                text = f"#{punch['number']} {punch['arm'].upper()} {punch['type'].upper()} (vis:{visibility:.2f})"
                cv2.putText(data_window, text, (50, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Positioning tips
        tips_y = 540
        cv2.putText(data_window, 'POSITIONING TIPS:', (50, tips_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        tips = guidance.get('suggestions', ['Position looks good!'])[:2]  # Show up to 2 tips
        for i, tip in enumerate(tips):
            cv2.putText(data_window, f'• {tip}', (50, tips_y + 20 + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Controls
        cv2.putText(data_window, 'R=Reset | Q=Quit', (600, 590), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return data_window
    
    def process_frame(self, frame: np.ndarray):
        """Process frame for shadowboxing."""
        frame_height, frame_width = frame.shape[:2]
        
        # Pose detection with high sensitivity
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
            print("RESET: All shadowbox counts cleared!")
        
        return True
    
    def run(self):
        """Main shadowboxing loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*70)
        print("SHADOWBOXING TRAINER")
        print("="*70)
        print("OPTIMIZED FOR LANDMARK DETECTION:")
        print("• NO heavy bag required - pure shadowboxing")
        print("• Ultra-low visibility thresholds (0.1 minimum)")
        print("• Maximum MediaPipe complexity for best detection")
        print("• Real-time positioning guidance")
        print("• Very relaxed angle thresholds (150°→170°)")
        print("")
        print("POSITIONING TIPS:")
        print("• Stand 2-3 feet from camera")
        print("• Face good lighting (window or lamps)")
        print("• Keep arms visible and unobstructed")
        print("• Watch visibility scores on hands")
        print("")
        print("CONTROLS:")
        print("• R = Reset counts")
        print("• Q = Quit")
        print("="*70 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.update_setup_phase()
                processed_frame, landmarks_dict = self.process_frame(frame)
                
                # Get positioning guidance
                guidance = self.assess_positioning(landmarks_dict)
                
                if landmarks_dict:
                    self.detect_punch_shadowbox(landmarks_dict, 'left')
                    self.detect_punch_shadowbox(landmarks_dict, 'right')
                
                camera_display = self.draw_camera_view(processed_frame, landmarks_dict, guidance)
                data_display = self.create_data_window(guidance)
                
                self.update_fps()
                
                cv2.imshow('Shadowbox Camera', camera_display)
                cv2.imshow('Shadowbox Data', data_display)
                cv2.moveWindow('Shadowbox Camera', 50, 50)
                cv2.moveWindow('Shadowbox Data', 700, 50)
                
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
        trainer = ShadowboxTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())