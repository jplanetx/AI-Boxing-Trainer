#!/usr/bin/env python3
"""
Improved AI Boxing Trainer - Fixes punch detection and UI visibility issues
"""

import cv2
import time
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_trainer.pose_tracker import PoseTracker
from ai_trainer.utils import calculate_angle


class ImprovedBoxingTrainer:
    """Improved boxing trainer with better punch detection and UI."""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        
        # Initialize pose tracker
        self.pose_tracker = PoseTracker(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.4,
            model_complexity=1
        )
        
        # Simplified punch tracking - just count extensions
        self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
        self.arm_states = {'left': 'unknown', 'right': 'unknown'}
        self.last_angles = {'left': 0, 'right': 0}
        
        # Punch type tracking (separate system)
        self.punch_types = {'jab': 0, 'cross': 0, 'hook': 0, 'uppercut': 0}
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # UI state - make feedback persistent
        self.recent_punches = []  # List of recent punch info
        self.feedback_messages = []
        self.mode_display_time = 0
        self.current_mode = "DETECTING..."
        
        # Session stats
        self.session_start = time.time()
        
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
    
    def detect_punch_reliable(self, landmarks_dict: dict, arm: str) -> bool:
        """Very simple and reliable punch detection."""
        if not landmarks_dict:
            return False
            
        try:
            # Get required landmarks
            shoulder_key = f'{arm}_shoulder'
            elbow_key = f'{arm}_elbow'
            wrist_key = f'{arm}_wrist'
            
            if not all(key in landmarks_dict for key in [shoulder_key, elbow_key, wrist_key]):
                return False
            
            shoulder = landmarks_dict[shoulder_key]
            elbow = landmarks_dict[elbow_key] 
            wrist = landmarks_dict[wrist_key]
            
            # Check minimum visibility
            if any(lm.get('visibility', 0) < 0.4 for lm in [shoulder, elbow, wrist]):
                return False
            
            # Calculate elbow angle
            current_angle = calculate_angle(
                [shoulder['x'], shoulder['y']],
                [elbow['x'], elbow['y']],
                [wrist['x'], wrist['y']]
            )
            
            # Simple state machine: bent -> extended = punch
            if self.arm_states[arm] == 'unknown':
                self.arm_states[arm] = 'bent' if current_angle < 120 else 'extended'
                self.last_angles[arm] = current_angle
                return False
            
            # Detect punch: transition from bent to extended
            if self.arm_states[arm] == 'bent' and current_angle > 150:
                # Punch detected!
                self.arm_states[arm] = 'extended'
                self.punch_counts[arm] += 1
                self.punch_counts['total'] += 1
                
                # Add to recent punches for UI
                punch_info = {
                    'arm': arm,
                    'time': time.time(),
                    'type': self.classify_simple_punch_type(arm, current_angle),
                    'angle': current_angle
                }
                self.recent_punches.append(punch_info)
                
                # Keep only last 5 punches
                if len(self.recent_punches) > 5:
                    self.recent_punches.pop(0)
                
                # Update punch type count
                punch_type = punch_info['type']
                if punch_type in self.punch_types:
                    self.punch_types[punch_type] += 1
                
                print(f"PUNCH DETECTED: {arm} {punch_type} (angle: {current_angle:.1f})")
                return True
                
            # Return to bent position
            elif self.arm_states[arm] == 'extended' and current_angle < 120:
                self.arm_states[arm] = 'bent'
            
            self.last_angles[arm] = current_angle
            return False
            
        except Exception as e:
            print(f"Error in punch detection for {arm}: {e}")
            return False
    
    def classify_simple_punch_type(self, arm: str, angle: float) -> str:
        """Simple punch type classification."""
        # Basic classification based on arm and extension
        if arm == 'left':
            return 'jab' if angle > 160 else 'hook'
        else:
            return 'cross' if angle > 160 else 'hook'
    
    def update_mode_detection(self, landmarks_dict: dict):
        """Simple mode detection."""
        if not landmarks_dict:
            self.current_mode = "NO POSE DETECTED"
            return
        
        # Count visible landmarks
        left_visible = sum(1 for key in ['left_shoulder', 'left_elbow', 'left_wrist'] 
                          if key in landmarks_dict and landmarks_dict[key].get('visibility', 0) > 0.5)
        right_visible = sum(1 for key in ['right_shoulder', 'right_elbow', 'right_wrist'] 
                           if key in landmarks_dict and landmarks_dict[key].get('visibility', 0) > 0.5)
        
        if left_visible >= 2 and right_visible >= 2:
            self.current_mode = "SHADOWBOXING MODE"
        elif left_visible >= 2 or right_visible >= 2:
            self.current_mode = "HEAVY BAG MODE"
        else:
            self.current_mode = "POOR VISIBILITY"
            
        self.mode_display_time = time.time()
    
    def update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame: np.ndarray, landmarks_dict: dict) -> np.ndarray:
        """Draw improved UI with better visibility."""
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Header with better background
        cv2.rectangle(frame, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.8, frame, 0.2, 0)
        
        # Title
        cv2.putText(frame, 'AI BOXING TRAINER - IMPROVED', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # FPS and session time
        session_duration = int(current_time - self.session_start)
        minutes, seconds = divmod(session_duration, 60)
        cv2.putText(frame, f'FPS: {self.current_fps:.1f} | Time: {minutes:02d}:{seconds:02d}', 
                   (width - 250, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mode display (persistent)
        mode_color = (0, 255, 0) if "BAG" in self.current_mode or "SHADOW" in self.current_mode else (255, 255, 0)
        cv2.putText(frame, self.current_mode, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Main punch counter (LEFT SIDE - BIG AND CLEAR)
        cv2.rectangle(frame, (30, 120), (400, 320), (0, 0, 0), -1)
        cv2.rectangle(frame, (30, 120), (400, 320), (0, 255, 0), 3)
        
        cv2.putText(frame, 'PUNCH COUNTS', (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Large, clear counters
        cv2.putText(frame, f'LEFT:  {self.punch_counts["left"]}', (50, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 3)
        cv2.putText(frame, f'RIGHT: {self.punch_counts["right"]}', (50, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 255), 3)
        cv2.putText(frame, f'TOTAL: {self.punch_counts["total"]}', (50, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Punch types (RIGHT SIDE)
        cv2.rectangle(frame, (width - 300, 120), (width - 30, 300), (0, 0, 0), -1)
        cv2.rectangle(frame, (width - 300, 120), (width - 30, 300), (255, 255, 0), 2)
        
        cv2.putText(frame, 'PUNCH TYPES', (width - 280, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_offset = 180
        for punch_type, count in self.punch_types.items():
            color = (200, 255, 200) if count > 0 else (100, 100, 100)
            cv2.putText(frame, f'{punch_type.upper()}: {count}', (width - 280, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30
        
        # Recent punches display (PERSISTENT - bottom center)
        if self.recent_punches:
            cv2.rectangle(frame, (width//2 - 200, height - 120), (width//2 + 200, height - 20), (0, 0, 0), -1)
            cv2.rectangle(frame, (width//2 - 200, height - 120), (width//2 + 200, height - 20), (0, 255, 255), 2)
            
            cv2.putText(frame, 'RECENT PUNCHES', (width//2 - 80, height - 95), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show last 3 punches
            for i, punch in enumerate(self.recent_punches[-3:]):
                age = current_time - punch['time']
                if age < 10:  # Show for 10 seconds
                    alpha = max(0.3, 1.0 - age / 10)
                    color_intensity = int(255 * alpha)
                    color = (color_intensity, color_intensity, 255)
                    
                    punch_text = f"{punch['arm'].upper()} {punch['type'].upper()}"
                    cv2.putText(frame, punch_text, (width//2 - 150, height - 65 + i * 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Instructions (persistent, top right)
        instructions = ['Q: Quit', 'R: Reset', 'H: Help']
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (width - 120, 120 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('r'):  # Reset
            self.punch_counts = {'left': 0, 'right': 0, 'total': 0}
            self.punch_types = {'jab': 0, 'cross': 0, 'hook': 0, 'uppercut': 0}
            self.recent_punches = []
            self.arm_states = {'left': 'unknown', 'right': 'unknown'}
            self.session_start = time.time()
            print("All counts reset!")
        elif key == ord('h'):  # Help
            self.show_help()
        
        return True
    
    def show_help(self):
        """Show help information."""
        print("\n" + "="*50)
        print("AI BOXING TRAINER - IMPROVED VERSION")
        print("="*50)
        print("CONTROLS:")
        print("  Q or ESC: Quit application")
        print("  R: Reset all counts and statistics")
        print("  H: Show this help")
        print("\nTIPS FOR BEST RESULTS:")
        print("  • Stand 3-6 feet from camera")
        print("  • Ensure good lighting")
        print("  • Make clear punching motions (bent arm → extended)")
        print("  • Works for both shadowboxing and heavy bag")
        print("  • Recent punches shown for 10 seconds")
        print("="*50 + "\n")
    
    def run(self):
        """Main application loop."""
        if not self.initialize_camera():
            return
        
        print("Improved AI Boxing Trainer - Starting...")
        print("This version has more reliable punch detection and better UI")
        print("Press 'H' for detailed help, 'R' to reset, 'Q' to quit")
        self.show_help()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame, landmarks_dict = self.pose_tracker.process_frame(frame)
                
                # Update mode detection
                self.update_mode_detection(landmarks_dict)
                
                # Detect punches for both arms
                if landmarks_dict:
                    self.detect_punch_reliable(landmarks_dict, 'left')
                    self.detect_punch_reliable(landmarks_dict, 'right')
                
                # Draw UI
                final_frame = self.draw_ui(processed_frame, landmarks_dict)
                
                # Update FPS
                self.update_fps()
                
                # Display
                cv2.imshow('AI Boxing Trainer - Improved', final_frame)
                
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
        self.pose_tracker.release()
        
        # Final session summary
        session_duration = int(time.time() - self.session_start)
        minutes, seconds = divmod(session_duration, 60)
        
        print(f"\n" + "="*50)
        print("SESSION SUMMARY")
        print("="*50)
        print(f"Duration: {minutes:02d}:{seconds:02d}")
        print(f"Left Punches: {self.punch_counts['left']}")
        print(f"Right Punches: {self.punch_counts['right']}")
        print(f"Total Punches: {self.punch_counts['total']}")
        print("\nPunch Types:")
        for punch_type, count in self.punch_types.items():
            if count > 0:
                print(f"  {punch_type.capitalize()}: {count}")
        print("="*50)


def main():
    """Main entry point."""
    try:
        trainer = ImprovedBoxingTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())