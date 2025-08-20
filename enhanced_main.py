#!/usr/bin/env python3
"""
Enhanced AI Boxing Trainer - Integrated Real-time Application
Combines advanced pose tracking, punch classification, and form analysis.
"""

import cv2
import time
import numpy as np
from typing import Optional, Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_trainer.pose_tracker import PoseTracker
from ai_trainer.punch_classifier import PunchClassifier, PunchType
from ai_trainer.form_analyzer import FormAnalyzer


class EnhancedBoxingTrainer:
    """
    Enhanced AI Boxing Trainer with integrated 3D pose tracking,
    punch classification, and real-time form analysis.
    """
    
    def __init__(self, camera_id: int = 0):
        """Initialize the enhanced boxing trainer."""
        self.camera_id = camera_id
        self.cap = None
        
        # Initialize AI components
        self.pose_tracker = PoseTracker(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=2
        )
        
        self.punch_classifier = PunchClassifier()
        self.form_analyzer = FormAnalyzer()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Training session statistics
        self.session_stats = {
            'total_punches': 0,
            'punch_types': {punch_type.value: 0 for punch_type in PunchType if punch_type != PunchType.UNKNOWN},
            'avg_form_score': 0.0,
            'session_start': time.time()
        }
        
        # UI state
        self.show_landmarks = True
        self.show_form_feedback = True
        self.last_form_feedback = None
        self.last_punch_detection = None
        self.feedback_display_time = 0
        
    def initialize_camera(self) -> bool:
        """Initialize camera capture."""
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
    
    def update_fps(self):
        """Update FPS calculation."""
        self.fps_counter += 1
        if self.fps_counter >= 30:
            current_time = time.time()
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_ui(self, frame: np.ndarray, landmarks_dict: Optional[Dict]) -> np.ndarray:
        """Draw comprehensive UI overlay."""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for UI elements
        overlay = frame.copy()
        
        # Header bar with session info
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, 'AI BOXING TRAINER - ENHANCED', (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # FPS and Performance
        cv2.putText(frame, f'FPS: {self.current_fps:.1f}', (width - 150, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Training mode indicator
        training_mode = self.pose_tracker.get_training_mode()
        mode_color = (0, 255, 255) if training_mode.value != 'unknown' else (128, 128, 128)
        cv2.putText(frame, f'Mode: {training_mode.value.upper()}', (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Session time
        session_duration = int(time.time() - self.session_stats['session_start'])
        minutes, seconds = divmod(session_duration, 60)
        cv2.putText(frame, f'Time: {minutes:02d}:{seconds:02d}', (width - 150, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Punch counters (left side)
        self.draw_punch_counters(frame, 50, 150)
        
        # Form feedback (right side)
        if self.show_form_feedback and self.last_form_feedback:
            self.draw_form_feedback(frame, width - 400, 150)
        
        # Recent punch detection
        if self.last_punch_detection and time.time() - self.feedback_display_time < 2.0:
            self.draw_punch_detection(frame, width // 2 - 100, height - 100)
        
        # Setup guidance if pose not properly detected
        if landmarks_dict:
            guidance = self.pose_tracker.get_setup_guidance(landmarks_dict)
            if guidance:
                self.draw_setup_guidance(frame, guidance, 50, height - 150)
        
        return frame
    
    def draw_punch_counters(self, frame: np.ndarray, x: int, y: int):
        """Draw punch count statistics."""
        # Background for punch counters
        cv2.rectangle(frame, (x - 10, y - 30), (x + 300, y + 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 10, y - 30), (x + 300, y + 200), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, 'PUNCH STATS', (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Get punch statistics
        left_stats = self.punch_classifier.get_punch_statistics('left')
        right_stats = self.punch_classifier.get_punch_statistics('right')
        
        y_offset = 40
        cv2.putText(frame, f"Left Punches: {left_stats['count']}", (x, y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Right Punches: {right_stats['count']}", (x, y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
        
        y_offset += 30
        total_punches = left_stats['count'] + right_stats['count']
        cv2.putText(frame, f"Total: {total_punches}", (x, y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Punch type breakdown
        y_offset += 40
        cv2.putText(frame, 'TYPES:', (x, y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, (punch_type, count) in enumerate(self.session_stats['punch_types'].items()):
            y_offset += 25
            color = (150, 255, 150) if count > 0 else (100, 100, 100)
            cv2.putText(frame, f"{punch_type.capitalize()}: {count}", (x, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def draw_form_feedback(self, frame: np.ndarray, x: int, y: int):
        """Draw form analysis feedback."""
        feedback = self.last_form_feedback
        if not feedback:
            return
        
        # Background for form feedback
        cv2.rectangle(frame, (x - 10, y - 30), (x + 380, y + 250), (0, 0, 0), -1)
        cv2.rectangle(frame, (x - 10, y - 30), (x + 380, y + 250), (255, 255, 255), 2)
        
        # Title with grade
        grade_color = {
            'A': (0, 255, 0), 'B': (100, 255, 100), 'C': (255, 255, 0),
            'D': (255, 150, 0), 'F': (255, 0, 0)
        }.get(feedback.technique_grade, (255, 255, 255))
        
        cv2.putText(frame, f'FORM ANALYSIS - {feedback.technique_grade}', (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, grade_color, 2)
        
        # Overall score
        y_offset = 40
        score_color = (0, 255, 0) if feedback.overall_score >= 80 else (255, 255, 0) if feedback.overall_score >= 60 else (255, 0, 0)
        cv2.putText(frame, f"Score: {feedback.overall_score:.1f}%", (x, y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
        
        # Feedback messages
        y_offset += 40
        cv2.putText(frame, 'FEEDBACK:', (x, y + y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, message in enumerate(feedback.feedback_messages[:3]):
            y_offset += 25
            # Wrap long messages
            if len(message) > 40:
                message = message[:37] + "..."
            cv2.putText(frame, f"• {message}", (x, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_punch_detection(self, frame: np.ndarray, x: int, y: int):
        """Draw recent punch detection notification."""
        if not self.last_punch_detection:
            return
        
        punch_info = self.last_punch_detection
        punch_type = punch_info.get('punch_type', 'unknown')
        confidence = punch_info.get('confidence', 0.0)
        arm = punch_info.get('arm', 'unknown')
        
        # Animated background
        alpha = max(0.3, 1.0 - (time.time() - self.feedback_display_time) / 2.0)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 20, y - 20), (x + 220, y + 60), (0, 255, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Punch detection text
        cv2.putText(frame, f'{arm.upper()} {punch_type.upper()}!', (x, y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, f'Confidence: {confidence:.1%}', (x, y + 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    def draw_setup_guidance(self, frame: np.ndarray, guidance: list, x: int, y: int):
        """Draw setup guidance messages."""
        if not guidance:
            return
        
        # Background
        cv2.rectangle(frame, (x - 10, y - 20), (x + 400, y + len(guidance) * 25 + 10), (0, 0, 200), -1)
        cv2.rectangle(frame, (x - 10, y - 20), (x + 400, y + len(guidance) * 25 + 10), (255, 255, 255), 2)
        
        cv2.putText(frame, 'SETUP GUIDANCE:', (x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, message in enumerate(guidance[:3]):
            y_offset = (i + 1) * 25
            cv2.putText(frame, f"• {message}", (x, y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, frame: np.ndarray):
        """Process a single frame with AI analysis."""
        # Pose tracking
        processed_frame, landmarks_dict = self.pose_tracker.process_frame(frame)
        
        if landmarks_dict:
            # Punch classification
            punch_result = self.punch_classifier.process_frame(landmarks_dict, self.pose_tracker)
            
            if punch_result:
                # Update session statistics
                self.session_stats['total_punches'] += 1
                punch_type = punch_result.get('punch_type', 'unknown')
                if punch_type in self.session_stats['punch_types']:
                    self.session_stats['punch_types'][punch_type] += 1
                
                # Store for UI display
                self.last_punch_detection = punch_result
                self.feedback_display_time = time.time()
                
                # Form analysis
                if punch_type != 'unknown':
                    try:
                        punch_enum = PunchType(punch_type)
                        arm = punch_result.get('arm', 'left')
                        form_feedback = self.form_analyzer.analyze_form(
                            landmarks_dict, punch_enum, arm
                        )
                        self.last_form_feedback = form_feedback
                        
                        # Update average form score
                        if form_feedback.overall_score > 0:
                            current_avg = self.session_stats['avg_form_score']
                            total_punches = self.session_stats['total_punches']
                            self.session_stats['avg_form_score'] = (
                                (current_avg * (total_punches - 1) + form_feedback.overall_score) / total_punches
                            )
                    except ValueError:
                        pass  # Invalid punch type
        
        return processed_frame, landmarks_dict
    
    def handle_keyboard_input(self) -> bool:
        """Handle keyboard input. Returns False to quit."""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            return False
        elif key == ord('l'):  # Toggle landmarks
            self.show_landmarks = not self.show_landmarks
        elif key == ord('f'):  # Toggle form feedback
            self.show_form_feedback = not self.show_form_feedback
        elif key == ord('r'):  # Reset statistics
            self.reset_session()
        elif key == ord('h'):  # Show help
            self.print_help()
        
        return True
    
    def reset_session(self):
        """Reset session statistics."""
        self.session_stats = {
            'total_punches': 0,
            'punch_types': {punch_type.value: 0 for punch_type in PunchType if punch_type != PunchType.UNKNOWN},
            'avg_form_score': 0.0,
            'session_start': time.time()
        }
        self.punch_classifier.reset_statistics()
        self.last_form_feedback = None
        self.last_punch_detection = None
        print("Session statistics reset")
    
    def print_help(self):
        """Print keyboard controls help."""
        print("\n=== AI BOXING TRAINER CONTROLS ===")
        print("Q/ESC: Quit application")
        print("L: Toggle landmark display")
        print("F: Toggle form feedback display")
        print("R: Reset session statistics")
        print("H: Show this help")
        print("==================================\n")
    
    def run(self):
        """Main application loop."""
        if not self.initialize_camera():
            return
        
        print("AI Boxing Trainer Enhanced - Starting...")
        print("Press 'H' for help, 'Q' or ESC to quit")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame with AI
                processed_frame, landmarks_dict = self.process_frame(frame)
                
                # Draw UI overlay
                final_frame = self.draw_ui(processed_frame, landmarks_dict)
                
                # Update performance metrics
                self.update_fps()
                
                # Display frame
                cv2.imshow('AI Boxing Trainer - Enhanced', final_frame)
                
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
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.pose_tracker.release()
        
        # Print session summary
        duration = int(time.time() - self.session_stats['session_start'])
        minutes, seconds = divmod(duration, 60)
        
        print(f"\n=== SESSION SUMMARY ===")
        print(f"Duration: {minutes:02d}:{seconds:02d}")
        print(f"Total Punches: {self.session_stats['total_punches']}")
        print(f"Average Form Score: {self.session_stats['avg_form_score']:.1f}%")
        print("Punch Types:")
        for punch_type, count in self.session_stats['punch_types'].items():
            if count > 0:
                print(f"  {punch_type.capitalize()}: {count}")
        print("=======================")


def main():
    """Main entry point."""
    try:
        trainer = EnhancedBoxingTrainer(camera_id=0)
        trainer.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())