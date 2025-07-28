"""
AI Boxing Trainer - Main Application Entry Point
Modular version using 3D pose tracking, punch classification, and form analysis.
"""

import cv2
import time
import numpy as np
from typing import Optional

from .pose_tracker import PoseTracker
from .punch_classifier import PunchClassifier, PunchType
from .form_analyzer import FormAnalyzer
from .heavy_bag_optimizer import TrainingMode


class AIBoxingTrainer:
    """
    Main application class that orchestrates all components.
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize the AI Boxing Trainer application.
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
        """
        # Initialize components
        self.pose_tracker = PoseTracker(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
            model_complexity=2  # Use heavy model for best accuracy
        )
        
        self.punch_classifier = PunchClassifier(
            trajectory_buffer_size=30,
            fps=30
        )
        
        self.form_analyzer = FormAnalyzer()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
        
        # Set camera properties for optimal performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # UI settings
        self.show_form_feedback = True
        self.show_detailed_stats = True
        self.show_setup_guidance = True
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def run(self) -> None:
        """
        Main application loop.
        """
        print("🥊 AI Boxing Trainer Started!")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset statistics")
        print("  'f' - Toggle form feedback")
        print("  's' - Toggle detailed stats")
        print("  'g' - Toggle setup guidance")
        print("=" * 50)
        
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from camera")
                    break
                
                # Process frame through pose tracker
                processed_frame, landmarks_dict = self.pose_tracker.process_frame(frame)
                
                # Get current training mode
                training_mode = self.pose_tracker.get_training_mode()
                
                # Analyze both arms with error handling
                for arm in ['left', 'right']:
                    try:
                        if landmarks_dict:
                            # Classify punches with training mode context
                            punch_type, count, score = self.punch_classifier.classify_punch(
                                landmarks_dict, arm, training_mode
                            )
                            
                            # Analyze form if punch detected
                            if (punch_type != PunchType.UNKNOWN and 
                                self.punch_classifier.punch_stages[arm].value == 'punching'):
                                
                                form_feedback = self.form_analyzer.analyze_form(
                                    landmarks_dict, punch_type, arm
                                )
                                
                                # Store form feedback for display
                                setattr(self, f'{arm}_form_feedback', form_feedback)
                    except Exception as e:
                        # Log error but continue processing
                        if hasattr(self, 'error_count'):
                            self.error_count += 1
                        else:
                            self.error_count = 1
                        
                        # Only print error every 30 frames to avoid spam
                        if self.error_count % 30 == 1:
                            print(f"⚠️  Processing error for {arm} arm: {e}")
                        continue
                
                # Draw UI elements
                self._draw_ui(processed_frame, landmarks_dict, training_mode)
                
                # Update FPS counter
                self._update_fps()
                
                # Show frame
                cv2.imshow('AI Boxing Trainer - 3D Enhanced', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self._reset_statistics()
                elif key == ord('f'):
                    self.show_form_feedback = not self.show_form_feedback
                elif key == ord('s'):
                    self.show_detailed_stats = not self.show_detailed_stats
                elif key == ord('g'):
                    self.show_setup_guidance = not self.show_setup_guidance
        
        except KeyboardInterrupt:
            print("\n⏹️  Training stopped by user")
        
        finally:
            self._cleanup()
    
    def _draw_ui(self, frame: np.ndarray, landmarks_dict: Optional[dict], 
                training_mode: TrainingMode) -> None:
        """
        Draw all UI elements on the frame.
        
        Args:
            frame: Video frame to draw on
            landmarks_dict: Current pose landmarks
            training_mode: Current detected training mode
        """
        frame_height, frame_width = frame.shape[:2]
        
        # Draw main stats panel
        self._draw_stats_panel(frame)
        
        # Draw training mode indicator
        self._draw_training_mode(frame, training_mode, frame_width)
        
        # Draw setup guidance if enabled
        if self.show_setup_guidance:
            self._draw_setup_guidance(frame, landmarks_dict)
        
        # Draw form feedback if enabled
        if self.show_form_feedback:
            self._draw_form_feedback(frame)
        
        # Draw performance info
        self._draw_performance_info(frame, frame_width, frame_height)
        
        # Draw pose detection status
        self._draw_pose_status(frame, landmarks_dict, frame_width)
    
    def _draw_training_mode(self, frame: np.ndarray, training_mode: TrainingMode, width: int) -> None:
        """Draw training mode indicator."""
        mode_text = f"MODE: {training_mode.value.upper()}"
        mode_color = (0, 255, 255) if training_mode == TrainingMode.HEAVY_BAG else (255, 255, 255)
        
        cv2.putText(frame, mode_text, (width - 250, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)
    
    def _draw_setup_guidance(self, frame: np.ndarray, landmarks_dict: Optional[dict]) -> None:
        """Draw setup guidance panel."""
        if not landmarks_dict:
            return
        
        try:
            guidance = self.pose_tracker.get_setup_guidance(landmarks_dict)
            if not guidance:
                return
            
            # Setup guidance panel
            panel_y = 250
            panel_height = len(guidance) * 25 + 20
            
            # Semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, panel_y), (500, panel_y + panel_height), (40, 40, 40), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Title
            cv2.putText(frame, "SETUP GUIDANCE:", (20, panel_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Guidance messages
            for i, message in enumerate(guidance[:3]):  # Show max 3 messages
                cv2.putText(frame, message, (20, panel_y + 45 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            # Silently skip guidance if error occurs
            pass    
    def _draw_stats_panel(self, frame: np.ndarray) -> None:
        """Draw the main statistics panel."""
        # Background panel
        cv2.rectangle(frame, (0, 0), (600, 120), (245, 117, 16), -1)
        
        # Get statistics for both arms
        left_stats = self.punch_classifier.get_punch_statistics('left')
        right_stats = self.punch_classifier.get_punch_statistics('right')
        
        # Left arm stats
        cv2.putText(frame, 'LEFT ARM', (15, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(frame, f"Count: {left_stats['count']}", (15, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Score: {left_stats['score']}", (15, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Type: {left_stats['last_type'].upper()}", (15, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Right arm stats
        cv2.putText(frame, 'RIGHT ARM', (300, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        cv2.putText(frame, f"Count: {right_stats['count']}", (300, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Score: {right_stats['score']}", (300, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Type: {right_stats['last_type'].upper()}", (300, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def _draw_form_feedback(self, frame: np.ndarray) -> None:
        """Draw form analysis feedback."""
        y_offset = 140
        
        for arm in ['left', 'right']:
            if hasattr(self, f'{arm}_form_feedback'):
                feedback = getattr(self, f'{arm}_form_feedback')
                
                # Draw feedback panel
                panel_width = 400
                panel_height = 100
                x_pos = 20 if arm == 'left' else frame.shape[1] - panel_width - 20
                
                # Semi-transparent background
                overlay = frame.copy()
                cv2.rectangle(overlay, (x_pos, y_offset), 
                            (x_pos + panel_width, y_offset + panel_height), 
                            (50, 50, 50), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                
                # Form score and grade
                score_color = self._get_score_color(feedback.overall_score)
                cv2.putText(frame, f"{arm.upper()} FORM: {feedback.technique_grade}", 
                           (x_pos + 10, y_offset + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
                
                cv2.putText(frame, f"Score: {feedback.overall_score:.1f}%", 
                           (x_pos + 10, y_offset + 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Top feedback message
                if feedback.feedback_messages:
                    cv2.putText(frame, feedback.feedback_messages[0][:35], 
                               (x_pos + 10, y_offset + 75), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _draw_performance_info(self, frame: np.ndarray, width: int, height: int) -> None:
        """Draw performance information."""
        # FPS counter
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (width - 120, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Model info
        cv2.putText(frame, "3D BlazePose GHUM", 
                   (width - 180, height - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_pose_status(self, frame: np.ndarray, landmarks_dict: Optional[dict], width: int) -> None:
        """Draw pose detection status indicator."""
        if landmarks_dict and self.pose_tracker.is_pose_detected(landmarks_dict):
            status_text = "POSE DETECTED"
            status_color = (0, 255, 0)
        else:
            status_text = "NO POSE"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (width - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    def _get_score_color(self, score: float) -> tuple:
        """Get color based on form score."""
        if score >= 80:
            return (0, 255, 0)  # Green
        elif score >= 60:
            return (0, 255, 255)  # Yellow
        else:
            return (0, 0, 255)  # Red
    
    def _update_fps(self) -> None:
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _reset_statistics(self) -> None:
        """Reset all training statistics."""
        self.punch_classifier.reset_statistics()
        print("📊 Statistics reset!")
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        print("🧹 Cleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose_tracker.release()
        print("✅ Cleanup complete!")


def main():
    """Main entry point for the application."""
    try:
        trainer = AIBoxingTrainer(camera_id=0)
        trainer.run()
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure your camera is connected and not used by another application")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
