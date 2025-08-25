#!/usr/bin/env python3
"""
Balanced Punch Detector
Medium sensitivity with label persistence for better accuracy evaluation
"""

import cv2
import numpy as np
import time
import sys
import os
from collections import deque

# MediaPipe imports
import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing

class BalancedPunchDetector:
    """Balanced punch detector with medium sensitivity and label persistence"""
    
    def __init__(self):
        print("Initializing balanced punch detector...")
        
        # MediaPipe setup
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.mediapipe_pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3
        )
        
        # Camera setup
        self.cap = None
        self.frame_count = 0
        
        # Punch tracking
        self.punch_history = deque(maxlen=5)
        self.last_punch_type = "guard"
        self.last_punch_time = 0
        
        # Label persistence system
        self.current_label = "GUARD"
        self.label_start_time = time.time()
        self.label_duration = 1.0  # Hold labels for 1 second
        self.pending_label = None
        
        # Confidence tracking for stability
        self.confidence_buffer = deque(maxlen=5)  # Track confidence over time
        
        # Detection results storage
        self.results_log = []
        
    def init_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                
            # Set resolution and frame rate
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test capture
            ret, frame = self.cap.read()
            if ret:
                print(f"Camera initialized: {frame.shape}")
                return True
            return False
        except Exception as e:
            print(f"Camera init failed: {e}")
            return False
    
    def process_mediapipe_frame(self, frame):
        """Process frame with MediaPipe"""
        start_time = time.time()
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = self.mediapipe_pose.process(rgb_frame)
            
            processing_time = (time.time() - start_time) * 1000
            return results, processing_time
            
        except Exception as e:
            print(f"MediaPipe processing error: {e}")
            return None, 0
    
    def calculate_arm_angle(self, shoulder, elbow, wrist):
        """Calculate arm angle for punch detection"""
        try:
            # Convert to vectors
            v1 = np.array([shoulder[0] - elbow[0], shoulder[1] - elbow[1]])
            v2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            return np.degrees(angle)
        except:
            return None
    
    def analyze_mediapipe_results(self, results):
        """Analyze MediaPipe results for boxing accuracy"""
        if not results.pose_landmarks:
            return {}
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get key points
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            # Calculate angles
            left_angle = self.calculate_arm_angle(
                [left_shoulder.x, left_shoulder.y],
                [left_elbow.x, left_elbow.y], 
                [left_wrist.x, left_wrist.y]
            )
            right_angle = self.calculate_arm_angle(
                [right_shoulder.x, right_shoulder.y],
                [right_elbow.x, right_elbow.y],
                [right_wrist.x, right_wrist.y]
            )
            
            return {
                'left_angle': left_angle,
                'right_angle': right_angle,
                'left_confidence': left_wrist.visibility,
                'right_confidence': right_wrist.visibility,
                'left_wrist': [left_wrist.x, left_wrist.y],
                'right_wrist': [right_wrist.x, right_wrist.y]
            }
        except Exception as e:
            return {}
    
    def detect_punch_with_confidence(self, analysis):
        """Detect punch with medium thresholds and confidence tracking"""
        if not analysis:
            return "GUARD", "guard", 0.0
        
        left_angle = analysis.get('left_angle', 180)
        right_angle = analysis.get('right_angle', 180)
        left_conf = analysis.get('left_confidence', 0)
        right_conf = analysis.get('right_confidence', 0)
        left_wrist = analysis.get('left_wrist')
        right_wrist = analysis.get('right_wrist')
        
        # MEDIUM thresholds - balanced between too strict and too loose
        angle_threshold = 130  # Between 120 (too low) and 135 (too high)
        conf_threshold = 0.4   # Between 0.3 (too low) and 0.5 (too high)
        
        # Calculate punch confidence scores
        punch_confidence = 0.0
        best_punch = "GUARD"
        best_type = "guard"
        
        # Check for extended arms with confidence scoring
        left_extended = (left_angle and left_angle > angle_threshold and 
                        left_conf > conf_threshold)
        right_extended = (right_angle and right_angle > angle_threshold and 
                         right_conf > conf_threshold)
        
        # Position-based hook detection with confidence
        left_hook = False
        right_hook = False
        
        if left_wrist and left_conf > conf_threshold:
            if left_wrist[0] < 0.35:  # Left 35% of screen
                left_hook = True
        
        if right_wrist and right_conf > conf_threshold:
            if right_wrist[0] > 0.65:  # Right 35% of screen
                right_hook = True
        
        # Score different punch types
        if left_extended and right_extended:
            punch_confidence = min(left_conf, right_conf) * 0.9  # Combo slightly less confident
            best_punch = "COMBO PUNCH"
            best_type = "combo"
        elif left_extended:
            punch_confidence = left_conf
            best_punch = "LEFT JAB"
            best_type = "left_jab"
        elif right_extended:
            punch_confidence = right_conf
            best_punch = "RIGHT CROSS"
            best_type = "right_cross"
        elif left_hook:
            punch_confidence = left_conf * 0.8  # Hooks slightly less confident
            best_punch = "LEFT HOOK"
            best_type = "left_hook"
        elif right_hook:
            punch_confidence = right_conf * 0.8
            best_punch = "RIGHT HOOK"
            best_type = "right_hook"
        else:
            punch_confidence = max(left_conf, right_conf) * 0.5  # Guard state
            best_punch = "GUARD"
            best_type = "guard"
        
        return best_punch, best_type, punch_confidence
    
    def update_persistent_label(self, detected_punch, punch_type, confidence):
        """Update label with persistence - hold labels for evaluation"""
        current_time = time.time()
        
        # Add confidence to buffer
        self.confidence_buffer.append(confidence)
        avg_confidence = sum(self.confidence_buffer) / len(self.confidence_buffer)
        
        # Only change label if we have high confidence AND enough time has passed
        label_age = current_time - self.label_start_time
        confidence_threshold = 0.6  # Require strong confidence to change labels
        
        # Force label change if current label has been shown long enough
        if label_age >= self.label_duration:
            if avg_confidence > confidence_threshold and punch_type != "guard":
                # Strong punch detected - change to punch label
                self.current_label = detected_punch
                self.label_start_time = current_time
                self.pending_label = None
            else:
                # Default back to guard if no strong punch
                self.current_label = "GUARD"
                self.label_start_time = current_time
                self.pending_label = None
        elif punch_type != "guard" and avg_confidence > confidence_threshold:
            # Set pending label but don't change yet
            self.pending_label = detected_punch
        
        return self.current_label
    
    def update_punch_history(self, punch_display, punch_type):
        """Update punch history with timing"""
        current_time = time.time()
        
        # Only log when label actually changes to a punch (not just detected)
        if (punch_type != "guard" and 
            punch_display == self.current_label and
            punch_type != self.last_punch_type and
            current_time - self.last_punch_time > 0.5):  # 500ms minimum
            
            self.punch_history.append({
                'action': punch_display,
                'time': current_time,
                'type': punch_type
            })
            self.last_punch_time = current_time
            print(f"PUNCH LOGGED: {punch_display}")  # Console feedback
        
        self.last_punch_type = punch_type
    
    def draw_punch_history(self, frame):
        """Draw punch history sidebar with fading effect"""
        h, w = frame.shape[:2]
        
        # History sidebar background (right side)
        sidebar_width = 220
        cv2.rectangle(frame, (w - sidebar_width, 0), (w, h), (20, 20, 20), -1)
        
        # Title
        cv2.putText(frame, "PUNCH SEQUENCE", 
                   (w - sidebar_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Show total count
        cv2.putText(frame, f"Count: {len(self.punch_history)}", 
                   (w - sidebar_width + 10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw history with decay effect
        y_start = 85
        for i, punch_data in enumerate(reversed(list(self.punch_history))):
            age = time.time() - punch_data['time']
            
            # Calculate fade/size based on position (0 = most recent)
            scale = 1.0 - (i * 0.15)  # Scale: 1.0, 0.85, 0.7, 0.55, 0.4
            alpha = 1.0 - (i * 0.2)   # Fade: 1.0, 0.8, 0.6, 0.4, 0.2
            
            if scale < 0.4:  # Skip if too small
                continue
            
            # Color based on punch type
            punch_type = punch_data['type']
            if 'left' in punch_type:
                color = (0, 255, 255)  # Yellow
            elif 'right' in punch_type:
                color = (255, 0, 255)  # Magenta
            else:
                color = (0, 255, 0)    # Green
            
            # Apply alpha fade
            faded_color = tuple(int(c * alpha) for c in color)
            
            # Draw punch with size scaling
            font_scale = 0.45 * scale
            thickness = max(1, int(2 * scale))
            
            y_pos = y_start + (i * 45)
            
            # Punch text
            cv2.putText(frame, punch_data['action'], 
                       (w - sidebar_width + 15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, faded_color, thickness)
            
            # Time ago
            time_ago = f"{age:.1f}s"
            cv2.putText(frame, time_ago, 
                       (w - sidebar_width + 15, y_pos + 18), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3 * scale, 
                       (150, 150, 150), 1)
    
    def draw_current_status(self, frame, persistent_label, detected_punch, analysis, confidence, processing_time):
        """Draw persistent label with detection info"""
        h, w = frame.shape[:2]
        sidebar_width = 220
        
        # Main status area background (top left)
        cv2.rectangle(frame, (10, 10), (w - sidebar_width - 20, 220), (0, 0, 0), -1)
        
        # Persistent label - LARGE and stable
        color_map = {
            'left_jab': (0, 255, 255),     # Yellow
            'right_cross': (255, 0, 255),  # Magenta
            'left_hook': (0, 255, 0),      # Green  
            'right_hook': (255, 100, 0),   # Orange
            'combo': (255, 255, 0),        # Cyan
            'guard': (255, 255, 255)       # White
        }
        
        # Determine color based on persistent label
        label_type = persistent_label.lower().replace(' ', '_').replace('punch', '').replace('combo_', 'combo').strip()
        color = color_map.get(label_type, (255, 255, 255))
        
        cv2.putText(frame, f"CURRENT: {persistent_label}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        
        # Show label timing
        label_age = time.time() - self.label_start_time
        time_remaining = max(0, self.label_duration - label_age)
        cv2.putText(frame, f"Label age: {label_age:.1f}s (hold for {self.label_duration}s)", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show what's being detected vs what's displayed
        cv2.putText(frame, f"Detecting: {detected_punch} (conf: {confidence:.2f})", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        
        if self.pending_label:
            cv2.putText(frame, f"Pending: {self.pending_label}", 
                       (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 1)
        
        # Technical details
        if analysis:
            left_angle = analysis.get('left_angle', 0)
            right_angle = analysis.get('right_angle', 0)
            left_conf = analysis.get('left_confidence', 0)
            right_conf = analysis.get('right_confidence', 0)
            
            # Show if arms meet punch criteria (angle>130, conf>0.4)
            left_ready = left_angle > 130 and left_conf > 0.4
            right_ready = right_angle > 130 and right_conf > 0.4
            
            left_color = (0, 255, 0) if left_ready else (255, 255, 255)
            right_color = (0, 255, 0) if right_ready else (255, 255, 255)
            
            cv2.putText(frame, f"Left: {left_angle:.0f}° ({left_conf:.2f}) {'✓' if left_ready else '✗'}", 
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
            cv2.putText(frame, f"Right: {right_angle:.0f}° ({right_conf:.2f}) {'✓' if right_ready else '✗'}", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
        
        cv2.putText(frame, f"Processing: {processing_time:.1f}ms", 
                   (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Show settings
        cv2.putText(frame, "BALANCED: Angle>130°, Conf>0.4, Labels persist 1s", 
                   (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw wrist markers
        if analysis:
            left_wrist = analysis.get('left_wrist')
            right_wrist = analysis.get('right_wrist')
            
            if left_wrist:
                x, y = int(left_wrist[0] * w), int(left_wrist[1] * h)
                cv2.circle(frame, (x, y), 12, (0, 255, 255), -1)
                cv2.putText(frame, "L", (x-8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            if right_wrist:
                x, y = int(right_wrist[0] * w), int(right_wrist[1] * h)
                cv2.circle(frame, (x, y), 12, (255, 0, 255), -1)
                cv2.putText(frame, "R", (x-8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def draw_instructions(self, frame):
        """Draw instructions at bottom"""
        h, w = frame.shape[:2]
        sidebar_width = 220
        
        # Instructions background
        cv2.rectangle(frame, (10, h-80), (w - sidebar_width - 20, h-10), (0, 0, 0), -1)
        
        cv2.putText(frame, "BALANCED PUNCH DETECTOR - Labels Persist for Evaluation", 
                   (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Watch: Current label holds for 1s, shows what's detected vs displayed", 
                   (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Controls: 's' = save  |  'q' = quit", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def run_test(self):
        """Run the balanced punch detector"""
        if not self.init_camera():
            print("Failed to initialize camera")
            return
        
        print("\n" + "="*80)
        print("BALANCED PUNCH DETECTOR - MEDIUM SENSITIVITY + LABEL PERSISTENCE")
        print("="*80)
        print("FEATURES:")
        print("• BALANCED thresholds: Angle>130°, Confidence>0.4")
        print("• Label persistence: Hold labels for 1 second for evaluation")
        print("• Shows detected vs displayed labels")
        print("• Confidence smoothing over 5 frames")
        print("• Only logs punches when labels actually change")
        print("="*80)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process with MediaPipe
                mp_results, mp_time = self.process_mediapipe_frame(frame)
                
                # Analyze results
                analysis = self.analyze_mediapipe_results(mp_results)
                
                # Detect punch with confidence
                detected_punch, punch_type, confidence = self.detect_punch_with_confidence(analysis)
                
                # Update persistent label
                persistent_label = self.update_persistent_label(detected_punch, punch_type, confidence)
                
                # Update history only when persistent label changes to punch
                persistent_type = persistent_label.lower().replace(' ', '_').replace('punch', '').replace('combo_', 'combo').strip()
                self.update_punch_history(persistent_label, persistent_type)
                
                # Draw all components
                self.draw_current_status(frame, persistent_label, detected_punch, analysis, confidence, mp_time)
                self.draw_punch_history(frame)
                self.draw_instructions(frame)
                
                # Show frame
                cv2.imshow('Balanced Punch Detector', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save analysis snapshot
                    timestamp = time.strftime("%H:%M:%S")
                    self.results_log.append({
                        'time': timestamp,
                        'persistent_label': persistent_label,
                        'detected_punch': detected_punch,
                        'confidence': confidence,
                        'analysis': analysis,
                        'processing_time': mp_time,
                        'punch_history': list(self.punch_history)
                    })
                    print(f"Analysis snapshot saved at {timestamp}")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("\nTest interrupted")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*60)
        print("BALANCED PUNCH DETECTOR TEST COMPLETE")
        print("="*60)
        print(f"Frames processed: {self.frame_count}")
        print(f"Total punches logged: {len(self.punch_history)}")
        print(f"Analysis snapshots saved: {len(self.results_log)}")
        
        if self.punch_history:
            print("\nPunch sequence logged:")
            for i, punch in enumerate(self.punch_history):
                print(f"  {i+1}. {punch['action']} at {time.strftime('%H:%M:%S', time.localtime(punch['time']))}")


if __name__ == "__main__":
    detector = BalancedPunchDetector()
    detector.run_test()