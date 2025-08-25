#!/usr/bin/env python3
"""
Smart Boxing Detector
Multi-modal punch detection including hooks with velocity and position tracking
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

class SmartBoxingDetector:
    """Smart boxing detector with hook support and reduced jumpiness"""
    
    def __init__(self):
        print("Initializing smart boxing detector...")
        
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
        
        # Punch tracking with velocity
        self.punch_history = deque(maxlen=5)
        self.wrist_history = deque(maxlen=10)  # For velocity calculation
        self.last_punch_type = "guard"
        self.last_punch_time = 0
        
        # State smoothing
        self.state_buffer = deque(maxlen=3)  # Smooth over 3 frames
        self.confidence_threshold = 0.6
        
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
    
    def calculate_wrist_velocity(self, current_wrist, frame_time, hand='left'):
        """Calculate wrist velocity for hook detection"""
        if len(self.wrist_history) < 2:
            return 0, 0
        
        # Get previous wrist position
        prev_data = self.wrist_history[-1]
        prev_wrist = prev_data[f'{hand}_wrist'] if prev_data[f'{hand}_wrist'] else current_wrist
        prev_time = prev_data['time']
        
        # Calculate velocity (pixels per second)
        dt = frame_time - prev_time
        if dt <= 0:
            return 0, 0
        
        vx = (current_wrist[0] - prev_wrist[0]) / dt
        vy = (current_wrist[1] - prev_wrist[1]) / dt
        
        return vx, vy
    
    def detect_punch_type_smart(self, analysis, frame_time):
        """Smart punch detection using multiple signals"""
        if not analysis:
            return "guard", 0.0
        
        left_wrist = analysis.get('left_wrist')
        right_wrist = analysis.get('right_wrist')
        left_angle = analysis.get('left_angle', 180)
        right_angle = analysis.get('right_angle', 180)
        left_conf = analysis.get('left_confidence', 0)
        right_conf = analysis.get('right_confidence', 0)
        
        # Store wrist positions for velocity tracking
        self.wrist_history.append({
            'left_wrist': left_wrist,
            'right_wrist': right_wrist,
            'time': frame_time
        })
        
        punch_scores = {
            'guard': 1.0,  # Default state
            'left_jab': 0.0,
            'right_cross': 0.0,
            'left_hook': 0.0,
            'right_hook': 0.0
        }
        
        # === LEFT ARM ANALYSIS ===
        if left_wrist and left_conf > 0.5:
            # Calculate velocity
            left_vx, left_vy = self.calculate_wrist_velocity(left_wrist, frame_time, 'left')
            left_speed = np.sqrt(left_vx**2 + left_vy**2)
            
            # Jab detection (straight forward movement)
            if left_angle and left_angle > 135:  # Arm extended
                if abs(left_vx) < 200 and left_speed > 100:  # Forward motion
                    punch_scores['left_jab'] = min(1.0, left_speed / 300) * left_conf
            
            # Hook detection (lateral movement + position)
            if left_wrist[0] < 0.3:  # Left side of screen
                if abs(left_vx) > 150:  # Lateral movement
                    punch_scores['left_hook'] = min(1.0, abs(left_vx) / 250) * left_conf
        
        # === RIGHT ARM ANALYSIS ===
        if right_wrist and right_conf > 0.5:
            # Calculate velocity
            right_vx, right_vy = self.calculate_wrist_velocity(right_wrist, frame_time, 'right')
            right_speed = np.sqrt(right_vx**2 + right_vy**2)
            
            # Cross detection (straight forward movement)
            if right_angle and right_angle > 135:  # Arm extended
                if abs(right_vx) < 200 and right_speed > 100:  # Forward motion
                    punch_scores['right_cross'] = min(1.0, right_speed / 300) * right_conf
            
            # Hook detection (lateral movement + position)
            if right_wrist[0] > 0.7:  # Right side of screen
                if abs(right_vx) > 150:  # Lateral movement
                    punch_scores['right_hook'] = min(1.0, abs(right_vx) / 250) * right_conf
        
        # Find best punch type
        best_punch = max(punch_scores.keys(), key=lambda k: punch_scores[k])
        best_score = punch_scores[best_punch]
        
        return best_punch, best_score
    
    def smooth_detection(self, punch_type, confidence):
        """Smooth detection over multiple frames to reduce jumpiness"""
        self.state_buffer.append((punch_type, confidence, time.time()))
        
        # If buffer not full, return current state
        if len(self.state_buffer) < 3:
            return punch_type if confidence > self.confidence_threshold else "guard"
        
        # Count occurrences of each punch type
        type_counts = {}
        total_confidence = 0
        
        for p_type, p_conf, p_time in self.state_buffer:
            if p_conf > self.confidence_threshold:
                type_counts[p_type] = type_counts.get(p_type, 0) + 1
                total_confidence += p_conf
        
        # Require at least 2 out of 3 frames to agree
        for p_type, count in type_counts.items():
            if count >= 2:
                return p_type
        
        return "guard"
    
    def update_punch_history(self, final_punch_type):
        """Update punch history with timing"""
        current_time = time.time()
        
        # Only log actual punches (not guard) and prevent spam
        if (final_punch_type != "guard" and 
            final_punch_type != self.last_punch_type and
            current_time - self.last_punch_time > 0.4):  # 400ms minimum
            
            # Convert to display format
            display_names = {
                'left_jab': 'LEFT JAB',
                'right_cross': 'RIGHT CROSS', 
                'left_hook': 'LEFT HOOK',
                'right_hook': 'RIGHT HOOK'
            }
            
            display_name = display_names.get(final_punch_type, final_punch_type.upper())
            
            self.punch_history.append({
                'action': display_name,
                'time': current_time,
                'type': final_punch_type
            })
            self.last_punch_time = current_time
        
        self.last_punch_type = final_punch_type
        return final_punch_type
    
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
    
    def draw_punch_history(self, frame):
        """Draw punch history sidebar with fading effect"""
        h, w = frame.shape[:2]
        
        # History sidebar background (right side)
        sidebar_width = 220
        cv2.rectangle(frame, (w - sidebar_width, 0), (w, h), (20, 20, 20), -1)
        
        # Title
        cv2.putText(frame, "PUNCH SEQUENCE", 
                   (w - sidebar_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw history with decay effect
        y_start = 70
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
    
    def draw_current_status(self, frame, current_action, analysis, processing_time, debug_info=None):
        """Draw large, visible current status with debug info"""
        h, w = frame.shape[:2]
        sidebar_width = 220
        
        # Main status area background (top left)
        cv2.rectangle(frame, (10, 10), (w - sidebar_width - 20, 180), (0, 0, 0), -1)
        
        # Current action - LARGE and prominent
        color_map = {
            'left_jab': (0, 255, 255),     # Yellow
            'right_cross': (255, 0, 255),  # Magenta
            'left_hook': (0, 255, 0),      # Green  
            'right_hook': (255, 100, 0),   # Orange
            'guard': (255, 255, 255)       # White
        }
        
        display_names = {
            'left_jab': 'LEFT JAB',
            'right_cross': 'RIGHT CROSS',
            'left_hook': 'LEFT HOOK', 
            'right_hook': 'RIGHT HOOK',
            'guard': 'GUARD'
        }
        
        display_text = display_names.get(current_action, current_action.upper())
        color = color_map.get(current_action, (255, 255, 255))
        
        cv2.putText(frame, f"CURRENT: {display_text}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        # Technical details
        if analysis:
            cv2.putText(frame, f"Left: {analysis.get('left_angle', 0):.0f}° ({analysis.get('left_confidence', 0):.2f})", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"Right: {analysis.get('right_angle', 0):.0f}° ({analysis.get('right_confidence', 0):.2f})", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Processing: {processing_time:.1f}ms", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Debug velocity info
        if debug_info and len(self.wrist_history) >= 2:
            cv2.putText(frame, f"L Speed: {debug_info.get('left_speed', 0):.0f} px/s", 
                       (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"R Speed: {debug_info.get('right_speed', 0):.0f} px/s", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Draw enhanced wrist markers with zones
        if analysis:
            left_wrist = analysis.get('left_wrist')
            right_wrist = analysis.get('right_wrist')
            
            # Draw hook zones
            cv2.rectangle(frame, (0, int(h*0.2)), (int(w*0.3), int(h*0.8)), (0, 100, 100), 2)  # Left hook zone
            cv2.rectangle(frame, (int(w*0.7), int(h*0.2)), (w, int(h*0.8)), (100, 0, 100), 2)  # Right hook zone
            
            if left_wrist:
                x, y = int(left_wrist[0] * w), int(left_wrist[1] * h)
                cv2.circle(frame, (x, y), 15, (0, 255, 255), -1)
                cv2.putText(frame, "L", (x-8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            if right_wrist:
                x, y = int(right_wrist[0] * w), int(right_wrist[1] * h)
                cv2.circle(frame, (x, y), 15, (255, 0, 255), -1)
                cv2.putText(frame, "R", (x-8, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    def draw_instructions(self, frame):
        """Draw instructions at bottom"""
        h, w = frame.shape[:2]
        sidebar_width = 220
        
        # Instructions background
        cv2.rectangle(frame, (10, h-80), (w - sidebar_width - 20, h-10), (0, 0, 0), -1)
        
        cv2.putText(frame, "SMART BOXING DETECTOR - Hook Support Enabled", 
                   (20, h-60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Jabs/Cross: Straight forward • Hooks: Wide lateral movement", 
                   (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "Controls: 's' = save analysis  |  'q' = quit", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def run_test(self):
        """Run the smart boxing detector"""
        if not self.init_camera():
            print("Failed to initialize camera")
            return
        
        print("\n" + "="*80)
        print("SMART BOXING DETECTOR - WITH HOOK SUPPORT")
        print("="*80)
        print("ENHANCED FEATURES:")
        print("• Multi-modal detection: angle + velocity + position")
        print("• Hook detection using lateral movement patterns")
        print("• Smoothed detection to reduce jumpiness") 
        print("• Punch sequence tracking with visual decay")
        print("• Support for: Jab, Cross, Left Hook, Right Hook")
        print("="*80)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                frame_time = time.time()
                
                # Process with MediaPipe
                mp_results, mp_time = self.process_mediapipe_frame(frame)
                
                # Analyze results
                analysis = self.analyze_mediapipe_results(mp_results)
                
                # Smart punch detection
                raw_punch_type, confidence = self.detect_punch_type_smart(analysis, frame_time)
                
                # Smooth detection to reduce jumpiness
                final_punch_type = self.smooth_detection(raw_punch_type, confidence)
                
                # Update history
                self.update_punch_history(final_punch_type)
                
                # Calculate debug info
                debug_info = {}
                if analysis and len(self.wrist_history) >= 2:
                    left_wrist = analysis.get('left_wrist')
                    right_wrist = analysis.get('right_wrist')
                    if left_wrist:
                        left_vx, left_vy = self.calculate_wrist_velocity(left_wrist, frame_time, 'left')
                        debug_info['left_speed'] = np.sqrt(left_vx**2 + left_vy**2)
                    if right_wrist:
                        right_vx, right_vy = self.calculate_wrist_velocity(right_wrist, frame_time, 'right')
                        debug_info['right_speed'] = np.sqrt(right_vx**2 + right_vy**2)
                
                # Draw all components
                self.draw_current_status(frame, final_punch_type, analysis, mp_time, debug_info)
                self.draw_punch_history(frame)
                self.draw_instructions(frame)
                
                # Show frame
                cv2.imshow('Smart Boxing Detector', frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save analysis snapshot
                    timestamp = time.strftime("%H:%M:%S")
                    self.results_log.append({
                        'time': timestamp,
                        'current_action': final_punch_type,
                        'raw_detection': raw_punch_type,
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
        print("SMART BOXING DETECTOR TEST COMPLETE")
        print("="*60)
        print(f"Frames processed: {self.frame_count}")
        print(f"Total punches logged: {len([p for p in self.punch_history])}")
        print(f"Analysis snapshots saved: {len(self.results_log)}")
        
        if self.punch_history:
            print("\nPunch sequence detected:")
            for i, punch in enumerate(self.punch_history):
                print(f"  {i+1}. {punch['action']} at {time.strftime('%H:%M:%S', time.localtime(punch['time']))}")
        
        if self.results_log:
            processing_times = [r['processing_time'] for r in self.results_log if r['processing_time'] > 0]
            if processing_times:
                print(f"Average processing time: {np.mean(processing_times):.1f}ms")


if __name__ == "__main__":
    detector = SmartBoxingDetector()
    detector.run_test()