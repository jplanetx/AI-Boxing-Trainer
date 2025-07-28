#!/usr/bin/env python3
"""
Debug Version - Minimal AI Boxing Trainer for troubleshooting
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def debug_trainer():
    """Minimal trainer for debugging landmark issues."""
    print("🔧 Debug Mode - AI Boxing Trainer")
    print("=" * 50)
    print("This version shows raw landmark detection without classification")
    print("Press 'q' to quit")
    print()
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1  # Use lighter model for stability
    )
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Key landmarks we need
    key_landmarks = {
        'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
        'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
        'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
        'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
        'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
        'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
        'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
        'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
    }
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            
            # Process pose
            results = pose.process(rgb_frame)
            
            # Convert back to BGR
            rgb_frame.flags.writeable = True
            processed_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
            # Extract and display landmark info
            landmarks_found = {}
            if results.pose_landmarks:
                try:
                    for name, landmark_idx in key_landmarks.items():
                        landmark = results.pose_landmarks.landmark[landmark_idx.value]
                        
                        x = landmark.x * frame_width
                        y = landmark.y * frame_height
                        z = landmark.z * frame_width
                        visibility = landmark.visibility
                        
                        if visibility > 0.1:  # Very low threshold
                            landmarks_found[name] = {
                                'x': x, 'y': y, 'z': z, 
                                'visibility': visibility
                            }
                    
                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        processed_frame, 
                        results.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS
                    )
                    
                except Exception as e:
                    print(f"Frame {frame_count}: Landmark extraction error: {e}")
            
            # Display landmark count and status
            cv2.rectangle(processed_frame, (0, 0), (400, 100), (0, 0, 0), -1)
            
            landmarks_count = len(landmarks_found)
            cv2.putText(processed_frame, f"Landmarks Found: {landmarks_count}/8", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show which landmarks are missing
            missing = []
            for name in key_landmarks.keys():
                if name not in landmarks_found:
                    missing.append(name.replace('_', ' ').title())
            
            if missing:
                missing_text = f"Missing: {', '.join(missing[:3])}"
                if len(missing) > 3:
                    missing_text += "..."
                cv2.putText(processed_frame, missing_text, 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(processed_frame, "All landmarks detected!", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show pose detection status
            pose_status = "POSE DETECTED" if results.pose_landmarks else "NO POSE"
            status_color = (0, 255, 0) if results.pose_landmarks else (0, 0, 255)
            cv2.putText(processed_frame, pose_status, 
                       (frame_width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            cv2.imshow('Debug Mode - AI Boxing Trainer', processed_frame)
            
            frame_count += 1
            
            # Print landmark status every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}: {landmarks_count}/8 landmarks, Missing: {missing}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("🧹 Debug session ended")

if __name__ == "__main__":
    debug_trainer()
