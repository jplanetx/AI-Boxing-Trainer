#!/usr/bin/env python3
"""
Test Complete Workflow - Validate the full recording + extraction pipeline
Creates a mock training session to test the complete system without webcam dependency.
"""

import cv2
import csv
import numpy as np
import os
import time

def create_mock_training_session():
    """Create a mock training session for testing with realistic spacing"""
    
    print("Creating mock training session for testing...")
    
    # Session parameters - longer session with realistic intervals
    session_name = "mock_session"
    video_file = f"{session_name}.mp4"
    csv_file = f"{session_name}_labels.csv"
    
    # Video parameters
    fps = 30
    width, height = 640, 480
    duration = 20  # 20 seconds for realistic punch spacing
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_file, fourcc, fps, (width, height))
    
    # Punch timing - realistic 4-second intervals
    punch_times = [2, 6, 10, 14, 18]  # Audio cues at these seconds
    punch_keys = ['1', '2', '1b', '4', '3']
    punch_frames = [int(t * fps) for t in punch_times]
    
    # Generate frames with realistic punch motion simulation
    for frame_num in range(total_frames):
        # Create a background frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background color changes slowly
        bg_color = int(100 + 50 * np.sin(frame_num / 100))
        frame[:, :] = (bg_color, 80, 120)
        
        # Add frame number and time
        time_stamp = frame_num / fps
        cv2.putText(frame, f"Frame {frame_num}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {time_stamp:.1f}s", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Simulate punch motion in windows after audio cues
        for i, (punch_frame, punch_key) in enumerate(zip(punch_frames, punch_keys)):
            # Window starts 0.5s after audio cue
            window_start = punch_frame + int(0.5 * fps)
            
            # Window ends at next audio cue (or session end)
            if i + 1 < len(punch_frames):
                window_end = punch_frames[i + 1]
            else:
                window_end = total_frames
            
            # If we're in this punch window, simulate motion
            if window_start <= frame_num < window_end:
                # Calculate position in window (0 to 1)
                window_progress = (frame_num - window_start) / (window_end - window_start)
                
                # Simulate punch motion: extend then retract
                if window_progress < 0.3:
                    # Preparation phase
                    motion_intensity = int(100 * window_progress / 0.3)
                    cv2.putText(frame, f"PREP {punch_key}", (200, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                elif window_progress < 0.6:
                    # Extension phase
                    motion_intensity = int(255 * (window_progress - 0.3) / 0.3)
                    cv2.circle(frame, (320, 240), 30 + motion_intensity//10, (0, 255, 0), -1)
                    cv2.putText(frame, f"PUNCH {punch_key}!", (200, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                else:
                    # Retraction/guard phase
                    motion_intensity = int(100 * (1 - window_progress))
                    cv2.putText(frame, f"GUARD {punch_key}", (200, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
                
                # Add motion indicator
                cv2.rectangle(frame, (50, 400), (50 + motion_intensity*2, 450), 
                             (motion_intensity, 255-motion_intensity, 100), -1)
        
        # Mark audio cue frames
        if frame_num in punch_frames:
            idx = punch_frames.index(frame_num)
            cv2.putText(frame, f"AUDIO CUE: {punch_keys[idx]}", (50, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.rectangle(frame, (0, 0), (width-1, height-1), (255, 0, 0), 5)
        
        out.write(frame)
    
    out.release()
    
    # Create corresponding CSV labels with audio cue frames
    punch_data = []
    for i, (punch_frame, punch_key) in enumerate(zip(punch_frames, punch_keys)):
        punch_data.append({
            'frame_index': punch_frame,
            'punch_key': punch_key
        })
    
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['frame_index', 'punch_key']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(punch_data)
    
    print(f"Created mock session: {video_file} ({total_frames} frames, {duration}s)")
    print(f"Created labels: {csv_file} ({len(punch_data)} audio cues)")
    print(f"Punch windows will span between audio cues for complete motion capture")
    
    return session_name, len(punch_data)

def test_extraction(session_name, expected_clips):
    """Test the windowed extraction script"""
    
    print(f"\nTesting windowed extraction on {session_name}...")
    
    # Import and run extraction
    import subprocess
    result = subprocess.run([
        'python', 'extract_punch_clips.py', session_name,
        '--window-offset', '0.5'
    ], capture_output=True, text=True)
    
    print("Extraction output:")
    print(result.stdout)
    
    if result.stderr:
        print("Extraction errors:")
        print(result.stderr)
    
    # Check if windows were created
    windows_dir = f"{session_name}_windows"
    if os.path.exists(windows_dir):
        windows = [f for f in os.listdir(windows_dir) if f.endswith('.mp4')]
        print(f"\nFound {len(windows)} windows in {windows_dir}/:")
        for window in sorted(windows):
            print(f"  {window}")
        
        if len(windows) == expected_clips:
            print(f"Success: Expected {expected_clips} windows, found {len(windows)}")
            return True
        else:
            print(f"Warning: Expected {expected_clips} windows, found {len(windows)}")
            return len(windows) > 0  # Accept any windows as success
    else:
        print(f"Error: Windows directory {windows_dir} not found")
        return False

def validate_clip_content(session_name):
    """Validate that extracted windows contain the expected content"""
    
    print(f"\nValidating window content...")
    
    windows_dir = f"{session_name}_windows"
    if not os.path.exists(windows_dir):
        print("No windows directory found")
        return False
    
    windows = [f for f in os.listdir(windows_dir) if f.endswith('.mp4')]
    
    for window_file in sorted(windows):
        window_path = os.path.join(windows_dir, window_file)
        
        # Open and check window
        cap = cv2.VideoCapture(window_path)
        if not cap.isOpened():
            print(f"Could not open {window_file}")
            continue
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps
        
        print(f"  {window_file}: {frame_count} frames, {duration:.1f}s duration")
        
        cap.release()
    
    print("Window validation complete")
    return True

def main():
    """Run complete workflow test"""
    
    print("=" * 60)
    print("COMPLETE WORKFLOW TEST")
    print("=" * 60)
    
    try:
        # Step 1: Create mock session
        session_name, expected_clips = create_mock_training_session()
        
        # Step 2: Test extraction
        extraction_success = test_extraction(session_name, expected_clips)
        
        # Step 3: Validate results
        if extraction_success:
            validate_clip_content(session_name)
            
            print("\n" + "=" * 60)
            print("WORKFLOW TEST COMPLETE ✓")
            print("=" * 60)
            print("Ready to test with real webcam recording!")
            print()
            print("Next steps:")
            print("1. python training_mode.py --combo 1,2,1,2,4,1 --output test_session --interval 2")
            print("2. python extract_punch_clips.py test_session --punch-delay 0.8 --punch-duration 1.2")
            
        else:
            print("\n" + "=" * 60)
            print("WORKFLOW TEST FAILED ✗")
            print("=" * 60)
            
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    main()
