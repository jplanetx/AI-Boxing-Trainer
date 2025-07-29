#!/usr/bin/env python3
"""
INDIVIDUAL PUNCH RECORDER - Clean ML Training Data Generator
Simple, focused tool for recording individual punches for ML training.

Usage:
    python individual_punch_recorder.py --punch-type jab --output-dir training_data

Features:
- Records 4-second clips (1s prep + punch + 1s return)
- 60fps for optimal motion capture
- Auto-incrementing filenames
- Perfect labeling (filename = punch type)
- No timing ambiguity or windowing

Author: Elevate AI
Target: High-quality ML training data for $1,800+ revenue
"""

import cv2
import time
import os
import argparse
from pathlib import Path

# Punch type mapping
PUNCH_TYPES = {
    '1': 'jab',
    '2': 'cross', 
    '3': 'hook',
    '4': 'uppercut'
}

def get_next_filename(output_dir, punch_type):
    """Get next available filename for this punch type."""
    counter = 1
    while True:
        filename = f"{punch_type}_{counter:03d}.mp4"
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            return filepath
        counter += 1

def record_single_punch(output_dir, punch_type, duration=4.0):
    """Record a single punch clip."""
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get output filename
    output_path = get_next_filename(output_dir, punch_type)
    filename = os.path.basename(output_path)
    
    print(f"Ready to record: {punch_type.upper()} → {filename}")
    print("Position yourself, then press SPACE to start 4-second recording...")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        return False
    
    # Set high frame rate for better motion capture
    cap.set(cv2.CAP_PROP_FPS, 60)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera: {width}x{height} @ {fps}fps")
    
    # Wait for user to position and press space
    positioning = True
    while positioning:
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, f"Ready: {punch_type.upper()} - Press SPACE to record", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Output: {filename}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.imshow('Individual Punch Recorder', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                positioning = False
            elif key == ord('q'):
                print("Cancelled by user")
                cap.release()
                cv2.destroyAllWindows()
                return False
    
    # Start recording
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"ERROR: Could not create video writer for {output_path}")
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    print(f"🔴 RECORDING {punch_type.upper()} for {duration} seconds...")
    
    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Lost webcam connection!")
            break
        
        out.write(frame)
        frame_count += 1
        
        elapsed = time.time() - start_time
        remaining = duration - elapsed
        
        # Show recording status
        cv2.putText(frame, f"REC: {elapsed:.1f}s / {duration}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Remaining: {remaining:.1f}s", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow('Individual Punch Recorder', frame)
        
        # Check for completion
        if elapsed >= duration:
            break
        
        # Allow early exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Recording stopped by user")
            break
    
    # Cleanup
    final_duration = time.time() - start_time
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Recorded {frame_count} frames in {final_duration:.2f}s")
    print(f"📁 Saved: {output_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Individual Punch Recorder for ML Training')
    parser.add_argument('--punch-type', '-p', choices=['1', '2', '3', '4', 'jab', 'cross', 'hook', 'uppercut'],
                       help='Punch type: 1=jab, 2=cross, 3=hook, 4=uppercut')
    parser.add_argument('--output-dir', '-o', default='training_data',
                       help='Output directory for video clips')
    parser.add_argument('--duration', '-d', type=float, default=4.0,
                       help='Recording duration in seconds (default: 4.0)')
    parser.add_argument('--batch', '-b', type=int, default=1,
                       help='Number of clips to record in batch')
    
    args = parser.parse_args()
    
    # Convert punch type if needed
    if args.punch_type in PUNCH_TYPES:
        punch_type = PUNCH_TYPES[args.punch_type]
    else:
        punch_type = args.punch_type
    
    print("=" * 60)
    print("INDIVIDUAL PUNCH RECORDER - CLEAN ML TRAINING DATA")
    print("=" * 60)
    print(f"Punch Type: {punch_type.upper()}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Duration: {args.duration}s per clip")
    print(f"Batch Size: {args.batch} clips")
    print("=" * 60)
    
    success_count = 0
    for i in range(args.batch):
        if args.batch > 1:
            print(f"\n--- Clip {i+1}/{args.batch} ---")
        
        if record_single_punch(args.output_dir, punch_type, args.duration):
            success_count += 1
            if i < args.batch - 1:  # Not the last clip
                print("Press ENTER for next clip, or 'q' + ENTER to quit...")
                user_input = input().strip().lower()
                if user_input == 'q':
                    break
        else:
            print(f"Failed to record clip {i+1}")
            break
    
    print(f"\n🎯 Session Complete: {success_count}/{args.batch} clips recorded")
    print(f"📁 Location: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
