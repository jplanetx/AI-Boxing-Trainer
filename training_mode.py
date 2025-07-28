#!/usr/bin/env python3
"""
Training Mode - Automated Boxing Training Session Recorder
Records webcam footage while providing TTS prompts for boxing combinations.
Outputs synchronized video and frame-indexed labels.
"""

import cv2
import pyttsx3
import argparse
import csv
import time
import os
from pathlib import Path

# Boxing combination mapping
COMBO_MAP = {
    (1, 1.1): "jab_head",
    (1, 1.2): "jab_body", 
    (2, 2.1): "cross_head",
    (2, 2.2): "cross_body",
    (3, 3.1): "left_hook_head",
    (3, 3.2): "left_hook_body",
    (4, 4.1): "right_hook_head", 
    (4, 4.2): "right_hook_body",
    (5, 5.1): "left_uppercut_head",
    (5, 5.2): "left_uppercut_body",
    (6, 6.1): "right_uppercut_head",
    (6, 6.2): "right_uppercut_body"
}

def parse_combo_string(combo_str):
    """Parse combo string like '1,2,3.1,4.2' into list of punch keys."""
    combo_keys = []
    
    for item in combo_str.split(','):
        item = item.strip()
        if '.' in item:
            # Handle decimal notation (e.g., '3.1')
            main_num, sub_num = item.split('.')
            key = (int(main_num), float(item))
        else:
            # Handle integer notation (e.g., '1' -> '1.1')
            key = (int(item), float(f"{item}.1"))
        
        if key in COMBO_MAP:
            combo_keys.append(key)
        else:
            print(f"Warning: Unknown combo key {item}, skipping...")
    
    return combo_keys

def initialize_tts_engine():
    """Initialize and configure the TTS engine."""
    engine = pyttsx3.init()
    
    # Configure speech rate and volume
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)  # Slightly slower speech
    engine.setProperty('volume', 0.9)
    
    return engine

def get_next_session_number():
    """Find the next available session number."""
    session_num = 1
    while os.path.exists(f"session{session_num}.mp4"):
        session_num += 1
    return session_num

def main():
    parser = argparse.ArgumentParser(description="Training Mode - Automated Boxing Training Recorder")
    parser.add_argument('--combo', type=str, required=True,
                       help='Comma-separated combo sequence (e.g., "1,2,3.1,4.2")')
    parser.add_argument('--output', type=str, 
                       help='Output base name (default: sessionX)')
    parser.add_argument('--interval', type=float, default=3.0,
                       help='Interval between prompts in seconds (default: 3.0)')
    
    args = parser.parse_args()
    
    # Parse the combo sequence
    combo_keys = parse_combo_string(args.combo)
    if not combo_keys:
        print("Error: No valid combo keys found!")
        return
    
    # Determine output filename
    if args.output:
        base_name = args.output
    else:
        session_num = get_next_session_number()
        base_name = f"session{session_num}"
    
    video_filename = f"{base_name}.mp4"
    csv_filename = f"{base_name}_labels.csv"
    
    print(f"Starting training session...")
    print(f"Combo sequence: {[COMBO_MAP[key] for key in combo_keys]}")
    print(f"Video output: {video_filename}")
    print(f"Labels output: {csv_filename}")
    print(f"Interval: {args.interval} seconds")
    print("-" * 50)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Get webcam properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam initialized: {width}x{height} @ {fps}fps")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    # Initialize TTS engine
    tts_engine = initialize_tts_engine()
    
    # Prepare CSV file
    csv_data = []
    frame_count = 0
    
    try:
        print("\nStarting training session in 3 seconds...")
        time.sleep(3)
        
        # Main training loop
        for i, combo_key in enumerate(combo_keys):
            punch_name = COMBO_MAP[combo_key]
            prompt = f"Throw {punch_name.replace('_', ' ')}"
            
            print(f"[{i+1}/{len(combo_keys)}] Frame {frame_count}: {prompt}")
            
            # BEFORE speaking: capture frame and write to video
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                
                # Log this frame with the punch key
                csv_data.append({
                    'frame_index': frame_count,
                    'punch_key': f"{combo_key[0]}.{str(combo_key[1]).split('.')[1]}"
                })
                
                frame_count += 1
            
            # THEN speak the prompt
            tts_engine.say(prompt)
            tts_engine.runAndWait()
            
            # Sleep for the specified interval
            if i < len(combo_keys) - 1:  # Don't sleep after the last prompt
                sleep_start = time.time()
                while time.time() - sleep_start < args.interval:
                    # Continue capturing frames during sleep
                    ret, frame = cap.read()
                    if ret:
                        out.write(frame)
                        frame_count += 1
                    time.sleep(0.01)  # Small delay to prevent excessive CPU usage
        
        # Continue capturing for 1 additional second to finalize
        print("Finalizing recording...")
        finalize_start = time.time()
        while time.time() - finalize_start < 1.0:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                frame_count += 1
            time.sleep(0.01)
        
        # Write CSV data
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['frame_index', 'punch_key']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\nTraining session completed!")
        print(f"Total frames recorded: {frame_count}")
        print(f"Video saved: {video_filename}")
        print(f"Labels saved: {csv_filename}")
        
    except KeyboardInterrupt:
        print("\nTraining session interrupted by user.")
        
    except Exception as e:
        print(f"Error during training session: {e}")
        
    finally:
        # Clean up resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        # Cleanup TTS engine
        try:
            tts_engine.stop()
        except:
            pass

if __name__ == "__main__":
    main()
