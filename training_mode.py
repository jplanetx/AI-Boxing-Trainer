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

def speak_prompt(text):
    """Speak text using a fresh TTS engine instance to avoid threading issues."""
    try:
        # Create fresh engine for each prompt to avoid Windows SAPI issues
        engine = pyttsx3.init()
        
        # Quick setup
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        
        # Set voice
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        print(f"SPEAKING: {text}")
        engine.say(text)
        engine.runAndWait()
        
        # Cleanup
        engine.stop()
        del engine
        
        # Ensure completion
        time.sleep(0.3)
        
    except Exception as e:
        print(f"TTS Error: {e}")
        print(f"FALLBACK PROMPT: {text}")
        # Audio beep as fallback
        try:
            import winsound
            winsound.Beep(800, 200)  # High beep
        except:
            pass

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
    
    # Test TTS before starting
    print("Testing TTS system...")
    speak_prompt("TTS test - ready to train")
    
    # Prepare CSV file
    csv_data = []
    frame_count = 0
    
    try:
        print("\nStarting training session in 3 seconds...")
        print("Position yourself in the webcam view, then press SPACE to start or 'q' to quit...")
        
        # Show webcam preview for positioning
        preview_start = time.time()
        while time.time() - preview_start < 10:  # 10 second preview window
            ret, frame = cap.read()
            if ret:
                # Add instruction overlay
                cv2.putText(frame, "Position yourself in frame", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Press SPACE to start, 'q' to quit", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Training Mode - Position Check', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space to start
                    break
                elif key == ord('q'):  # Quit
                    print("Training cancelled by user.")
                    return
        
        cv2.destroyAllWindows()
        print("Starting training sequence...")
        time.sleep(1)
        
        # Main training loop
        for i, combo_key in enumerate(combo_keys):
            punch_name = COMBO_MAP[combo_key]
            prompt = f"Throw {punch_name.replace('_', ' ')}"
            
            print(f"[{i+1}/{len(combo_keys)}] Frame {frame_count}: {prompt}")
            
            # BEFORE speaking: capture frame and write to video
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                
                # Show live preview during training (smaller window)
                preview_frame = cv2.resize(frame, (320, 240))
                cv2.putText(preview_frame, f"Recording: {prompt}", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(preview_frame, f"Frame: {frame_count}", (5, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.imshow('Training Live', preview_frame)
                cv2.waitKey(1)
                
                # Log this frame with the punch key
                csv_data.append({
                    'frame_index': frame_count,
                    'punch_key': f"{combo_key[0]}.{str(combo_key[1]).split('.')[1]}"
                })
                
                frame_count += 1
            
            # THEN speak the prompt using improved TTS function
            speak_prompt(prompt)
            
            # Sleep for the specified interval
            if i < len(combo_keys) - 1:  # Don't sleep after the last prompt
                sleep_start = time.time()
                while time.time() - sleep_start < args.interval:
                    # Continue capturing frames during sleep
                    ret, frame = cap.read()
                    if ret:
                        out.write(frame)
                        
                        # Update live preview
                        preview_frame = cv2.resize(frame, (320, 240))
                        remaining_time = args.interval - (time.time() - sleep_start)
                        cv2.putText(preview_frame, f"Next in: {remaining_time:.1f}s", (5, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                        cv2.imshow('Training Live', preview_frame)
                        cv2.waitKey(1)
                        
                        frame_count += 1
                    time.sleep(0.03)  # ~30fps capture rate
        
        # Continue capturing for 1 additional second to finalize
        print("Finalizing recording...")
        finalize_start = time.time()
        while time.time() - finalize_start < 1.0:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                
                # Update live preview
                preview_frame = cv2.resize(frame, (320, 240))
                cv2.putText(preview_frame, "Finalizing...", (5, 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.imshow('Training Live', preview_frame)
                cv2.waitKey(1)
                
                frame_count += 1
            time.sleep(0.03)
        
        cv2.destroyAllWindows()
        
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
        try:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    main()
