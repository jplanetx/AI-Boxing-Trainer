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
    "1":  "jab_head",
    "1b": "jab_body",
    "2":  "cross_head",
    "2b": "cross_body",
    "3":  "lead_hook_head",
    "3b": "lead_hook_body",
    "4":  "rear_hook_head",
    "4b": "rear_hook_body",
    "5":  "lead_uppercut_head",
    "5b": "lead_uppercut_body",
    "6":  "rear_uppercut_head",
    "6b": "rear_uppercut_body",
}
def parse_combo_string(combo_str):
    """Parse combo string like '1,2,3b,4b' into list of punch keys."""
    combo_keys = []
    
    for item in combo_str.split(','):
        key = item.strip()
        
        if key in COMBO_MAP:
            combo_keys.append(key)
        else:
            print(f"Warning: Unknown combo key '{key}', skipping...")
    
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

def create_speak_text(label):
    """Convert punch label to natural speech text."""
    speak_text = label.replace("_", " ").replace("head", "to the head").replace("body", "to the body")
    return f"{speak_text} — go!"

def get_next_session_number():
    """Find the next available session number."""
    session_num = 1
    while os.path.exists(f"session{session_num}.mp4"):
        session_num += 1
    return session_num

def main():
    parser = argparse.ArgumentParser(description="Training Mode - Automated Boxing Training Recorder")
    parser.add_argument('--combo', type=str, required=True,
                       help='Comma-separated combo sequence (e.g., "1,2,3b,4b")')
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
        
        # Record ENTIRE session continuously, log TTS timestamps for post-processing
        start_time = time.time()
        prompt_times = [start_time + i * args.interval for i in range(len(combo_keys))]
        prompt_idx = 0
        tts_timestamps = []  # Log when each TTS actually fired
        
        print(f"Recording started at {start_time:.2f}")
        print("Strategy: Record everything, extract punch windows post-processing")
        
        while True:
            # Record every frame continuously
            ret, frame = cap.read()
            if not ret:
                print("Error: Lost webcam connection!")
                break
                
            out.write(frame)
            now = time.time()
            
            # Calculate current frame index based on time and FPS
            frame_idx = int((now - start_time) * fps)
            
            # Show live preview
            preview_frame = cv2.resize(frame, (320, 240))
            cv2.putText(preview_frame, f"Recording: Frame {frame_idx}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Trigger TTS when time arrives for next prompt
            if prompt_idx < len(prompt_times) and now >= prompt_times[prompt_idx]:
                combo_key = combo_keys[prompt_idx]
                label = COMBO_MAP[combo_key]
                speak_text = create_speak_text(label)
                
                # Log the ACTUAL TTS timestamp and frame for post-processing
                tts_timestamp = {
                    'prompt_idx': prompt_idx,
                    'combo_key': combo_key,
                    'tts_time': now - start_time,
                    'tts_frame': frame_idx,
                    'prompt_text': speak_text
                }
                tts_timestamps.append(tts_timestamp)
                
                print(f"[TTS {prompt_idx+1}/{len(combo_keys)}] t={tts_timestamp['tts_time']:.2f}s, Frame {frame_idx}: {speak_text}")
                
                # Update preview with current prompt
                cv2.putText(preview_frame, f"TTS: {speak_text}", (5, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                
                # Trigger TTS (non-blocking)
                speak_prompt(speak_text)
                prompt_idx += 1
            
            # Show countdown to next prompt or session end
            if prompt_idx < len(prompt_times):
                time_to_next = prompt_times[prompt_idx] - now
                cv2.putText(preview_frame, f"Next TTS: {time_to_next:.1f}s", (5, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                # After all prompts, show finalization countdown
                time_remaining = (prompt_times[-1] + 3.0) - now  # 3 second tail for final punch
                cv2.putText(preview_frame, f"Finishing: {time_remaining:.1f}s", (5, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            cv2.imshow('Training Live', preview_frame)
            cv2.waitKey(1)
            
            # Exit condition: all prompts complete + 3 second tail for final punch
            if prompt_idx >= len(prompt_times) and now >= prompt_times[-1] + 3.0:
                print("Recording completed with 3-second tail")
                break
                
            # Emergency exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user")
                break
        
        
        cv2.destroyAllWindows()
        
        # Write labels CSV in format expected by extract_punch_clips.py
        csv_data = []
        for ts in tts_timestamps:
            csv_data.append({
                'frame_index': ts['tts_frame'],
                'punch_key': ts['combo_key']
            })
        
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ['frame_index', 'punch_key']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\nSession completed!")
        print(f"Total duration: {final_duration:.2f} seconds")
        print(f"Total frames recorded: {final_frame_count}")
        print(f"Video saved: {video_filename}")
        print(f"TTS timestamps saved: {csv_filename}")
        print(f"Total prompts: {len(tts_timestamps)}")
        print()
        print("POST-PROCESSING: Use the timestamps to extract punch windows from the full video")
        print("Example: Extract frames [tts_frame + 30 : tts_frame + 75] for each punch (1-1.5s after TTS)")
        
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
