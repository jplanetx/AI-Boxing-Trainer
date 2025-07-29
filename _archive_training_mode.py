#!/usr/bin/env python3
"""
FIXED Training Mode - Simple continuous recording with proper session length
"""

import cv2
import pyttsx3
import argparse
import csv
import threading
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
    """Speak text using TTS"""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)
        
        print(f"SPEAKING: {text}")
        engine.say(text)
        engine.runAndWait()
        engine.stop()
        del engine
        time.sleep(0.3)
    except Exception as e:
        print(f"TTS Error: {e}")

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
    parser = argparse.ArgumentParser(description="Training Mode - Record Full Boxing Session")
    parser.add_argument('--combo', type=str, required=True,
                       help='Comma-separated combo sequence (e.g., "1,2,1,2")')
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
    
    print("=" * 60)
    print("FIXED TRAINING MODE - CONTINUOUS RECORDING")
    print("=" * 60)
    print(f"Combo sequence: {combo_keys}")
    print(f"Video output: {video_filename}")
    print(f"Labels output: {csv_filename}")
    print(f"Interval: {args.interval} seconds")
    
    # Calculate total session time
    total_prompts = len(combo_keys)
    estimated_duration = (total_prompts - 1) * args.interval + 5  # +5s tail
    print(f"Estimated session duration: {estimated_duration:.1f} seconds")
    print("=" * 60)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam!")
        return
    
    # Get webcam properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Webcam: {width}x{height} @ {fps}fps")
    
    # Initialize video writer - USE RELIABLE CODEC
    print("Using mp4v codec...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"ERROR: Could not create video writer for {video_filename}")
        return
    
    print("Video writer initialized successfully")
    
    try:
        print("\nPress SPACE to start recording, or 'q' to quit...")
        
        # Positioning phase
        positioning = True
        while positioning:
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, "Position yourself - Press SPACE to start", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Training Mode', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    positioning = False
                elif key == ord('q'):
                    print("Cancelled by user")
                    return
        
        cv2.destroyAllWindows()
        print("Starting recording NOW...")
        
        # MAIN RECORDING PHASE
        start_time = time.time()
        prompt_times = [start_time + i * args.interval for i in range(len(combo_keys))]
        prompt_idx = 0
        tts_timestamps = []
        
        print(f"Recording started at {time.strftime('%H:%M:%S', time.localtime(start_time))}")
        print(f"Will record for {estimated_duration} seconds...")
        print(f"DEBUG: Expected session duration = {estimated_duration}s")
        
        frame_count = 0
        last_debug_time = start_time
        
        while True:
            # Record frame
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Lost webcam connection!")
                break
            
            # CRITICAL: Check if video writer is working
            write_success = out.write(frame)
            if write_success is False:
                print(f"ERROR: Video writer failed at frame {frame_count}!")
                break
                
            frame_count += 1
            now = time.time()
            elapsed = now - start_time
            frame_idx = int(elapsed * fps)
            
            # Debug every 0.5 seconds for detailed tracking
            if now - last_debug_time >= 0.5:
                session_duration = (len(combo_keys) - 1) * args.interval + 5.0
                remaining = session_duration - elapsed
                print(f"FRAME DEBUG: t={elapsed:.2f}s, frames_written={frame_count}, target={session_duration:.1f}s")
                last_debug_time = now
            
            # Show preview
            preview = cv2.resize(frame, (320, 240))
            cv2.putText(preview, f"REC: {elapsed:.1f}s / Frame {frame_count}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # TTS trigger
            if prompt_idx < len(prompt_times) and now >= prompt_times[prompt_idx]:
                combo_key = combo_keys[prompt_idx]
                label = COMBO_MAP[combo_key]
                speak_text = create_speak_text(label)
                
                # Log timestamp
                tts_timestamps.append({
                    'prompt_idx': prompt_idx,
                    'combo_key': combo_key,
                    'tts_time': elapsed,
                    'tts_frame': frame_idx,
                    'prompt_text': speak_text
                })
                
                print(f"[{prompt_idx+1}/{len(combo_keys)}] t={elapsed:.1f}s: {speak_text}")
                
                cv2.putText(preview, f"TTS: {speak_text}", (5, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                
                # NON-BLOCKING TTS
                threading.Thread(target=speak_prompt, args=(speak_text,), daemon=True).start()
                prompt_idx += 1
            
            # Show status
            if prompt_idx < len(prompt_times):
                next_in = prompt_times[prompt_idx] - now
                cv2.putText(preview, f"Next: {next_in:.1f}s", (5, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            else:
                # Show remaining time in session
                session_end_time = start_time + (len(combo_keys) - 1) * args.interval + 5.0
                remaining = session_end_time - now
                cv2.putText(preview, f"Ending: {remaining:.1f}s", (5, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            cv2.imshow('Recording', preview)
            
            # Check for early exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("TERMINATION: Recording stopped by user")
                break
            
            # Normal exit: session duration reached
            session_duration = (len(combo_keys) - 1) * args.interval + 5.0
            if elapsed >= session_duration:
                print(f"TERMINATION: Recording completed after {elapsed:.1f} seconds (target: {session_duration:.1f}s)")
                break
        
        final_duration = time.time() - start_time
        
        print(f"CLEANUP: Releasing video writer after {frame_count} frames...")
        
        # Ensure all frames are written
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"FINAL: Video should contain {frame_count} frames over {final_duration:.2f} seconds")
        
        # Verify file exists and has size
        if os.path.exists(video_filename):
            file_size = os.path.getsize(video_filename)
            print(f"VIDEO FILE: {video_filename} created, size: {file_size} bytes")
        else:
            print(f"ERROR: Video file {video_filename} was not created!")
            return
        
        # Save CSV
        csv_data = []
        for ts in tts_timestamps:
            csv_data.append({
                'frame_index': ts['tts_frame'],
                'punch_key': ts['combo_key']
            })
        
        with open(csv_filename, 'w', newline='') as f:
            fieldnames = ['frame_index', 'punch_key']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print("=" * 60)
        print("RECORDING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Duration: {final_duration:.1f} seconds")
        print(f"Frames recorded: {frame_count}")
        print(f"TTS prompts: {len(tts_timestamps)}")
        print(f"Video: {video_filename}")
        print(f"Labels: {csv_filename}")
        print()
        print("Next step: Extract punch windows with:")
        print(f"python extract_punch_clips.py {base_name} --window-offset 1.0")
        
    except KeyboardInterrupt:
        print("\nRecording interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
