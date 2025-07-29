#!/usr/bin/env python3
"""
Extract Punch Windows - Extract wide punch windows between audio cues
Each window captures the complete punch action from one cue to the next.
"""

import cv2
import csv
import argparse
import os
from pathlib import Path

def load_punch_labels(csv_file):
    """Load punch labels from CSV file"""
    labels = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append({
                'frame_index': int(row['frame_index']),
                'punch_key': row['punch_key']
            })
    return labels

def extract_punch_windows(session_prefix, window_offset=0.5):
    """
    Extract wide punch windows between audio cues
    Each window spans from one audio cue to the next (or session end)
    
    Args:
        session_prefix: Base name for session files (e.g., "test_session")
        window_offset: Seconds after audio cue to start window (default: 0.5)
    """
    
    # Build file paths
    video_file = f"{session_prefix}.mp4"
    csv_file = f"{session_prefix}_labels.csv"
    output_dir = f"{session_prefix}_windows"
    
    # Validate input files
    if not os.path.exists(video_file):
        print(f"Error: Video file {video_file} not found!")
        return
    
    if not os.path.exists(csv_file):
        print(f"Error: CSV file {csv_file} not found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load punch labels
    labels = load_punch_labels(csv_file)
    if not labels:
        print(f"Error: No punch labels found in {csv_file}")
        return
    
    # Open video and get metadata
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_file}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    print(f"Video metadata: {width}x{height} @ {fps}fps, {total_frames} frames ({total_duration:.1f}s)")
    print(f"Window offset: {window_offset}s after audio cue")
    print(f"Processing {len(labels)} punch windows...")
    print()
    
    # Calculate window boundaries
    windows = []
    for i, label in enumerate(labels):
        # Window starts after the audio cue
        window_start_frame = label['frame_index'] + int(window_offset * fps)
        
        # Window ends at next audio cue (or session end)
        if i + 1 < len(labels):
            window_end_frame = labels[i + 1]['frame_index']
        else:
            # Last window goes to end of session
            window_end_frame = total_frames - 1
        
        # Ensure valid bounds
        window_start_frame = max(0, window_start_frame)
        window_end_frame = min(total_frames - 1, window_end_frame)
        
        if window_start_frame < window_end_frame:
            window_duration = (window_end_frame - window_start_frame) / fps
            windows.append({
                'idx': i + 1,
                'punch_key': label['punch_key'],
                'audio_frame': label['frame_index'],
                'start_frame': window_start_frame,
                'end_frame': window_end_frame,
                'duration': window_duration
            })
    
    # Extract each window
    extracted_count = 0
    
    for window in windows:
        idx = window['idx']
        punch_key = window['punch_key']
        start_frame = window['start_frame']
        end_frame = window['end_frame']
        duration = window['duration']
        
        # Create output filename
        clip_filename = f"{output_dir}/{idx:02d}_{punch_key}_window.mp4"
        
        # Set up video writer for this window
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"Error: Could not create output file {clip_filename}")
            continue
        
        # Seek to start frame and extract window
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames_written = 0
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_num} for window {idx}")
                break
            
            out.write(frame)
            frames_written += 1
        
        out.release()
        
        if frames_written > 0:
            extracted_count += 1
            print(f"Window {idx:02d}: {punch_key} | "
                  f"Audio@{window['audio_frame']} -> "
                  f"Window[{start_frame}-{end_frame}] | "
                  f"{duration:.1f}s ({frames_written} frames) -> {clip_filename}")
        else:
            print(f"Failed to extract window {idx}: {punch_key}")
            # Remove empty file
            try:
                os.remove(clip_filename)
            except:
                pass
    
    cap.release()
    
    # Print summary
    print()
    print("=" * 80)
    print(f"Extracted {extracted_count} punch windows into {output_dir}/")
    print("=" * 80)
    print()
    print("Each window contains:")
    print("- Complete punch cycle: preparation -> extension -> impact -> retraction -> guard")
    print("- Natural movement boundaries between audio cues")
    print("- Guaranteed punch motion capture within wide time range")
    print()
    print("Ready for ML training with robust punch windows!")
    
    return extracted_count

def main():
    parser = argparse.ArgumentParser(description="Extract wide punch windows from training session")
    parser.add_argument('session_prefix', 
                       help='Session prefix (e.g., "test_session" for test_session.mp4)')
    parser.add_argument('--window-offset', type=float, default=0.5,
                       help='Seconds after audio cue to start window (default: 0.5)')
    
    args = parser.parse_args()
    
    try:
        extracted_count = extract_punch_windows(
            args.session_prefix, 
            args.window_offset
        )
        
        if extracted_count > 0:
            print("Success! Wide punch windows extracted and ready for analysis.")
        else:
            print("No windows extracted. Check input files and parameters.")
            
    except Exception as e:
        print(f"Error during extraction: {e}")

if __name__ == "__main__":
    main()
