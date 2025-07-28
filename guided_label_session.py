#!/usr/bin/env python3
"""
Guided Labeling Session for AI Boxing Trainer
Generates audio prompts for punch combinations and logs ground truth data.
"""

import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime
from typing import List, Union

try:
    import pyttsx3
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Please install requirements with: pip install -r requirements.txt")
    sys.exit(1)

# Punch combination mapping
COMBO_MAP = {
    1: "jab_head",
    1.1: "jab_body", 
    2: "cross_head",
    2.1: "cross_body",
    3: "lead_hook_head",
    3.1: "lead_hook_body",
    4: "rear_hook_head", 
    4.1: "rear_hook_body",
    5: "lead_uppercut_head",
    5.1: "lead_uppercut_body",
    6: "rear_uppercut_head",
    6.1: "rear_uppercut_body"
}

class GuidedLabelSession:
    """Manages guided labeling sessions with TTS prompts and data logging."""
    
    def __init__(self, output_file: str = "ground_truth.csv", prompt_interval: float = 1.0):
        self.output_file = output_file
        self.prompt_interval = prompt_interval
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_index = 0
        
        # Initialize TTS engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)  # Adjust speech rate
            self.tts_engine.setProperty('volume', 0.8)  # Adjust volume
        except Exception as e:
            print(f"Warning: TTS initialization failed: {e}")
            self.tts_engine = None
        
        # Initialize CSV file with headers if it doesn't exist
        self._init_csv_file()
    
    def _init_csv_file(self):
        """Initialize CSV file with headers if it doesn't exist."""
        file_exists = os.path.exists(self.output_file)
        
        with open(self.output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow(['session', 'frame_index', 'punch_key'])
    
    def _speak_prompt(self, text: str):
        """Use TTS to speak the given text."""
        if self.tts_engine:
            try:
                print(f"Speaking: {text}")
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
                print(f"Prompt: {text}")
        else:
            print(f"PROMPT: {text}")
    
    def _log_punch(self, punch_key: Union[int, float]):
        """Log a punch to the CSV file."""
        with open(self.output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([self.session_id, self.frame_index, punch_key])
        
        print(f"Logged: Session {self.session_id}, Frame {self.frame_index}, Punch {punch_key}")
        self.frame_index += 1
    
    def _format_punch_prompt(self, punch_key: Union[int, float]) -> str:
        """Convert punch key to human-readable prompt."""
        if punch_key not in COMBO_MAP:
            return f"Unknown punch {punch_key}"
        
        punch_name = COMBO_MAP[punch_key]
        
        # Convert to natural speech
        parts = punch_name.split('_')
        if len(parts) >= 2:
            # Handle compound punch types
            if parts[0] == "lead" and len(parts) >= 3:
                punch_type = f"lead {parts[1]}"  # lead hook, lead uppercut
                target = parts[2]  # head, body
            elif parts[0] == "rear" and len(parts) >= 3:
                punch_type = f"rear {parts[1]}"  # rear hook, rear uppercut  
                target = parts[2]  # head, body
            else:
                punch_type = parts[0]  # jab, cross
                target = parts[1]  # head, body
            
            return f"{punch_type.title()} to the {target} - GO!"
        
        return f"{punch_name.replace('_', ' ').title()} - GO!"
    
    def run_combo_session(self, combo_keys: List[Union[int, float]], warmup_time: float = 3.0):
        """Run a guided labeling session for a specific combination."""
        print(f"\n=== Guided Labeling Session ===")
        print(f"Session ID: {self.session_id}")
        print(f"Combination: {combo_keys}")
        print(f"Total punches: {len(combo_keys)}")
        print(f"Interval: {self.prompt_interval} seconds")
        print(f"Output file: {self.output_file}")
        
        # Warmup countdown
        print(f"\nWarmup period - Get ready!")
        for i in range(int(warmup_time), 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
        
        print("\n=== SESSION STARTED ===")
        
        try:
            # Execute the combination
            for i, punch_key in enumerate(combo_keys):
                # Generate and speak prompt
                prompt = self._format_punch_prompt(punch_key)
                print(f"\nPunch {i+1}/{len(combo_keys)}: {prompt}")
                
                # Speak the prompt (includes debug print and runAndWait)
                self._speak_prompt(prompt)
                
                # Log the punch
                self._log_punch(punch_key)
                
                # Wait for next prompt (except for last punch)
                if i < len(combo_keys) - 1:
                    print(f"Waiting {self.prompt_interval} seconds...")
                    time.sleep(self.prompt_interval)
            
            print(f"\n=== SESSION COMPLETED ===")
            print(f"Logged {len(combo_keys)} punches to {self.output_file}")
            
        except KeyboardInterrupt:
            print(f"\n=== SESSION INTERRUPTED ===")
            print(f"Logged {self.frame_index} punches before interruption")
    
    def run_interactive_session(self):
        """Run an interactive session where user can input punch keys manually."""
        print(f"\n=== Interactive Labeling Session ===")
        print(f"Session ID: {self.session_id}")
        print("Available punch keys:")
        for key, name in COMBO_MAP.items():
            print(f"  {key}: {name}")
        print("\nEnter punch keys (space-separated) or 'quit' to exit:")
        
        try:
            while True:
                user_input = input("\nPunch keys: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                try:
                    # Parse input
                    punch_keys = []
                    for key_str in user_input.split():
                        try:
                            key = float(key_str) if '.' in key_str else int(key_str)
                            if key in COMBO_MAP:
                                punch_keys.append(key)
                            else:
                                print(f"Warning: Invalid punch key {key}")
                        except ValueError:
                            print(f"Warning: Could not parse '{key_str}'")
                    
                    if punch_keys:
                        self.run_combo_session(punch_keys, warmup_time=2.0)
                    
                except Exception as e:
                    print(f"Error processing input: {e}")
        
        except KeyboardInterrupt:
            pass
        
        print(f"\n=== INTERACTIVE SESSION ENDED ===")


def parse_combo_keys(combo_input: str) -> List[Union[int, float]]:
    """Parse combo keys from string input."""
    keys = []
    for key_str in combo_input.split(','):
        key_str = key_str.strip()
        try:
            key = float(key_str) if '.' in key_str else int(key_str)
            if key in COMBO_MAP:
                keys.append(key)
            else:
                print(f"Warning: Invalid punch key {key}")
        except ValueError:
            print(f"Warning: Could not parse '{key_str}'")
    return keys


def main():
    parser = argparse.ArgumentParser(description="Guided Labeling Session for AI Boxing Trainer")
    parser.add_argument('--combo', type=str, help='Comma-separated punch keys (e.g., "1,2,1.1,2.1,4,1")')
    parser.add_argument('--json-file', type=str, help='JSON file containing punch key array')
    parser.add_argument('--output', type=str, default='ground_truth.csv', help='Output CSV file')
    parser.add_argument('--interval', type=float, default=1.0, help='Interval between prompts (seconds)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Create session
    session = GuidedLabelSession(output_file=args.output, prompt_interval=args.interval)
    
    # Determine input method
    combo_keys = []
    
    if args.interactive:
        session.run_interactive_session()
        return
    
    if args.json_file:
        try:
            with open(args.json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    combo_keys = [float(k) if isinstance(k, float) else int(k) for k in data if k in COMBO_MAP]
                else:
                    print("Error: JSON file must contain an array of punch keys")
                    return
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return
    
    elif args.combo:
        combo_keys = parse_combo_keys(args.combo)
    
    else:
        print("No input provided. Use --combo, --json-file, or --interactive")
        print("\nExample usage:")
        print("  python guided_label_session.py --combo '1,2,1.1,2.1,4,1'")
        print("  python guided_label_session.py --json-file combo.json")
        print("  python guided_label_session.py --interactive")
        return
    
    if combo_keys:
        session.run_combo_session(combo_keys)
    else:
        print("No valid punch keys found")


if __name__ == "__main__":
    main()
