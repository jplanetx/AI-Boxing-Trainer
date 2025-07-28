#!/usr/bin/env python3
"""
Simple TTS test for guided_label_session.py
"""

import time
import pyttsx3

def speak_text(text):
    """Speak text using TTS with fresh engine initialization."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.8)
        
        print(f"Speaking: '{text}'")
        engine.say(text)
        engine.runAndWait()
        
        engine.stop()
        del engine
        
        print(f"Speech completed: {text}")
        return True
        
    except Exception as e:
        print(f"TTS Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing TTS with multiple prompts...")
    
    test_prompts = ["jab head go", "cross head go", "rear hook head go"]
    
    for i, prompt in enumerate(test_prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        success = speak_text(prompt)
        if success:
            print("✓ Audio successful")
        else:
            print("✗ Audio failed")
        
        if i < len(test_prompts) - 1:
            print("Waiting 1 second...")
            time.sleep(1)
    
    print("\nTTS test complete!")
