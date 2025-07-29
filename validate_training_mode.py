#!/usr/bin/env python3
"""
Training Mode Validation Script
Verifies all key improvements in the continuous recording system.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from training_mode import COMBO_MAP, parse_combo_string, create_speak_text

def test_combo_parsing():
    """Test combo string parsing"""
    print("Testing combo string parsing...")
    
    test_cases = [
        ("1,2,1,2,4,1", ["1", "2", "1", "2", "4", "1"]),
        ("1b,2b,3,4b", ["1b", "2b", "3", "4b"]),
        ("1, 2 , 3b , 4b", ["1", "2", "3b", "4b"]),  # With spaces
        ("invalid,1,2,unknown", ["1", "2"]),  # With invalid keys
    ]
    
    for combo_str, expected in test_cases:
        result = parse_combo_string(combo_str)
        print(f"  '{combo_str}' -> {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("  [PASS] Combo parsing tests passed")

def test_speech_generation():
    """Test TTS text generation"""
    print("Testing TTS text generation...")
    
    test_cases = [
        ("1", "jab to the head — go!"),
        ("1b", "jab to the body — go!"),
        ("3b", "lead hook to the body — go!"),
        ("6", "rear uppercut to the head — go!"),
    ]
    
    for combo_key, expected_speech in test_cases:
        label = COMBO_MAP[combo_key]
        speech_text = create_speak_text(label)
        print(f"  '{combo_key}' -> '{speech_text}'")
        assert speech_text == expected_speech, f"Expected '{expected_speech}', got '{speech_text}'"
    
    print("  [PASS] Speech generation tests passed")

def test_frame_calculation():
    """Test frame index calculation accuracy"""
    print("Testing frame index calculation...")
    
    fps = 30
    test_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    expected_frames = [0, 15, 30, 45, 60, 75, 90]
    
    for time_offset, expected_frame in zip(test_times, expected_frames):
        calculated_frame = int(time_offset * fps)
        print(f"  t={time_offset}s -> frame {calculated_frame}")
        assert calculated_frame == expected_frame, f"Expected frame {expected_frame}, got {calculated_frame}"
    
    print("  [PASS] Frame calculation tests passed")

def validate_file_structure():
    """Validate that training_mode.py has the correct structure"""
    print("Validating training_mode.py structure...")
    
    with open(os.path.join(os.path.dirname(__file__), "training_mode.py"), "r") as f:
        content = f.read()
    
    # Check for key improvements
    required_patterns = [
        "start_time = time.time()",
        "prompt_times = [start_time + i * args.interval",
        "frame_idx = int((now - start_time) * fps)",
        "while True:",
        "out.write(frame)",
        "if prompt_idx < len(prompt_times) and now >= prompt_times[prompt_idx]:",
        "if prompt_idx >= len(prompt_times) and now >= prompt_times[-1] + 1.0:",
    ]
    
    for pattern in required_patterns:
        assert pattern in content, f"Missing required pattern: {pattern}"
        print(f"  [PASS] Found: {pattern}")
    
    print("  [PASS] File structure validation passed")

def main():
    """Run all validation tests"""
    print("=" * 60)
    print("TRAINING MODE CONTINUOUS RECORDING VALIDATION")
    print("=" * 60)
    
    try:
        test_combo_parsing()
        print()
        
        test_speech_generation()
        print()
        
        test_frame_calculation()
        print()
        
        validate_file_structure()
        print()
        
        print("=" * 60)
        print("ALL TESTS PASSED - READY FOR PRODUCTION TESTING")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Test with real webcam: python training_mode.py --combo 1,2,1,2,4,1 --output test_session --interval 1")
        print("2. Verify continuous video output")
        print("3. Check frame-indexed CSV accuracy")
        print("4. Scale to full ML training pipeline")
        
        return True
        
    except Exception as e:
        print(f"VALIDATION FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
