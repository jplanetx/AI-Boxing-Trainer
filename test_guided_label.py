#!/usr/bin/env python3
"""
Test script for guided_label_session.py
Tests combo parsing and TTS functionality.
"""

import sys
import os

# Add the current directory to the path so we can import guided_label_session
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from guided_label_session import GuidedLabelSession, parse_combo_keys, COMBO_MAP

def test_combo_parsing():
    """Test combo parsing function."""
    print("=== Testing Combo Parsing ===")
    
    test_input = "1,2,1.1,2.1,4,1"
    parsed = parse_combo_keys(test_input)
    
    print(f"Input: {test_input}")
    print(f"Parsed: {parsed}")
    print(f"Length: {len(parsed)}")
    
    expected = [1, 2, 1.1, 2.1, 4, 1]
    assert parsed == expected, f"Expected {expected}, got {parsed}"
    print("✅ Combo parsing test passed!")
    
    return parsed

def test_tts_all_prompts():
    """Test TTS for all combo entries."""
    print("\n=== Testing TTS for All Prompts ===")
    
    # Use a short test combo
    test_combo = [1, 2, 1.1]
    
    print(f"Test combo: {test_combo}")
    print(f"Will speak {len(test_combo)} prompts with 0.5 second intervals")
    
    # Create session with short interval for testing
    session = GuidedLabelSession(output_file="test_ground_truth.csv", prompt_interval=0.5)
    
    # Run the session
    session.run_combo_session(test_combo, warmup_time=1.0)
    
    print("✅ TTS test completed!")

def test_combo_map():
    """Test that all keys in test combo exist in COMBO_MAP."""
    print("\n=== Testing COMBO_MAP ===")
    
    test_keys = [1, 2, 1.1, 2.1, 4, 1]
    
    for key in test_keys:
        assert key in COMBO_MAP, f"Key {key} not found in COMBO_MAP"
        print(f"  {key}: {COMBO_MAP[key]}")
    
    print("✅ COMBO_MAP test passed!")

if __name__ == "__main__":
    print("Testing guided_label_session.py functionality...\n")
    
    try:
        # Test 1: Combo parsing
        test_combo = test_combo_parsing()
        
        # Test 2: COMBO_MAP
        test_combo_map()
        
        # Test 3: TTS (interactive)
        print(f"\n=== Ready for TTS Test ===")
        user_input = input("Run TTS test with sample combo? [y/N]: ").strip().lower()
        
        if user_input in ['y', 'yes']:
            test_tts_all_prompts()
        else:
            print("Skipping TTS test")
        
        print(f"\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)
