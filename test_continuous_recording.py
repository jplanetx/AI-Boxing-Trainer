#!/usr/bin/env python3
"""
Test script to validate continuous recording logic without webcam dependency
"""

import time

def test_timing_logic():
    """Test the timing calculation logic"""
    print("Testing continuous recording timing logic...")
    
    # Simulate test parameters
    combo_keys = ["1", "2", "1", "2", "4", "1"]
    interval = 1.0  # 1 second intervals
    fps = 30
    
    # Calculate prompt times
    start_time = time.time()
    prompt_times = [start_time + i * interval for i in range(len(combo_keys))]
    prompt_idx = 0
    
    print(f"Start time: {start_time:.2f}")
    print(f"Prompt times: {[t - start_time for t in prompt_times]}")
    print(f"Total session duration: {prompt_times[-1] - start_time + 1.0:.1f}s")
    
    # Simulate frame recording loop
    frame_count = 0
    recorded_prompts = []
    
    while True:
        now = time.time()
        
        # Calculate frame index like the real code
        frame_idx = int((now - start_time) * fps)
        frame_count += 1
        
        # Check if it's time for next prompt
        if prompt_idx < len(prompt_times) and now >= prompt_times[prompt_idx]:
            combo_key = combo_keys[prompt_idx]
            actual_time = now - start_time
            expected_time = prompt_idx * interval
            timing_error = actual_time - expected_time
            
            recorded_prompts.append({
                'prompt_idx': prompt_idx,
                'combo_key': combo_key,
                'frame_idx': frame_idx,
                'actual_time': actual_time,
                'expected_time': expected_time,
                'timing_error': timing_error
            })
            
            print(f"Prompt {prompt_idx + 1}: {combo_key} at frame {frame_idx} "
                  f"(t={actual_time:.3f}s, error={timing_error:.3f}s)")
            
            prompt_idx += 1
        
        # Exit condition: all prompts + 1 second tail
        if prompt_idx >= len(prompt_times) and now >= prompt_times[-1] + 1.0:
            break
            
        # Simulate 30fps (~33ms per frame)
        time.sleep(0.033)
    
    total_duration = time.time() - start_time
    final_frame_count = int(total_duration * fps)
    
    print(f"\nTest Results:")
    print(f"Total duration: {total_duration:.2f}s")
    print(f"Final frame count: {final_frame_count}")
    print(f"Prompts recorded: {len(recorded_prompts)}/{len(combo_keys)}")
    print(f"Max timing error: {max([p['timing_error'] for p in recorded_prompts]):.3f}s")
    
    # Validate results
    assert len(recorded_prompts) == len(combo_keys), "Missing prompts!"
    assert all(abs(p['timing_error']) < 0.1 for p in recorded_prompts), "Timing too inaccurate!"
    
    print("✅ Continuous recording logic validated successfully!")
    return True

if __name__ == "__main__":
    test_timing_logic()
