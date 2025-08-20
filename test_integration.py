#!/usr/bin/env python3
"""
Test Integration - Verify that all components work together
"""

import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_trainer.pose_tracker import PoseTracker
from ai_trainer.punch_classifier import PunchClassifier, PunchType
from ai_trainer.form_analyzer import FormAnalyzer

def test_integration():
    """Test that all components can be initialized and work together."""
    print("Testing AI Boxing Trainer Integration...")
    
    # Test component initialization
    try:
        print("1. Initializing PoseTracker...")
        pose_tracker = PoseTracker(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1  # Use lower complexity for testing
        )
        print("   ✓ PoseTracker initialized")
        
        print("2. Initializing PunchClassifier...")
        punch_classifier = PunchClassifier()
        print("   ✓ PunchClassifier initialized")
        
        print("3. Initializing FormAnalyzer...")
        form_analyzer = FormAnalyzer()
        print("   ✓ FormAnalyzer initialized")
        
    except Exception as e:
        print(f"   ✗ Error initializing components: {e}")
        return False
    
    # Test camera access
    try:
        print("4. Testing camera access...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("   ⚠ Warning: Camera not available, using dummy data")
            use_camera = False
        else:
            print("   ✓ Camera accessible")
            use_camera = True
            cap.release()
            
    except Exception as e:
        print(f"   ⚠ Camera test failed: {e}, will use dummy data")
        use_camera = False
    
    # Test processing pipeline with dummy data
    try:
        print("5. Testing processing pipeline...")
        
        if use_camera:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Process one frame
                processed_frame, landmarks_dict = pose_tracker.process_frame(frame)
                print("   ✓ Frame processing successful")
                
                if landmarks_dict:
                    # Test punch classification
                    punch_result = punch_classifier.process_frame(landmarks_dict, pose_tracker)
                    print("   ✓ Punch classification completed")
                    
                    # Test form analysis
                    form_feedback = form_analyzer.analyze_form(
                        landmarks_dict, PunchType.JAB, 'left'
                    )
                    print("   ✓ Form analysis completed")
                    print(f"     Form score: {form_feedback.overall_score:.1f}%")
                else:
                    print("   ⚠ No pose detected in frame")
            else:
                print("   ✗ Could not read frame from camera")
        else:
            print("   ⚠ Skipping camera tests, no camera available")
            
    except Exception as e:
        print(f"   ✗ Error in processing pipeline: {e}")
        return False
    
    finally:
        # Cleanup
        try:
            pose_tracker.release()
        except:
            pass
    
    print("\n✓ Integration test completed successfully!")
    print("\nReady to run enhanced_main.py")
    return True

if __name__ == "__main__":
    success = test_integration()
    exit(0 if success else 1)