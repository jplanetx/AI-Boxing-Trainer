#!/usr/bin/env python3
"""
Robust Heavy Bag Trainer - Enhanced error handling for pose transitions
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from ai_trainer.main import AIBoxingTrainer
    from ai_trainer.heavy_bag_optimizer import TrainingMode
    
    def launch_robust_trainer():
        """Launch trainer with enhanced error handling for pose transitions."""
        print("🥊 AI Boxing Trainer - Heavy Bag Enhanced (Robust Mode)")
        print("=" * 60)
        print()
        print("🛡️  ENHANCED STABILITY:")
        print("   ✅ Robust error handling during pose transitions")
        print("   ✅ Graceful degradation when landmarks are missing")
        print("   ✅ Continues running even during camera movement")
        print("   ✅ Auto-recovery when pose detection resumes")
        print()
        print("📋 TRANSITION TIPS:")
        print("   • Walk slowly when moving from computer to heavy bag")
        print("   • Keep yourself visible to camera during transition")
        print("   • App will show 'NO POSE' briefly - this is normal")
        print("   • Heavy bag mode activates automatically when positioned")
        print()
        print("🎯 OPTIMAL POSITIONING (Right-handed):")
        print("   • Stand with LEFT SIDE facing camera")
        print("   • 6-8 feet from camera")
        print("   • Heavy bag between you and camera is OK")
        print("   • Ensure your punching arm side is visible")
        print()
        input("Press ENTER to start the robust trainer...")
        
        try:
            # Launch the enhanced trainer
            trainer = AIBoxingTrainer(camera_id=0)
            trainer.run()
        except Exception as e:
            print(f"❌ Trainer error: {e}")
            print("💡 This may be due to camera issues or dependency problems")
            print("   Try:")
            print("   1. Ensure camera is not used by another application")
            print("   2. Check camera permissions")
            print("   3. Restart the application")
    
    if __name__ == "__main__":
        launch_robust_trainer()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you've installed the requirements:")
    print("   pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
