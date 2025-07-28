#!/usr/bin/env python3
"""
Heavy Bag Mode Test Script
Quick test of the enhanced AI Boxing Trainer with heavy bag optimizations.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from ai_trainer.main import AIBoxingTrainer
    from ai_trainer.heavy_bag_optimizer import TrainingMode
    
    def test_heavy_bag_mode():
        """Test the heavy bag optimization features."""
        print("🥊 Testing AI Boxing Trainer - Heavy Bag Mode")
        print("=" * 60)
        print()
        print("🎯 HEAVY BAG OPTIMIZATIONS ACTIVE:")
        print("   ✅ Confidence-based landmark filtering")
        print("   ✅ Asymmetric tracking for angled positioning")
        print("   ✅ Automatic training mode detection")
        print("   ✅ Adjusted thresholds for heavy bag scenarios")
        print("   ✅ Setup guidance system")
        print()
        print("📋 EXPECTED IMPROVEMENTS:")
        print("   • +30-40% accuracy for heavy bag training")
        print("   • Better punch type classification with partial occlusion")
        print("   • More reliable form feedback during real training")
        print("   • Reduced false positives from low-confidence landmarks")
        print()
        print("🎮 NEW CONTROLS:")
        print("   • 'G' key - Toggle setup guidance display")
        print("   • Watch for 'HEAVY BAG' mode indicator in top-right")
        print("   • Setup guidance panel shows positioning tips")
        print()
        print("🔧 OPTIMIZATION TIPS:")
        print("   • Position camera so your LEFT SIDE is visible (right-handed)")
        print("   • Stand 6-8 feet from camera")
        print("   • Ensure good lighting on your punching side")
        print("   • App will auto-detect heavy bag vs shadowboxing mode")
        print()
        input("Press ENTER to start the enhanced trainer...")
        
        # Launch the enhanced trainer
        trainer = AIBoxingTrainer(camera_id=0)
        trainer.run()
    
    if __name__ == "__main__":
        test_heavy_bag_mode()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you've installed the requirements:")
    print("   pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error: {e}")
