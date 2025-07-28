#!/usr/bin/env python3
"""
AI Boxing Trainer Launcher Script
Launches the modular 3D-enhanced boxing trainer application.
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from ai_trainer.main import main
    
    if __name__ == "__main__":
        print("🥊 Starting AI Boxing Trainer (3D Enhanced)")
        print("=" * 50)
        main()
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("💡 Make sure you've installed the requirements:")
    print("   pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Error starting trainer: {e}")
