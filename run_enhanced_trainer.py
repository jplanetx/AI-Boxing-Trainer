#!/usr/bin/env python3
"""
Enhanced AI Boxing Trainer Launcher
Quick start script for the enhanced boxing trainer application.
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_main import main
    
    if __name__ == "__main__":
        print("=== AI Boxing Trainer Enhanced ===")
        print("Starting enhanced boxing trainer...")
        print("Controls:")
        print("  Q/ESC: Quit")
        print("  L: Toggle landmarks")
        print("  F: Toggle form feedback")
        print("  R: Reset session")
        print("  H: Show help")
        print("===================================\n")
        
        exit_code = main()
        sys.exit(exit_code)
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1)