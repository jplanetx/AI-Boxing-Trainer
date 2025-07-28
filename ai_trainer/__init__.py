"""
AI Boxing Trainer - Modular AI-powered boxing training system.

This package provides real-time pose tracking, punch classification, 
and form analysis for boxing training applications.
"""

from .pose_tracker import PoseTracker
from .punch_classifier import PunchClassifier
from .form_analyzer import FormAnalyzer
from .heavy_bag_optimizer import HeavyBagOptimizer, TrainingMode
from .utils import calculate_angle, calculate_distance

__version__ = "1.0.0"
__all__ = [
    "PoseTracker",
    "PunchClassifier", 
    "FormAnalyzer",
    "HeavyBagOptimizer",
    "TrainingMode",
    "calculate_angle",
    "calculate_distance"
]
