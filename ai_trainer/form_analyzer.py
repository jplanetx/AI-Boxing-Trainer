"""
Form Analysis System - Provides real-time technique feedback for boxing punches.
Implements biomechanical analysis based on ideal angles and form principles.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .utils import calculate_angle
from .punch_classifier import PunchType


@dataclass
class FormFeedback:
    """Structure for form analysis feedback."""
    overall_score: float
    feedback_messages: List[str]
    angle_deviations: Dict[str, float]
    technique_grade: str


class FormAnalyzer:
    """
    Real-time form analysis and technique feedback system.
    Analyzes punch biomechanics and provides corrective guidance.
    """
    
    def __init__(self):
        """Initialize the form analyzer with ideal biomechanical parameters."""
        
        # Ideal angles for each punch type (in degrees)
        # Based on boxing biomechanics and coaching principles
        self.ideal_angles = {
            PunchType.JAB: {
                'shoulder_angle': 45,    # Shoulder extension from body
                'hip_rotation': 15,      # Minimal hip engagement
                'elbow_extension': 175,  # Nearly full extension
                'wrist_alignment': 0     # Straight wrist
            },
            PunchType.CROSS: {
                'shoulder_angle': 90,    # Full shoulder rotation
                'hip_rotation': 45,      # Significant hip drive
                'elbow_extension': 178,  # Full extension
                'wrist_alignment': 0     # Straight wrist
            },
            PunchType.HOOK: {
                'shoulder_angle': 90,    # Horizontal shoulder movement
                'hip_rotation': 30,      # Moderate hip engagement
                'elbow_angle': 90,       # 90-degree elbow angle
                'wrist_alignment': 0     # Straight wrist
            },
            PunchType.UPPERCUT: {
                'shoulder_angle': 75,    # Upward shoulder drive
                'hip_rotation': 25,      # Moderate hip engagement
                'elbow_angle': 45,       # Bent elbow for upward trajectory
                'wrist_alignment': 0     # Straight wrist
            }
        }
        
        # Tolerance ranges for scoring (in degrees)
        self.tolerance_ranges = {
            'excellent': 5,    # Within 5 degrees
            'good': 10,        # Within 10 degrees
            'fair': 20,        # Within 20 degrees
            'poor': 30         # Within 30 degrees
        }
        
        # Weight factors for different aspects of form
        self.scoring_weights = {
            'shoulder_angle': 0.3,
            'hip_rotation': 0.2,
            'elbow_extension': 0.3,
            'elbow_angle': 0.3,     # Alternative to extension for hooks/uppercuts
            'wrist_alignment': 0.2
        }
    
    def analyze_form(self, landmarks_dict: Dict, punch_type: PunchType, arm: str) -> FormFeedback:
        """
        Analyze punch form and provide comprehensive feedback.
        
        Args:
            landmarks_dict: 3D landmarks from pose tracker
            punch_type: Type of punch being analyzed
            arm: 'left' or 'right'
            
        Returns:
            FormFeedback object with score and suggestions
        """
        if not landmarks_dict or punch_type == PunchType.UNKNOWN:
            return FormFeedback(
                overall_score=0.0,
                feedback_messages=["Unable to analyze form - pose not detected"],
                angle_deviations={},
                technique_grade="N/A"
            )
        
        # Calculate current angles
        current_angles = self._calculate_current_angles(landmarks_dict, arm, punch_type)
        
        # Get ideal angles for this punch type
        ideal_angles = self.ideal_angles.get(punch_type, {})
        
        # Calculate deviations and score
        angle_deviations = {}
        total_score = 100.0
        feedback_messages = []
        
        for angle_name, ideal_value in ideal_angles.items():
            if angle_name in current_angles:
                current_value = current_angles[angle_name]
                deviation = abs(current_value - ideal_value)
                angle_deviations[angle_name] = deviation
                
                # Calculate score reduction based on deviation
                weight = self.scoring_weights.get(angle_name, 0.25)
                score_reduction = self._calculate_score_reduction(deviation)
                total_score -= score_reduction * weight * 100
                
                # Generate feedback message if significant deviation
                if deviation > self.tolerance_ranges['good']:
                    feedback_msg = self._generate_feedback_message(
                        angle_name, deviation, ideal_value, current_value, punch_type
                    )
                    if feedback_msg:
                        feedback_messages.append(feedback_msg)
        
        # Ensure score doesn't go below 0
        total_score = max(0.0, total_score)
        
        # Determine technique grade
        technique_grade = self._calculate_technique_grade(total_score)
        
        # Add general technique tips if score is low
        if total_score < 70:
            feedback_messages.extend(self._get_general_tips(punch_type))
        
        return FormFeedback(
            overall_score=total_score,
            feedback_messages=feedback_messages[:3],  # Limit to top 3 suggestions
            angle_deviations=angle_deviations,
            technique_grade=technique_grade
        )
    
    def _calculate_current_angles(self, landmarks_dict: Dict, arm: str, punch_type: PunchType) -> Dict[str, float]:
        """
        Calculate current angles from landmarks.
        
        Args:
            landmarks_dict: 3D landmarks
            arm: 'left' or 'right'
            punch_type: Type of punch for context
            
        Returns:
            Dictionary of calculated angles
        """
        try:
            # Get landmark coordinates with safety checks
            shoulder = landmarks_dict.get(f'{arm}_shoulder')
            elbow = landmarks_dict.get(f'{arm}_elbow')
            wrist = landmarks_dict.get(f'{arm}_wrist')
            hip = landmarks_dict.get(f'{arm}_hip')
            
            # Check if essential landmarks are available
            if not all([shoulder, elbow, wrist]):
                return {}
            
            # Calculate angles
            angles = {}
            
            # Shoulder angle (hip-shoulder-elbow) - only if hip available
            if hip and hip.get('visibility', 0) > 0.5:
                angles['shoulder_angle'] = calculate_angle(
                    [hip['x'], hip['y']],
                    [shoulder['x'], shoulder['y']],
                    [elbow['x'], elbow['y']]
                )
            
            # Elbow extension angle (shoulder-elbow-wrist)
            elbow_angle = calculate_angle(
                [shoulder['x'], shoulder['y']],
                [elbow['x'], elbow['y']],
                [wrist['x'], wrist['y']]
            )
            
            # Store appropriate angle based on punch type
            if punch_type in [PunchType.HOOK, PunchType.UPPERCUT]:
                angles['elbow_angle'] = elbow_angle
            else:
                angles['elbow_extension'] = elbow_angle
            
            # Hip rotation (approximate from shoulder positions)
            if ('left_hip' in landmarks_dict and 'right_hip' in landmarks_dict and
                'left_shoulder' in landmarks_dict and 'right_shoulder' in landmarks_dict):
                
                left_hip = landmarks_dict['left_hip']
                right_hip = landmarks_dict['right_hip']
                left_shoulder = landmarks_dict['left_shoulder']
                right_shoulder = landmarks_dict['right_shoulder']
                
                # Check visibility of all required landmarks
                if all(lm.get('visibility', 0) > 0.5 for lm in [left_hip, right_hip, left_shoulder, right_shoulder]):
                    # Calculate hip-shoulder alignment angle
                    hip_vector = [right_hip['x'] - left_hip['x'], right_hip['y'] - left_hip['y']]
                    shoulder_vector = [right_shoulder['x'] - left_shoulder['x'], right_shoulder['y'] - left_shoulder['y']]
                    
                    hip_rotation = abs(np.arctan2(shoulder_vector[1], shoulder_vector[0]) - 
                                     np.arctan2(hip_vector[1], hip_vector[0])) * 180 / np.pi
                    
                    angles['hip_rotation'] = min(hip_rotation, 180 - hip_rotation)
            
            # Wrist alignment (simplified - assume straight for now)
            angles['wrist_alignment'] = 0  # Placeholder - would need hand landmarks
            
            return angles
            
        except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
            return {}
    
    def _calculate_score_reduction(self, deviation: float) -> float:
        """
        Calculate score reduction based on angle deviation.
        
        Args:
            deviation: Angle deviation in degrees
            
        Returns:
            Score reduction factor (0-1)
        """
        if deviation <= self.tolerance_ranges['excellent']:
            return 0.0
        elif deviation <= self.tolerance_ranges['good']:
            return 0.1
        elif deviation <= self.tolerance_ranges['fair']:
            return 0.3
        elif deviation <= self.tolerance_ranges['poor']:
            return 0.6
        else:
            return 1.0
    
    def _generate_feedback_message(self, angle_name: str, deviation: float, 
                                 ideal_value: float, current_value: float, 
                                 punch_type: PunchType) -> str:
        """
        Generate specific feedback message for angle deviation.
        
        Args:
            angle_name: Name of the angle
            deviation: Deviation amount
            ideal_value: Ideal angle value
            current_value: Current angle value
            punch_type: Type of punch
            
        Returns:
            Feedback message string
        """
        feedback_map = {
            'shoulder_angle': {
                'too_low': "Rotate your shoulder more - drive through the punch",
                'too_high': "Don't over-rotate your shoulder - stay balanced"
            },
            'hip_rotation': {
                'too_low': "Engage your hips more - power comes from the core",
                'too_high': "Don't over-rotate your hips - maintain balance"
            },
            'elbow_extension': {
                'too_low': "Extend your arm more fully - reach for the target",
                'too_high': "Don't hyperextend - slight bend protects your elbow"
            },
            'elbow_angle': {
                'too_low': "Keep your elbow higher for better hook/uppercut form",
                'too_high': "Lower your elbow - tighter angle for more power"
            }
        }
        
        if angle_name not in feedback_map:
            return ""
        
        # Determine if current value is too high or too low
        direction = 'too_high' if current_value > ideal_value else 'too_low'
        
        return feedback_map[angle_name].get(direction, "")
    
    def _calculate_technique_grade(self, score: float) -> str:
        """
        Convert numerical score to letter grade.
        
        Args:
            score: Overall technique score (0-100)
            
        Returns:
            Letter grade string
        """
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def _get_general_tips(self, punch_type: PunchType) -> List[str]:
        """
        Get general technique tips for each punch type.
        
        Args:
            punch_type: Type of punch
            
        Returns:
            List of general tips
        """
        tips = {
            PunchType.JAB: [
                "Keep your jab quick and snappy",
                "Return to guard position immediately"
            ],
            PunchType.CROSS: [
                "Rotate your hip and shoulder together",
                "Keep your chin down and protected"
            ],
            PunchType.HOOK: [
                "Turn your hip and pivot on your front foot",
                "Keep your elbow parallel to the ground"
            ],
            PunchType.UPPERCUT: [
                "Drive upward with your legs and core",
                "Keep your other hand up for protection"
            ]
        }
        
        return tips.get(punch_type, ["Focus on proper form and technique"])
    
    def get_technique_summary(self, punch_counts: Dict[str, int]) -> Dict[str, str]:
        """
        Get a summary of overall technique performance.
        
        Args:
            punch_counts: Dictionary of punch counts by type
            
        Returns:
            Summary of technique recommendations
        """
        total_punches = sum(punch_counts.values())
        
        if total_punches == 0:
            return {"summary": "No punches detected yet"}
        
        # Analyze punch distribution
        summary = {}
        
        most_common = max(punch_counts, key=punch_counts.get)
        summary["most_used"] = f"Most used: {most_common} ({punch_counts[most_common]} punches)"
        
        # Suggest variety if too focused on one punch
        if punch_counts[most_common] > total_punches * 0.7:
            summary["variety_tip"] = "Try mixing in more punch combinations"
        
        return summary
