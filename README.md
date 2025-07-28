# AI Boxing Trainer - 3D Enhanced v1.0

An advanced AI-powered boxing trainer that uses 3D pose estimation to provide real-time punch classification, form analysis, and technique feedback. This modular system transforms your webcam into a sophisticated boxing coach.

## 🥊 Features

### Core Capabilities
- **3D Pose Tracking**: Advanced BlazePose GHUM model for accurate depth-aware pose estimation
- **Punch Classification**: Distinguishes between jab, cross, hook, and uppercut punches
- **Form Analysis**: Real-time biomechanical analysis with technique scoring
- **Performance Metrics**: Speed scoring, punch counting, and progression tracking

### Technical Improvements
- **Modular Architecture**: Clean separation of pose tracking, classification, and analysis
- **Trajectory Analysis**: 3D movement vectors for accurate punch type detection
- **Stance Detection**: Automatic orthodox/southpaw stance recognition
- **Smoothing Algorithms**: Reduced jitter and improved stability

## 🏗️ Project Structure

```
AI_Boxing_Trainer/
├── ai_trainer/                 # Main package
│   ├── __init__.py            # Package initialization
│   ├── pose_tracker.py        # 3D pose tracking engine
│   ├── punch_classifier.py    # Punch type classification
│   ├── form_analyzer.py       # Technique analysis
│   ├── utils.py              # Shared utilities
│   └── main.py               # Application entry point
├── main.py                   # Legacy single-file version
├── run_trainer.py           # New launcher script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### Option 1: Run New Modular Version (Recommended)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the enhanced trainer
python run_trainer.py
```

### Option 2: Run Legacy Version
```bash
# Run the original single-file version
python main.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `R` | Reset all statistics |
| `F` | Toggle form feedback display |
| `S` | Toggle detailed statistics |

## 📊 What You'll See

### Main Interface
- **Left/Right Arm Stats**: Punch counts, speed scores, and punch types
- **Form Analysis**: Real-time technique grades (A-F) and feedback
- **Performance Metrics**: FPS counter and pose detection status
- **3D Pose Visualization**: Enhanced landmark display with depth

### Form Feedback System
- **Technique Scoring**: 0-100% score based on biomechanical analysis
- **Real-time Tips**: Specific suggestions for improving form
- **Grade System**: A-F grades for instant performance feedback

## 🔧 Technical Details

### Punch Classification Algorithm
The system uses 3D trajectory analysis to distinguish punch types:

1. **Uppercut**: High vertical movement component (>70%)
2. **Hook**: Horizontal lateral movement with high width-to-depth ratio
3. **Jab**: Lead hand straight punch (stance-dependent)
4. **Cross**: Rear hand straight punch with hip rotation

### Form Analysis Metrics
- **Shoulder Angle**: Hip-shoulder-elbow alignment
- **Hip Rotation**: Core engagement measurement
- **Elbow Extension**: Full extension for straights, proper angles for hooks/uppercuts
- **Wrist Alignment**: Proper striking surface orientation

### 3D Pose Tracking
- **Model**: MediaPipe BlazePose GHUM (Heavy complexity)
- **Landmarks**: 33 3D points with visibility confidence
- **Smoothing**: 5-frame temporal smoothing for stability
- **Performance**: 30 FPS target on modern hardware

## 🎯 Training Tips

### Getting Started
1. **Position yourself**: Stand 6-8 feet from camera, full body visible
2. **Lighting**: Ensure good lighting, avoid backlighting
3. **Background**: Plain background works best for pose detection
4. **Stance**: Start in orthodox (left foot forward) or southpaw stance

### Improving Accuracy
- Keep movements controlled and deliberate
- Return to guard position between punches
- Focus on proper form over speed initially
- Use the form feedback to refine technique

## 🛠️ Development

### Module Overview

**`pose_tracker.py`**
- 3D landmark extraction and smoothing
- Pose validation and confidence checking
- Real-time pose visualization

**`punch_classifier.py`**
- Trajectory buffer management
- Punch type classification logic
- State machine for punch detection
- Performance scoring algorithms

**`form_analyzer.py`**
- Biomechanical angle calculations
- Form scoring based on ideal technique
- Real-time feedback generation
- Technique grading system

**`utils.py`**
- Mathematical utilities (angles, distances)
- Position smoothing algorithms
- Coordinate system conversions
- Helper functions for pose analysis

### Extending the System

To add new features:

1. **New Punch Types**: Extend `PunchType` enum and add classification logic
2. **Additional Metrics**: Add new measurements to `FormAnalyzer`
3. **UI Enhancements**: Modify drawing functions in `main.py`
4. **Performance Optimizations**: Adjust buffer sizes and thresholds

## 📈 Performance Requirements

### Hardware
- **CPU**: Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)
- **RAM**: 8GB minimum, 16GB recommended
- **Camera**: 720p webcam minimum, 1080p preferred
- **GPU**: Optional, but improves performance

### Software
- **Python**: 3.7 or later
- **OpenCV**: 4.8.0 or later
- **MediaPipe**: 0.10.0 or later
- **NumPy**: 1.24.0 or later

## 🐛 Troubleshooting

### Common Issues

**"Could not open camera"**
- Check if camera is connected and not used by another app
- Try different camera_id values (0, 1, 2...)
- Restart the application

**Low FPS or lag**
- Reduce model_complexity in `PoseTracker` (2 → 1 → 0)
- Lower camera resolution
- Close other applications using CPU/camera

**Inaccurate punch detection**
- Ensure good lighting conditions
- Check camera positioning (full body visible)
- Calibrate detection thresholds in `punch_classifier.py`

**Form analysis not working**
- Verify pose detection is active (green "POSE DETECTED" indicator)
- Check landmark visibility scores
- Ensure proper stance and camera angle

## 🚀 Future Enhancements

### Phase 2 Roadmap
- [ ] Combination detection (1-2, 1-2-3, etc.)
- [ ] Defensive movement tracking (slips, ducks, weaves)
- [ ] Footwork analysis and scoring
- [ ] Fatigue detection and pacing recommendations

### Phase 3 Roadmap
- [ ] Mobile app development (Flutter/React Native)
- [ ] Cloud sync and progress tracking
- [ ] Multiplayer training sessions
- [ ] AI coaching with personalized training plans

## 📄 License

This project is for educational and personal use. Commercial use requires additional licensing considerations for MediaPipe and other dependencies.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request with detailed description

---

**Ready to train? Run `python run_trainer.py` and start your AI-powered boxing journey! 🥊**
