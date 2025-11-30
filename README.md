# Individual Punch Recorder

**Clean, focused ML training data generator for boxing punch tracking.**

## 🎯 Purpose
Generate high-quality training data for ML punch tracking models by recording individual punch clips with perfect labeling.

## 🚀 Quick Start

### Record Individual Punches
```bash
# Record jabs
python individual_punch_recorder.py --punch-type jab --batch 10

# Record crosses  
python individual_punch_recorder.py --punch-type cross --batch 10

# Record hooks
python individual_punch_recorder.py --punch-type hook --batch 10

# Record uppercuts
python individual_punch_recorder.py --punch-type uppercut --batch 10
```

### Batch Collection Strategy
```bash
# Session 1: Jabs (50 clips)
python individual_punch_recorder.py -p jab -b 50 -o training_data

# Session 2: Crosses (50 clips) 
python individual_punch_recorder.py -p cross -b 50 -o training_data

# Session 3: Hooks (50 clips)
python individual_punch_recorder.py -p hook -b 50 -o training_data

# Session 4: Uppercuts (50 clips)
python individual_punch_recorder.py -p uppercut -b 50 -o training_data
```

## 📁 Output Structure
```
training_data/
├── jab_001.mp4
├── jab_002.mp4
├── cross_001.mp4
├── cross_002.mp4
├── hook_001.mp4
├── uppercut_001.mp4
└── ...
```

## 🎬 Recording Process
1. **Position**: Get in stance, ready to punch
2. **Press SPACE**: Starts 4-second recording
3. **Execute**: 1s prep → punch → 1s return to guard  
4. **Auto-save**: Perfect filename with punch type
5. **Repeat**: Continue for batch size

## ⚡ Features
- **Perfect Labeling**: Filename = punch type (no timing ambiguity)
- **60fps Recording**: Optimal motion capture for ML training
- **4-second Clips**: Complete punch motion with prep/recovery
- **Auto-increment**: No filename conflicts
- **Batch Mode**: Record multiple clips efficiently
- **Quality Control**: Consistent timing and format

## 🧠 ML Training Benefits
- ✅ **Zero timing uncertainty** (entire clip = one punch)
- ✅ **Complete motion capture** (prep → execution → recovery)
- ✅ **Consistent data format** (all clips identical duration/fps)
- ✅ **Scalable collection** (batch recording sessions)
- ✅ **Clean labels** (filename contains ground truth)

## 🎯 Target: 200+ Training Clips
- **50+ jabs** (left/right variants, different speeds)
- **50+ crosses** (power shots, speed variations)  
- **50+ hooks** (body/head targets, angles)
- **50+ uppercuts** (close range, different trajectories)

## 🚀 Next Steps
1. **Collect Training Data**: 200+ individual punch clips
2. **Train ML Model**: Use clean, labeled data
3. **Deploy Real-time Tracking**: Production system
4. **Launch Elevate AI**: $1,800+ monthly revenue target

## 🔧 Requirements
```bash
pip install opencv-python
```

## 💡 Usage Tips
- **Consistent Setup**: Same lighting, background, camera distance
- **Multiple Sessions**: Spread data collection across days
- **Speed Variations**: Record slow, medium, fast punches
- **Quality over Quantity**: Better to have 200 perfect clips than 500 mediocre ones

---
**Elevate AI - Focused execution for ML training excellence** 🥊⚡
