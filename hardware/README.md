# AI Boxing Trainer - Hardware Module

ESP32 + BNO085 IMU sensors for real-time punch tracking and motion analysis.

## Overview

This module provides wrist-mounted IMU sensors that stream motion data to enhance the Boxing Trainer's punch detection capabilities. The system uses WiFi UDP for low-latency data transmission.

```
┌─────────────────┐     WiFi UDP      ┌─────────────────┐
│  Left Hand      │ ─────────────────►│                 │
│  ESP32 + BNO085 │                   │   PC Running    │
└─────────────────┘                   │  Boxing Trainer │
                                      │                 │
┌─────────────────┐     WiFi UDP      │   + IMU Data    │
│  Right Hand     │ ─────────────────►│   Integration   │
│  ESP32 + BNO085 │                   │                 │
└─────────────────┘                   └─────────────────┘
```

## Quick Start

### 1. Wire the Hardware
See [docs/wiring_guide.md](docs/wiring_guide.md)

### 2. Configure WiFi
Edit `shared/config.h`:
```cpp
#define WIFI_SSID       "YourNetwork"
#define WIFI_PASSWORD   "YourPassword"
#define RECEIVER_IP     "192.168.1.100"  // Your PC's IP
```

### 3. Flash Firmware
```bash
# Left hand
cd firmware_left && pio run --target upload

# Right hand
cd ../firmware_right && pio run --target upload
```

### 4. Test Reception
```bash
cd receiver
python pc_test_listener.py
```

## Directory Structure

```
hardware/
├── README.md               # This file
├── firmware_left/          # Left hand ESP32 firmware
│   ├── platformio.ini      # Build configuration
│   └── src/
│       └── main.cpp        # Main firmware code
├── firmware_right/         # Right hand ESP32 firmware
│   ├── platformio.ini
│   └── src/
│       └── main.cpp
├── shared/                 # Shared code between devices
│   ├── config.h            # Configuration (WiFi, pins, rates)
│   ├── imu_data.h          # Data structures and protocol
│   └── filters.h           # Signal processing filters
├── receiver/               # PC receiver utilities
│   ├── pc_test_listener.py # Standalone test tool
│   └── imu_receiver.py     # Integration module
└── docs/                   # Documentation
    ├── wiring_guide.md     # Hardware connections
    └── SETUP.md            # Build and flash instructions
```

## Data Format

Each UDP packet (64 bytes) contains:

```json
{
  "device": "left_hand",
  "timestamp": 123456.789,
  "accel": [x, y, z],      // m/s²
  "gyro": [x, y, z],       // rad/s
  "quat": [w, x, y, z],    // orientation
  "accuracy": 3,           // 0-3 calibration level
  "status": {
    "motion": true,
    "calibrated": true
  }
}
```

## Specifications

| Parameter | Value |
|-----------|-------|
| IMU Sample Rate | 100 Hz |
| Network Transmit Rate | 50 Hz |
| Protocol | UDP (low latency) |
| Packet Size | 64 bytes |
| Latency | < 10ms typical |
| Power | 3.3V via USB or LiPo |

## Integration

```python
from hardware.receiver.imu_receiver import IMUReceiver

receiver = IMUReceiver(port=5555)
receiver.start()

# Get latest data
left, right = receiver.get_latest()
if left:
    print(f"Left accel: {left.accel_magnitude:.2f} m/s²")
```

## Hardware Cost

| Component | Qty | Cost |
|-----------|-----|------|
| ESP32 DevKit | 2 | ~$10 |
| BNO085 IMU | 2 | ~$40 |
| Misc (wires, USB) | - | ~$15 |
| **Total** | | **~$65** |

## Documentation

- [Wiring Guide](docs/wiring_guide.md) - Hardware connections
- [Setup Guide](docs/SETUP.md) - Build and flash instructions

## Next Steps

1. **Movement Classification**: Train ML models on IMU data
2. **Sensor Fusion**: Combine IMU with vision for better accuracy
3. **Wireless Power**: Add LiPo battery support
4. **3D Printed Enclosure**: Design wrist-mount cases
