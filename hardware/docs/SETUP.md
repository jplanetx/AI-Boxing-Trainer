# AI Boxing Trainer - Firmware Setup Guide

Complete instructions for building, flashing, and testing the ESP32 IMU firmware.

## Prerequisites

### Software Requirements

1. **PlatformIO** (recommended)
   - Install VS Code: https://code.visualstudio.com/
   - Install PlatformIO extension from VS Code marketplace
   - OR install CLI: `pip install platformio`

2. **Alternative: Arduino IDE**
   - Download from: https://www.arduino.cc/en/software
   - Add ESP32 board support (see below)

3. **Python 3.8+** (for test utilities)
   - Install from: https://www.python.org/
   - Required packages: `pip install numpy`

### Hardware Requirements

- 2x ESP32 DevKit boards (one per hand)
- 2x BNO085 IMU sensors
- USB cables for programming
- See [wiring_guide.md](./wiring_guide.md) for connections

---

## Quick Start

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/AI-Boxing-Trainer.git
cd AI-Boxing-Trainer/hardware
```

### Step 2: Configure WiFi

Edit `shared/config.h`:

```cpp
// Update these with your network credentials
#define WIFI_SSID           "YourWiFiName"
#define WIFI_PASSWORD       "YourWiFiPassword"

// Update with your PC's IP address
#define RECEIVER_IP         "192.168.1.100"
```

**Finding Your PC's IP:**
- Windows: `ipconfig` in Command Prompt
- Mac/Linux: `ifconfig` or `ip addr` in Terminal
- Look for IPv4 address on your local network (usually 192.168.x.x)

### Step 3: Flash Left Hand Firmware

```bash
cd firmware_left

# Build and upload
pio run --target upload

# Monitor output (optional)
pio device monitor
```

### Step 4: Flash Right Hand Firmware

```bash
cd ../firmware_right

# Build and upload
pio run --target upload

# Monitor output
pio device monitor
```

### Step 5: Test Data Reception

```bash
cd ../receiver

# Run test listener
python pc_test_listener.py
```

Expected output:
```
========================================
  AI Boxing Trainer - IMU Test Listener
========================================

Listening on 0.0.0.0:5555
Waiting for IMU packets... (Ctrl+C to stop)

[LEFT ] #     1 | Accel: [  0.12,   0.05,   9.82] | Gyro: [  0.01,  -0.02,   0.00] |        CAL
[RIGHT] #     1 | Accel: [  0.08,  -0.03,   9.79] | Gyro: [  0.00,   0.01,  -0.01] |        CAL
```

---

## Detailed Instructions

### PlatformIO Setup (Recommended)

#### Installing PlatformIO

**Option A: VS Code Extension**
1. Open VS Code
2. Go to Extensions (Ctrl+Shift+X)
3. Search for "PlatformIO IDE"
4. Click Install
5. Restart VS Code

**Option B: Command Line**
```bash
pip install platformio
```

#### Building with PlatformIO

```bash
# Navigate to firmware directory
cd hardware/firmware_left

# Build only
pio run

# Build and upload
pio run --target upload

# Clean build
pio run --target clean
```

#### Serial Monitor

```bash
# Start monitor
pio device monitor

# With specific port
pio device monitor --port /dev/ttyUSB0

# With specific baud rate
pio device monitor --baud 115200
```

### Arduino IDE Setup (Alternative)

#### Adding ESP32 Board Support

1. Open Arduino IDE
2. Go to File → Preferences
3. Add to "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Go to Tools → Board → Board Manager
5. Search for "esp32" and install "ESP32 by Espressif Systems"

#### Installing Libraries

1. Go to Sketch → Include Library → Manage Libraries
2. Search and install:
   - "Adafruit BNO08x" by Adafruit
   - "Adafruit BusIO" by Adafruit

#### Building with Arduino IDE

1. Open `hardware/firmware_left/src/main.cpp`
2. Select Tools → Board → ESP32 Dev Module
3. Select the correct COM port
4. Click Upload button

**Note:** You'll need to manually include the shared headers. Copy files from `hardware/shared/` to your Arduino libraries folder.

---

## Configuration Options

### config.h Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WIFI_SSID` | "YOUR_WIFI_SSID" | Your WiFi network name |
| `WIFI_PASSWORD` | "YOUR_WIFI_PASSWORD" | Your WiFi password |
| `RECEIVER_IP` | "192.168.1.100" | IP of PC running Boxing Trainer |
| `RECEIVER_PORT` | 5555 | UDP port for data transmission |
| `IMU_SAMPLE_RATE_HZ` | 100 | IMU polling rate |
| `TRANSMIT_RATE_HZ` | 50 | Network transmission rate |
| `I2C_SDA_PIN` | 21 | I2C data pin |
| `I2C_SCL_PIN` | 22 | I2C clock pin |
| `DEBUG_MODE` | true | Enable serial debug output |
| `DEBUG_PRINT_DATA` | false | Print IMU data to serial |

### Adjusting Sample Rate

For different use cases:

```cpp
// High precision (more CPU/bandwidth)
#define IMU_SAMPLE_RATE_HZ  200
#define TRANSMIT_RATE_HZ    100

// Power saving (longer battery life)
#define IMU_SAMPLE_RATE_HZ  50
#define TRANSMIT_RATE_HZ    25
```

### Debug Mode

Enable detailed serial output:

```cpp
#define DEBUG_MODE          true
#define DEBUG_PRINT_DATA    true   // Warning: reduces performance
#define DEBUG_PRINT_RATE_HZ 10     // Print 10 times per second
```

---

## Testing

### Test 1: Serial Monitor

After flashing, open Serial Monitor (115200 baud):

```
========================================
  AI Boxing Trainer - Left Hand IMU
========================================
Device: left_hand
Firmware: v1.0.0

Initializing BNO085 IMU...
BNO085 found!
Enabling sensor reports...
Sensor reports enabled.
IMU initialized successfully!
Connecting to WiFi...
  SSID: YourNetwork
.....
WiFi connected!
  IP: 192.168.1.105
  Target: 192.168.1.100:5555
Performing calibration...
Keep the sensor still!
Calibration complete! (100 samples)
  Gyro offsets: [0.0012, -0.0008, 0.0003]

Setup complete! Starting main loop...
========================================
```

### Test 2: PC Listener

Run the test listener on your PC:

```bash
cd hardware/receiver

# Basic mode
python pc_test_listener.py

# Verbose mode
python pc_test_listener.py --verbose

# JSON output (for integration testing)
python pc_test_listener.py --json

# Log to file
python pc_test_listener.py --log session_data.csv
```

### Test 3: Motion Detection

1. Start the listener in verbose mode
2. Hold the sensor still - should show "CAL" (calibrated)
3. Move the sensor quickly - should show "MOTION"
4. Verify acceleration and gyro values change with movement

### Test 4: Punch Detection

Use the integration receiver:

```bash
python hardware/receiver/imu_receiver.py
```

Throw punches with the sensor attached:
- Should detect peak acceleration
- Should report punch events with timing

---

## Troubleshooting

### Build Errors

**Error: "Adafruit_BNO08x.h: No such file or directory"**
```bash
# PlatformIO automatically handles dependencies
# If issues persist:
pio lib install "Adafruit BNO08x"
```

**Error: "cannot open source file 'config.h'"**
- Ensure you're building from the correct directory
- Check that `build_flags` includes `-I../shared`

### Upload Errors

**Error: "Failed to connect to ESP32"**
1. Hold BOOT button while clicking Upload
2. Release BOOT after "Connecting..." appears
3. Check USB cable (some are charge-only)
4. Try different USB port

**Error: "A fatal error occurred: Timed out"**
- Install CP210x or CH340 USB driver
- Windows: Device Manager → Update driver
- Mac: Install from Silicon Labs website

### WiFi Connection Issues

**Stuck on "Connecting to WiFi..."**
1. Verify SSID and password (case-sensitive)
2. Ensure 2.4GHz network (ESP32 doesn't support 5GHz)
3. Move closer to router
4. Check if router has MAC filtering

**WiFi connects but no data received**
1. Verify RECEIVER_IP matches your PC's IP
2. Check firewall allows UDP port 5555
3. Ensure PC and ESP32 are on same network subnet

### IMU Issues

**"Failed to find BNO085!"**
- Check wiring (see wiring_guide.md)
- Verify 3.3V power (not 5V)
- Try I2C address 0x4B in config.h

**Erratic sensor data**
- Shorten I2C wires
- Add pull-up resistors (4.7kΩ)
- Reduce I2C clock speed to 100kHz

---

## Production Deployment

### Optimizing for Battery Life

```cpp
// In config.h
#define IMU_SAMPLE_RATE_HZ  50      // Lower sample rate
#define TRANSMIT_RATE_HZ    25      // Lower transmit rate
#define DEBUG_MODE          false   // Disable serial output
```

### OTA Updates

For wireless firmware updates, add to platformio.ini:

```ini
upload_protocol = espota
upload_port = 192.168.1.105  ; ESP32's IP address
```

### Security Considerations

For production use:
1. Use WPA2 networks only
2. Consider adding packet encryption
3. Implement device authentication
4. Use unique device identifiers

---

## Integration with Boxing Trainer

### Python Integration

```python
from hardware.receiver.imu_receiver import IMUReceiver

# Create receiver
receiver = IMUReceiver(port=5555)

# Set up callbacks
receiver.on_punch = lambda p: print(f"Punch: {p.hand}")

# Start receiving
receiver.start()

# In your main loop
left_data, right_data = receiver.get_latest()
if left_data:
    # Use IMU data for enhanced punch detection
    pass
```

### Combining with Vision Data

The IMU data can enhance the vision-based punch detection:

1. **Validation**: Confirm vision-detected punches with IMU acceleration spikes
2. **Speed**: IMU detects punch faster than vision (lower latency)
3. **Accuracy**: Combine both signals for more reliable detection
4. **Blind spots**: IMU works when hands are outside camera view

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01 | Initial release |

---

*AI Boxing Trainer Hardware Team*
