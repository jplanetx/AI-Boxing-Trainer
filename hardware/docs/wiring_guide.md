# AI Boxing Trainer - Wiring Guide

Complete wiring instructions for ESP32 + BNO085 IMU sensors.

## Hardware Requirements

### Per Hand Unit (x2 total)

| Component | Quantity | Notes |
|-----------|----------|-------|
| ESP32 DevKit V1 | 1 | 30-pin or 38-pin version |
| BNO085 IMU Breakout | 1 | Adafruit or SparkFun board recommended |
| Jumper Wires | 4-6 | Female-to-female for breadboard |
| Micro USB Cable | 1 | For programming and power |
| Optional: LiPo Battery | 1 | 3.7V 500mAh+ for wireless operation |

### Recommended Boards

**ESP32:**
- ESP32-DevKitC (official Espressif)
- ESP32-WROOM-32 DevKit
- NodeMCU ESP32

**BNO085 IMU:**
- [Adafruit BNO085](https://www.adafruit.com/product/4754) - Recommended
- [SparkFun BNO085](https://www.sparkfun.com/products/14686)
- Generic BNO085 breakout boards

---

## Wiring Diagram

### ESP32 to BNO085 I2C Connection

```
ESP32 DevKit                    BNO085 Breakout
┌─────────────┐                ┌─────────────┐
│             │                │             │
│  3.3V ──────┼───────────────►│ VIN (3.3V)  │
│             │                │             │
│  GND ───────┼───────────────►│ GND         │
│             │                │             │
│  GPIO 21 ───┼───────────────►│ SDA         │
│  (SDA)      │                │             │
│             │                │             │
│  GPIO 22 ───┼───────────────►│ SCL         │
│  (SCL)      │                │             │
│             │                │             │
│  GPIO 4 ────┼───────────────►│ INT         │ (Optional)
│  (INT)      │                │             │
│             │                │             │
│  GPIO 5 ────┼───────────────►│ RST         │ (Optional)
│  (RST)      │                │             │
└─────────────┘                └─────────────┘
```

### Pin Mapping Table

| ESP32 Pin | BNO085 Pin | Wire Color (suggested) | Notes |
|-----------|------------|------------------------|-------|
| 3.3V | VIN | Red | Power supply |
| GND | GND | Black | Ground |
| GPIO 21 | SDA | Blue | I2C Data |
| GPIO 22 | SCL | Yellow | I2C Clock |
| GPIO 4 | INT | Green | Interrupt (optional) |
| GPIO 5 | RST | White | Reset (optional) |

---

## Detailed Wiring Instructions

### Step 1: Power Connections

1. Connect ESP32 `3.3V` pin to BNO085 `VIN` pin
2. Connect ESP32 `GND` pin to BNO085 `GND` pin

**IMPORTANT:**
- The BNO085 operates at 3.3V logic level
- Do NOT connect to 5V - this may damage the sensor
- Ensure good, solid connections

### Step 2: I2C Data Connections

1. Connect ESP32 `GPIO 21` to BNO085 `SDA` (I2C Data)
2. Connect ESP32 `GPIO 22` to BNO085 `SCL` (I2C Clock)

**Notes:**
- These are the default I2C pins on most ESP32 boards
- Internal pull-up resistors are enabled in firmware
- Keep wires short (< 20cm) for reliable I2C communication

### Step 3: Optional Connections

For enhanced reliability (recommended for production):

1. Connect ESP32 `GPIO 4` to BNO085 `INT` (Interrupt)
2. Connect ESP32 `GPIO 5` to BNO085 `RST` (Reset)

These allow:
- Hardware reset capability
- Interrupt-driven data reading (lower latency)

---

## BNO085 I2C Address Configuration

The default I2C address is `0x4A`. If you need to use a different address:

| ADR Pin | I2C Address |
|---------|-------------|
| LOW (default) | 0x4A |
| HIGH | 0x4B |

To change the address, connect the ADR pin to 3.3V for address 0x4B.

---

## Physical Assembly Tips

### Wrist Mount Orientation

For accurate punch detection, mount the sensor with consistent orientation:

```
         ┌──────────────┐
         │    BNO085    │
         │   ┌─────┐    │
         │   │     │    │   ▲
         │   │  ●  │    │   │ Y+ (toward fingers)
         │   │     │    │   │
         │   └─────┘    │
         │  X+ ────►    │
         └──────────────┘
              Z+ (up, away from wrist)
```

**Mounting Position:**
- Secure on back of wrist/hand
- Sensor flat against hand
- X-axis pointing toward thumb
- Y-axis pointing toward fingers
- Z-axis pointing away from body

### Enclosure Recommendations

1. **3D Printed Case:**
   - Design should allow airflow for heat dissipation
   - Include Velcro strap mounting points
   - Ensure USB port access for charging

2. **Wrist Strap:**
   - Use adjustable elastic band (20-30mm width)
   - Secure attachment to prevent shifting during punches
   - Consider sweat-resistant materials

---

## Testing the Connection

### 1. Visual Verification

Before powering on, verify:
- [ ] All connections are secure
- [ ] No exposed wires touching
- [ ] Correct voltage (3.3V, not 5V)
- [ ] Ground connected

### 2. I2C Scanner Test

Upload this test sketch to verify I2C communication:

```cpp
#include <Wire.h>

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);  // SDA, SCL

  Serial.println("I2C Scanner");
}

void loop() {
  Serial.println("Scanning...");

  for (byte addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.printf("Device found at 0x%02X\n", addr);
    }
  }

  Serial.println("Scan complete\n");
  delay(5000);
}
```

Expected output:
```
I2C Scanner
Scanning...
Device found at 0x4A    <- This is the BNO085
Scan complete
```

### 3. Full Firmware Test

After flashing the Boxing Trainer firmware:

1. Open Serial Monitor (115200 baud)
2. Look for successful initialization messages
3. Verify IMU data is being read
4. Run PC test listener to confirm UDP transmission

---

## Troubleshooting

### Problem: "Failed to find BNO085!"

**Causes & Solutions:**

1. **Incorrect wiring**
   - Double-check SDA/SCL connections
   - Verify power connections (3.3V and GND)

2. **Wrong I2C address**
   - Check ADR pin state
   - Try address 0x4B in config.h

3. **Defective sensor**
   - Test with I2C scanner
   - Try a different BNO085 board

4. **I2C bus issue**
   - Shorten wire length
   - Add 4.7kΩ pull-up resistors on SDA/SCL

### Problem: Intermittent Data / Errors

1. **Loose connections**
   - Solder connections for reliability
   - Use strain relief on wires

2. **Electrical noise**
   - Add 0.1µF capacitor between VIN and GND
   - Keep wires away from motors/high-current paths

3. **I2C speed too high**
   - Reduce I2C clock in config.h (try 100000)

### Problem: WiFi Not Connecting

1. **Incorrect credentials**
   - Update WIFI_SSID and WIFI_PASSWORD in config.h
   - Ensure no hidden characters in password

2. **Network issues**
   - Verify 2.4GHz network (ESP32 doesn't support 5GHz)
   - Check router allows new connections

3. **Signal strength**
   - Move closer to router for testing
   - Consider external antenna ESP32 variant

---

## Bill of Materials

### Basic Setup (2 sensors)

| Item | Qty | Est. Cost | Link |
|------|-----|-----------|------|
| ESP32 DevKit | 2 | $10 | [Amazon](https://amazon.com) |
| Adafruit BNO085 | 2 | $40 | [Adafruit](https://www.adafruit.com/product/4754) |
| Jumper Wires | 1 pack | $5 | Various |
| USB Cables | 2 | $10 | Various |
| **Total** | | **~$65** | |

### Wireless Setup (with batteries)

| Item | Qty | Est. Cost | Link |
|------|-----|-----------|------|
| ESP32 DevKit | 2 | $10 | [Amazon](https://amazon.com) |
| Adafruit BNO085 | 2 | $40 | [Adafruit](https://www.adafruit.com/product/4754) |
| LiPo Battery 500mAh | 2 | $15 | [Adafruit](https://www.adafruit.com) |
| TP4056 Charger | 2 | $5 | Various |
| 3D Printed Case | 2 | $10 | Print yourself |
| Wrist Straps | 2 | $5 | Various |
| **Total** | | **~$85** | |

---

## Next Steps

After completing wiring:

1. **Flash Firmware**: See [SETUP.md](./SETUP.md)
2. **Configure Network**: Update config.h with your WiFi
3. **Test Streaming**: Run pc_test_listener.py
4. **Integrate**: Add IMU data to main Boxing Trainer

---

## Safety Notes

- Always disconnect power before modifying wiring
- Handle LiPo batteries with care
- Do not short circuit any connections
- Ensure adequate ventilation during charging

---

*AI Boxing Trainer Hardware Team*
