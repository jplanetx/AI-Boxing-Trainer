/**
 * AI Boxing Trainer - ESP32 IMU Configuration
 * Shared configuration for left and right hand devices
 *
 * Hardware: ESP32 + BNO085 IMU
 * Protocol: WiFi UDP streaming
 */

#ifndef BOXING_TRAINER_CONFIG_H
#define BOXING_TRAINER_CONFIG_H

// =============================================================================
// WiFi Configuration
// =============================================================================
// IMPORTANT: Update these values for your network
#define WIFI_SSID           "YOUR_WIFI_SSID"
#define WIFI_PASSWORD       "YOUR_WIFI_PASSWORD"

// Target receiver (PC running the Boxing Trainer app)
#define RECEIVER_IP         "192.168.1.100"  // Update to your PC's IP
#define RECEIVER_PORT       5555

// =============================================================================
// IMU Configuration (BNO085)
// =============================================================================
// I2C Pins for ESP32
#define I2C_SDA_PIN         21
#define I2C_SCL_PIN         22
#define BNO085_I2C_ADDR     0x4A  // Default BNO085 address (0x4B if ADR pin high)
#define BNO085_INT_PIN      4     // Optional interrupt pin
#define BNO085_RESET_PIN    5     // Optional reset pin

// I2C Clock Speed
#define I2C_CLOCK_SPEED     400000  // 400kHz Fast Mode

// =============================================================================
// Sampling Configuration
// =============================================================================
#define IMU_SAMPLE_RATE_HZ  100    // IMU polling rate (Hz)
#define TRANSMIT_RATE_HZ    50     // Network transmission rate (Hz)
#define SAMPLE_INTERVAL_US  (1000000 / IMU_SAMPLE_RATE_HZ)
#define TRANSMIT_INTERVAL_MS (1000 / TRANSMIT_RATE_HZ)

// =============================================================================
// Data Processing
// =============================================================================
// Moving average filter window size
#define ACCEL_FILTER_SIZE   5
#define GYRO_FILTER_SIZE    3

// Calibration samples
#define CALIBRATION_SAMPLES 100

// Motion detection thresholds
#define ACCEL_MOTION_THRESHOLD  2.0f   // m/s^2 - acceleration threshold for motion
#define GYRO_MOTION_THRESHOLD   0.5f   // rad/s - angular velocity threshold

// =============================================================================
// LED Status Indicators (optional)
// =============================================================================
#define LED_STATUS_PIN      2      // Built-in LED on most ESP32 boards
#define LED_WIFI_CONNECTED  true   // LED on when WiFi connected
#define LED_BLINK_ON_TX     true   // Blink on data transmission

// =============================================================================
// Debug Configuration
// =============================================================================
#define DEBUG_MODE          true   // Enable serial debug output
#define DEBUG_BAUD_RATE     115200
#define DEBUG_PRINT_DATA    false  // Print IMU data to serial (reduces performance)
#define DEBUG_PRINT_RATE_HZ 10     // Rate for debug prints when enabled

// =============================================================================
// Protocol Version
// =============================================================================
#define PROTOCOL_VERSION    1
#define PACKET_MAGIC        0xB0C5  // "BOXS" in hex

// =============================================================================
// Device Identification (override in device-specific main.cpp)
// =============================================================================
#ifndef DEVICE_ID
#define DEVICE_ID           "unknown"
#endif

#ifndef DEVICE_HAND
#define DEVICE_HAND         0  // 0 = left, 1 = right
#endif

#endif // BOXING_TRAINER_CONFIG_H
