/**
 * AI Boxing Trainer - Right Hand IMU Firmware
 *
 * ESP32 + BNO085 IMU sensor for real-time motion tracking
 * Streams accelerometer, gyroscope, and quaternion data via WiFi UDP
 *
 * Hardware:
 *   - ESP32 DevKit (or compatible)
 *   - BNO085 IMU (I2C connection)
 *
 * Author: AI Boxing Trainer Team
 * License: MIT
 */

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <Adafruit_BNO08x.h>

// Device identification
#define DEVICE_ID "right_hand"
#define DEVICE_HAND 1  // 0 = left, 1 = right

// Include shared headers
#include "config.h"
#include "imu_data.h"
#include "filters.h"

// =============================================================================
// Global Objects
// =============================================================================

// BNO085 IMU sensor
Adafruit_BNO08x bno08x(BNO085_RESET_PIN);
sh2_SensorValue_t sensorValue;

// WiFi UDP
WiFiUDP udp;

// Data packet
IMUPacket packet;
uint32_t sequence_number = 0;

// Filters
Vector3Filter accel_filter(0.3f);
Vector3Filter gyro_filter(0.2f);

// Calibration
CalibrationData calibration;

// Timing
unsigned long last_sample_time = 0;
unsigned long last_transmit_time = 0;
unsigned long last_debug_time = 0;

// Status
bool imu_ready = false;
bool wifi_connected = false;
uint8_t device_status = 0;

// =============================================================================
// Function Declarations
// =============================================================================

void setupWiFi();
void setupIMU();
void enableReports();
bool readIMUData();
void transmitPacket();
void updateStatusLED();
void performCalibration();
void printDebugInfo();

// =============================================================================
// Setup
// =============================================================================

void setup() {
    // Initialize serial for debugging
    Serial.begin(DEBUG_BAUD_RATE);
    delay(100);

    Serial.println();
    Serial.println("========================================");
    Serial.println("  AI Boxing Trainer - Right Hand IMU");
    Serial.println("========================================");
    Serial.printf("Device: %s\n", DEVICE_ID);
    Serial.printf("Firmware: v1.0.0\n");
    Serial.println();

    // Initialize status LED
    pinMode(LED_STATUS_PIN, OUTPUT);
    digitalWrite(LED_STATUS_PIN, LOW);

    // Initialize I2C
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
    Wire.setClock(I2C_CLOCK_SPEED);

    // Setup components
    setupIMU();
    setupWiFi();

    // Initialize packet
    init_packet(&packet, DEVICE_HAND);

    // Perform initial calibration
    if (imu_ready) {
        performCalibration();
    }

    Serial.println();
    Serial.println("Setup complete! Starting main loop...");
    Serial.println("========================================");
    Serial.println();
}

// =============================================================================
// Main Loop
// =============================================================================

void loop() {
    unsigned long current_time = micros();

    // Sample IMU at configured rate
    if (current_time - last_sample_time >= SAMPLE_INTERVAL_US) {
        last_sample_time = current_time;

        if (imu_ready && readIMUData()) {
            // Data successfully read and stored in packet
        }
    }

    // Transmit at configured rate (lower than sample rate)
    unsigned long current_ms = millis();
    if (current_ms - last_transmit_time >= TRANSMIT_INTERVAL_MS) {
        last_transmit_time = current_ms;

        if (wifi_connected && imu_ready) {
            transmitPacket();
        }
    }

    // Debug output (if enabled)
    if (DEBUG_MODE && DEBUG_PRINT_DATA) {
        if (current_ms - last_debug_time >= (1000 / DEBUG_PRINT_RATE_HZ)) {
            last_debug_time = current_ms;
            printDebugInfo();
        }
    }

    // Update status LED
    updateStatusLED();

    // Check WiFi connection
    if (WiFi.status() != WL_CONNECTED) {
        if (wifi_connected) {
            Serial.println("WiFi disconnected! Reconnecting...");
            wifi_connected = false;
            device_status &= ~STATUS_WIFI_OK;
        }
        WiFi.reconnect();
        delay(100);
    }

    // Small delay to prevent watchdog issues
    yield();
}

// =============================================================================
// WiFi Setup
// =============================================================================

void setupWiFi() {
    Serial.println("Connecting to WiFi...");
    Serial.printf("  SSID: %s\n", WIFI_SSID);

    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    // Wait for connection with timeout
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 30) {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED) {
        wifi_connected = true;
        device_status |= STATUS_WIFI_OK;

        Serial.println();
        Serial.println("WiFi connected!");
        Serial.printf("  IP: %s\n", WiFi.localIP().toString().c_str());
        Serial.printf("  Target: %s:%d\n", RECEIVER_IP, RECEIVER_PORT);

        // Initialize UDP
        udp.begin(RECEIVER_PORT);
    } else {
        Serial.println();
        Serial.println("WiFi connection failed!");
        Serial.println("Check SSID and password in config.h");
    }
}

// =============================================================================
// IMU Setup
// =============================================================================

void setupIMU() {
    Serial.println("Initializing BNO085 IMU...");

    // Try to initialize BNO085
    if (!bno08x.begin_I2C(BNO085_I2C_ADDR, &Wire)) {
        Serial.println("Failed to find BNO085!");
        Serial.println("Check wiring:");
        Serial.printf("  SDA: GPIO %d\n", I2C_SDA_PIN);
        Serial.printf("  SCL: GPIO %d\n", I2C_SCL_PIN);
        Serial.printf("  Address: 0x%02X\n", BNO085_I2C_ADDR);
        return;
    }

    Serial.println("BNO085 found!");

    // Enable required reports
    enableReports();

    imu_ready = true;
    device_status |= STATUS_IMU_OK;

    Serial.println("IMU initialized successfully!");
}

void enableReports() {
    Serial.println("Enabling sensor reports...");

    // Enable accelerometer (100Hz)
    if (!bno08x.enableReport(SH2_ACCELEROMETER, 10000)) {  // 10ms = 100Hz
        Serial.println("Could not enable accelerometer!");
    }

    // Enable gyroscope (100Hz)
    if (!bno08x.enableReport(SH2_GYROSCOPE_CALIBRATED, 10000)) {
        Serial.println("Could not enable gyroscope!");
    }

    // Enable rotation vector (quaternion from sensor fusion, 100Hz)
    if (!bno08x.enableReport(SH2_ROTATION_VECTOR, 10000)) {
        Serial.println("Could not enable rotation vector!");
    }

    // Enable game rotation vector (faster, no magnetometer)
    if (!bno08x.enableReport(SH2_GAME_ROTATION_VECTOR, 10000)) {
        Serial.println("Could not enable game rotation vector!");
    }

    Serial.println("Sensor reports enabled.");
}

// =============================================================================
// IMU Data Reading
// =============================================================================

bool readIMUData() {
    bool got_data = false;

    // Read all available sensor events
    while (bno08x.getSensorEvent(&sensorValue)) {
        got_data = true;

        switch (sensorValue.sensorId) {
            case SH2_ACCELEROMETER:
                packet.accel_x = sensorValue.un.accelerometer.x;
                packet.accel_y = sensorValue.un.accelerometer.y;
                packet.accel_z = sensorValue.un.accelerometer.z;

                // Apply calibration offsets
                if (calibration.is_calibrated) {
                    packet.accel_x -= calibration.accel_offset_x;
                    packet.accel_y -= calibration.accel_offset_y;
                    packet.accel_z -= calibration.accel_offset_z;
                }

                // Apply filter
                accel_filter.update(packet.accel_x, packet.accel_y, packet.accel_z);
                break;

            case SH2_GYROSCOPE_CALIBRATED:
                packet.gyro_x = sensorValue.un.gyroscope.x;
                packet.gyro_y = sensorValue.un.gyroscope.y;
                packet.gyro_z = sensorValue.un.gyroscope.z;

                // Apply calibration offsets
                if (calibration.is_calibrated) {
                    packet.gyro_x -= calibration.gyro_offset_x;
                    packet.gyro_y -= calibration.gyro_offset_y;
                    packet.gyro_z -= calibration.gyro_offset_z;
                }

                // Apply filter
                gyro_filter.update(packet.gyro_x, packet.gyro_y, packet.gyro_z);
                break;

            case SH2_ROTATION_VECTOR:
            case SH2_GAME_ROTATION_VECTOR:
                packet.quat_w = sensorValue.un.rotationVector.real;
                packet.quat_x = sensorValue.un.rotationVector.i;
                packet.quat_y = sensorValue.un.rotationVector.j;
                packet.quat_z = sensorValue.un.rotationVector.k;
                packet.accuracy = sensorValue.un.rotationVector.accuracy;
                break;
        }
    }

    // Update motion detection status
    float accel_magnitude = sqrt(
        packet.accel_x * packet.accel_x +
        packet.accel_y * packet.accel_y +
        packet.accel_z * packet.accel_z
    );

    // Remove gravity (~9.8) and check for motion
    if (fabs(accel_magnitude - 9.81f) > ACCEL_MOTION_THRESHOLD) {
        device_status |= STATUS_MOTION;
    } else {
        device_status &= ~STATUS_MOTION;
    }

    return got_data;
}

// =============================================================================
// Data Transmission
// =============================================================================

void transmitPacket() {
    // Update packet metadata
    packet.sequence = sequence_number++;
    packet.timestamp_ms = millis();
    packet.timestamp_us = micros() % 1000;
    packet.status = device_status;

    // Calculate checksum
    packet.checksum = calculate_checksum(&packet);

    // Send UDP packet
    udp.beginPacket(RECEIVER_IP, RECEIVER_PORT);
    udp.write((uint8_t*)&packet, sizeof(IMUPacket));
    udp.endPacket();

    // Blink LED on transmission (if enabled)
    if (LED_BLINK_ON_TX && (sequence_number % 10 == 0)) {
        digitalWrite(LED_STATUS_PIN, !digitalRead(LED_STATUS_PIN));
    }
}

// =============================================================================
// Calibration
// =============================================================================

void performCalibration() {
    Serial.println("Performing calibration...");
    Serial.println("Keep the sensor still!");

    float accel_sum[3] = {0, 0, 0};
    float gyro_sum[3] = {0, 0, 0};
    int samples = 0;

    // Collect samples for calibration
    unsigned long start_time = millis();
    while (samples < CALIBRATION_SAMPLES && (millis() - start_time) < 5000) {
        if (bno08x.getSensorEvent(&sensorValue)) {
            if (sensorValue.sensorId == SH2_ACCELEROMETER) {
                accel_sum[0] += sensorValue.un.accelerometer.x;
                accel_sum[1] += sensorValue.un.accelerometer.y;
                accel_sum[2] += sensorValue.un.accelerometer.z;
            } else if (sensorValue.sensorId == SH2_GYROSCOPE_CALIBRATED) {
                gyro_sum[0] += sensorValue.un.gyroscope.x;
                gyro_sum[1] += sensorValue.un.gyroscope.y;
                gyro_sum[2] += sensorValue.un.gyroscope.z;
                samples++;
            }
        }
        delay(10);
    }

    if (samples > 0) {
        // Calculate offsets (gyro should be ~0, accel should show gravity on Z)
        calibration.gyro_offset_x = gyro_sum[0] / samples;
        calibration.gyro_offset_y = gyro_sum[1] / samples;
        calibration.gyro_offset_z = gyro_sum[2] / samples;

        // For accelerometer, we don't offset Z (gravity axis)
        // This assumes sensor is flat during calibration
        calibration.accel_offset_x = accel_sum[0] / samples;
        calibration.accel_offset_y = accel_sum[1] / samples;
        calibration.accel_offset_z = 0;  // Don't offset gravity

        calibration.is_calibrated = true;
        calibration.calibration_timestamp = millis();
        device_status |= STATUS_CALIBRATED;

        Serial.printf("Calibration complete! (%d samples)\n", samples);
        Serial.printf("  Gyro offsets: [%.4f, %.4f, %.4f]\n",
            calibration.gyro_offset_x,
            calibration.gyro_offset_y,
            calibration.gyro_offset_z);
    } else {
        Serial.println("Calibration failed - no samples received!");
    }
}

// =============================================================================
// Status LED
// =============================================================================

void updateStatusLED() {
    static unsigned long last_blink = 0;
    static bool led_state = false;

    if (!wifi_connected) {
        // Fast blink when WiFi disconnected
        if (millis() - last_blink > 200) {
            last_blink = millis();
            led_state = !led_state;
            digitalWrite(LED_STATUS_PIN, led_state);
        }
    } else if (!imu_ready) {
        // Slow blink when IMU not ready
        if (millis() - last_blink > 1000) {
            last_blink = millis();
            led_state = !led_state;
            digitalWrite(LED_STATUS_PIN, led_state);
        }
    } else if (LED_WIFI_CONNECTED) {
        // Solid on when everything is working
        // (transmission blink handled in transmitPacket)
    }
}

// =============================================================================
// Debug Output
// =============================================================================

void printDebugInfo() {
    Serial.println("----------------------------------------");
    Serial.printf("Seq: %lu | Time: %lu ms\n", packet.sequence, packet.timestamp_ms);
    Serial.printf("Accel: [%7.3f, %7.3f, %7.3f] m/s^2\n",
        packet.accel_x, packet.accel_y, packet.accel_z);
    Serial.printf("Gyro:  [%7.3f, %7.3f, %7.3f] rad/s\n",
        packet.gyro_x, packet.gyro_y, packet.gyro_z);
    Serial.printf("Quat:  [%6.3f, %6.3f, %6.3f, %6.3f]\n",
        packet.quat_w, packet.quat_x, packet.quat_y, packet.quat_z);
    Serial.printf("Status: 0x%02X | Accuracy: %d\n",
        packet.status, packet.accuracy);
}
