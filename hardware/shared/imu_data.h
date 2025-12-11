/**
 * AI Boxing Trainer - IMU Data Structures
 * Shared data structures for sensor communication
 */

#ifndef IMU_DATA_H
#define IMU_DATA_H

#include <stdint.h>

// =============================================================================
// Packet Structure for UDP Transmission
// =============================================================================

#pragma pack(push, 1)  // Ensure no padding in struct

/**
 * IMU Data Packet
 * Total size: 64 bytes (efficient for UDP)
 */
typedef struct {
    // Header (8 bytes)
    uint16_t magic;           // 0xB0C5 packet identifier
    uint8_t  version;         // Protocol version
    uint8_t  device_hand;     // 0 = left, 1 = right
    uint32_t sequence;        // Packet sequence number

    // Timestamp (8 bytes)
    uint32_t timestamp_ms;    // Milliseconds since boot
    uint32_t timestamp_us;    // Microsecond precision within ms

    // Accelerometer data (12 bytes) - m/s^2
    float accel_x;
    float accel_y;
    float accel_z;

    // Gyroscope data (12 bytes) - rad/s
    float gyro_x;
    float gyro_y;
    float gyro_z;

    // Quaternion orientation (16 bytes) - from BNO085 sensor fusion
    float quat_w;
    float quat_x;
    float quat_y;
    float quat_z;

    // Status (4 bytes)
    uint8_t  accuracy;        // BNO085 calibration accuracy (0-3)
    uint8_t  status;          // Device status flags
    uint16_t checksum;        // Simple checksum for validation

    // Reserved (4 bytes) - for future expansion
    uint8_t  reserved[4];

} IMUPacket;

#pragma pack(pop)

// Status flags
#define STATUS_IMU_OK        0x01
#define STATUS_WIFI_OK       0x02
#define STATUS_CALIBRATED    0x04
#define STATUS_MOTION        0x08
#define STATUS_ERROR         0x80

// Accuracy levels (from BNO085)
#define ACCURACY_UNRELIABLE  0
#define ACCURACY_LOW         1
#define ACCURACY_MEDIUM      2
#define ACCURACY_HIGH        3

// =============================================================================
// Calibration Data Structure
// =============================================================================

typedef struct {
    // Accelerometer offsets
    float accel_offset_x;
    float accel_offset_y;
    float accel_offset_z;

    // Gyroscope offsets
    float gyro_offset_x;
    float gyro_offset_y;
    float gyro_offset_z;

    // Calibration state
    bool is_calibrated;
    uint32_t calibration_timestamp;

} CalibrationData;

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Calculate checksum for packet validation
 */
inline uint16_t calculate_checksum(const IMUPacket* packet) {
    const uint8_t* data = (const uint8_t*)packet;
    uint16_t sum = 0;
    // Sum all bytes except the checksum field itself
    for (size_t i = 0; i < offsetof(IMUPacket, checksum); i++) {
        sum += data[i];
    }
    return sum;
}

/**
 * Validate packet checksum
 */
inline bool validate_checksum(const IMUPacket* packet) {
    return packet->checksum == calculate_checksum(packet);
}

/**
 * Initialize packet with default values
 */
inline void init_packet(IMUPacket* packet, uint8_t hand) {
    packet->magic = 0xB0C5;
    packet->version = 1;
    packet->device_hand = hand;
    packet->sequence = 0;
    packet->timestamp_ms = 0;
    packet->timestamp_us = 0;
    packet->accel_x = 0;
    packet->accel_y = 0;
    packet->accel_z = 0;
    packet->gyro_x = 0;
    packet->gyro_y = 0;
    packet->gyro_z = 0;
    packet->quat_w = 1.0f;  // Identity quaternion
    packet->quat_x = 0;
    packet->quat_y = 0;
    packet->quat_z = 0;
    packet->accuracy = 0;
    packet->status = 0;
    packet->checksum = 0;
    for (int i = 0; i < 4; i++) packet->reserved[i] = 0;
}

#endif // IMU_DATA_H
