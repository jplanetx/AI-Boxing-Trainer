#!/usr/bin/env python3
"""
AI Boxing Trainer - PC Test Listener

Receives and displays IMU data from ESP32 sensors via UDP.
Use this script to verify sensor connections and data streaming.

Usage:
    python pc_test_listener.py                    # Default settings
    python pc_test_listener.py --port 5555        # Custom port
    python pc_test_listener.py --json             # JSON output mode
    python pc_test_listener.py --log data.csv     # Log to file

Author: AI Boxing Trainer Team
License: MIT
"""

import socket
import struct
import argparse
import json
import csv
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Any
import sys

# =============================================================================
# Protocol Constants (must match firmware)
# =============================================================================

PACKET_MAGIC = 0xB0C5
PACKET_SIZE = 64  # bytes
PROTOCOL_VERSION = 1

# Status flags
STATUS_IMU_OK = 0x01
STATUS_WIFI_OK = 0x02
STATUS_CALIBRATED = 0x04
STATUS_MOTION = 0x08
STATUS_ERROR = 0x80

# Hand identification
HAND_LEFT = 0
HAND_RIGHT = 1
HAND_NAMES = {HAND_LEFT: "left_hand", HAND_RIGHT: "right_hand"}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class IMUPacket:
    """Parsed IMU packet from ESP32 sensor."""
    magic: int
    version: int
    device_hand: int
    sequence: int
    timestamp_ms: int
    timestamp_us: int
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float
    accuracy: int
    status: int
    checksum: int

    @property
    def device_name(self) -> str:
        return HAND_NAMES.get(self.device_hand, "unknown")

    @property
    def is_valid(self) -> bool:
        return self.magic == PACKET_MAGIC and self.version == PROTOCOL_VERSION

    @property
    def has_motion(self) -> bool:
        return bool(self.status & STATUS_MOTION)

    @property
    def is_calibrated(self) -> bool:
        return bool(self.status & STATUS_CALIBRATED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "device": self.device_name,
            "timestamp": self.timestamp_ms + self.timestamp_us / 1000,
            "sequence": self.sequence,
            "accel": [self.accel_x, self.accel_y, self.accel_z],
            "gyro": [self.gyro_x, self.gyro_y, self.gyro_z],
            "quat": [self.quat_w, self.quat_x, self.quat_y, self.quat_z],
            "accuracy": self.accuracy,
            "status": {
                "motion": self.has_motion,
                "calibrated": self.is_calibrated,
                "raw": self.status
            }
        }


def parse_packet(data: bytes) -> Optional[IMUPacket]:
    """Parse raw UDP packet into IMUPacket structure."""
    if len(data) != PACKET_SIZE:
        return None

    try:
        # Unpack binary data according to packet structure
        # Format: HBBIIIfffffffffBBH4s
        # H = uint16 (magic)
        # B = uint8 (version, hand)
        # I = uint32 (sequence, timestamps)
        # f = float (sensor data)
        # B = uint8 (accuracy, status)
        # H = uint16 (checksum)
        # 4s = 4 bytes reserved

        unpacked = struct.unpack('<HBBIIIfffffffffffffffffBBH4s', data)

        return IMUPacket(
            magic=unpacked[0],
            version=unpacked[1],
            device_hand=unpacked[2],
            sequence=unpacked[3],
            timestamp_ms=unpacked[4],
            timestamp_us=unpacked[5],
            accel_x=unpacked[6],
            accel_y=unpacked[7],
            accel_z=unpacked[8],
            gyro_x=unpacked[9],
            gyro_y=unpacked[10],
            gyro_z=unpacked[11],
            quat_w=unpacked[12],
            quat_x=unpacked[13],
            quat_y=unpacked[14],
            quat_z=unpacked[15],
            accuracy=unpacked[16],
            status=unpacked[17],
            checksum=unpacked[18]
        )
    except struct.error as e:
        print(f"Failed to parse packet: {e}")
        return None


# =============================================================================
# Display Functions
# =============================================================================

def print_header():
    """Print startup header."""
    print("\n" + "=" * 60)
    print("  AI Boxing Trainer - IMU Test Listener")
    print("=" * 60)
    print()


def print_packet_table(packet: IMUPacket, packet_count: Dict[int, int]):
    """Print packet data in table format."""
    hand = "LEFT " if packet.device_hand == HAND_LEFT else "RIGHT"
    motion = "MOTION" if packet.has_motion else "      "
    cal = "CAL" if packet.is_calibrated else "   "

    # Clear line and print
    print(f"\r[{hand}] #{packet.sequence:6d} | "
          f"Accel: [{packet.accel_x:7.2f}, {packet.accel_y:7.2f}, {packet.accel_z:7.2f}] | "
          f"Gyro: [{packet.gyro_x:6.2f}, {packet.gyro_y:6.2f}, {packet.gyro_z:6.2f}] | "
          f"{motion} {cal}", end="")


def print_packet_verbose(packet: IMUPacket):
    """Print detailed packet information."""
    print("-" * 60)
    print(f"Device: {packet.device_name} (hand={packet.device_hand})")
    print(f"Sequence: {packet.sequence}")
    print(f"Timestamp: {packet.timestamp_ms}.{packet.timestamp_us:03d} ms")
    print(f"Accelerometer: [{packet.accel_x:.4f}, {packet.accel_y:.4f}, {packet.accel_z:.4f}] m/s^2")
    print(f"Gyroscope:     [{packet.gyro_x:.4f}, {packet.gyro_y:.4f}, {packet.gyro_z:.4f}] rad/s")
    print(f"Quaternion:    [{packet.quat_w:.4f}, {packet.quat_x:.4f}, {packet.quat_y:.4f}, {packet.quat_z:.4f}]")
    print(f"Accuracy: {packet.accuracy} | Status: 0x{packet.status:02X}")
    print(f"  - Motion: {packet.has_motion}")
    print(f"  - Calibrated: {packet.is_calibrated}")


def print_packet_json(packet: IMUPacket):
    """Print packet as JSON."""
    print(json.dumps(packet.to_dict()))


def print_stats(stats: Dict[str, Any]):
    """Print session statistics."""
    print("\n" + "=" * 60)
    print("  Session Statistics")
    print("=" * 60)
    print(f"  Duration: {stats['duration']:.1f} seconds")
    print(f"  Total packets: {stats['total_packets']}")
    print(f"  Left hand packets: {stats['left_packets']}")
    print(f"  Right hand packets: {stats['right_packets']}")
    print(f"  Invalid packets: {stats['invalid_packets']}")
    print(f"  Packet rate: {stats['packet_rate']:.1f} Hz")
    print("=" * 60)
    print()


# =============================================================================
# Main Listener
# =============================================================================

class IMUListener:
    """UDP listener for IMU sensor data."""

    def __init__(self, port: int = 5555, host: str = '0.0.0.0'):
        self.port = port
        self.host = host
        self.socket = None
        self.running = False

        # Statistics
        self.start_time = None
        self.packet_counts = {HAND_LEFT: 0, HAND_RIGHT: 0}
        self.invalid_count = 0

        # Callbacks
        self.on_packet = None

    def start(self):
        """Start listening for UDP packets."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.settimeout(1.0)  # 1 second timeout for graceful shutdown

        self.running = True
        self.start_time = time.time()

        print(f"Listening on {self.host}:{self.port}")
        print("Waiting for IMU packets... (Ctrl+C to stop)")
        print()

    def stop(self):
        """Stop listening."""
        self.running = False
        if self.socket:
            self.socket.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        duration = time.time() - self.start_time if self.start_time else 0
        total = self.packet_counts[HAND_LEFT] + self.packet_counts[HAND_RIGHT]

        return {
            'duration': duration,
            'total_packets': total,
            'left_packets': self.packet_counts[HAND_LEFT],
            'right_packets': self.packet_counts[HAND_RIGHT],
            'invalid_packets': self.invalid_count,
            'packet_rate': total / duration if duration > 0 else 0
        }

    def receive_one(self) -> Optional[IMUPacket]:
        """Receive and parse a single packet."""
        try:
            data, addr = self.socket.recvfrom(PACKET_SIZE + 16)  # Extra buffer
            packet = parse_packet(data)

            if packet and packet.is_valid:
                self.packet_counts[packet.device_hand] += 1
                return packet
            else:
                self.invalid_count += 1
                return None

        except socket.timeout:
            return None
        except Exception as e:
            print(f"\nError receiving packet: {e}")
            return None

    def run(self, callback=None):
        """Run the listener loop."""
        self.start()

        try:
            while self.running:
                packet = self.receive_one()
                if packet:
                    if callback:
                        callback(packet)
                    elif self.on_packet:
                        self.on_packet(packet)

        except KeyboardInterrupt:
            print("\n\nStopping listener...")

        finally:
            self.stop()
            return self.get_stats()


# =============================================================================
# CSV Logger
# =============================================================================

class CSVLogger:
    """Log IMU data to CSV file."""

    def __init__(self, filename: str):
        self.filename = filename
        self.file = None
        self.writer = None

    def start(self):
        """Open file and write header."""
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            'timestamp', 'device', 'sequence',
            'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'quat_w', 'quat_x', 'quat_y', 'quat_z',
            'accuracy', 'motion', 'calibrated'
        ])

    def log(self, packet: IMUPacket):
        """Log a packet to CSV."""
        if self.writer:
            self.writer.writerow([
                datetime.now().isoformat(),
                packet.device_name,
                packet.sequence,
                packet.accel_x, packet.accel_y, packet.accel_z,
                packet.gyro_x, packet.gyro_y, packet.gyro_z,
                packet.quat_w, packet.quat_x, packet.quat_y, packet.quat_z,
                packet.accuracy,
                int(packet.has_motion),
                int(packet.is_calibrated)
            ])

    def stop(self):
        """Close file."""
        if self.file:
            self.file.close()


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='AI Boxing Trainer - IMU Test Listener',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pc_test_listener.py                 # Listen with default settings
  python pc_test_listener.py --port 5555     # Custom port
  python pc_test_listener.py --verbose       # Detailed output
  python pc_test_listener.py --json          # JSON output for integration
  python pc_test_listener.py --log data.csv  # Log data to CSV file
        """
    )

    parser.add_argument('--port', type=int, default=5555,
                        help='UDP port to listen on (default: 5555)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to bind (default: 0.0.0.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output mode')
    parser.add_argument('--json', '-j', action='store_true',
                        help='JSON output mode')
    parser.add_argument('--log', '-l', type=str, metavar='FILE',
                        help='Log data to CSV file')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode (no console output)')

    args = parser.parse_args()

    print_header()

    # Setup logger if requested
    logger = None
    if args.log:
        logger = CSVLogger(args.log)
        logger.start()
        print(f"Logging to: {args.log}")

    # Create listener
    listener = IMUListener(port=args.port, host=args.host)

    # Define packet handler
    def handle_packet(packet: IMUPacket):
        if not args.quiet:
            if args.json:
                print_packet_json(packet)
            elif args.verbose:
                print_packet_verbose(packet)
            else:
                print_packet_table(packet, listener.packet_counts)

        if logger:
            logger.log(packet)

    # Run listener
    stats = listener.run(callback=handle_packet)

    # Cleanup
    if logger:
        logger.stop()
        print(f"Data saved to: {args.log}")

    # Print statistics
    print_stats(stats)


if __name__ == '__main__':
    main()
