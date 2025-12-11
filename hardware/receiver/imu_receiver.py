#!/usr/bin/env python3
"""
AI Boxing Trainer - IMU Receiver Module

Integration module for receiving IMU data from ESP32 sensors
and feeding it into the Boxing Trainer application.

This module can be imported into the main application to provide
real-time IMU data alongside computer vision pose tracking.

Usage:
    from hardware.receiver.imu_receiver import IMUReceiver, IMUData

    receiver = IMUReceiver(port=5555)
    receiver.start()

    # In your main loop:
    left_data, right_data = receiver.get_latest()
    if left_data:
        print(f"Left hand accel: {left_data.accel}")

Author: AI Boxing Trainer Team
License: MIT
"""

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Callable
from collections import deque
import numpy as np

# =============================================================================
# Constants
# =============================================================================

PACKET_MAGIC = 0xB0C5
PACKET_SIZE = 64
HAND_LEFT = 0
HAND_RIGHT = 1


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class IMUData:
    """Processed IMU data for application use."""

    device: str  # "left_hand" or "right_hand"
    timestamp: float  # seconds
    sequence: int

    # 3D vectors
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))  # m/s^2
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))   # rad/s
    quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))  # w, x, y, z

    # Status
    accuracy: int = 0
    has_motion: bool = False
    is_calibrated: bool = False

    @property
    def accel_magnitude(self) -> float:
        """Get acceleration magnitude (without gravity ~9.81)."""
        return np.linalg.norm(self.accel) - 9.81

    @property
    def gyro_magnitude(self) -> float:
        """Get angular velocity magnitude."""
        return np.linalg.norm(self.gyro)

    def get_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = self.quat
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])


@dataclass
class PunchEvent:
    """Detected punch event from IMU data."""
    hand: str  # "left" or "right"
    timestamp: float
    peak_accel: float
    peak_gyro: float
    duration_ms: float
    confidence: float


# =============================================================================
# IMU Receiver Class
# =============================================================================

class IMUReceiver:
    """
    Threaded UDP receiver for IMU sensor data.

    Provides non-blocking access to the latest IMU data from both hands.
    Can be integrated with the main Boxing Trainer application.
    """

    def __init__(self, port: int = 5555, host: str = '0.0.0.0',
                 buffer_size: int = 100):
        """
        Initialize IMU receiver.

        Args:
            port: UDP port to listen on
            host: Host address to bind
            buffer_size: Number of packets to buffer for each hand
        """
        self.port = port
        self.host = host
        self.buffer_size = buffer_size

        # Socket
        self.socket = None
        self.running = False
        self.thread = None

        # Data buffers (thread-safe with deque)
        self.buffers: Dict[int, deque] = {
            HAND_LEFT: deque(maxlen=buffer_size),
            HAND_RIGHT: deque(maxlen=buffer_size)
        }

        # Latest data (for quick access)
        self.latest: Dict[int, Optional[IMUData]] = {
            HAND_LEFT: None,
            HAND_RIGHT: None
        }

        # Statistics
        self.packet_counts = {HAND_LEFT: 0, HAND_RIGHT: 0}
        self.start_time = None

        # Callbacks
        self.on_data: Optional[Callable[[IMUData], None]] = None
        self.on_punch: Optional[Callable[[PunchEvent], None]] = None

        # Punch detection state
        self._punch_detector = PunchDetector()

    def start(self) -> bool:
        """Start the receiver thread. Returns True on success."""
        if self.running:
            return True

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(0.1)

            self.running = True
            self.start_time = time.time()

            self.thread = threading.Thread(target=self._receive_loop, daemon=True)
            self.thread.start()

            return True

        except Exception as e:
            print(f"Failed to start IMU receiver: {e}")
            return False

    def stop(self):
        """Stop the receiver thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.socket:
            self.socket.close()

    def get_latest(self) -> Tuple[Optional[IMUData], Optional[IMUData]]:
        """
        Get the latest data from both hands.

        Returns:
            Tuple of (left_hand_data, right_hand_data)
            Either or both may be None if no data received.
        """
        return (self.latest[HAND_LEFT], self.latest[HAND_RIGHT])

    def get_buffer(self, hand: int) -> List[IMUData]:
        """Get buffered data for a specific hand."""
        return list(self.buffers[hand])

    def get_stats(self) -> Dict:
        """Get receiver statistics."""
        duration = time.time() - self.start_time if self.start_time else 0
        total = self.packet_counts[HAND_LEFT] + self.packet_counts[HAND_RIGHT]

        return {
            'duration': duration,
            'total_packets': total,
            'left_packets': self.packet_counts[HAND_LEFT],
            'right_packets': self.packet_counts[HAND_RIGHT],
            'packet_rate': total / duration if duration > 0 else 0,
            'running': self.running
        }

    def _receive_loop(self):
        """Internal receive loop (runs in thread)."""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(PACKET_SIZE + 16)
                imu_data = self._parse_packet(data)

                if imu_data:
                    hand = HAND_LEFT if imu_data.device == "left_hand" else HAND_RIGHT

                    # Update buffers
                    self.buffers[hand].append(imu_data)
                    self.latest[hand] = imu_data
                    self.packet_counts[hand] += 1

                    # Callback
                    if self.on_data:
                        self.on_data(imu_data)

                    # Punch detection
                    punch = self._punch_detector.process(imu_data)
                    if punch and self.on_punch:
                        self.on_punch(punch)

            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")

    def _parse_packet(self, data: bytes) -> Optional[IMUData]:
        """Parse raw packet into IMUData."""
        if len(data) != PACKET_SIZE:
            return None

        try:
            unpacked = struct.unpack('<HBBIIIfffffffffffffffffBBH4s', data)

            magic = unpacked[0]
            if magic != PACKET_MAGIC:
                return None

            hand = unpacked[2]
            device_name = "left_hand" if hand == HAND_LEFT else "right_hand"

            return IMUData(
                device=device_name,
                timestamp=unpacked[4] / 1000.0 + unpacked[5] / 1000000.0,
                sequence=unpacked[3],
                accel=np.array([unpacked[6], unpacked[7], unpacked[8]]),
                gyro=np.array([unpacked[9], unpacked[10], unpacked[11]]),
                quat=np.array([unpacked[12], unpacked[13], unpacked[14], unpacked[15]]),
                accuracy=unpacked[16],
                has_motion=bool(unpacked[17] & 0x08),
                is_calibrated=bool(unpacked[17] & 0x04)
            )

        except struct.error:
            return None


# =============================================================================
# Punch Detection from IMU Data
# =============================================================================

class PunchDetector:
    """
    Detects punches from IMU acceleration and gyroscope data.

    Uses acceleration spikes and angular velocity patterns to identify punches.
    This provides a complement to the vision-based punch detection.
    """

    def __init__(self):
        # Detection thresholds
        self.accel_threshold = 15.0  # m/s^2 above baseline
        self.gyro_threshold = 5.0    # rad/s
        self.cooldown_ms = 200       # minimum time between punches

        # State for each hand
        self.state = {
            'left_hand': {'in_punch': False, 'start_time': 0, 'peak_accel': 0, 'peak_gyro': 0},
            'right_hand': {'in_punch': False, 'start_time': 0, 'peak_accel': 0, 'peak_gyro': 0}
        }
        self.last_punch_time = {'left_hand': 0, 'right_hand': 0}

    def process(self, data: IMUData) -> Optional[PunchEvent]:
        """
        Process IMU data and detect punches.

        Args:
            data: IMUData from sensor

        Returns:
            PunchEvent if punch detected, None otherwise
        """
        state = self.state[data.device]

        # Calculate acceleration magnitude (subtract gravity)
        accel_mag = data.accel_magnitude
        gyro_mag = data.gyro_magnitude

        current_time = data.timestamp * 1000  # Convert to ms

        # Check cooldown
        if current_time - self.last_punch_time[data.device] < self.cooldown_ms:
            return None

        # State machine for punch detection
        if not state['in_punch']:
            # Look for punch start (acceleration spike)
            if accel_mag > self.accel_threshold:
                state['in_punch'] = True
                state['start_time'] = current_time
                state['peak_accel'] = accel_mag
                state['peak_gyro'] = gyro_mag

        else:
            # Update peak values
            state['peak_accel'] = max(state['peak_accel'], accel_mag)
            state['peak_gyro'] = max(state['peak_gyro'], gyro_mag)

            # Check for punch end (acceleration back to baseline)
            if accel_mag < self.accel_threshold * 0.3:
                # Punch completed
                duration = current_time - state['start_time']

                # Validate punch (duration should be reasonable)
                if 50 < duration < 500:  # 50-500ms is reasonable punch duration
                    # Calculate confidence based on acceleration and duration
                    confidence = min(1.0, state['peak_accel'] / 30.0)

                    punch = PunchEvent(
                        hand=data.device.replace('_hand', ''),
                        timestamp=data.timestamp,
                        peak_accel=state['peak_accel'],
                        peak_gyro=state['peak_gyro'],
                        duration_ms=duration,
                        confidence=confidence
                    )

                    self.last_punch_time[data.device] = current_time
                    state['in_punch'] = False

                    return punch

                state['in_punch'] = False

        return None


# =============================================================================
# Example Usage / Test
# =============================================================================

if __name__ == '__main__':
    print("AI Boxing Trainer - IMU Receiver Test")
    print("=" * 50)

    def on_data(data: IMUData):
        print(f"[{data.device}] Accel: {data.accel_magnitude:6.2f} m/s^2 | "
              f"Gyro: {data.gyro_magnitude:5.2f} rad/s | "
              f"Motion: {data.has_motion}")

    def on_punch(punch: PunchEvent):
        print(f"\n*** PUNCH DETECTED: {punch.hand.upper()} ***")
        print(f"    Peak accel: {punch.peak_accel:.1f} m/s^2")
        print(f"    Duration: {punch.duration_ms:.0f} ms")
        print(f"    Confidence: {punch.confidence:.2f}\n")

    receiver = IMUReceiver(port=5555)
    receiver.on_data = on_data
    receiver.on_punch = on_punch

    if receiver.start():
        print(f"Listening on port 5555...")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                time.sleep(1)
                stats = receiver.get_stats()
                if stats['total_packets'] > 0:
                    print(f"\rPackets: L={stats['left_packets']} R={stats['right_packets']} "
                          f"({stats['packet_rate']:.1f} Hz)", end="")
        except KeyboardInterrupt:
            pass
        finally:
            receiver.stop()
            print("\n\nFinal stats:", receiver.get_stats())
    else:
        print("Failed to start receiver")
