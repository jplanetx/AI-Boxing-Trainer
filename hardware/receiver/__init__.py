"""
AI Boxing Trainer - Hardware Receiver Module

Provides IMU data reception from ESP32 sensors.

Usage:
    from hardware.receiver import IMUReceiver, IMUData

    receiver = IMUReceiver(port=5555)
    receiver.start()

    left, right = receiver.get_latest()
"""

from .imu_receiver import IMUReceiver, IMUData, PunchDetector, PunchEvent

__all__ = ['IMUReceiver', 'IMUData', 'PunchDetector', 'PunchEvent']
