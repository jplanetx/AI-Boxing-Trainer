import cv2
import numpy as np
import time

def test_camera_settings():
    """Test different camera settings to find optimal configuration"""
    print("Testing camera settings...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Test different exposure settings
    exposure_values = [-11, -8, -6, -4, -2, 0]
    brightness_values = [0, 25, 50, 75]
    
    print(f"Original settings:")
    print(f"Exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Auto Exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)}")
    
    for exposure in exposure_values:
        for brightness in brightness_values:
            # Reset settings
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual mode
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
            
            # Give camera time to adjust
            time.sleep(0.5)
            
            # Capture frame
            ret, frame = cap.read()
            if ret:
                # Calculate average brightness
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                avg_brightness = np.mean(gray)
                
                print(f"Exposure: {exposure}, Brightness: {brightness}, Avg Frame Brightness: {avg_brightness:.1f}")
                
                # Show frame for visual inspection
                cv2.putText(frame, f"Exp:{exposure} Bright:{brightness} Avg:{avg_brightness:.1f}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Camera Test', frame)
                
                # Wait briefly
                key = cv2.waitKey(1000)  # 1 second delay
                if key == ord('q'):
                    break
            
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test complete")

if __name__ == "__main__":
    test_camera_settings()