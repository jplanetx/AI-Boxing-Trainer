import cv2
import mediapipe as mp
import numpy as np
import time

# --- FUNCTIONS ---
def calculate_angle(a,b,c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle 

# --- INITIALIZATIONS ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# --- TRACKING VARIABLES ---
# More robust state machine: 'guard' -> 'punching' -> 'returning'
punch_stages = {'left': 'guard', 'right': 'guard'} 
punch_counts = {'left': 0, 'right': 0}
last_punch_time = {'left': 0, 'right': 0}
punch_scores = {'left': 0, 'right': 0}

# --- MAIN APP LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        for arm in ['left', 'right']:
            # Define landmarks
            shoulder_lm = mp_pose.PoseLandmark[f'{arm.upper()}_SHOULDER']
            elbow_lm = mp_pose.PoseLandmark[f'{arm.upper()}_ELBOW']
            wrist_lm = mp_pose.PoseLandmark[f'{arm.upper()}_WRIST']
            hip_lm = mp_pose.PoseLandmark[f'{arm.upper()}_HIP']

            # Get coordinates
            shoulder = [landmarks[shoulder_lm.value].x, landmarks[shoulder_lm.value].y]
            elbow = [landmarks[elbow_lm.value].x, landmarks[elbow_lm.value].y]
            wrist = [landmarks[wrist_lm.value].x, landmarks[wrist_lm.value].y]
            hip = [landmarks[hip_lm.value].x, landmarks[hip_lm.value].y]
            
            # --- CALCULATE ANGLES ---
            elbow_angle = calculate_angle(shoulder, elbow, wrist)
            shoulder_angle = calculate_angle(hip, shoulder, elbow)

            # --- NEW MULTI-CRITERIA PUNCH LOGIC ---
            
            # 1. CHECK FOR GUARD POSITION
            # Is the arm flexed and close to the body?
            if elbow_angle < 60 and shoulder_angle < 45:
                punch_stages[arm] = 'guard'

            # 2. DETECT THE PUNCH
            # Did the arm just extend from a guard position?
            if punch_stages[arm] == 'guard' and elbow_angle > 160:
                punch_stages[arm] = 'punching'
                punch_counts[arm] += 1
                last_punch_time[arm] = time.time() # Start timer

            # 3. CALCULATE SCORE ON RETURN
            # Did the arm just return to guard after a punch?
            if punch_stages[arm] == 'punching' and elbow_angle < 60:
                punch_stages[arm] = 'guard' # Reset state
                duration = time.time() - last_punch_time[arm]
                if duration > 0:
                    punch_scores[arm] = (1 / duration) * 100 # Higher score for faster punch
                
    except:
        pass # Ignore frames where landmarks aren't detected
    
    # --- UI DISPLAY ---
    cv2.rectangle(image, (0,0), (480,80), (245,117,16), -1)
    
    # Left Arm Stats
    cv2.putText(image, 'L PUNCH', (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(punch_counts['left']), (25,65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, 'L SCORE', (150,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(int(punch_scores['left'])), (160,65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

    # Right Arm Stats
    cv2.putText(image, 'R PUNCH', (285,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(punch_counts['right']), (295,65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, 'R SCORE', (420,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(int(punch_scores['right'])), (430,65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
    cv2.imshow('AI Boxing Trainer', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()