import cv2
import math
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Toggle for enabling/disabling hand tracking
tracking_enabled = True

# Colors for each finger
finger_colors = {
    "thumb": (255, 0, 0),  # Red
    "index": (0, 255, 0),  # Green
    "middle": (0, 0, 255), # Blue
    "ring": (255, 255, 0), # Yellow
    "pinky": (255, 0, 255) # Magenta
}

# Define finger landmarks
finger_indices = {
    "thumb": [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP,
               mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP],
    "index": [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
               mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP],
    "middle": [mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
    "ring": [mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP,
              mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP],
    "pinky": [mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP,
               mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP]
}

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    vector1 = (x2 - x1, y2 - y1)
    vector2 = (x3 - x2, y3 - y2)

    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    if magnitude1 * magnitude2 == 0:
        return 180

    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

# Function to check if a finger is bent
def check_finger_bend(landmarks, indices):
    points = [landmarks[i] for i in indices]
    angles = [calculate_angle((points[i].x, points[i].y), (points[i+1].x, points[i+1].y), (points[i+2].x, points[i+2].y))
              for i in range(2)]
    return all(angle < 85 for angle in angles)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks and tracking_enabled:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for finger, indices in finger_indices.items():
                if check_finger_bend(hand_landmarks.landmark, indices):
                    cv2.putText(frame, f"{finger.capitalize()} bent", (50, 50 + list(finger_indices.keys()).index(finger) * 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "Press 'T' to toggle tracking", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Tracking is ON" if tracking_enabled else "Tracking is OFF", (50, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Hand Tracking with Finger Bend Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled

cap.release()
cv2.destroyAllWindows()
