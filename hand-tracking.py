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
    "middle": (0, 0, 255),  # Blue
    "ring": (255, 255, 0),  # Yellow
    "pinky": (255, 0, 255)  # Magenta
}

# Available colors for drawing
drawing_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (255, 255, 255), (0, 0, 0)]
selected_color = (255, 255, 255)  # Default color (White)

drawing = False

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

ret, frame = cap.read()
frame_height, frame_width, _ = frame.shape
canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

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

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

            if drawing:
                cv2.circle(canvas, (x, y), 5, selected_color, -1)

            if index_tip.y < 0.1:
                selected_color = drawing_colors[int(index_tip.x * len(drawing_colors)) % len(drawing_colors)]
                cv2.putText(frame, "Color changed!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, selected_color, 2)

    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow('Hand Tracking with Drawing', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled
    elif key == ord('d'):
        drawing = not drawing
    elif key == ord('c'):
        canvas[:] = 0

cap.release()
cv2.destroyAllWindows()
