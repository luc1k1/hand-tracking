import cv2
import math
import mediapipe as mp
import numpy as np

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
selected_color = (255, 255, 255)  # Default color - White

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

    # Draw color selection bar
    for i, color in enumerate(drawing_colors):
        cv2.rectangle(frame, (i * 50, 0), ((i + 1) * 50, 50), color, -1)

    if results.multi_hand_landmarks and tracking_enabled:
        for hand_landmarks in results.multi_hand_landmarks:
            if not hand_landmarks or len(hand_landmarks.landmark) < 21:
                continue

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

            if drawing:
                cv2.circle(canvas, (x, y), 5, selected_color, -1)

            if y < 50:
                color_index = x // 50
                if 0 <= color_index < len(drawing_colors):
                    selected_color = drawing_colors[color_index]
                    cv2.putText(frame, "Color changed!", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, selected_color, 2)

            # Check bent fingers
            for finger, indices in finger_indices.items():
                try:
                    p1, p2, p3, p4 = [hand_landmarks.landmark[i] for i in indices]
                    angle1 = math.degrees(math.atan2(p2.y - p1.y, p2.x - p1.x) - math.atan2(p3.y - p2.y, p3.x - p2.x))
                    angle2 = math.degrees(math.atan2(p3.y - p2.y, p3.x - p2.x) - math.atan2(p4.y - p3.y, p4.x - p3.x))
                    if abs(angle1) < 85 and abs(angle2) < 85:
                        cv2.putText(frame, f"{finger.capitalize()} bent",
                                    (50, 100 + list(finger_indices.keys()).index(finger) * 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, selected_color, 2)
                except IndexError:
                    continue

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
