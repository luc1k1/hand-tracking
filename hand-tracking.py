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

# Color selection for drawing
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (255, 255, 255), (0, 0, 0)]
color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "White", "Black"]
selected_color = (255, 255, 255)  # Default color (White)

# Function to draw the color selector on the screen
def draw_color_selector(frame):
    global selected_color
    x_start, y_start = 10, 10
    box_size = 40

    cv2.putText(frame, "Choose color:", (x_start, y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    for i, color in enumerate(colors):
        x, y = x_start, y_start + i * (box_size + 10)
        cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, -1)
        cv2.putText(frame, color_names[i], (x + box_size + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return (x_start, y_start, box_size)

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # Vectors from p1 to p2 and p2 to p3
    vector1 = (x2 - x1, y2 - y1)
    vector2 = (x3 - x2, y3 - y2)

    # Dot product and magnitudes for angle calculation
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
    magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

    if magnitude1 * magnitude2 == 0:
        return 180

    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)  # Clamp to avoid numerical errors
    return math.degrees(math.acos(cos_angle))

# Function to check if a finger is bent based on its angles
def check_finger_bend(landmarks, finger_indices):
    p1 = landmarks[finger_indices[0]]
    p2 = landmarks[finger_indices[1]]
    p3 = landmarks[finger_indices[2]]
    p4 = landmarks[finger_indices[3]]

    # Calculate angles between joints of the finger
    angle1 = calculate_angle((p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y))
    angle2 = calculate_angle((p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y))

    # If both angles are less than 85 degrees, consider the finger bent
    return angle1 < 85 and angle2 < 85

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe requires RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(frame_rgb)

    # Check if hands are detected and tracking is enabled
    if results.multi_hand_landmarks and tracking_enabled:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Indices for each finger's key points
            thumb_indices = [mp_hands.HandLandmark.THUMB_CMC, mp_hands.HandLandmark.THUMB_MCP,
                             mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_TIP]
            index_finger_indices = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                                    mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_indices = [mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                                     mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_indices = [mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP,
                                   mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_finger_indices = [mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP,
                                    mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_TIP]

            # Check for bent fingers and display messages
            if check_finger_bend(hand_landmarks.landmark, thumb_indices):
                cv2.putText(frame, "Thumb bent", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if check_finger_bend(hand_landmarks.landmark, index_finger_indices):
                cv2.putText(frame, "Index Finger bent", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if check_finger_bend(hand_landmarks.landmark, middle_finger_indices):
                cv2.putText(frame, "Middle Finger bent", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if check_finger_bend(hand_landmarks.landmark, ring_finger_indices):
                cv2.putText(frame, "Ring Finger bent", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if check_finger_bend(hand_landmarks.landmark, pinky_finger_indices):
                cv2.putText(frame, "Pinky Finger bent", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display tracking status
    status_text = "Tracking is ON" if tracking_enabled else "Tracking is OFF"
    cv2.putText(frame, status_text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Show instructions
    cv2.putText(frame, "Press 'T' to toggle tracking", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Hand Tracking with Finger Bend Detection', frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('t'):
        tracking_enabled = not tracking_enabled

# Release resources
cap.release()
cv2.destroyAllWindows()
