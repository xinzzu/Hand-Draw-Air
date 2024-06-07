import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Drawing variables
ix, iy = 200, 200
mask = None
color = (0, 255, 0)
c = 0
drawing = False
moving = False
hand_info = ""

# For smoothing the drawing
previous_points = deque(maxlen=5)

cv2.namedWindow("draw")

cap = cv2.VideoCapture(0)

# Initial setup
while True:
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    cv2.imshow("draw", frm)
    if cv2.waitKey(1) == 27:
        old_gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(frm)
        break

cv2.destroyAllWindows()

def moving_average(points):
    if len(points) == 0:
        return None
    avg_x = int(sum([p[0] for p in points]) / len(points))
    avg_y = int(sum([p[1] for p in points]) / len(points))
    return (avg_x, avg_y)

# Main loop
while True:
    _, new_frm = cap.read()
    new_frm = cv2.flip(new_frm, 1)
    rgb_frame = cv2.cvtColor(new_frm, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hands
    result = hands.process(rgb_frame)
    
    hand_labels = []

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            # Get the hand label (left or right)
            hand_label = result.multi_handedness[idx].classification[0].label
            hand_labels.append(hand_label)

            # Get the tips of the index finger, middle finger, and all five fingertips
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

            # Convert normalized coordinates to image coordinates
            index_x = int(index_tip.x * new_frm.shape[1])
            index_y = int(index_tip.y * new_frm.shape[0])
            middle_x = int(middle_tip.x * new_frm.shape[1])
            middle_y = int(middle_tip.y * new_frm.shape[0])
            thumb_x = int(thumb_tip.x * new_frm.shape[1])
            thumb_y = int(thumb_tip.y * new_frm.shape[0])
            ring_x = int(ring_tip.x * new_frm.shape[1])
            ring_y = int(ring_tip.y * new_frm.shape[0])
            pinky_x = int(pinky_tip.x * new_frm.shape[1])
            pinky_y = int(pinky_tip.y * new_frm.shape[0])

            # Check if five fingers are up for erasing
            if (abs(index_x - middle_x) < 40 and abs(index_y - middle_y) < 40 and
                abs(index_x - thumb_x) < 40 and abs(index_y - thumb_y) < 40 and
                abs(index_x - ring_x) < 40 and abs(index_y - ring_y) < 40 and
                abs(index_x - pinky_x) < 40 and abs(index_y - pinky_y) < 40):
                cv2.circle(mask, (index_x, index_y), 50, (0, 0, 0), -1)
                drawing = False
                moving = False
            # Check if two fingers (index and middle) are up for moving
            elif abs(index_x - middle_x) < 40 and abs(index_y - middle_y) < 40:
                drawing = False
                moving = True
            # Otherwise, use the index finger for drawing
            else:
                drawing = True
                moving = False

            # Add the current point to the deque for smoothing
            previous_points.append((index_x, index_y))

            # Get the smoothed point
            avg_point = moving_average(previous_points)

            if avg_point and drawing:
                cv2.line(mask, (ix, iy), avg_point, color, 8)
                ix, iy = avg_point
            elif not drawing:
                ix, iy = index_x, index_y  # Update ix, iy to the latest point if not drawing

            # Draw hand landmarks
            mp_drawing.draw_landmarks(new_frm, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display hand information
    if len(hand_labels) == 1:
        hand_info = f"{hand_labels[0]} hand detected"
    elif len(hand_labels) == 2:
        hand_info = "Both hands detected"
    else:
        hand_info = ""

    if hand_info:
        cv2.putText(new_frm, hand_info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    key = cv2.waitKey(1)

    if key == ord('e'):
        mask = np.zeros_like(new_frm)
        drawing = False
    elif key == ord('c'):
        color = (0, 0, 0)
        lst = list(color)
        c += 1
        lst[c % 3] = 255
        color = tuple(lst)
    elif key == ord('g'):
        drawing = not drawing  # Toggle drawing mode

    new_frm = cv2.addWeighted(new_frm, 0.8, mask, 0.2, 0.1)
    
    cv2.imshow("draw", new_frm)
    cv2.imshow("drawing", mask)
    
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
hands.close()
