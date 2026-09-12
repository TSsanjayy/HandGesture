import math
import time
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# min_detection_confidence=0.3 allows instant detection of ANY gesture (fist, pointing, sideways)
# model_complexity=1 provides high-precision 3D keypoints from all camera angles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)

# Finger joint sets: (MCP, PIP, TIP)
FINGER_JOINTS = [
    (5, 6, 8),     # Index
    (9, 10, 12),   # Middle
    (13, 14, 16),  # Ring
    (17, 18, 20)   # Pinky
]

TIP_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
TIP_IDS = [4, 8, 12, 16, 20]


def get_3d_dist(p1, p2):
    """Calculates 3D Euclidean distance between two landmarks."""
    return math.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2 +
        (p1.z - p2.z) ** 2
    )


def get_joint_angle(p1, p2, p3):
    """
    Calculates the 3D intrinsic angle (in degrees) at joint p2 between p1-p2 and p3-p2.
    ~180 deg = extended/straight.
    <125 deg = bent/curled.
    This calculation is 100% rotation invariant.
    """
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 == 0 or n2 == 0:
        return 0.0

    dot = np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def fingers_up(hand_landmarks):
    landmarks = hand_landmarks.landmark
    fingers = []

    # 1. THUMB CHECK (3D Joint Angle + distance relative to Pinky MCP 17)
    thumb_angle = get_joint_angle(landmarks[2], landmarks[3], landmarks[4])
    d_tip_pinky = get_3d_dist(landmarks[4], landmarks[17])
    d_mcp_pinky = get_3d_dist(landmarks[2], landmarks[17])

    if thumb_angle > 140 and d_tip_pinky > d_mcp_pinky:
        fingers.append(1)  # Thumb open
    else:
        fingers.append(0)  # Thumb closed

    # 2. FOUR FINGERS CHECK (Pure 3D Intrinsic Joint Angle)
    for mcp_id, pip_id, tip_id in FINGER_JOINTS:
        angle = get_joint_angle(landmarks[mcp_id], landmarks[pip_id], landmarks[tip_id])

        if angle > 145:
            fingers.append(1)  # Straight / Extended
        elif angle < 125:
            fingers.append(0)  # Bent / Curled
        else:
            # Fallback 3D distance check for intermediate angles
            d_tip_wrist = get_3d_dist(landmarks[tip_id], landmarks[0])
            d_pip_wrist = get_3d_dist(landmarks[pip_id], landmarks[0])
            fingers.append(1 if d_tip_wrist > d_pip_wrist else 0)

    return fingers


def classify_gesture(fingers, landmarks):
    # 1. Special Check: OK Sign (Thumb Tip & Index Tip touching)
    d_thumb_index = get_3d_dist(landmarks[4], landmarks[8])
    d_palm = get_3d_dist(landmarks[0], landmarks[9])
    if d_thumb_index < 0.35 * d_palm and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
        return "OK Sign"

    # 2. Thumbs Up vs Thumbs Down Check
    if fingers == [1, 0, 0, 0, 0]:
        if landmarks[4].y > landmarks[2].y:
            return "Thumbs Down"
        else:
            return "Thumbs Up"

    # 3. Pattern Matching
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"

    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"

    elif fingers == [0, 1, 0, 0, 0]:
        return "Pointing"

    elif fingers in ([0, 1, 1, 0, 0], [1, 1, 1, 0, 0]):
        return "Peace"

    elif fingers == [0, 1, 0, 0, 1]:
        return "Rock On"

    elif fingers == [1, 1, 0, 0, 1]:
        return "I Love You"

    elif fingers == [1, 0, 0, 0, 1]:
        return "Call Me / Shaka"

    elif fingers == [1, 1, 0, 0, 0]:
        return "Gun / Pistol"

    elif fingers == [0, 1, 1, 1, 0]:
        return "Three"

    elif fingers == [0, 1, 1, 1, 1]:
        return "Four"

    elif fingers == [0, 0, 1, 0, 0]:
        return "Middle Finger"

    else:
        return "Unknown"


# Start camera (CAP_DSHOW prevents camera freeze on Windows)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read camera frame")
        break

    h, w, _ = frame.shape

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Memory optimization for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    results = hands.process(rgb_frame)

    rgb_frame.flags.writeable = True

    gesture_text = "No hand detected"
    finger_state_str = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            landmarks = hand_landmarks.landmark
            fingers = fingers_up(hand_landmarks)
            gesture_text = classify_gesture(fingers, landmarks)
            finger_state_str = f"Fingers: {fingers}"

            # Draw Wrist Coordinate
            wx, wy = int(landmarks[0].x * w), int(landmarks[0].y * h)
            cv2.putText(frame, f"Wrist: ({wx},{wy})", (wx + 10, wy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            # Draw HUD panel for fingertip coordinates in top-right corner
            cv2.rectangle(frame, (w - 210, 10), (w - 10, 150), (0, 0, 0), -1)
            cv2.putText(frame, "Coordinates (X,Y):", (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            for idx, (name, tip_id) in enumerate(zip(TIP_NAMES, TIP_IDS)):
                cx, cy = int(landmarks[tip_id].x * w), int(landmarks[tip_id].y * h)

                # Draw coordinate tag next to each fingertip
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx},{cy})", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

                # Draw in HUD panel
                cv2.putText(frame, f"{name}: ({cx}, {cy})", (w - 200, 52 + idx * 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # FPS Calculation
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # Display Gesture & Debug Info
    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
    if finger_state_str:
        cv2.putText(frame, finger_state_str, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame, f"FPS: {fps}", (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)

    cv2.imshow("Gesture Recognition", frame)

    # Press 'Q' or close window button (X) to exit cleanly
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty("Gesture Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()