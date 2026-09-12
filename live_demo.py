import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from collections import deque

# -----------------------------
# Load trained engineered model
# -----------------------------
model = joblib.load("model_engineered.joblib")

# -----------------------------
# MediaPipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# Feature extraction
# SAME 20 features used in training
# -----------------------------
def distance(a, b):
    return np.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def extract_engineered_features(landmarks):
    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    wrist = points[0]

    # Normalize using wrist -> middle MCP distance
    scale = distance(points[0], points[9])

    if scale < 1e-6:
        scale = 1e-6

    # 5 fingertips
    fingertips = [4, 8, 12, 16, 20]

    # 5 MCP joints
    mcps = [5, 9, 13, 17, 1]

    features = []

    # 1. Fingertip -> wrist distances (5)
    for idx in fingertips:
        features.append(distance(points[idx], wrist) / scale)

    # 2. Fingertip -> MCP distances (5)
    for tip, mcp in zip(fingertips, mcps):
        features.append(distance(points[tip], points[mcp]) / scale)

    # 3. Pairwise fingertip distances (10)
    for i in range(len(fingertips)):
        for j in range(i + 1, len(fingertips)):
            features.append(
                distance(points[fingertips[i]], points[fingertips[j]]) / scale
            )

    return np.array(features, dtype=np.float32)


# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -----------------------------
# Prediction smoothing
# -----------------------------
prediction_history = deque(maxlen=7)

# FPS measurement
frame_count = 0
fps_start = time.perf_counter()
fps = 0.0

# Classifier timing
classifier_times = []

print("Live gesture recognition started.")
print("Press Q to quit.")


# -----------------------------
# Main loop
# -----------------------------
while True:

    loop_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe processing
    result = hands.process(rgb)

    gesture = "No hand"

    if result.multi_hand_landmarks:

        hand_landmarks = result.multi_hand_landmarks[0]

        # Draw hand landmarks
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract engineered features
        features = extract_engineered_features(
            hand_landmarks.landmark
        )

        # -----------------------------
        # Classifier timing
        # -----------------------------
        classifier_start = time.perf_counter()

        prediction = model.predict(
            features.reshape(1, -1)
        )[0]

        classifier_end = time.perf_counter()

        classifier_time = (
            classifier_end - classifier_start
        ) * 1000

        classifier_times.append(classifier_time)

        # Smooth predictions
        prediction_history.append(prediction)

        gesture = max(
            set(prediction_history),
            key=prediction_history.count
        )

    # -----------------------------
    # FPS calculation
    # -----------------------------
    frame_count += 1

    elapsed = time.perf_counter() - fps_start

    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        fps_start = time.perf_counter()

    # -----------------------------
    # Display
    # -----------------------------
    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Pipeline FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    if classifier_times:
        avg_classifier_time = np.mean(classifier_times)

        cv2.putText(
            frame,
            f"Classifier: {avg_classifier_time:.2f} ms",
            (20, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow(
        "ML Hand Gesture Recognition",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n==============================")
print("LIVE DEMO RESULTS")
print("==============================")

print(f"Average Pipeline FPS: {fps:.2f}")

if classifier_times:
    print(
        f"Average Classifier Time: "
        f"{np.mean(classifier_times):.4f} ms"
    )

print("==============================")