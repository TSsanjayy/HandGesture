"""
Live real-time gesture classification demo -- polished UI version.

Feature extraction logic below mirrors features.py exactly (wrist-centered,
hand-size normalized), but implemented per-single-frame instead of batch,
so it can run live without re-processing the whole dataset on import.

Usage:
    python live_demo_ui.py --model engineered
    python live_demo_ui.py --model raw
"""

import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import argparse
from collections import deque

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["raw", "engineered"], default="engineered")
args = parser.parse_args()

clf = joblib.load(f"model_{args.model}.joblib")
class_names = clf.classes_

FINGERTIPS = [4, 8, 12, 16, 20]
MCP_POINTS = [2, 5, 9, 13, 17]


def landmarks_to_array(hand_landmarks):
    """MediaPipe hand_landmarks -> (21, 3) numpy array."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    return pts


def raw_features_single(points):
    """Matches create_raw_features(): wrist-centered, hand-size normalized, flattened."""
    pts = points - points[0:1, :]  # center on wrist
    hand_size = np.linalg.norm(pts[9] - pts[0])
    if hand_size == 0:
        hand_size = 1
    pts = pts / hand_size
    return pts.flatten()


def engineered_features_single(points):
    """Matches create_engineered_features(): 20 distance-based features, hand-size normalized."""
    wrist = points[0]
    features = []

    for tip in FINGERTIPS:
        features.append(np.linalg.norm(points[tip] - wrist))

    for tip, mcp in zip(FINGERTIPS, MCP_POINTS):
        features.append(np.linalg.norm(points[tip] - points[mcp]))

    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            features.append(np.linalg.norm(points[FINGERTIPS[i]] - points[FINGERTIPS[j]]))

    features = np.array(features)
    wrist_middle = np.linalg.norm(points[9] - wrist)
    if wrist_middle == 0:
        wrist_middle = 1
    return features / wrist_middle


feature_fn = raw_features_single if args.model == "raw" else engineered_features_single

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    max_num_hands=1,           # we only ever use hand_landmarks[0] -- don't track a 2nd hand
    model_complexity=0,        # "lite" landmark model: much faster, plenty accurate for this
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # smaller frames = faster capture + inference
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

fps = 0
avg_classifier_ms = 0.0
history = deque(maxlen=8)
CONFIDENCE_THRESHOLD = 0.5

# --- Design palette ---
ACCENT = (255, 210, 80)      # gold accent (BGR) for labels/branding
TEXT_LIGHT = (235, 235, 235)
TEXT_DIM = (150, 150, 150)
BORDER_COLOR = (255, 210, 80)


def shadow_text(frame, text, pos, font, scale, color, thickness=1, shadow_offset=1):
    x, y = pos
    cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset), font, scale,
                (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_rounded_bar(frame, x, y, w, h, fill_ratio, fill_color, bg_color=(90, 90, 90)):
    radius = h // 2
    cv2.ellipse(frame, (x + radius, y + radius), (radius, radius), 0, 90, 270, bg_color, 1, cv2.LINE_AA)
    cv2.ellipse(frame, (x + w - radius, y + radius), (radius, radius), 0, -90, 90, bg_color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + radius, y), (x + w - radius, y), bg_color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + radius, y + h), (x + w - radius, y + h), bg_color, 1, cv2.LINE_AA)

    filled_w = int(w * fill_ratio)
    if filled_w > radius:
        cv2.ellipse(frame, (x + radius, y + radius), (radius, radius), 0, 90, 270, fill_color, -1, cv2.LINE_AA)
        cv2.rectangle(frame, (x + radius, y), (x + filled_w, y + h), fill_color, -1, cv2.LINE_AA)
        if filled_w >= w - radius:
            cv2.ellipse(frame, (x + w - radius, y + radius), (radius, radius), 0, -90, 90, fill_color, -1, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h, alpha=0.55, color=(20, 20, 20)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def confidence_color(conf):
    if conf >= 0.8:
        return (80, 220, 100)
    elif conf >= 0.5:
        return (60, 200, 230)
    else:
        return (60, 60, 230)


# White joint dots (small, no outline) + thin green connections.
HAND_DOT_COLOR = (225, 225, 235)   # warm off-white
HAND_LINE_COLOR = (60, 200, 60)    # green, BGR


def draw_scaled_hand(frame, points_norm, w_frame, h_frame):
    """Draw hand landmarks: thin green connections with small plain white
    joint dots on top, sized proportional to the hand's actual size on
    screen instead of a fixed pixel radius."""
    points_px = np.array([[p[0] * w_frame, p[1] * h_frame] for p in points_norm]).astype(int)

    wrist = points_px[0]
    middle_mcp = points_px[9]
    hand_size_px = np.linalg.norm(middle_mcp - wrist)

    radius = max(2, int(hand_size_px * 0.025))          # small dots
    thickness = max(1, int(hand_size_px * 0.008))       # thin lines

    for connection in mp_hands.HAND_CONNECTIONS:
        s, e = connection
        cv2.line(frame, tuple(points_px[s]), tuple(points_px[e]), HAND_LINE_COLOR, thickness, cv2.LINE_AA)

    for pt in points_px:
        cv2.circle(frame, tuple(pt), radius, HAND_DOT_COLOR, -1, cv2.LINE_AA)


while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h_frame, w_frame = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture_text = "No hand detected"
    confidence = 0.0
    classifier_ms = 0.0

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        points = landmarks_to_array(hand_landmarks)
        draw_scaled_hand(frame, points, w_frame, h_frame)

        feat = feature_fn(points).reshape(1, -1)

        t0 = time.perf_counter()
        probs = clf.predict_proba(feat)[0]
        classifier_ms = (time.perf_counter() - t0) * 1000
        avg_classifier_ms = avg_classifier_ms * 0.9 + classifier_ms * 0.1  # smoothed average

        best_idx = np.argmax(probs)
        gesture_text = class_names[best_idx]
        confidence = probs[best_idx]

        if confidence >= CONFIDENCE_THRESHOLD:
            history.append(gesture_text)

    now = time.time()
    instant_fps = 1.0 / (now - loop_start + 1e-6)
    fps = fps * 0.9 + instant_fps * 0.1

    # --- Top info panel (compact, branded) ---
    PANEL_H = 78
    draw_panel(frame, 0, 0, w_frame, PANEL_H)
    color = confidence_color(confidence)

    # Small branded title
    cv2.putText(frame, "GESTURE RECOGNITION", (18, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, ACCENT, 1, cv2.LINE_AA)

    # Gesture label (consistent accent color, shadowed for depth)
    shadow_text(frame, gesture_text.upper(), (18, 42),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, TEXT_LIGHT, 2)

    # Rounded confidence bar
    draw_rounded_bar(frame, 18, 52, 180, 10, confidence, color)
    cv2.putText(frame, f"{confidence*100:.0f}%", (18 + 180 + 10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1, cv2.LINE_AA)

    # Right-side column: status dot + model/FPS/ms, right-aligned by measured text width
    RIGHT_MARGIN = 14

    hand_detected = results.multi_hand_landmarks is not None
    dot_color = (100, 220, 120) if hand_detected else (110, 110, 110)
    status_text = "LIVE" if hand_detected else "IDLE"
    (status_w, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    status_x = w_frame - RIGHT_MARGIN - status_w
    cv2.circle(frame, (status_x - 12, 14), 5, dot_color, -1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (status_x, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1, cv2.LINE_AA)

    info_lines = [
        f"{args.model[:3].upper()} model",
        f"{fps:.1f} FPS",
        f"{avg_classifier_ms:.1f} ms"
    ]
    for i, line in enumerate(info_lines):
        (line_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        line_x = w_frame - RIGHT_MARGIN - line_w
        cv2.putText(frame, line, (line_x, 34 + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1, cv2.LINE_AA)

    # Thin accent line under the panel
    cv2.line(frame, (0, PANEL_H), (w_frame, PANEL_H), ACCENT, 1, cv2.LINE_AA)

    # --- Bottom history strip (slimmer) ---
    STRIP_H = 32
    draw_panel(frame, 0, h_frame - STRIP_H, w_frame, STRIP_H, alpha=0.5)
    history_text = "  ->  ".join(history)
    cv2.putText(frame, history_text[-90:], (15, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_DIM, 1, cv2.LINE_AA)

    # --- Frame border (app-window look) ---
    cv2.rectangle(frame, (0, 0), (w_frame - 1, h_frame - 1), BORDER_COLOR, 2, cv2.LINE_AA)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()