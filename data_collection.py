"""
Data collection script.

Run this TWICE, on two separate occasions (different lighting/time/session),
to build data/session_a.csv and data/session_b.csv.

Usage:
    python data_collection.py --session a
    python data_collection.py --session b

Controls while running:
    1 = fist
    2 = open_palm
    3 = peace
    4 = thumbs_up
    5 = pointing
    x = stop capturing (no active label)
    q = quit and save CSV

While a label is active (after pressing 1-5), every detected frame is
automatically saved as a training sample under that label. Move your hand
slightly between captures (angle, distance, position) to get varied samples.
Aim for at least 150-200 samples per gesture per session.
"""

import cv2
import mediapipe as mp
import csv
import os
import argparse
import time

LABELS = {
    ord('1'): "fist",
    ord('2'): "open_palm",
    ord('3'): "peace",
    ord('4'): "thumbs_up",
    ord('5'): "pointing",
}

parser = argparse.ArgumentParser()
parser.add_argument("--session", required=True, help="Session name, e.g. a or b")
args = parser.parse_args()

os.makedirs("data", exist_ok=True)
out_path = f"data/session_{args.session}.csv"

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# CSV header: label, session, then x0,y0,z0 ... x20,y20,z20 (63 raw values)
header = ["label", "session"]
for i in range(21):
    header += [f"x{i}", f"y{i}", f"z{i}"]

file_exists = os.path.exists(out_path)
csv_file = open(out_path, "a", newline="")
writer = csv.writer(csv_file)
if not file_exists:
    writer.writerow(header)

cap = cv2.VideoCapture(0)
current_label = None
sample_count = 0

print("Press 1-5 to select a gesture label, x to pause, q to quit and save.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if current_label is not None:
            row = [current_label, args.session]
            for lm in hand_landmarks.landmark:
                row += [lm.x, lm.y, lm.z]
            writer.writerow(row)
            sample_count += 1

    status = f"Label: {current_label or 'NONE'}  |  Samples this run: {sample_count}"
    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "1-5=label  x=pause  q=quit", (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('x'):
        current_label = None
    elif key in LABELS:
        current_label = LABELS[key]

csv_file.close()
cap.release()
cv2.destroyAllWindows()
print(f"Saved {sample_count} samples to {out_path}")