import pandas as pd
import numpy as np
import os

# -----------------------------
# Load data
# -----------------------------

os.makedirs("features", exist_ok=True)

session_a = pd.read_csv("data/session_a.csv")
session_b = pd.read_csv("data/session_b.csv")
session_c = pd.read_csv("data/session_c.csv")

data = pd.concat(
    [session_a, session_b, session_c],
    ignore_index=True
)

# -----------------------------
# Landmark columns
# -----------------------------

landmark_columns = []

for i in range(21):
    landmark_columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

# -----------------------------
# RAW FEATURES
# -----------------------------

def create_raw_features(df):

    X = df[landmark_columns].values.astype(float)

    # Reshape:
    # samples × 21 landmarks × 3 coordinates
    X = X.reshape(-1, 21, 3)

    # Make wrist the origin
    X = X - X[:, 0:1, :]

    # Normalize by hand size
    hand_size = np.linalg.norm(
        X[:, 9, :] - X[:, 0, :],
        axis=1,
        keepdims=True
    )

    hand_size[hand_size == 0] = 1

    X = X / hand_size[:, :, None]

    return X.reshape(len(X), -1)


# -----------------------------
# ENGINEERED FEATURES
# -----------------------------

def create_engineered_features(df):

    X = df[landmark_columns].values.astype(float)
    X = X.reshape(-1, 21, 3)

    features = []

    # Important landmarks
    wrist = X[:, 0, :]

    fingertips = [
        4,   # thumb
        8,   # index
        12,  # middle
        16,  # ring
        20   # pinky
    ]

    # --------------------------------
    # Finger tip distances from wrist
    # --------------------------------

    for tip in fingertips:

        distance = np.linalg.norm(
            X[:, tip, :] - wrist,
            axis=1
        )

        features.append(distance)

    # --------------------------------
    # Finger tip distances from MCP
    # --------------------------------

    mcp_points = [
        2,   # thumb
        5,   # index
        9,   # middle
        13,  # ring
        17   # pinky
    ]

    for tip, mcp in zip(fingertips, mcp_points):

        distance = np.linalg.norm(
            X[:, tip, :] - X[:, mcp, :],
            axis=1
        )

        features.append(distance)

    # --------------------------------
    # Distances between fingertips
    # --------------------------------

    for i in range(len(fingertips)):

        for j in range(i + 1, len(fingertips)):

            distance = np.linalg.norm(
                X[:, fingertips[i], :] -
                X[:, fingertips[j], :],
                axis=1
            )

            features.append(distance)

    # --------------------------------
    # Normalize engineered features
    # --------------------------------

    features = np.array(features).T

    wrist_middle = np.linalg.norm(
        X[:, 9, :] - wrist,
        axis=1,
        keepdims=True
    )

    wrist_middle[wrist_middle == 0] = 1

    features = features / wrist_middle

    return features


# -----------------------------
# Create features
# -----------------------------

raw_features = create_raw_features(data)

engineered_features = create_engineered_features(data)

labels = data["label"].values
sessions = data["session"].values

# -----------------------------
# Save
# -----------------------------

np.savez(
    "features/raw_features.npz",
    X=raw_features,
    y=labels,
    session=sessions
)

np.savez(
    "features/engineered_features.npz",
    X=engineered_features,
    y=labels,
    session=sessions
)

print("Feature extraction complete!")
print()
print("Total samples:", len(data))
print("Raw feature shape:", raw_features.shape)
print("Engineered feature shape:", engineered_features.shape)
print()
print("Saved:")
print("features/raw_features.npz")
print("features/engineered_features.npz")