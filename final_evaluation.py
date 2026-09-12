import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import time


# --------------------------------------------------
# Load features
# --------------------------------------------------

raw = np.load(
    "features/raw_features.npz",
    allow_pickle=True
)

engineered = np.load(
    "features/engineered_features.npz",
    allow_pickle=True
)

X_raw = raw["X"]
y_raw = raw["y"]
sessions_raw = raw["session"]

X_eng = engineered["X"]
y_eng = engineered["y"]
sessions_eng = engineered["session"]

print("Total samples:", len(y_raw))
print("Raw features:", X_raw.shape)
print("Engineered features:", X_eng.shape)


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate(X, y, sessions, name):

    print()
    print("=" * 55)
    print(f"FEATURE SET: {name}")
    print("=" * 55)

    # ==================================================
    # SAME SESSION
    # Train 80% of Session A
    # Test 20% of Session A
    # ==================================================

    mask_a = sessions == "a"

    X_a = X[mask_a]
    y_a = y[mask_a]

    X_train, X_test, y_train, y_test = train_test_split(
        X_a,
        y_a,
        test_size=0.20,
        random_state=42,
        stratify=y_a
    )

    model_same = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model_same.fit(X_train, y_train)

    pred_same = model_same.predict(X_test)

    same_accuracy = accuracy_score(
        y_test,
        pred_same
    )

    print(
        f"Same-session accuracy: "
        f"{same_accuracy:.4f}"
    )


    # ==================================================
    # SEPARATE SESSION
    #
    # TRAIN = Session A + Session C
    # TEST  = Session B
    #
    # Session B is completely unseen.
    # ==================================================

    mask_b = sessions == "b"
    mask_c = sessions == "c"

    X_b = X[mask_b]
    y_b = y[mask_b]

    X_c = X[mask_c]
    y_c = y[mask_c]

    X_train_cross = np.concatenate(
        [X_a, X_c],
        axis=0
    )

    y_train_cross = np.concatenate(
        [y_a, y_c],
        axis=0
    )

    model_cross = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model_cross.fit(
        X_train_cross,
        y_train_cross
    )

    pred_b = model_cross.predict(X_b)

    cross_accuracy = accuracy_score(
        y_b,
        pred_b
    )

    print(
        f"Separate-session accuracy: "
        f"{cross_accuracy:.4f}"
    )


    # ==================================================
    # CLASSIFIER INFERENCE TIME
    # ==================================================

    sample = X_b[0].reshape(1, -1)

    # Warm-up
    for _ in range(10):
        model_cross.predict(sample)

    start = time.perf_counter()

    for _ in range(100):
        model_cross.predict(sample)

    end = time.perf_counter()

    avg_time_ms = (
        (end - start) / 100
    ) * 1000

    print(
        f"Average classifier inference time: "
        f"{avg_time_ms:.4f} ms"
    )

    return (
        model_cross,
        same_accuracy,
        cross_accuracy,
        avg_time_ms
    )


# --------------------------------------------------
# RAW LANDMARK MODEL
# --------------------------------------------------

raw_model, raw_same, raw_cross, raw_time = evaluate(
    X_raw,
    y_raw,
    sessions_raw,
    "RAW LANDMARKS"
)


# --------------------------------------------------
# ENGINEERED FEATURE MODEL
# --------------------------------------------------

eng_model, eng_same, eng_cross, eng_time = evaluate(
    X_eng,
    y_eng,
    sessions_eng,
    "ENGINEERED FEATURES"
)


# --------------------------------------------------
# Save final models
# These are trained on A + C
# --------------------------------------------------

joblib.dump(
    raw_model,
    "model_raw.joblib"
)

joblib.dump(
    eng_model,
    "model_engineered.joblib"
)


# --------------------------------------------------
# FINAL TABLE
# --------------------------------------------------

print()
print("=" * 70)
print("FINAL ACCURACY TABLE")
print("=" * 70)

print()

print(
    f"{'Feature Set':<25}"
    f"{'Same Session':<20}"
    f"{'Separate Session':<20}"
)

print("-" * 70)

print(
    f"{'Raw Landmarks':<25}"
    f"{raw_same * 100:>8.2f}%"
    f"{raw_cross * 100:>18.2f}%"
)

print(
    f"{'Engineered Features':<25}"
    f"{eng_same * 100:>8.2f}%"
    f"{eng_cross * 100:>18.2f}%"
)

print()
print("Classifier time:")

print(
    f"Raw:        {raw_time:.4f} ms"
)

print(
    f"Engineered: {eng_time:.4f} ms"
)

print()
print("Final models saved:")
print("model_raw.joblib")
print("model_engineered.joblib")
