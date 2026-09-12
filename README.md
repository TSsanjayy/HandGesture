Yep 👍 Here is the **complete README.md in one block**. Just copy everything below and paste it into your `README.md`.

````markdown
# ✋ Real-Time Hand Gesture Recognition

### MediaPipe • Computer Vision • Machine Learning • Real-Time Interaction

A real-time hand gesture recognition system built using **Python, OpenCV, MediaPipe Hands, and Random Forest machine learning**.

The system recognizes **5 hand gestures** from MediaPipe's 21 hand landmarks and compares two feature representations to study **accuracy, generalization, and inference speed**.

---

## 🚀 Project Highlights

| Feature | Result |
|---|---|
| 🎯 Gesture Classes | **5** |
| 🖐️ MediaPipe Landmarks | **21 per hand** |
| 📊 Raw Features | **63** |
| 📐 Engineered Features | **20** |
| 🧠 Classifier | **Random Forest** |
| 🏆 Best Separate-Session Accuracy | **94.84%** |
| ⚡ Classifier Inference | **19.57 ms** |
| 🎥 Real-Time Webcam Demo | **Yes** |

### 🏆 Key Result

The engineered geometric feature model achieved:

**94.84% accuracy on a completely unseen recording session.**

Compared with raw landmark features, engineered features improved separate-session accuracy by:

**+8.57 percentage points**

This demonstrates stronger generalization across different recording sessions.

---

# 👋 Gestures Recognized

The system recognizes five gestures:

- ✊ **Fist**
- 🖐️ **Open Palm**
- ✌️ **Peace**
- 👍 **Thumbs Up**
- ☝️ **Pointing**

---

# 🎥 Real-Time Demo

The application processes webcam input in real time and displays:

- Hand landmarks
- Gesture prediction
- Prediction confidence
- Pipeline FPS
- Classifier inference time
- Recent gesture history

### Recommended Model

```bash
python live_demo_ui.py --model engineered
````

Press **Q** to exit.

### Raw Landmark Model

```bash
python live_demo_ui.py --model raw
```

---

# 🧠 How the System Works

```text
                         Webcam
                            │
                            ▼
                   ┌─────────────────┐
                   │   OpenCV Camera │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ MediaPipe Hands │
                   │   21 Landmarks  │
                   └────────┬────────┘
                            │
                     Feature Extraction
                            │
                 ┌──────────┴──────────┐
                 │                     │
                 ▼                     ▼
          Raw Landmarks       Engineered Features
             63 features           20 features
                 │                     │
                 └──────────┬──────────┘
                            ▼
                   ┌─────────────────┐
                   │ Random Forest   │
                   │   Classifier    │
                   └────────┬────────┘
                            │
                            ▼
                   Gesture + Confidence
```

---

# 📐 Feature Engineering

Two different feature representations were evaluated.

## 1. Raw Landmark Features

MediaPipe Hands provides **21 landmarks** for a detected hand.

Each landmark contains:

```text
x
y
z
```

Therefore:

```text
21 × 3 = 63 features
```

The landmarks are:

1. Translated relative to the wrist
2. Normalized using the wrist-to-middle-MCP distance
3. Flattened into a 63-dimensional feature vector

This reduces sensitivity to the position and scale of the hand.

---

## 2. Engineered Geometric Features

The second representation describes the **shape of the hand** using geometric distances.

It contains **20 features**:

```text
5  fingertip → wrist distances
5  fingertip → MCP distances
10 pairwise fingertip distances
```

All distances are normalized using the wrist-to-middle-MCP distance.

This representation focuses on **hand geometry rather than absolute landmark coordinates**.

---

# 📊 Dataset

Three independently recorded sessions were collected:

```text
Session A
Session B
Session C
```

### Final Dataset

**3,986 samples**

The dataset contains samples from all five gesture classes.

Session B was deliberately kept as an **unseen independent session** for the final separate-session evaluation.

This allows the project to test whether the model generalizes to a new recording session rather than only performing well on data recorded under the same conditions as the training data.

---

# 🧪 Evaluation Method

Two evaluation scenarios were used.

## Same-Session Evaluation

Session A was divided into:

```text
80% Training
20% Testing
```

using a stratified split.

A fixed random seed of **42** was used for reproducibility.

---

## Separate-Session Evaluation

For the stronger generalization test:

```text
Training → Session A + Session C
Testing  → Session B
```

Session B was completely unseen during training.

This provides a more realistic measure of how the model performs when the recording conditions change.

---

# 🏆 Final Results

| Feature Representation  | Same-Session Accuracy | Separate-Session Accuracy |
| ----------------------- | --------------------: | ------------------------: |
| **Raw Landmarks**       |            **99.29%** |                **86.27%** |
| **Engineered Features** |            **97.16%** |                **94.84%** |

### Results Analysis

The raw landmark model achieved the highest same-session accuracy:

**99.29%**

However, its performance decreased to:

**86.27%**

on the completely unseen session.

The engineered feature model achieved:

**97.16%**

same-session accuracy and:

**94.84%**

separate-session accuracy.

This shows that the engineered geometric representation generalizes better across independently recorded sessions.

### 📈 Generalization Improvement

Compared with raw landmarks, engineered features improved separate-session accuracy by:

**94.84% − 86.27% = 8.57 percentage points**

This indicates substantially stronger cross-session robustness.

---

# ⚡ Inference Performance

| Feature Representation | Average Classifier Time |
| ---------------------- | ----------------------: |
| Raw Landmarks          |            **21.35 ms** |
| Engineered Features    |            **19.57 ms** |

The engineered representation also required less average classifier inference time.

---

# 🛠️ Technologies Used

* **Python 3.11**
* **OpenCV** — webcam capture and visualization
* **MediaPipe Hands** — 21-point hand landmark detection
* **NumPy** — numerical feature processing
* **Pandas** — dataset handling
* **Scikit-learn** — Random Forest classification
* **Joblib** — model serialization

---

# 📁 Project Structure

```text
HandGesture/
│
├── data/
│   ├── session_a.csv
│   ├── session_b.csv
│   └── session_c.csv
│
├── features/
│   ├── raw_features.npz
│   └── engineered_features.npz
│
├── data_collection.py
├── features.py
├── final_evaluation.py
├── train_and_evaluate.py
│
├── gesture_recognition.py
├── hand_tracking.py
├── test_camera.py
│
├── live_demo.py
├── live_demo_ui.py
│
├── model_raw.joblib
├── model_engineered.joblib
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/TSsanjayy/HandGesture.git
cd HandGesture
```

## 2. Create a Virtual Environment

```bash
python -m venv gesture-env
```

## 3. Activate the Environment on Windows

```bash
gesture-env\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Collect Data

To collect a session:

```bash
python data_collection.py --session a
```

Use `b` or `c` to record another session.

Example:

```bash
python data_collection.py --session c
```

---

## Generate Features

Run:

```bash
python features.py
```

This generates:

```text
features/raw_features.npz
features/engineered_features.npz
```

---

## Train and Evaluate

Run:

```bash
python final_evaluation.py
```

This evaluates:

* Same-session accuracy
* Separate-session accuracy
* Classifier inference time

It also saves:

```text
model_raw.joblib
model_engineered.joblib
```

---

## Run the Live Recognition System

Recommended:

```bash
python live_demo_ui.py --model engineered
```

Alternative:

```bash
python live_demo_ui.py --model raw
```

Press:

```text
Q
```

to close the application.

---

# 🔬 Reproducibility

The project uses pinned dependency versions through `requirements.txt`.

Machine learning experiments use:

```text
Random State = 42
```

The same feature extraction and normalization procedures are used during offline evaluation and live recognition.

---

# 📦 Requirements

```text
mediapipe==0.10.21
numpy==1.26.4
opencv-python==4.11.0.86
scikit-learn==1.9.0
pandas==3.0.5
joblib==1.6.0
```

---

# 🎯 Project Objective

The objective of this project is to develop a **real-time and robust hand gesture recognition system** and investigate whether engineered geometric features can improve generalization across independently recorded sessions.

The experimental results show an important trade-off:

* Raw landmarks provide extremely high accuracy on familiar data.
* Engineered geometric features provide substantially better generalization to unseen recording sessions.
* Engineered features also provide slightly faster classifier inference.

The final real-time system therefore uses the **engineered feature representation**.

---

# 👨‍💻 Author

## Sanjay TS

GitHub:

[https://github.com/TSsanjayy](https://github.com/TSsanjayy)

---

## ⭐ Project Summary

**5 gestures • 21 landmarks • 2 feature representations • Random Forest • 3,986 samples • 94.84% unseen-session accuracy • Real-time webcam recognition**

```

After pasting, press **Ctrl + S**.

Then tell me **saved**.
```
