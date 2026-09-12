# ✋ Real-Time Hand Gesture Recognition

### MediaPipe • Computer Vision • Machine Learning • Real-Time Interaction

A real-time hand gesture recognition system built using **Python, OpenCV, MediaPipe Hands, and Random Forest machine learning**.

The system recognizes **five hand gestures** using MediaPipe's 21 hand landmarks and evaluates two different feature representations for **accuracy, cross-session generalization, and inference speed**.

---

## 🚀 Project Highlights

| Feature | Result |
|---|---|
| 🎯 Gesture Classes | **5** |
| 🖐️ Hand Landmarks | **21** |
| 📊 Raw Features | **63** |
| 📐 Engineered Features | **20** |
| 🧠 Classifier | **Random Forest** |
| 🏆 Best Separate-Session Accuracy | **94.84%** |
| ⚡ Best Classifier Inference | **19.57 ms** |
| 🎥 Real-Time Webcam Demo | **Yes** |

### 🏆 Key Result

The engineered geometric feature representation achieved **94.84% accuracy on a completely unseen recording session**.

Compared with raw landmark features, this improved separate-session accuracy by **8.57 percentage points**.

---

## 👋 Gestures Recognized

✊ **Fist**  
🖐️ **Open Palm**  
✌️ **Peace**  
👍 **Thumbs Up**  
☝️ **Pointing**

---

## 🎥 Real-Time Recognition

The system provides a live webcam interface with:

- Hand landmark visualization
- Gesture prediction
- Prediction confidence
- Pipeline FPS
- Classifier inference time
- Recent gesture history

The recommended model uses engineered geometric features.

Run `python live_demo_ui.py --model engineered`.

The raw-landmark model can be tested using `python live_demo_ui.py --model raw`.

Press **Q** to exit the application.

---

## 🧠 System Pipeline

**Webcam → OpenCV → MediaPipe Hands → 21 Landmarks → Feature Extraction → Random Forest → Gesture Prediction**

The system contains two feature-processing approaches:

**Raw Landmark Representation**  
21 landmarks × 3 coordinates = **63 features**

**Engineered Geometric Representation**  
20 normalized distance-based features describing hand shape.

---

## 📐 Feature Engineering

### Raw Landmark Features

MediaPipe provides 21 landmarks, with each landmark containing X, Y, and Z coordinates.

The landmarks are:

- Centered relative to the wrist
- Normalized using the wrist-to-middle-MCP distance
- Flattened into a 63-dimensional feature vector

This reduces sensitivity to hand position and scale.

### Engineered Features

The engineered representation contains **20 geometric features**:

- 5 fingertip-to-wrist distances
- 5 fingertip-to-MCP distances
- 10 pairwise fingertip distances

All distances are normalized using the wrist-to-middle-MCP distance.

This focuses the model on **hand shape and finger relationships** rather than absolute coordinates.

---

## 📊 Dataset

Three independently recorded sessions were used:

**Session A • Session B • Session C**

The final dataset contains **3,986 samples** covering all five gesture classes.

Session B was kept completely unseen during the final separate-session evaluation.

### Final Evaluation Setup

**Same-session:**  
Session A → 80% training / 20% testing

**Separate-session:**  
Training → Session A + Session C  
Testing → Session B

This separate-session setup evaluates how well the model generalizes to independently recorded data.

---

## 🏆 Final Results

| Feature Representation | Same-Session Accuracy | Separate-Session Accuracy |
|---|---:|---:|
| **Raw Landmarks** | **99.29%** | **86.27%** |
| **Engineered Features** | **97.16%** | **94.84%** |

### Results Analysis

The raw landmark representation achieved the highest same-session accuracy at **99.29%**.

However, its accuracy decreased to **86.27%** on the unseen session.

The engineered feature representation achieved **97.16%** same-session accuracy and **94.84%** separate-session accuracy.

This demonstrates that engineered geometric features provide substantially better **cross-session generalization**.

### 📈 Generalization Improvement

**94.84% − 86.27% = 8.57 percentage points**

The engineered representation therefore provides a significant improvement when the model encounters data from a different recording session.

---

## ⚡ Inference Performance

| Feature Representation | Average Classifier Time |
|---|---:|
| Raw Landmarks | **21.35 ms** |
| Engineered Features | **19.57 ms** |

The engineered representation also provides lower average classifier inference time.

---

## 🛠️ Technologies Used

**Python 3.11** — Development language

**OpenCV** — Webcam capture and real-time visualization

**MediaPipe Hands** — 21-point hand landmark detection

**NumPy** — Numerical processing and feature extraction

**Pandas** — Dataset handling

**Scikit-learn** — Random Forest classification

**Joblib** — Saving and loading trained models

---

## 📁 Project Structure

```text
HandGesture/
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
├── gesture_recognition.py
├── hand_tracking.py
├── test_camera.py
├── live_demo.py
├── live_demo_ui.py
├── model_raw.joblib
├── model_engineered.joblib
├── requirements.txt
├── .gitignore
└── README.md
⚙️ Installation

Clone the repository using git clone https://github.com/TSsanjayy/HandGesture.git and enter the project directory with cd HandGesture.

Create a virtual environment using python -m venv gesture-env.

On Windows, activate it with gesture-env\Scripts\activate.

Install the required dependencies using pip install -r requirements.txt.

▶️ Running the Project
Data Collection

Collect a session using python data_collection.py --session a.

The same script can be used with sessions b and c.

Feature Extraction

Run python features.py to generate the raw and engineered feature files.

Model Evaluation

Run python final_evaluation.py to perform the final same-session and separate-session evaluation and measure classifier inference time.

Live Recognition

Run python live_demo_ui.py --model engineered for the recommended real-time system.

Use python live_demo_ui.py --model raw to test the raw landmark model.

🔬 Reproducibility

The project uses pinned dependency versions in requirements.txt.

Machine learning experiments use Random State = 42.

The same landmark normalization and engineered-feature procedures are used for offline evaluation and real-time prediction.

📦 Requirements

The project uses:

mediapipe==0.10.21
numpy==1.26.4
opencv-python==4.11.0.86
scikit-learn==1.9.0
pandas==3.0.5
joblib==1.6.0
🎯 Project Objective

The objective of this project is to develop a real-time and robust hand gesture recognition system while investigating whether geometric feature engineering can improve generalization across independently recorded sessions.

The experimental results demonstrate an important trade-off:

Raw landmarks provide extremely high accuracy on familiar data.

Engineered geometric features provide substantially better performance on unseen recording sessions while also providing faster classifier inference.

Based on these results, the engineered feature model is used for the final real-time recognition system.

👨‍💻 Author
Sanjay TS

GitHub: TSsanjayy

⭐ Project Summary

5 Gestures • 21 Landmarks • 2 Feature Representations • Random Forest • 3,986 Samples • 94.84% Unseen-Session Accuracy • Real-Time Webcam Recognition