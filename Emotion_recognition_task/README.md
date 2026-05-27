# Emotion Recognition from Speech
### CodeAlpha ML Internship — Task 2

---

## Objective
Recognize human emotions from speech audio using deep learning and signal processing techniques.

---

## Dataset
**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 24 professional actors (12 male, 12 female)
- 1,380 audio files (.wav format)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised

### Filename Convention
Each filename encodes metadata. The 3rd number is the emotion:
```
03-01-05-01-01-01-02.wav
         ↑
     emotion code (05 = angry)
```

| Code | Emotion |
|---|---|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## Project Structure
```
Emotion_recognition_task/
│
├── Actor_01/ ... Actor_24/    # RAVDESS audio files
├── emotion_recognition.py     # Main model code
└── README.md                  # This file
```

---

## Approach

### 1. Feature Extraction
Raw audio is converted into 180 numerical features per file:

| Feature | Count | What it captures |
|---|---|---|
| MFCCs | 40 | Voice tone and timbre |
| Chroma | 12 | Pitch information |
| Mel Spectrogram | 128 | Energy at different frequencies |
| **Total** | **180** | |

### 2. Preprocessing
- Labels encoded with `LabelEncoder` (strings → numbers)
- Features scaled with `StandardScaler` to normalize ranges
- 80/20 train/test split

### 3. Neural Network Architecture
```
Input (180 features)
    ↓
Dense(256) + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(128) + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(64) + ReLU
    ↓
Dense(8) + Softmax → emotion prediction
```

- **ReLU** — activation function that decides which neurons fire
- **Dropout** — randomly disables 30% of neurons to prevent overfitting
- **Softmax** — converts final outputs to percentages (e.g. 82% angry, 6% sad)

### 4. Training
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 100
- Batch size: 32

---

## Results

| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| Neutral | 0.57 | 0.44 | 0.50 |
| Calm | 0.83 | 0.85 | 0.84 |
| Happy | 0.57 | 0.71 | 0.63 |
| Sad | 0.55 | 0.53 | 0.54 |
| Angry | 0.82 | 0.78 | 0.79 |
| Fearful | 0.75 | 0.66 | 0.70 |
| Disgust | 0.70 | 0.69 | 0.69 |
| Surprised | 0.70 | 0.76 | 0.73 |
| **Overall** | | | **0.70** |

**Overall Accuracy: 70%** — within industry standard range (60–75%) for this dataset.

### Key Observations
- **Angry** and **calm** are easiest to detect — they have strong, distinct audio signals
- **Neutral** and **sad** are hardest — they lack strong distinguishing features, even for humans

---

## How to Run

1. Install required libraries:
```bash
pip install librosa tensorflow scikit-learn numpy
```

2. Place RAVDESS actor folders in the same directory as `emotion_recognition.py`

3. Update `DATASET_PATH` in the script to your local path

4. Run:
```bash
python emotion_recognition.py
```

---

## Libraries Used
- `librosa` — audio loading and feature extraction
- `tensorflow/keras` — neural network
- `scikit-learn` — preprocessing and evaluation
- `numpy` — numerical operations