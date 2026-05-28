# CodeAlpha ML Internship
### Machine Learning Projects

---

## Projects

### 📊 Task 1 — Credit Scoring Model
Predict whether a customer will default on a loan using past financial data.

- **Dataset:** Give Me Some Credit (Kaggle) — 150,000 rows
- **Models:** Logistic Regression, Decision Tree, Random Forest
- **Best Result:** Logistic Regression (balanced) — ROC-AUC: 0.71, Recall: 0.65
- **Key Learning:** Handling imbalanced data with `class_weight='balanced'`

📁 [View Project](./Credit_scoring_task/)

---

### 🎙️ Task 2 — Emotion Recognition from Speech
Recognize human emotions from speech audio using deep learning.

- **Dataset:** RAVDESS — 1,380 audio files, 8 emotions
- **Model:** Neural Network (Dense layers) with MFCCs + Chroma + Mel features
- **Best Result:** 70% accuracy — within industry standard range
- **Key Learning:** Audio feature extraction with librosa, neural networks with TensorFlow

📁 [View Project](./Emotion_recognition_task/)

---

## Tech Stack
- Python 3.13
- pandas, numpy, scikit-learn
- librosa, TensorFlow/Keras
