import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# ── 1. Configuration ──────────────────────────────────────────────────────────
DATASET_PATH = r"C:\Users\Priyanshi\OneDrive\Desktop\Codealpha_tasks\Emotion_recognition_task"

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

# ── 2. Feature Extraction ─────────────────────────────────────────────────────
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050, duration=3)

    # MFCCs — 40 numbers capturing voice tone and timbre
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40), axis=1)

    # Chroma — 12 numbers capturing pitch information
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)

    # Mel Spectrogram — 128 numbers capturing energy at different frequencies
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr), axis=1)

    return np.concatenate([mfccs, chroma, mel])  # 180 features total

# ── 3. Load Dataset ───────────────────────────────────────────────────────────
print("Loading dataset...")
X = []
y = []

actors = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]

for actor in actors:
    actor_path = os.path.join(DATASET_PATH, actor)
    for filename in os.listdir(actor_path):
        if not filename.endswith(".wav"):
            continue
        emotion_code = filename.split("-")[2]
        file_path = os.path.join(actor_path, filename)
        features = extract_features(file_path)
        X.append(features)
        y.append(emotion_code)

X = np.array(X)
print(f"Done! Total samples: {len(X)}, Features per sample: {X.shape[1]}")

# ── 4. Encode Labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Emotions found: {list(le.classes_)}")

# ── 5. Train/Test Split ───────────────────────────────────────────────────────
print("\nSplitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ── 6. Scale Features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── 7. Build Neural Network ───────────────────────────────────────────────────
print("\nBuilding model...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(180,)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(8, activation='softmax')   # 8 emotions
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ── 8. Train Model ────────────────────────────────────────────────────────────
print("\nTraining model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

# ── 9. Evaluate ───────────────────────────────────────────────────────────────
print("\n── Evaluation ──")
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes,
      target_names=list(EMOTION_MAP.values())))