import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore

print("--- Starting Model Training Script ---")

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

DATA_PATH = "./Audio_Speech_Actors_01-24/"
actor_folders = os.listdir(DATA_PATH)

emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
EMOTIONS_LIST = ["angry", "sad", "happy", "neutral", "fearful"]

data = []
for actor_dir in actor_folders:
    if not actor_dir.startswith("Actor_"):
        continue
    file_list = os.listdir(os.path.join(DATA_PATH, actor_dir))
    for file_name in file_list:
        parts = file_name.split('.')[0].split('-')
        emotion_code = parts[2]
        emotion = emotion_map.get(emotion_code)
        if emotion in EMOTIONS_LIST:
            file_path = os.path.join(DATA_PATH, actor_dir, file_name)
            features = extract_features(file_path)
            if features is not None:
                data.append([features, emotion])

print(f"--- Extracted features for {len(data)} audio files ---")

df = pd.DataFrame(data, columns=['features', 'emotion'])
X = np.array(df['features'].tolist())
y = np.array(df['emotion'].tolist())

encoder = OneHotEncoder()
y_onehot = encoder.fit_transform(y.reshape(-1, 1)).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Reshape for RNN input: (samples, timesteps, features)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)

print(f"--- Training data shape: {X_train.shape} ---")
print(f"--- Testing data shape: {X_test.shape} ---")

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print("--- Starting Model Training ---")
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', save_best_only=True, mode='max')

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint]
)

# Save model architecture to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save label encoder categories
np.save('emotion_labels.npy', encoder.categories_)

print("--- Training complete! Model and weights saved. ---")
print("You can now run 'python3 app.py'")
