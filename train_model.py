import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from preprocessing import preprocess_video


dataset_path = "C:\Users\HP\Documents\Sem 2, 2025-2026\Deepfake_Detection_Website\dataset"

data = []
labels = []

classes = ["real", "fake"]

print("Loading dataset...")

for label, category in enumerate(classes):

    folder = os.path.join(dataset_path, category)

    for file in os.listdir(folder):

        video_path = os.path.join(folder, file)

        print("Processing:", video_path)

        try:
            features = preprocess_video(video_path)

            data.append(features)
            labels.append(label)

        except:
            print("Error processing:", video_path)


X = np.array(data)
y = np.array(labels)

y = to_categorical(y)

print("Dataset Loaded")


# Handle imbalanced dataset

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))


# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)


# Build model

model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Train model

print("Training model...")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=8,
    class_weight=class_weights
)


# Save model

os.makedirs("model", exist_ok=True)

model.save("model/deepfake_model.h5")

print("Model Saved Successfully")
