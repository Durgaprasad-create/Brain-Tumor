import os
import cv2
import numpy as np
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset paths
TUMOR_PATH = "dataset/brain_tumor_dataset/yes"
NO_TUMOR_PATH = "dataset/brain_tumor_dataset/no"

MODEL_SAVE_PATH = "model/brain_tumor_model.pkl"


def load_images(folder, label):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    images, labels = [], []

    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, (64, 64))
            img = img.flatten() / 255.0   # Normalization (IMPORTANT)
            images.append(img)
            labels.append(label)

    return images, labels


print("Loading dataset...")

tumor_imgs, tumor_labels = load_images(TUMOR_PATH, 1)
no_imgs, no_labels = load_images(NO_TUMOR_PATH, 0)

X = np.array(tumor_imgs + no_imgs)
y = np.array(tumor_labels + no_labels)

print(f"Total Samples: {len(X)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {round(accuracy * 100, 2)}%")

# Save model + metadata
model_data = {
    "model": model,
    "accuracy": accuracy
}

if not os.path.exists("model"):
    os.makedirs("model")

with open(MODEL_SAVE_PATH, "wb") as f:
    pickle.dump(model_data, f)

print("✅ Model trained and saved successfully!")