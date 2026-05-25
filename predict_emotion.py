import cv2
import numpy as np
from tensorflow.keras.models import load_model

try:
    model = load_model("emotion_model.h5")
except Exception as e:
    print(f"Warning: could not load model: {e}")
    model = None

IMG_SIZE = 48
LABELS = ["angry", "happy", "neutral", "sad", "surprise"]

model = load_model("emotion_model.h5")


def predict_emotion(image_path):
    if model is None:
        return "model_missing", 0.0
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        return "error", 0.0

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]
    class_index = int(np.argmax(prediction))

    emotion = LABELS[class_index]
    confidence = float(prediction[class_index])

    return emotion, confidence