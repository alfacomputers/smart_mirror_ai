import cv2
import numpy as np
from pathlib import Path

IMG_SIZE = 48
LABELS = ["angry", "happy", "neutral", "sad", "surprise"]
MODEL_ONNX = Path("emotion_model.onnx")

model = None
backend = None


def _preprocess_face(face_gray):
    face = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    return face


def _load_onnx_model():
    global model, backend
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError("onnxruntime must be installed to use ONNX model support") from error

    session = ort.InferenceSession(str(MODEL_ONNX), providers=["CPUExecutionProvider"])
    model = {"type": "onnx", "session": session}
    backend = "onnx"
    return model


def _fallback_predict(face_gray):
    if face_gray is None or face_gray.size == 0:
        return "error", 0.0

    average = float(np.mean(face_gray)) / 255.0
    if average > 0.65:
        emotion = "happy"
        confidence = min(0.95, 0.5 + (average - 0.65) * 1.2)
    elif average < 0.35:
        emotion = "sad"
        confidence = min(0.85, 0.5 + (0.35 - average) * 1.5)
    else:
        emotion = "neutral"
        confidence = 0.65

    return emotion, float(max(0.0, min(confidence, 1.0)))


def load_emotion_model():
    global model
    if model is not None:
        return model

    if MODEL_ONNX.exists():
        try:
            return _load_onnx_model()
        except Exception as exc:
            print(f"Warning: could not load ONNX model: {exc}")

    print("Warning: no compatible model found; using lightweight CPU fallback.")
    model = {"type": "fallback"}
    return model


def _onnx_predict(face):
    session = model["session"]
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: face})
    prediction = np.asarray(outputs[0])[0]
    return prediction


def _keras_predict(face):
    prediction = model.predict(face, verbose=0)
    return np.asarray(prediction)[0]


def predict_face_emotion(face_gray):
    if model is None:
        load_emotion_model()

    if isinstance(model, dict):
        if model["type"] == "onnx":
            face = _preprocess_face(face_gray)
            prediction = _onnx_predict(face)
            class_index = int(np.argmax(prediction))
            emotion = LABELS[class_index]
            confidence = float(prediction[class_index])
            return emotion, confidence

        return _fallback_predict(face_gray)

    if hasattr(model, "predict"):
        face = _preprocess_face(face_gray)
        prediction = _keras_predict(face)
        class_index = int(np.argmax(prediction))
        emotion = LABELS[class_index]
        confidence = float(prediction[class_index])
        return emotion, confidence

    return _fallback_predict(face_gray)


def predict_emotion(image_path):
    if model is None:
        load_emotion_model()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "model_missing", 0.0

    return predict_face_emotion(img)
