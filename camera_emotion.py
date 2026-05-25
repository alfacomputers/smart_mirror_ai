import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 48
LABELS = ["angry", "happy", "neutral", "sad", "surprise"]

try:
    model = load_model("emotion_model.h5")
except Exception as e:
    print(f"Warning: could not load model: {e}")
    model = None

# face detector جاهز من OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if model is None:
    print("Model not loaded; exiting.")
    exit(1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face)[0]
        class_index = np.argmax(prediction)
        emotion = LABELS[class_index]
        confidence = prediction[class_index]

        # rectangle على الوجه
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # text
        text = f"{emotion} ({confidence:.2f})"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

    cv2.imshow("Smart Mirror AI", frame)

    # اضغط q للخروج
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()