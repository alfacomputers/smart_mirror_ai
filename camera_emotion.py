import cv2

from predict_emotion import load_emotion_model, predict_face_emotion

IMG_SIZE = 48
LABELS = ["angry", "happy", "neutral", "sad", "surprise"]

load_emotion_model()

# face detector جاهز من OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        emotion, confidence = predict_face_emotion(face_gray)

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