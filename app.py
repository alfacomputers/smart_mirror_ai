from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import threading
import time

from predict_emotion import load_emotion_model, predict_face_emotion

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

load_emotion_model()

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

current_emotion = "loading..."
latest_frame = None
lock = threading.Lock()


def camera_loop():
    global current_emotion, latest_frame

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        current_emotion = "camera_error"
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            current_emotion = "camera_error"
            continue

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5
            )

            display_frame = frame.copy()

            if len(faces) == 0:
                current_emotion = "no_face"
            else:
                x, y, w, h = faces[0]
                face_gray = gray[y:y + h, x:x + w]

                emotion, confidence = predict_face_emotion(face_gray)
                current_emotion = emotion

                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(
                    display_frame,
                    f"{emotion} ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2
                )

            with lock:
                latest_frame = display_frame

        except Exception as e:
            print(f"AI error: {e}")
            current_emotion = "error"


def generate_frames():
    global latest_frame

    while True:
        with lock:
            if latest_frame is None:
                # avoid busy spin when no frame is available
                time.sleep(0.05)
                continue
            frame = latest_frame.copy()

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "server is running"})


@app.route("/current-emotion", methods=["GET"])
def get_emotion():
    return jsonify({"emotion": current_emotion})


@app.route("/video_feed", methods=["GET"])
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    print("Starting camera thread...")
    thread = threading.Thread(target=camera_loop, daemon=True)
    thread.start()

    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False, threaded=True)