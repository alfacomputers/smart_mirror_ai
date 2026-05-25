import os
import tempfile
import numpy as np
import cv2

import predict_emotion as pe


def test_model_missing_behavior():
    # Ensure predictable state: model may be None if no .h5 present
    original_model = getattr(pe, "model", None)
    pe.model = None

    emotion, conf = pe.predict_emotion("non_existent_file.png")
    assert emotion == "model_missing"
    assert conf == 0.0

    pe.model = original_model


class DummyModel:
    def predict(self, x, verbose=0):
        # return batch of softmax-like outputs
        return np.array([[0.0, 0.99, 0.0, 0.0, 0.01]])


def test_predict_with_dummy_model(tmp_path):
    original_model = getattr(pe, "model", None)
    pe.model = DummyModel()

    img = (np.random.rand(48, 48) * 255).astype(np.uint8)
    tmpfile = tmp_path / "tmp.png"
    cv2.imwrite(str(tmpfile), img)

    emotion, conf = pe.predict_emotion(str(tmpfile))

    assert emotion == "happy"
    assert conf > 0.9

    pe.model = original_model
