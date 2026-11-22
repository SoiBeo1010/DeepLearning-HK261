from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image
import torch

from fer.app import annotate_frame, main, run_camera, run_image
from fer.config import InferenceConfig
from fer.predictor import EmotionRecognizer, Prediction, build_preprocess

ROOT = Path(__file__).resolve().parents[1]


class FixedLogits(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.last_input = None

    def forward(self, tensor):
        self.last_input = tensor
        logits = torch.full((tensor.shape[0], 7), -20.0)
        logits[:, self.index] = 20.0
        return logits


def fake_recognizer(index=4):
    # Exercise the actual preprocessing and prediction without loading 91 MB per test.
    from fer.config import EMOTION_LABELS

    recognizer = EmotionRecognizer.__new__(EmotionRecognizer)
    recognizer.device = torch.device("cpu")
    recognizer.labels = EMOTION_LABELS
    recognizer.preprocess = build_preprocess()
    recognizer.model = FixedLogits(index)
    return recognizer


class InferenceTests(unittest.TestCase):
    def test_final_three_class_indices_match_imagefolder(self):
        crop = np.zeros((48, 48, 3), dtype=np.uint8)
        for index, expected in ((4, "Neutral"), (5, "Sad"), (6, "Surprise")):
            with self.subTest(index=index):
                prediction = fake_recognizer(index).predict(crop)
                self.assertEqual(prediction.label, expected)
                self.assertAlmostEqual(prediction.confidence, 1.0)

    def test_bgr_preprocessing_matches_training_rgb(self):
        from fer.training import build_transforms

        crop = np.zeros((48, 48, 3), dtype=np.uint8)
        crop[:, :, 2] = 255  # Red in OpenCV's BGR order.
        recognizer = fake_recognizer()
        recognizer.predict(crop)
        expected = build_transforms()(Image.fromarray(crop[:, :, ::-1].copy()))
        torch.testing.assert_close(recognizer.model.last_input[0], expected)

    def test_empty_face_is_reported(self):
        with self.assertRaisesRegex(ValueError, "non-empty BGR"):
            fake_recognizer().predict(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_missing_checkpoint_has_actionable_message(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(FileNotFoundError, "--weights"):
                EmotionRecognizer(Path(folder) / "missing.pth")

    def test_overlapping_faces_are_predicted_before_drawing(self):
        frame = np.full((100, 100, 3), 127, dtype=np.uint8)
        original = frame.copy()
        detector = Mock()
        detector.detectMultiScale.return_value = [(10, 10, 50, 50), (20, 20, 50, 50)]
        crops = []
        recognizer = Mock()

        def predict(crop):
            crops.append(crop.copy())
            return Prediction("Neutral", 0.9)

        recognizer.predict.side_effect = predict
        results = annotate_frame(frame, recognizer, detector, 20)
        self.assertEqual(len(results), 2)
        np.testing.assert_array_equal(crops[0], original[10:60, 10:60])
        np.testing.assert_array_equal(crops[1], original[20:70, 20:70])
        self.assertFalse(np.array_equal(frame, original))

    def test_camera_is_released_after_read_failure(self):
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)
        with patch("fer.app.load_face_detector"), patch(
            "fer.app.cv2.VideoCapture", return_value=capture
        ), patch("fer.app.cv2.destroyAllWindows") as destroy:
            with self.assertRaisesRegex(RuntimeError, "Cannot read"):
                run_camera(InferenceConfig(), Mock())
            capture.release.assert_called_once()
            destroy.assert_called_once()

    def test_camera_is_released_when_open_fails(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch("fer.app.load_face_detector"), patch(
            "fer.app.cv2.VideoCapture", return_value=capture
        ), patch("fer.app.cv2.destroyAllWindows"):
            with self.assertRaisesRegex(RuntimeError, "Cannot open camera"):
                run_camera(InferenceConfig(), Mock())
            capture.release.assert_called_once()

    def test_image_mode_handles_unicode_paths_and_no_faces(self):
        detector = Mock()
        detector.detectMultiScale.return_value = []
        with tempfile.TemporaryDirectory() as folder:
            image = Path(folder) / "\u1ea3nh.png"
            output = Path(folder) / "out" / "k\u1ebft-qu\u1ea3.png"
            original = np.full((64, 64, 3), 80, dtype=np.uint8)
            cv2.imencode(".png", original)[1].tofile(image)
            with patch("fer.app.load_face_detector", return_value=detector):
                results = run_image(image, output, InferenceConfig(), Mock())
            self.assertEqual(results, [])
            saved = cv2.imdecode(np.fromfile(output, dtype=np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(saved, original)

    def test_cli_rejects_invalid_input_before_camera(self):
        with patch("fer.app.run_camera") as camera, self.assertLogs("fer.app", level="ERROR"):
            self.assertEqual(main(["--weights", "not-present.pth"]), 1)
            camera.assert_not_called()

    @unittest.skipUnless(
        (ROOT / "models/cct_resnet_fer2013_best.pth").is_file(), "Local checkpoint unavailable"
    )
    def test_original_checkpoint_loads_and_predicts_on_cpu(self):
        recognizer = EmotionRecognizer(ROOT / "models/cct_resnet_fer2013_best.pth", "cpu")
        self.assertFalse(recognizer.model.training)
        prediction = recognizer.predict(np.full((48, 48, 3), 128, dtype=np.uint8))
        self.assertIn(prediction.label, recognizer.labels)
        self.assertTrue(np.isfinite(prediction.confidence))
        self.assertGreaterEqual(prediction.confidence, 1 / 7)
        self.assertLessEqual(prediction.confidence, 1)


if __name__ == "__main__":
    unittest.main()
