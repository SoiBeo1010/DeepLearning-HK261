"""Command-line application for webcam and still-image emotion recognition."""

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np

from fer.config import DEFAULT_CHECKPOINT, InferenceConfig
from fer.predictor import EmotionRecognizer, Prediction

LOGGER = logging.getLogger(__name__)
WINDOW_TITLE = "Facial Emotion Recognition - press Q or Esc to exit"


@dataclass(frozen=True)
class FaceResult:
    box: tuple[int, int, int, int]
    prediction: Prediction


def load_face_detector() -> cv2.CascadeClassifier:
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise RuntimeError(f"Cannot load OpenCV face detector: {path}")
    return detector


def annotate_frame(
    frame: np.ndarray,
    recognizer: EmotionRecognizer,
    detector: cv2.CascadeClassifier,
    min_face_size: int,
) -> list[FaceResult]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(min_face_size, min_face_size)
    )
    # Predict all crops before drawing, so boxes/text cannot alter another face's crop.
    results = [
        FaceResult(
            tuple(int(value) for value in (x, y, width, height)),
            recognizer.predict(frame[y : y + height, x : x + width]),
        )
        for x, y, width, height in faces
    ]
    for result in results:
        x, y, width, height = result.box
        prediction = result.prediction
        color = (0, 255, 0) if prediction.label in {"Happy", "Neutral"} else (0, 0, 255)
        text = f"{prediction.label} ({prediction.confidence:.1%})"
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame, text, (x, max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2, cv2.LINE_AA,
        )
    return results


def run_camera(config: InferenceConfig, recognizer: EmotionRecognizer) -> None:
    detector = load_face_detector()
    capture = cv2.VideoCapture(config.camera)
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Cannot open camera {config.camera}. Check permissions or try --camera 1."
            )
        LOGGER.info("Camera %s opened. Press Q or Esc to exit.", config.camera)
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("Cannot read a frame from the camera.")
            if config.mirror:
                frame = cv2.flip(frame, 1)
            annotate_frame(frame, recognizer, detector, config.min_face_size)
            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run_image(
    image: Path, output: Path | None, config: InferenceConfig, recognizer: EmotionRecognizer
) -> list[FaceResult]:
    # imdecode/tofile supports Windows paths containing non-ASCII characters.
    frame = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Cannot decode image: {image}")
    results = annotate_frame(frame, recognizer, load_face_detector(), config.min_face_size)
    if not results:
        LOGGER.info("No faces detected. Try a clearer image or a smaller --min-face-size.")
    for result in results:
        LOGGER.info(
            "Face %s: %s (%.1f%%)", result.box,
            result.prediction.label, result.prediction.confidence * 100,
        )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(output.suffix or ".jpg", frame)
        if not ok:
            raise OSError(f"Cannot encode output image: {output}")
        encoded.tofile(output)
        LOGGER.info("Saved annotated image: %s", output)
    return results


def create_parser(default_checkpoint: Path = DEFAULT_CHECKPOINT) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FER2013 emotion recognition with CCT + ResNet18.")
    parser.add_argument("--weights", type=Path, default=default_checkpoint, help="Path to model .pth weights.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--no-mirror", action="store_true", help="Disable webcam mirroring.")
    parser.add_argument("--min-face-size", type=int, default=100, help="Minimum detected face size in pixels.")
    parser.add_argument("--image", type=Path, help="Process an image instead of opening the webcam.")
    parser.add_argument("--output", type=Path, help="Save the annotated image (requires --image).")
    return parser


def main(argv: list[str] | None = None, *, default_checkpoint: Path = DEFAULT_CHECKPOINT) -> int:
    parser = create_parser(default_checkpoint)
    args = parser.parse_args(argv)
    if args.output is not None and args.image is None:
        parser.error("--output requires --image")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = InferenceConfig(
            checkpoint=args.weights, device=args.device, camera=args.camera,
            mirror=not args.no_mirror, min_face_size=args.min_face_size,
        )
        if args.image is not None and not args.image.is_file():
            raise FileNotFoundError(f"Input image not found: {args.image}")
        recognizer = EmotionRecognizer(config.checkpoint, config.device)
        LOGGER.info("Model loaded on %s.", recognizer.device)
        if args.image is not None:
            run_image(args.image, args.output, config, recognizer)
        else:
            run_camera(config, recognizer)
    except KeyboardInterrupt:
        LOGGER.info("Stopped.")
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0
