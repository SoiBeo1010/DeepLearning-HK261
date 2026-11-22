"""Shared inference settings matching the FER2013 training notebook."""

from dataclasses import dataclass
from pathlib import Path


# torchvision.datasets.ImageFolder assigns indices in alphabetical folder order.
EMOTION_LABELS = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")
IMAGE_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
DEFAULT_CHECKPOINT = Path("models/cct_resnet_fer2013_best.pth")


@dataclass(frozen=True)
class InferenceConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    device: str = "auto"
    camera: int = 0
    mirror: bool = True
    min_face_size: int = 100

    def __post_init__(self) -> None:
        if self.camera < 0:
            raise ValueError("Camera index must be zero or greater.")
        if self.min_face_size < 1:
            raise ValueError("Minimum face size must be greater than zero.")
