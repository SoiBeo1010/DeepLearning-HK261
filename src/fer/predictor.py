"""Model loading and image preprocessing independent of the camera interface."""

from dataclasses import dataclass
from pathlib import Path
import pickle

import cv2
import numpy as np
import torch
from torchvision import transforms

from fer.config import EMOTION_LABELS, IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD
from fer.model import CCT_ResNet


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of: auto, cpu, cuda, mps.")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device cpu or install CUDA-enabled PyTorch.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable. Use --device cpu.")
    return torch.device(name)


def build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


class EmotionRecognizer:
    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint}. "
                "Place cct_resnet_fer2013_best.pth in models/ or pass --weights PATH."
            )
        self.device = resolve_device(device)
        self.labels = EMOTION_LABELS
        self.preprocess = build_preprocess()
        # Load on CPU first to avoid allocating a second checkpoint copy on the GPU.
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model = CCT_ResNet(num_classes=len(self.labels))
            self.model.load_state_dict(state, strict=True)
        except (RuntimeError, ValueError, TypeError, KeyError, pickle.UnpicklingError, EOFError) as exc:
            raise ValueError(
                f"Invalid checkpoint: {checkpoint}. Expected the CCT_ResNet state_dict "
                "from the FER2013 notebook."
            ) from exc
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, face_bgr: np.ndarray) -> Prediction:
        if face_bgr.ndim != 3 or face_bgr.shape[2] != 3 or face_bgr.size == 0:
            raise ValueError("Expected a non-empty BGR face image with three channels.")
        # OpenCV uses BGR; PIL/torchvision expects RGB before grayscale conversion.
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.preprocess(face_rgb).unsqueeze(0).to(self.device)
        probabilities = self.model(tensor).softmax(dim=1)
        confidence, index = probabilities.max(dim=1)
        return Prediction(self.labels[index.item()], confidence.item())
