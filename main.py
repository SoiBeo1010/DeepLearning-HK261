"""Run the application from a checkout, including from another working directory."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fer.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(
        main(default_checkpoint=PROJECT_ROOT / "models" / "cct_resnet_fer2013_best.pth")
    )
