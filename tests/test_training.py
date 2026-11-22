from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from fer.config import EMOTION_LABELS
from fer.training import FocalLoss, TrainConfig, build_loaders


def make_dataset(root, split, images_per_class):
    for label in EMOTION_LABELS:
        folder = root / split / label.lower()
        folder.mkdir(parents=True)
        for index in range(images_per_class):
            Image.fromarray(np.full((48, 48, 3), index, dtype=np.uint8)).save(
                folder / f"{index}.png"
            )


class TrainingTests(unittest.TestCase):
    def test_validation_split_is_disjoint_stratified_and_repeatable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_dataset(root, "train", 10)
            # No test directory: training must work without ever reading test data.
            config = TrainConfig(root, validation_fraction=0.2, batch_size=4)
            train, val = build_loaders(config)
            train2, val2 = build_loaders(config)
            self.assertFalse(set(train.dataset.indices) & set(val.dataset.indices))
            self.assertEqual(len(train.dataset) + len(val.dataset), 70)
            self.assertEqual(train.dataset.indices, train2.dataset.indices)
            self.assertEqual(val.dataset.indices, val2.dataset.indices)
            targets = np.array(val.dataset.dataset.targets)[val.dataset.indices]
            self.assertEqual(np.bincount(targets).tolist(), [2] * 7)
            self.assertIsNot(train.dataset.dataset, val.dataset.dataset)
            self.assertEqual(train.dataset.dataset.transform.transforms[-1].p, 0.1)
            self.assertNotIn("RandomErasing", str(val.dataset.dataset.transform))

    def test_explicit_val_directory_is_used(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_dataset(root, "train", 3)
            make_dataset(root, "val", 1)
            train, val = build_loaders(TrainConfig(root, batch_size=4))
            self.assertEqual(len(train.dataset), 21)
            self.assertEqual(len(val.dataset), 7)
            self.assertEqual(Path(val.dataset.root), root / "val")

    def test_focal_gamma_zero_matches_smoothed_cross_entropy(self):
        logits = torch.tensor([[1.0, -1.0, 0.3], [-0.1, 0.6, 0.9]], requires_grad=True)
        labels = torch.tensor([0, 2])
        loss = FocalLoss(gamma=0, label_smoothing=0.05)(logits, labels)
        expected = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
