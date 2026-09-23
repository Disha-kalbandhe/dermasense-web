import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import FocalCrossEntropy, MultiTaskLoss


def test_multitask_loss_is_finite_and_differentiable():
    weights = {"mainclass": torch.ones(8), "subclass": torch.ones(19), "disease": torch.ones(181)}
    criterion = MultiTaskLoss({"mainclass": 0.2, "subclass": 0.3, "disease": 1.0}, weights, 2.0, 0.1)
    outputs = {
        "mainclass": torch.randn(4, 8, requires_grad=True),
        "subclass": torch.randn(4, 19, requires_grad=True),
        "disease": torch.randn(4, 181, requires_grad=True),
    }
    targets = {"mainclass": torch.tensor([0, 1, 2, 3]), "subclass": torch.tensor([0, 1, 2, 3]), "disease": torch.tensor([0, 1, 2, 3])}
    loss = criterion(outputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert outputs["disease"].grad is not None
