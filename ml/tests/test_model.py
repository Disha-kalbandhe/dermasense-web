import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from model import build_model


def test_all_ablation_shapes():
    images = torch.randn(2, 3, 64, 64)
    tabular = torch.randn(2, 96)
    for fusion_mode, hierarchical, use_metadata in [
        ("concat", False, False),
        ("concat", False, True),
        ("cross_attention", False, True),
        ("concat", True, True),
        ("cross_attention", True, True),
    ]:
        model = build_model(181, pretrained=False, fusion_mode=fusion_mode, hierarchical=hierarchical, use_metadata=use_metadata)
        outputs = model(images, tabular)
        if hierarchical:
            assert {name: tuple(value.shape) for name, value in outputs.items()} == {
                "mainclass": (2, 8), "subclass": (2, 19), "disease": (2, 181)
            }
        else:
            assert outputs.shape == (2, 181)
