import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dataset import DermaSenseDataset


def test_real_sample_loads():
    labels = json.loads((ROOT / "data/processed/label2idx.json").read_text())
    hierarchy = json.loads((ROOT / "data/processed/hierarchy_map.json").read_text())
    hierarchy = {row["disease"]: row for row in hierarchy["rows"]}
    frame = pd.read_csv(ROOT / "data/processed/train.csv", nrows=1)
    dataset = DermaSenseDataset(
        frame,
        str(ROOT / "data/raw/DATASET/DATASET_0/DATASET_0"),
        labels,
        "Image_name",
        "Disease_label",
        img_size=64,
        train=False,
        extra_dirs=[str(ROOT / "data/raw/DATASET/DATASET_1/DATASET_1")],
        hierarchy_map=hierarchy,
    )
    image, tabular, targets = dataset[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert tuple(tabular.shape) == (96,)
    assert set(targets) == {"disease", "mainclass", "subclass"}
