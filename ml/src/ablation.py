"""CPU-reduced train-only ablation comparison for the reconciled dataset."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from dataset import DermaSenseDataset
from model import build_model
from train import MultiTaskLoss, effective_number_weights, run_epoch
from utils import set_seed


def main() -> None:
    set_seed(config.SEED)
    torch.set_num_threads(config.CPU_THREADS)
    processed = Path(config.DATA_PROCESSED)
    train = pd.read_csv(processed / "train.csv")
    val = pd.read_csv(processed / "val.csv")
    labels = json.loads((processed / "label2idx.json").read_text(encoding="utf-8"))
    hierarchy_rows = json.loads(Path(config.HIERARCHY_MAP_PATH).read_text(encoding="utf-8"))["rows"]
    hierarchy = {row["disease"]: row for row in hierarchy_rows}
    image_dirs = [config.DATASET_DIR_0, config.DATASET_DIR_1]
    image_dirs = [path for path in image_dirs if Path(path).exists()]
    rng = np.random.default_rng(config.SEED)
    subset_parts = []
    for _, group in train.groupby(config.LABEL_COL, sort=True):
        take = max(1, int(np.ceil(len(group) * 0.35)))
        subset_parts.append(group.iloc[rng.choice(len(group), size=take, replace=False)])
    subset = pd.concat(subset_parts).sort_values(config.IMG_COL).reset_index(drop=True)
    configs = {
        "A_image_only": ("concat", False, False),
        "B_concat_metadata": ("concat", False, True),
        "C_cross_attention": ("cross_attention", False, True),
        "D_cross_attention_hierarchical": ("cross_attention", True, True),
        "E_concat_hierarchical": ("concat", True, True),
    }
    results = []
    for name, (fusion_mode, hierarchical, use_metadata) in configs.items():
        started = time.perf_counter()
        train_ds = DermaSenseDataset(subset, image_dirs[0], labels, config.IMG_COL, config.LABEL_COL, 224, True, image_dirs[1:], hierarchy)
        val_ds = DermaSenseDataset(val, image_dirs[0], labels, config.IMG_COL, config.LABEL_COL, 224, False, image_dirs[1:], hierarchy)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
        model = build_model(len(labels), config.TABULAR_DIM, config.FUSION_DIM, config.DROPOUT, pretrained=False, fusion_mode=fusion_mode, hierarchical=hierarchical, use_metadata=use_metadata, num_main_classes=config.NUM_MAIN_CLASSES, num_subclass_classes=config.NUM_SUBCLASS_CLASSES, d_model=config.D_MODEL, n_attn_layers=config.N_ATTN_LAYERS, n_heads=config.N_HEADS).to("cpu")
        disease_labels = subset[config.LABEL_COL].map(labels).to_numpy()
        disease_weights = effective_number_weights(disease_labels, len(labels), config.CB_BETA)
        criterion = MultiTaskLoss(config.LOSS_WEIGHTS, {"disease": disease_weights, "mainclass": torch.ones(config.NUM_MAIN_CLASSES), "subclass": torch.ones(config.NUM_SUBCLASS_CLASSES)}, config.FOCAL_GAMMA, config.LABEL_SMOOTHING)
        optimizer = AdamW(model.parameters(), lr=config.LR)
        last = None
        for epoch in range(1, 3):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, torch.device("cpu"), True, 1)
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, torch.device("cpu"), False, 1)
            last = (train_metrics, val_metrics)
            print(f"ABLATION model={name} epoch={epoch} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} val_macro_f1={val_metrics['macro_f1']:.6f}")
        results.append({"model": name, "fusion_mode": fusion_mode, "hierarchical": hierarchical, "use_metadata": use_metadata, "subset_rows": len(subset), "epochs": 2, "train_loss_final": last[0]["loss"], "val_loss_final": last[1]["loss"], "val_macro_f1": last[1]["macro_f1"], "val_accuracy": last[1]["accuracy"], "wall_clock_seconds": time.perf_counter() - started})
    results.sort(key=lambda row: row["val_macro_f1"], reverse=True)
    report_dir = Path(config.REPORTS_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "ablation_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Reduced-Budget Ablation", "", f"Subset rows: {len(subset)} ({len(subset) / len(train):.3%} of reconciled train)", "Two epochs per configuration; model selection used validation macro-F1 only.", "", "| Model | Val macro-F1 | Val accuracy | Final val loss | Seconds |", "|---|---:|---:|---:|---:|"]
    lines.extend(f"| {row['model']} | {row['val_macro_f1']:.6f} | {row['val_accuracy']:.6f} | {row['val_loss_final']:.6f} | {row['wall_clock_seconds']:.2f} |" for row in results)
    lines.extend(["", f"Winner: **{results[0]['model']}** with validation macro-F1 {results[0]['val_macro_f1']:.6f}."])
    (report_dir / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ABLATION_WINNER={results[0]['model']} val_macro_f1={results[0]['val_macro_f1']:.6f}")


if __name__ == "__main__":
    main()