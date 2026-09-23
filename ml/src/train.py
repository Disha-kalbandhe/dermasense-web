"""Training utilities and CLI for DermaSense hierarchical models."""

from __future__ import annotations
import argparse
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from dataset import DermaSenseDataset
from model import build_model
from utils import file_sha256, get_device, set_seed

def effective_number_weights(labels: np.ndarray, num_classes: int, beta: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = (1.0 - beta) / (1.0 - np.power(beta, counts[present]))
    if present.any():
        weights[present] /= weights[present].mean()
    weights = np.clip(weights, 0.2, 5.0)
    return torch.tensor(weights, dtype=torch.float32)

class FocalCrossEntropy(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None, gamma: float = 0.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else torch.empty(0))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight if self.weight.numel() else None
        loss = nn.functional.cross_entropy(logits, targets, weight=weight, reduction="none", label_smoothing=self.label_smoothing)
        if self.gamma:
            loss = loss * (1.0 - torch.softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)).pow(self.gamma)
        return loss.mean()

class MultiTaskLoss(nn.Module):
    def __init__(self, weights, class_weights, gamma, label_smoothing):
        super().__init__()
        self.losses = nn.ModuleDict({
            "mainclass": FocalCrossEntropy(class_weights["mainclass"], label_smoothing=label_smoothing),
            "subclass": FocalCrossEntropy(class_weights["subclass"]),
            "disease": FocalCrossEntropy(class_weights["disease"], gamma=gamma),
        })
        self.weights = weights

    def forward(self, outputs, targets):
        if not isinstance(outputs, dict):
            return self.losses["disease"](outputs, targets["disease"])
        return sum(self.weights[name] * self.losses[name](outputs[name], targets[name]) for name in self.losses)

def _move_targets(targets, device):
    return {name: value.to(device, non_blocking=True) for name, value in targets.items()}

def run_epoch(model, loader, criterion, optimizer, device, train, accumulation_steps=1):
    model.train(train)
    total_loss = 0.0
    total = 0
    correct = 0
    predictions = []
    labels_seen = []
    for step, (images, tabular, targets) in enumerate(tqdm(loader, leave=False, desc="Train" if train else "Val")):
        images, tabular = images.to(device, non_blocking=True), tabular.to(device, non_blocking=True)
        targets = _move_targets(targets, device)
        with torch.set_grad_enabled(train):
            outputs = model(images, tabular)
            loss = criterion(outputs, targets)
            if train:
                (loss / accumulation_steps).backward()
                if (step + 1) % accumulation_steps == 0 or step + 1 == len(loader):
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        disease_logits = outputs["disease"] if isinstance(outputs, dict) else outputs
        total_loss += loss.item() * images.size(0)
        correct += (disease_logits.argmax(1) == targets["disease"]).sum().item()
        predictions.extend(disease_logits.argmax(1).detach().cpu().tolist())
        labels_seen.extend(targets["disease"].detach().cpu().tolist())
        total += images.size(0)
    return {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1), "macro_f1": f1_score(labels_seen, predictions, average="macro", zero_division=0)}

def build_optimizer(model, learning_rate=config.LR):
    backbone, other = [], []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("backbone") else other).append(parameter)
    return AdamW([{"params": backbone, "lr": learning_rate * 0.1}, {"params": other, "lr": learning_rate}], weight_decay=config.WEIGHT_DECAY)

def make_checkpoint(model, optimizer, epoch, metrics, config_values):
    serializable_config = {
        key: str(value) if isinstance(value, (Path, type(Path()))) else value
        for key, value in config_values.items()
        if isinstance(value, (str, int, float, bool, type(None), Path))
    }
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "config": serializable_config,
    "model_config": {"use_metadata": config_values.get("MODEL_USE_METADATA", True), "fusion_mode": config_values.get("MODEL_FUSION_MODE", "cross_attention"), "hierarchical": config_values.get("MODEL_HIERARCHICAL", True), "num_classes": config.NUM_CLASSES, "num_main_classes": config.NUM_MAIN_CLASSES, "num_subclass_classes": config.NUM_SUBCLASS_CLASSES},
    "hierarchy_version": file_sha256(config.HIERARCHY_MAP_PATH) if Path(config.HIERARCHY_MAP_PATH).exists() else None,
    "class_mapping_version": file_sha256(Path(config.DATA_PROCESSED) / "label2idx.json"),
    "metadata_encoding_version": "dermasense-tabular-v2",
    "val_metrics": metrics,
    }

def train(model, train_loader, val_loader, config_module=config, device=None, criterion=None):
    device = device or get_device(config_module.DEVICE)
    Path(config_module.CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = build_optimizer(model, config_module.LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config_module.NUM_EPOCHS, 1), eta_min=config_module.LR * 0.01)
    criterion = criterion or nn.CrossEntropyLoss(label_smoothing=config_module.LABEL_SMOOTHING)
    best = -1.0
    history = []
    for epoch in range(1, config_module.NUM_EPOCHS + 1):
        epoch_started = time.perf_counter()
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True, config_module.GRADIENT_ACCUMULATION_STEPS)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record))
        print(f"EPOCH_TIMING epoch={epoch} seconds={time.perf_counter() - epoch_started:.2f} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f} train_macro_f1={train_metrics['macro_f1']:.6f} val_macro_f1={val_metrics['macro_f1']:.6f}")
        torch.save(make_checkpoint(model, optimizer, epoch, val_metrics, vars(config_module)), config_module.LAST_MODEL)
        if val_metrics["macro_f1"] > best:
            best = val_metrics["macro_f1"]
            torch.save(make_checkpoint(model, optimizer, epoch, val_metrics, vars(config_module)), config_module.BEST_MODEL)
    return history

def _load_hierarchy(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {row["disease"]: row for row in payload["rows"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "ablation"], default="train")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fusion-mode", choices=["concat", "cross_attention"], default="cross_attention")
    parser.add_argument("--hierarchical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-metadata", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", choices=["auto", "gpu", "cpu_fast"], default="auto")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    if args.profile == "cpu_fast" or (args.profile == "auto" and not torch.cuda.is_available()):
        config.PROFILE = "cpu_fast"
        config.IMG_SIZE = 224
        config.BATCH_SIZE = 16
        config.GRADIENT_ACCUMULATION_STEPS = 2
        config.DEVICE = "cpu"
        config.PIN_MEMORY = False
        torch.set_num_threads(config.CPU_THREADS)
    elif args.profile == "gpu":
        config.PROFILE = "gpu"
        config.DEVICE = "cuda"
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    pretrained = args.pretrained if args.pretrained is not None else config.PROFILE == "gpu"
    config.MODEL_FUSION_MODE = args.fusion_mode
    config.MODEL_HIERARCHICAL = args.hierarchical
    config.MODEL_USE_METADATA = args.use_metadata
    print(json.dumps({"profile": config.PROFILE, "reason": "CUDA available" if config.PROFILE == "gpu" else "CUDA unavailable; CPU-adaptive reduced configuration", "device": config.DEVICE, "image_size": config.IMG_SIZE, "batch_size": config.BATCH_SIZE, "gradient_accumulation": config.GRADIENT_ACCUMULATION_STEPS, "epochs": config.NUM_EPOCHS, "num_classes": config.NUM_CLASSES, "pretrained": pretrained}))
    set_seed(config.SEED)
    from dataset_discovery import discover_dataset, validate_dataset

    discovered = discover_dataset(Path(config.DATA_RAW))
    validation = validate_dataset(discovered)
    image_dirs = list(discovered.image_dirs)
    train_df = pd.read_csv(Path(config.DATA_PROCESSED) / "train.csv")
    val_df = pd.read_csv(Path(config.DATA_PROCESSED) / "val.csv")
    labels = json.loads((Path(config.DATA_PROCESSED) / "label2idx.json").read_text())
    hierarchy = _load_hierarchy(Path(config.HIERARCHY_MAP_PATH))
    image_column = config.IMG_COL if config.IMG_COL in train_df.columns else validation["image_column"]
    label_column = config.LABEL_COL if config.LABEL_COL in train_df.columns else validation["label_column"]
    train_dataset = DermaSenseDataset(train_df, image_dirs[0], labels, image_column, label_column, config.IMG_SIZE, True, image_dirs[1:], hierarchy)
    val_dataset = DermaSenseDataset(val_df, image_dirs[0], labels, image_column, label_column, config.IMG_SIZE, False, image_dirs[1:], hierarchy)
    disease_labels = train_df[config.LABEL_COL].map(labels).to_numpy()
    class_weights = {"disease": effective_number_weights(disease_labels, len(labels), config.CB_BETA), "mainclass": torch.ones(config.NUM_MAIN_CLASSES), "subclass": torch.ones(config.NUM_SUBCLASS_CLASSES)}
    sampler_weights = class_weights["disease"][disease_labels]
    sampler = WeightedRandomSampler(sampler_weights, len(sampler_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    model = build_model(len(labels), config.TABULAR_DIM, config.FUSION_DIM, config.DROPOUT, pretrained=pretrained, fusion_mode=args.fusion_mode, hierarchical=args.hierarchical, use_metadata=args.use_metadata, num_main_classes=config.NUM_MAIN_CLASSES, num_subclass_classes=config.NUM_SUBCLASS_CLASSES, d_model=config.D_MODEL, n_attn_layers=config.N_ATTN_LAYERS, n_heads=config.N_HEADS)
    criterion = MultiTaskLoss(config.LOSS_WEIGHTS, class_weights, config.FOCAL_GAMMA, config.LABEL_SMOOTHING)
    if args.smoke_only:
        model.to(get_device(config.DEVICE)).train()
        optimizer = AdamW(model.parameters(), lr=config.LR)
        images, tabular, targets = next(iter(train_loader))
        device = get_device(config.DEVICE)
        images, tabular = images.to(device), tabular.to(device)
        targets = _move_targets(targets, device)
        losses = []
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images, tabular), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            print(f"SMOKE_STEP={step + 1} loss={losses[-1]:.6f}")
        print(f"SMOKE_LOSS_INITIAL={losses[0]:.6f} SMOKE_LOSS_FINAL={losses[-1]:.6f} SMOKE_FINITE={all(np.isfinite(losses))} SMOKE_DECREASED={losses[-1] < losses[0]}")
        return
    train(model, train_loader, val_loader, config, criterion=criterion)

if __name__ == "__main__":
    main()