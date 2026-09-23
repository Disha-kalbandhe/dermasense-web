"""Run DermaSense smoke validation, ablations, and staged winner training on Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler


def find_root() -> Path:
    candidates = [Path.cwd(), Path("/kaggle/working/dermasense-web"), Path("/kaggle/input/dermasense-web")]
    for candidate in candidates:
        if (candidate / "ml" / "src" / "model.py").exists():
            return candidate
    raise FileNotFoundError("Could not find a repository root containing ml/src/model.py")


def find_dataset_paths(root: Path):
    sys.path.insert(0, str(root / "ml" / "src"))
    from dataset_discovery import discover_dataset

    candidates = [root / "ml" / "data" / "raw", root, Path("/kaggle/input")]
    errors = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return discover_dataset(candidate)
        except FileNotFoundError as error:
            errors.append(str(error))
    raise FileNotFoundError(
        "Could not discover the DermaCon-IN dataset. "
        "Required train_split.csv, test_split.csv, and DATASET image directory. "
        + " | ".join(errors)
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(root: Path):
    ml_root = root / "ml"
    src = ml_root / "src"
    sys.path.insert(0, str(ml_root))
    sys.path.insert(0, str(src))
    import config
    import dataset
    import model
    import train
    import utils
    importlib.reload(config)
    importlib.reload(dataset)
    importlib.reload(model)
    importlib.reload(train)
    importlib.reload(utils)
    return ml_root, config, dataset, model, train, utils


def smoke(model_module, train_module, device):
    model = model_module.build_model(181, tabular_dim=96, pretrained=False, fusion_mode="concat", hierarchical=True).to(device).train()
    criterion = train_module.MultiTaskLoss(
        {"mainclass": 0.2, "subclass": 0.3, "disease": 1.0},
        {"mainclass": torch.ones(8), "subclass": torch.ones(19), "disease": torch.ones(181)},
        gamma=2.0,
        label_smoothing=0.1,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    images = torch.randn(2, 3, 64, 64, device=device)
    tabular = torch.randn(2, 96, device=device)
    targets = {
        "mainclass": torch.tensor([0, 1], device=device),
        "subclass": torch.tensor([0, 1], device=device),
        "disease": torch.tensor([0, 1], device=device),
    }
    optimizer.zero_grad(set_to_none=True)
    loss = criterion(model(images, tabular), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    if not torch.isfinite(loss):
        raise RuntimeError("Smoke loss was not finite")
    print(f"SMOKE_EPOCH_PASS loss={float(loss):.6f}")


def build_data(ml_root, config, dataset_module, train_module, utils_module, dataset_paths, validation):
    processed = ml_root / "data" / "processed"
    labels_path = processed / "label2idx.json"
    hierarchy_path = processed / "hierarchy_map.json"
    audit_path = ml_root / "outputs" / "reports" / "hierarchy_audit.md"
    if not hierarchy_path.exists():
        if dataset_paths.metadata_csv is None:
            raise FileNotFoundError(
                "SkinMetadata.csv was not found; cannot build hierarchy_map.json from native annotations."
            )
        from build_hierarchy import build_hierarchy
        build_hierarchy(dataset_paths.metadata_csv, labels_path, hierarchy_path, audit_path, dataset_paths.train_csv)
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    hierarchy_rows = json.loads(hierarchy_path.read_text(encoding="utf-8"))["rows"]
    hierarchy = {row["disease"]: row for row in hierarchy_rows}
    processed_splits = [processed / f"{name}.csv" for name in ("train", "val", "test")]
    if all(path.exists() for path in processed_splits):
        frames = {name: pd.read_csv(processed / f"{name}.csv") for name in ("train", "val", "test")}
    else:
        raw_train = validation["train"].copy()
        raw_test = validation["test"].copy()
        image_column = validation["image_column"]
        split_mask = raw_train[image_column].astype(str).map(
            lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 7 == 0
        )
        frames = {"train": raw_train.loc[~split_mask].reset_index(drop=True), "val": raw_train.loc[split_mask].reset_index(drop=True), "test": raw_test}
        print("TRAINING_SPLITS_SOURCE=raw_train_split_with_deterministic_validation_from_train_only")
    image_dirs = list(dataset_paths.image_dirs)
    image_column = config.IMG_COL if config.IMG_COL in frames["train"].columns else validation["image_column"]
    label_column = config.LABEL_COL if config.LABEL_COL in frames["train"].columns else validation["label_column"]
    def make_dataset(frame, train_mode):
        return dataset_module.DermaSenseDataset(
            frame, image_dirs[0], labels, image_column, label_column,
            config.IMG_SIZE, train=train_mode, extra_dirs=image_dirs[1:], hierarchy_map=hierarchy,
        )
    datasets = {"train": make_dataset(frames["train"], True), "val": make_dataset(frames["val"], False), "test": make_dataset(frames["test"], False)}
    disease_ids = frames["train"][config.LABEL_COL].map(labels).to_numpy()
    weights = train_module.effective_number_weights(disease_ids, len(labels), config.CB_BETA)
    sampler = WeightedRandomSampler(weights[disease_ids], len(disease_ids), replacement=True)
    device = utils_module.get_device("cuda")
    kwargs = {"batch_size": config.BATCH_SIZE, "num_workers": 0, "pin_memory": device.type == "cuda"}
    loaders = {"train": DataLoader(datasets["train"], sampler=sampler, **kwargs), "val": DataLoader(datasets["val"], shuffle=False, **kwargs), "test": DataLoader(datasets["test"], shuffle=False, **kwargs)}
    return processed, labels, hierarchy_rows, hierarchy, frames, loaders, device


def model_for(model_module, config, labels, spec, pretrained):
    return model_module.build_model(
        len(labels), tabular_dim=config.TABULAR_DIM, fusion_dim=config.FUSION_DIM,
        dropout=config.DROPOUT, pretrained=pretrained, num_main_classes=8,
        num_subclass_classes=19, d_model=config.D_MODEL,
        n_attn_layers=config.N_ATTN_LAYERS, n_heads=config.N_HEADS, **spec,
    )


def predictions(model, loader, device):
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for images, tabular, targets in loader:
            output = model(images.to(device), tabular.to(device))
            logits = output["disease"] if isinstance(output, dict) else output
            truth.extend(targets["disease"].tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    return np.asarray(truth), np.asarray(predicted)


def set_stage(model, stage):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = stage != 1
    if stage >= 2 and hasattr(model.backbone, "blocks"):
        for block in model.backbone.blocks[-2:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def checkpoint(model, optimizer, epoch, metrics, spec, config, hierarchy_path, labels_path):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": {"model": spec, "image_size": config.IMG_SIZE, "batch_size": config.BATCH_SIZE},
        "model_config": spec,
        "hierarchy_version": sha256(hierarchy_path),
        "class_mapping_version": sha256(labels_path),
        "metadata_encoding_version": "dermasense-tabular-v2",
        "val_metrics": metrics,
    }


def run_config(name, spec, epochs, loaders, labels, model_module, train_module, config, device, checkpoints, hierarchy_path, labels_path, train_frame):
    torch.manual_seed(config.SEED)
    current = model_for(model_module, config, labels, spec, pretrained=True).to(device)
    optimizer = train_module.build_optimizer(current, config.LR)
    class_ids = train_frame[config.LABEL_COL].map(labels).to_numpy()
    criterion = train_module.MultiTaskLoss(
        config.LOSS_WEIGHTS,
        {"mainclass": torch.ones(8), "subclass": torch.ones(19), "disease": train_module.effective_number_weights(class_ids, len(labels), config.CB_BETA)},
        config.FOCAL_GAMMA,
        config.LABEL_SMOOTHING,
    )
    best_f1 = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        set_stage(current, 1 if epoch <= config.WARMUP_EPOCHS else 2)
        train_metrics = train_module.run_epoch(current, loaders["train"], criterion, optimizer, device, True, config.GRADIENT_ACCUMULATION_STEPS)
        truth, predicted = predictions(current, loaders["val"], device)
        metrics = {"epoch": epoch, "train": train_metrics, "val_macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)), "val_accuracy": float(accuracy_score(truth, predicted))}
        history.append(metrics)
        print(name, json.dumps(metrics))
        if metrics["val_macro_f1"] > best_f1:
            best_f1 = metrics["val_macro_f1"]
            best_path = checkpoints / f"{name.lower()}_best_model.pth" if name != "FULL" else checkpoints / "best_model.pth"
            torch.save(checkpoint(current, optimizer, epoch, metrics, spec, config, hierarchy_path, labels_path), best_path)
        if name == "FULL":
            torch.save(checkpoint(current, optimizer, epoch, metrics, spec, config, hierarchy_path, labels_path), checkpoints / "last_model.pth")
    return {"name": name, "spec": spec, "best_val_macro_f1": best_f1, "history": history}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-tree", action="store_true")
    parser.add_argument("--ablation-epochs", type=int, default=5)
    parser.add_argument("--full-epochs", type=int, default=30)
    args = parser.parse_args()
    root = find_root()
    dataset_paths = find_dataset_paths(root)
    from dataset_discovery import print_tree, validate_dataset
    if not args.no_tree:
        print_tree(dataset_paths.root)
    validation = validate_dataset(dataset_paths)
    if args.validate_only:
        return
    ml_root, config, dataset_module, model_module, train_module, utils_module = setup(root)
    device = utils_module.get_device("cuda")
    smoke(model_module, train_module, device)
    if args.smoke_only:
        return
    processed, labels, hierarchy_rows, hierarchy, frames, loaders, device = build_data(
        ml_root, config, dataset_module, train_module, utils_module, dataset_paths, validation
    )
    checkpoints = ml_root / "checkpoints"
    reports = ml_root / "outputs" / "reports"
    checkpoints.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    hierarchy_path = processed / "hierarchy_map.json"
    labels_path = processed / "label2idx.json"
    ablations = {
        "A": {"use_metadata": False, "fusion_mode": "concat", "hierarchical": False},
        "B": {"use_metadata": True, "fusion_mode": "concat", "hierarchical": False},
        "C": {"use_metadata": True, "fusion_mode": "cross_attention", "hierarchical": False},
        "D": {"use_metadata": True, "fusion_mode": "concat", "hierarchical": True},
        "E": {"use_metadata": True, "fusion_mode": "cross_attention", "hierarchical": True},
    }
    results = [run_config(name, spec, args.ablation_epochs, loaders, labels, model_module, train_module, config, device, checkpoints, hierarchy_path, labels_path, frames["train"]) for name, spec in ablations.items()]
    summary = pd.DataFrame([{"model": result["name"], "val_macro_f1": result["best_val_macro_f1"], **result["spec"]} for result in results])
    summary.to_csv(reports / "ablation_results.csv", index=False)
    (reports / "ablation_report.md").write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")
    deployment_results = [result for result in results if result["spec"]["hierarchical"]]
    winner = max(deployment_results, key=lambda result: result["best_val_macro_f1"])
    print("WINNER", winner["name"], winner["best_val_macro_f1"])
    full = run_config("FULL", winner["spec"], args.full_epochs, loaders, labels, model_module, train_module, config, device, checkpoints, hierarchy_path, labels_path, frames["train"])
    (reports / "selected_config.json").write_text(json.dumps({"winner": winner["name"], "spec": winner["spec"], "history": full["history"]}, indent=2), encoding="utf-8")
    subprocess.run([sys.executable, str(ml_root / "src" / "evaluate.py"), "--checkpoint", str(checkpoints / "best_model.pth"), "--output", str(reports)], check=True)
    print("Checkpoint saved:", checkpoints / "best_model.pth")
    print("Evaluation reports saved:", reports)


if __name__ == "__main__":
    main()
