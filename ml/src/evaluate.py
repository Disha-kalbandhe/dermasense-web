"""Evaluate a checkpoint once on an untouched test CSV."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
	accuracy_score,
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from dataset import DermaSenseDataset
from model import build_model
from utils import get_device


def _outputs(model, images, tabular):
	outputs = model(images, tabular)
	return outputs if isinstance(outputs, dict) else {"disease": outputs}


def evaluate(model, loader, device, hierarchy_rows, output_dir):
	model.eval()
	predictions = {"disease": [], "mainclass": [], "subclass": []}
	targets = {"disease": [], "mainclass": [], "subclass": []}
	disease_probabilities = []
	with torch.no_grad():
		for images, tabular, batch_targets in loader:
			outputs = _outputs(model, images.to(device), tabular.to(device))
			disease_probabilities.extend(torch.softmax(outputs["disease"], dim=1).cpu().tolist())
			for name in predictions:
				if name in outputs:
					predictions[name].extend(outputs[name].argmax(1).cpu().tolist())
					targets[name].extend(batch_targets[name].tolist())

	disease_true = np.asarray(targets["disease"])
	disease_pred = np.asarray(predictions["disease"])
	disease_probabilities = np.asarray(disease_probabilities)
	top_indices = np.argsort(-disease_probabilities, axis=1)
	top_3 = np.any(top_indices[:, :3] == disease_true[:, None], axis=1)
	top_5 = np.any(top_indices[:, :5] == disease_true[:, None], axis=1)
	confidence = disease_probabilities.max(axis=1)
	correctness = (disease_pred == disease_true).astype(float)
	bins = np.linspace(0.0, 1.0, 11)
	ece = 0.0
	for lower, upper in zip(bins[:-1], bins[1:]):
		selected = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
		if selected.any():
			ece += selected.mean() * abs(correctness[selected].mean() - confidence[selected].mean())
	result = {
		"top_1_accuracy": float(accuracy_score(disease_true, disease_pred)),
		"top_3_accuracy": float(top_3.mean()),
		"top_5_accuracy": float(top_5.mean()),
		"macro_precision": float(precision_score(disease_true, disease_pred, average="macro", zero_division=0)),
		"macro_recall": float(recall_score(disease_true, disease_pred, average="macro", zero_division=0)),
		"macro_f1": float(f1_score(disease_true, disease_pred, average="macro", zero_division=0)),
		"weighted_f1": float(f1_score(disease_true, disease_pred, average="weighted", zero_division=0)),
		"balanced_accuracy": float(balanced_accuracy_score(disease_true, disease_pred)),
		"expected_calibration_error_10_bins": float(ece),
	}
	for name in ["mainclass", "subclass"]:
		if targets[name]:
			result[f"{name}_accuracy"] = float(accuracy_score(targets[name], predictions[name]))

	output_dir.mkdir(parents=True, exist_ok=True)
	report = classification_report(disease_true, disease_pred, output_dict=True, zero_division=0)
	with (output_dir / "per_class_report.csv").open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(["class", "precision", "recall", "f1-score", "support"])
		for label, values in report.items():
			if isinstance(values, dict):
				writer.writerow([label, values["precision"], values["recall"], values["f1-score"], values["support"]])

	matrix = confusion_matrix(disease_true, disease_pred)
	np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
	plt.figure(figsize=(14, 12))
	plt.imshow(matrix, interpolation="nearest", cmap="Blues")
	plt.title("Disease-head confusion matrix")
	plt.xlabel("Predicted class index")
	plt.ylabel("True class index")
	plt.colorbar()
	plt.tight_layout()
	plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
	plt.close()

	if targets["mainclass"] and targets["subclass"]:
		by_disease = {int(row["disease_idx"]): row for row in hierarchy_rows}
		consistent = []
		coarse_correct = []
		for index, disease_prediction in enumerate(disease_pred):
			mapping = by_disease[int(disease_prediction)]
			consistent.append(mapping["mainclass_idx"] == predictions["mainclass"][index] and mapping["subclass_idx"] == predictions["subclass"][index])
			coarse_correct.append(mapping["mainclass_idx"] == targets["mainclass"][index])
		result["hierarchical_consistency_rate"] = float(np.mean(consistent))
		result["coarse_correct_fraction"] = float(np.mean(coarse_correct))
		result["coarse_wrong_fraction"] = float(1.0 - np.mean(coarse_correct))

	(output_dir / "evaluation_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
	lines = ["# Evaluation Report", "", "Test set: reconciled `test.csv`; excluded raw test rows were not evaluated because their labels were outside the train-derived production class map.", "", "| Metric | Value |", "|---|---:|"]
	lines.extend(f"| {key} | {value:.6f} |" for key, value in result.items() if isinstance(value, (int, float)))
	lines.extend(["", "Auxiliary mainclass/subclass metrics are reported only when the selected checkpoint contains hierarchical heads.", "The calibration value is expected calibration error over 10 confidence bins."])
	(output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
	return result


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint", type=Path, default=Path(config.BEST_MODEL))
	parser.add_argument("--output", type=Path, default=Path(config.REPORTS_DIR))
	args = parser.parse_args()
	device = get_device(config.DEVICE)
	data_dir = Path(config.DATA_PROCESSED)
	labels = json.loads((data_dir / "label2idx.json").read_text())
	hierarchy = json.loads(Path(config.HIERARCHY_MAP_PATH).read_text())["rows"]
	from dataset_discovery import discover_dataset, validate_dataset

	discovered = discover_dataset(Path(config.DATA_RAW))
	validation = validate_dataset(discovered)
	test_df = pd.read_csv(data_dir / "test.csv")
	hierarchy_map = {row["disease"]: row for row in hierarchy}
	image_column = config.IMG_COL if config.IMG_COL in test_df.columns else validation["test_image_column"]
	label_column = config.LABEL_COL if config.LABEL_COL in test_df.columns else validation["test_label_column"]
	dataset = DermaSenseDataset(test_df, discovered.image_dirs[0], labels, image_column, label_column, config.IMG_SIZE, False, list(discovered.image_dirs[1:]), hierarchy_map)
	loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
	checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
	model_spec = checkpoint.get("model_config", {"use_metadata": True, "fusion_mode": "cross_attention", "hierarchical": True})
	model = build_model(len(labels), config.TABULAR_DIM, config.FUSION_DIM, config.DROPOUT, pretrained=False, fusion_mode=model_spec["fusion_mode"], hierarchical=model_spec["hierarchical"], use_metadata=model_spec["use_metadata"], num_main_classes=config.NUM_MAIN_CLASSES, num_subclass_classes=config.NUM_SUBCLASS_CLASSES, d_model=config.D_MODEL, n_attn_layers=config.N_ATTN_LAYERS, n_heads=config.N_HEADS)
	model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("model_state")))
	result = evaluate(model.to(device), loader, device, hierarchy, args.output)
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()
