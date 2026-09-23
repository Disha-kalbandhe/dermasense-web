"""Gradient-derived Grad-CAM++ overlays for the disease head."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
from PIL import Image
import pandas as pd
import json
from pathlib import Path


def _last_convolution(module):
	layers = [layer for layer in module.modules() if isinstance(layer, torch.nn.Conv2d)]
	if not layers:
		raise ValueError("The backbone has no convolutional layer for Grad-CAM++.")
	return layers[-1]


class GradCAMPlusPlus:
	def __init__(self, model):
		self.model = model
		self.activations = None
		self.gradients = None
		self.target_layer = _last_convolution(model.backbone)
		self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
		self.backward_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

	def _save_activation(self, _, __, output):
		self.activations = output

	def _save_gradient(self, _, __, grad_output):
		self.gradients = grad_output[0]

	def close(self):
		self.forward_handle.remove()
		self.backward_handle.remove()

	def __call__(self, image, tabular):
		self.model.zero_grad(set_to_none=True)
		outputs = self.model(image, tabular)
		logits = outputs["disease"] if isinstance(outputs, dict) else outputs
		class_index = int(logits.argmax(1).item())
		logits[:, class_index].sum().backward()
		activations = self.activations.float()
		gradients = self.gradients.float()
		spatial_sum = gradients.square().sum(dim=(2, 3), keepdim=True)
		alpha = gradients.square() / (2.0 * gradients.square() + (activations * gradients.pow(3)).sum(dim=(2, 3), keepdim=True) + 1e-7)
		weights = (alpha * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)
		heatmap = torch.relu((weights * activations).sum(dim=1, keepdim=True))
		heatmap = torch.nn.functional.interpolate(heatmap, size=image.shape[-2:], mode="bicubic", align_corners=False)
		heatmap = heatmap.squeeze().detach().cpu().numpy()
		heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
		return class_index, heatmap


def overlay_base64(original_rgb: Image.Image, heatmap: np.ndarray, alpha: float = 0.4):
	original = np.asarray(original_rgb.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0])))
	colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_INFERNO)
	blended = cv2.addWeighted(original[:, :, ::-1], 1 - alpha, colored, alpha, 0)[:, :, ::-1]
	buffer = io.BytesIO()
	Image.fromarray(blended).save(buffer, format="PNG")
	return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def generate_samples(checkpoint_path: Path, count: int = 8) -> None:
	import sys
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
	import config
	from dataset import get_transforms, find_image
	from model import build_model
	from utils import build_tabular_vector, metadata_columns

	device = torch.device("cpu")
	checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
	labels = json.loads((Path(config.DATA_PROCESSED) / "label2idx.json").read_text(encoding="utf-8"))
	spec = checkpoint["model_config"]
	model = build_model(len(labels), config.TABULAR_DIM, config.FUSION_DIM, config.DROPOUT, pretrained=False, fusion_mode=spec["fusion_mode"], hierarchical=spec["hierarchical"], use_metadata=spec["use_metadata"], num_main_classes=config.NUM_MAIN_CLASSES, num_subclass_classes=config.NUM_SUBCLASS_CLASSES, d_model=config.D_MODEL, n_attn_layers=config.N_ATTN_LAYERS, n_heads=config.N_HEADS)
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device).eval()
	cam = GradCAMPlusPlus(model)
	df = pd.read_csv(Path(config.DATA_PROCESSED) / "test.csv").head(count)
	columns = metadata_columns(df.columns)
	transform = get_transforms(config.IMG_SIZE, False)
	output = Path(config.OUTPUTS_DIR) / "gradcam_samples"
	output.mkdir(parents=True, exist_ok=True)
	for row_number, row in df.iterrows():
		image_path = find_image(config.DATASET_DIR_0, str(row[config.IMG_COL]), [config.DATASET_DIR_1])
		original = Image.open(image_path).convert("RGB")
		image = transform(image=np.asarray(original))["image"].unsqueeze(0).to(device)
		tabular = build_tabular_vector("Other", [], columns).unsqueeze(0).to(device)
		class_index, heatmap = cam(image, tabular)
		colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_INFERNO)
		base = np.asarray(original.resize((heatmap.shape[1], heatmap.shape[0])))[:, :, ::-1]
		blended = cv2.addWeighted(base, 0.6, colored, 0.4, 0)
		cv2.imwrite(str(output / f"sample_{row_number:02d}_{class_index}.png"), blended)
	cam.close()
	print(f"GRADCAM_SAMPLES={len(list(output.glob('sample_*.png')))} output={output}")


if __name__ == "__main__":
	generate_samples(Path(__file__).resolve().parents[1] / "checkpoints" / "best_model.pth")
