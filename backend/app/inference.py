from __future__ import annotations

import json
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2

ML_ROOT = Path(__file__).resolve().parents[2] / "ml"
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT / "src"))

from config import (  # noqa: E402
    BEST_MODEL,
    D_MODEL,
    DROPOUT,
    FUSION_DIM,
    HIERARCHY_MAP_PATH,
    IMG_SIZE,
    N_ATTN_LAYERS,
    N_HEADS,
    TABULAR_DIM,
    NUM_MAIN_CLASSES,
    NUM_SUBCLASS_CLASSES,
)
from gradcam import GradCAMPlusPlus, overlay_base64  # noqa: E402
from model import build_model  # noqa: E402
from utils import build_tabular_vector, file_sha256, metadata_columns  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = ML_ROOT / "data" / "processed"
LABEL2IDX = json.loads((DATA_DIR / "label2idx.json").read_text(encoding="utf-8"))
IDX2LABEL = {int(index): label for label, index in LABEL2IDX.items()}
HIERARCHY_ROWS = json.loads(Path(HIERARCHY_MAP_PATH).read_text(encoding="utf-8"))["rows"]
HIERARCHY_BY_INDEX = {int(row["disease_idx"]): row for row in HIERARCHY_ROWS}
TABULAR_COLUMNS = metadata_columns(pd.read_csv(DATA_DIR / "train.csv", nrows=1).columns)
_TRANSFORM = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
_model = None
_cam = None
_model_version = None


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    return _TRANSFORM(image=np.asarray(pil_image.convert("RGB")))["image"].unsqueeze(0)


def get_model():
    global _model, _cam, _model_version
    if _model is not None:
        return _model
    checkpoint_path = Path(BEST_MODEL)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint is not available: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_hierarchy = file_sha256(Path(HIERARCHY_MAP_PATH))
    expected_labels = file_sha256(DATA_DIR / "label2idx.json")
    if checkpoint.get("hierarchy_version") not in {None, expected_hierarchy}:
        raise RuntimeError("Checkpoint hierarchy_version does not match hierarchy_map.json")
    if checkpoint.get("class_mapping_version") not in {None, expected_labels}:
        raise RuntimeError("Checkpoint class_mapping_version does not match label2idx.json")
    model_spec = checkpoint.get("model_config", {"use_metadata": True, "fusion_mode": "cross_attention", "hierarchical": True})
    _model = build_model(
        len(LABEL2IDX), TABULAR_DIM, FUSION_DIM, DROPOUT,
        pretrained=False, fusion_mode=model_spec["fusion_mode"], hierarchical=model_spec["hierarchical"], use_metadata=model_spec["use_metadata"],
        num_main_classes=NUM_MAIN_CLASSES, num_subclass_classes=NUM_SUBCLASS_CLASSES,
        d_model=D_MODEL, n_attn_layers=N_ATTN_LAYERS, n_heads=N_HEADS,
    )
    _model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("model_state")), strict=True)
    _model.to(DEVICE).eval()
    _cam = GradCAMPlusPlus(_model)
    _model_version = checkpoint_path.name + ":" + file_sha256(checkpoint_path)[:12]
    return _model


def checkpoint_status() -> dict:
    return {"loaded": _model is not None, "available": Path(BEST_MODEL).exists(), "version": _model_version}


def run_inference(pil_image, age=None, gender=None, skin_tone=None, body_location="Other", duration=None, symptoms=None, top_k=3):
    model = get_model()
    symptoms = symptoms or []
    image_tensor = preprocess_image(pil_image).to(DEVICE)
    tabular = build_tabular_vector(body_location, symptoms, TABULAR_COLUMNS).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        raw_outputs = model(image_tensor, tabular)
        outputs = raw_outputs if isinstance(raw_outputs, dict) else {"disease": raw_outputs}
        disease_probs = torch.softmax(outputs["disease"], dim=1)[0]
        main_probs = torch.softmax(outputs["mainclass"], dim=1)[0] if "mainclass" in outputs else None
        subclass_probs = torch.softmax(outputs["subclass"], dim=1)[0] if "subclass" in outputs else None
    top_probs, top_indices = disease_probs.topk(min(top_k, len(LABEL2IDX)))
    primary_index = int(top_indices[0])
    hierarchy = HIERARCHY_BY_INDEX[primary_index]
    heatmap_index, heatmap = _cam(image_tensor, tabular)
    if heatmap_index != primary_index:
        primary_index = heatmap_index
        hierarchy = HIERARCHY_BY_INDEX[primary_index]
    predicted_main = int(outputs["mainclass"].argmax(1).item()) if "mainclass" in outputs else None
    predicted_subclass = int(outputs["subclass"].argmax(1).item()) if "subclass" in outputs else None
    return {
        "primary_condition": IDX2LABEL[primary_index],
        "primary_confidence": float(disease_probs[primary_index]),
        "disease_family": hierarchy["mainclass"],
        "disease_subgroup": hierarchy["subclass"],
        "hierarchy_path": [hierarchy["mainclass"], hierarchy["subclass"], IDX2LABEL[primary_index]],
        "hierarchy_confidence": {"mainclass": float(main_probs[predicted_main]), "subclass": float(subclass_probs[predicted_subclass])} if main_probs is not None and subclass_probs is not None else {},
        "differentials": [{"condition": IDX2LABEL[int(index)], "confidence": float(prob)} for prob, index in zip(top_probs, top_indices)],
        "hierarchical_consistency": predicted_main == hierarchy["mainclass_idx"] and predicted_subclass == hierarchy["subclass_idx"] if predicted_main is not None and predicted_subclass is not None else None,
        "explainability": {"available": True, "heatmap_base64": overlay_base64(pil_image, heatmap)},
        "model_version": _model_version,
        "disclaimer": "AI-assisted screening only; not a medical diagnosis. Consult a qualified dermatologist.",
    }
