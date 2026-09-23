import hashlib
import json
import os, random
import numpy as np
import torch

BODY_PART_PREFIX = "Body_part_"
DESCRIPTOR_PREFIX = "Descriptor_"

SYMPTOM_TO_DESCRIPTOR = {
    "Itching": ["Descriptor_Excoriation", "Descriptor_Xerosis"],
    "Pain": ["Descriptor_Erosion", "Descriptor_Ulcer"],
    "Bleeding": ["Descriptor_Purpura/Petechiae", "Descriptor_Erosion"],
    "Color change": ["Descriptor_Brown (Hyperpigmentation)", "Descriptor_White (Hypopigmentation)"],
    "Size increase": ["Descriptor_Nodule", "Descriptor_Plaque"],
    "Scaling": ["Descriptor_Scale", "Descriptor_Hyperkeratotic plaques"],
    "Discharge": ["Descriptor_Exudate", "Descriptor_Pustule"],
    "None": [],
}

LOCATION_TO_BODY = {
    "Face": ["Body_part_Head Cheeks", "Body_part_Head Forehead", "Body_part_Head Chin", "Body_part_Head Nose"],
    "Scalp": ["Body_part_Head Scalp"],
    "Neck": ["Body_part_Neck"],
    "Chest": ["Body_part_Trunk Chest (Thorax)"],
    "Back": ["Body_part_Back", "Body_part_Trunk Back (Dorsum)"],
    "Arm": ["Body_part_Upper Extremities Upper Arms (Brachium)", "Body_part_Upper Extremities Forearms (Antebrachium)"],
    "Hand": ["Body_part_Upper Extremities Hands (Manus)"],
    "Leg": ["Body_part_Lower Extremities Thighs (Femoral Region)", "Body_part_Lower Extremities Lower Legs (Crural Region)"],
    "Foot": ["Body_part_Lower Extremities Feet (Pedal Region)", "Body_part_Lower Extremities Soles (Plantar Region)"],
    "Other": [],
}


def metadata_columns(columns):
    body = sorted(c for c in columns if c.startswith(BODY_PART_PREFIX))
    descriptors = sorted(c for c in columns if c.startswith(DESCRIPTOR_PREFIX))
    return body + descriptors


def build_tabular_vector(body_location, symptoms, columns=None):
    columns = columns or sorted(set(sum(LOCATION_TO_BODY.values(), []) + sum(SYMPTOM_TO_DESCRIPTOR.values(), [])))
    vector = np.zeros(len(columns), dtype=np.float32)
    index = {name: position for position, name in enumerate(columns)}
    for name in LOCATION_TO_BODY.get(body_location, []):
        if name in index:
            vector[index[name]] = 1.0
    for symptom in symptoms:
        for name in SYMPTOM_TO_DESCRIPTOR.get(symptom, []):
            if name in index:
                vector[index[name]] = 1.0
    return torch.tensor(vector, dtype=torch.float32)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_label_map(series):
    classes = sorted(series.dropna().unique().tolist())
    l2i = {c: i for i, c in enumerate(classes)}
    i2l = {i: c for c, i in l2i.items()}
    return l2i, i2l


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"  ✅ Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer=None):
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
    model.load_state_dict(state)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    elif optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    print(f"  ✅ Checkpoint loaded ← {path}")
    return ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0.0)