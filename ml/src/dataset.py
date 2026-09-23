import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import metadata_columns


def get_tabular_cols(df_columns):
    return metadata_columns(df_columns)


def find_image(img_dir, filename, extra_dirs=None):
    path = os.path.join(img_dir, filename)
    if os.path.exists(path):
        return path
    if extra_dirs:
        for d in extra_dirs:
            p = os.path.join(d, filename)
            if os.path.exists(p):
                return p
    return None


def get_transforms(img_size: int, train: bool):
    if train:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(10, 30),
                hole_width_range=(10, 30),
                p=0.2
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=img_size, width=img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class DermaSenseDataset(Dataset):
    def __init__(self, df, img_dir, label2idx, img_col,
                 label_col, img_size=300, train=True, extra_dirs=None,
                 hierarchy_map=None):
        self.df         = df.reset_index(drop=True)
        self.img_dir    = img_dir
        self.extra_dirs = extra_dirs or []
        self.label2idx  = label2idx
        self.img_col    = img_col
        self.label_col  = label_col
        self.transforms = get_transforms(img_size, train)
        self.tab_cols   = get_tabular_cols(df.columns)
        self.hierarchy_map = hierarchy_map or {}

        if "Fitzpatrick" in self.df.columns:
            self.df["Fitzpatrick"] = (
                self.df["Fitzpatrick"]
                .astype(str)
                .str.extract(r"(\d+)")
                .astype(float)
            )
            median_fitz = self.df["Fitzpatrick"].median()
            self.df["Fitzpatrick"] = self.df["Fitzpatrick"].fillna(median_fitz)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = find_image(
            self.img_dir,
            str(row[self.img_col]),
            self.extra_dirs
        )
        if img_path is None:
            raise FileNotFoundError(
                f"Image not found: {row[self.img_col]}"
            )

        image   = np.array(Image.open(img_path).convert("RGB"))
        image   = self.transforms(image=image)["image"]

        tabular = torch.tensor(
            row[self.tab_cols].values.astype(np.float32),
            dtype=torch.float32
        )
        disease = row[self.label_col]
        label   = torch.tensor(
            self.label2idx[disease],
            dtype=torch.long
        )
        mapping = self.hierarchy_map.get(disease, {})
        targets = {
            "disease": label,
            "mainclass": torch.tensor(mapping.get("mainclass_idx", -1), dtype=torch.long),
            "subclass": torch.tensor(mapping.get("subclass_idx", -1), dtype=torch.long),
        }
        return image, tabular, targets