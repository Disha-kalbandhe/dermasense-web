"""Validate the raw DermaSense splits before any model selection or training."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import pandas as pd


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REQUIRED_MISSINGNESS_COLUMNS = ("Sex", "Age", "Fitzpatrick", "Bodypart", "Descriptors")


def _column(frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    lookup = {str(name).casefold(): str(name) for name in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    if required:
        raise ValueError(f"Missing required column; tried {candidates}")
    return None


def _missing_rate(frame: pd.DataFrame, candidates: tuple[str, ...], prefix: str) -> float | None:
    column = _column(frame, candidates, required=False)
    if column:
        return float(frame[column].isna().mean())
    matching = [name for name in frame.columns if str(name).startswith(prefix)]
    if matching:
        return float(frame[matching].isna().any(axis=1).mean())
    return None


def validate(raw_dir: Path, processed_dir: Path) -> int:
    raw_dir = raw_dir.resolve()
    train_path = raw_dir / "train_split.csv"
    test_path = raw_dir / "test_split.csv"
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    label2idx = json.loads((processed_dir / "label2idx.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((processed_dir / "hierarchy_map.json").read_text(encoding="utf-8"))
    hierarchy_rows = hierarchy.get("rows", [])
    processed_train = pd.read_csv(processed_dir / "train.csv", low_memory=False)
    processed_val = pd.read_csv(processed_dir / "val.csv", low_memory=False)
    processed_test = pd.read_csv(processed_dir / "test.csv", low_memory=False)

    image_column = _column(train, ("Image_name", "ImageName", "filename", "file_name"))
    test_image_column = _column(test, (image_column, "Image_name", "ImageName", "filename", "file_name"))
    label_column = _column(train, ("Disease_label", "Diseaselabel", "DiseaseLabel"))
    test_label_column = _column(test, (label_column, "Disease_label", "Diseaselabel", "DiseaseLabel"))
    subject_column = _column(train, ("Subject_ID", "SubjectID", "SubjectId"), required=False)
    test_subject_column = _column(test, (subject_column or "Subject_ID", "SubjectID", "SubjectId"), required=False)

    image_files = {path.name for path in (raw_dir / "DATASET").rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    train_names = set(train[image_column].astype(str))
    test_names = set(test[test_image_column].astype(str))
    train_labels = set(train[label_column].dropna())
    test_labels = set(test[test_label_column].dropna())
    expected_labels = set(label2idx)
    hierarchy_labels = {row.get("disease") for row in hierarchy_rows}

    print("RESOLVED_DATASET_PATHS")
    print(f"raw_dir={raw_dir}")
    print(f"train_csv={train_path.resolve()}")
    print(f"test_csv={test_path.resolve()}")
    print(f"image_root={(raw_dir / 'DATASET').resolve()}")
    print(f"metadata_csv={(raw_dir / 'Skin_Metadata.csv').resolve()}")
    print("RUNTIME")
    print(f"os={platform.platform()}")
    print(f"python={sys.version.split()[0]}")
    print(f"cpu_cores={os.cpu_count()}")
    try:
        import psutil

        print(f"available_ram_gb={psutil.virtual_memory().available / 1024**3:.2f}")
    except ImportError:
        print("available_ram_gb=unavailable (psutil is not installed)")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"gpu_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
    except ImportError:
        print("torch=unavailable")
    print("DATA_VALIDATION")
    print(f"train_rows={len(train)} test_rows={len(test)}")
    print(f"train_unique_labels={len(train_labels)} test_unique_labels={len(test_labels)} expected_labels={len(expected_labels)}")
    print(f"excluded_raw_train_labels={json.dumps(sorted(train_labels - expected_labels), ensure_ascii=True)}")
    print(f"excluded_raw_test_labels={json.dumps(sorted(test_labels - expected_labels), ensure_ascii=True)}")
    print(f"processed_train_rows={len(processed_train)} processed_val_rows={len(processed_val)} processed_test_rows={len(processed_test)}")
    processed_labels = set(processed_train[label_column]) | set(processed_val[label_column]) | set(processed_test[label_column])
    print(f"processed_unsupported_labels={json.dumps(sorted(processed_labels - expected_labels), ensure_ascii=True)}")
    print(f"processed_missing_expected_labels={json.dumps(sorted(expected_labels - processed_labels), ensure_ascii=True)}")
    print(f"hierarchy_rows={len(hierarchy_rows)} hierarchy_labels_match_map={hierarchy_labels == expected_labels}")
    print(f"image_files={len(image_files)} referenced_unique={len(train_names | test_names)}")
    print(f"missing_train_images={json.dumps(sorted(train_names - image_files)[:20], ensure_ascii=True)}")
    print(f"missing_test_images={json.dumps(sorted(test_names - image_files)[:20], ensure_ascii=True)}")
    print(f"duplicate_filenames_across_splits={len(train_names & test_names)}")
    counts = train[label_column].value_counts()
    print(f"train_class_counts_min={int(counts.min())} median={float(counts.median())} max={int(counts.max())} imbalance_ratio={float(counts.max() / counts.min()):.3f}")
    production_counts = processed_train[label_column].value_counts()
    print(f"production_train_class_counts_min={int(production_counts.min())} median={float(production_counts.median())} max={int(production_counts.max())} imbalance_ratio={float(production_counts.max() / production_counts.min()):.3f}")
    for name, rate in {
        "Sex": _missing_rate(train, ("Sex",), ""),
        "Age": _missing_rate(train, ("Age",), ""),
        "Fitzpatrick": _missing_rate(train, ("Fitzpatrick",), ""),
        "Bodypart": _missing_rate(train, ("Bodypart",), "Body_part_"),
        "Descriptors": _missing_rate(train, ("Descriptors",), "Descriptor_"),
    }.items():
        print(f"missing_rate_{name}={rate if rate is not None else 'column_not_present'}")
    if subject_column and test_subject_column:
        overlap = set(train[subject_column].dropna().astype(str)) & set(test[test_subject_column].dropna().astype(str))
        print(f"subject_column={subject_column} subject_overlap_count={len(overlap)}")
    else:
        overlap = set()
        print("subject_overlap_count=unavailable")

    blocking = {
        "missing_images": sorted((train_names | test_names) - image_files),
        "filename_overlap": sorted(train_names & test_names),
        "processed_unsupported_labels": sorted(processed_labels - expected_labels),
        "processed_missing_expected_labels": sorted(expected_labels - processed_labels),
        "subject_overlap": sorted(overlap),
    }
    blocking_items = {name: values for name, values in blocking.items() if values}
    print(f"BLOCKING_CHECKS_PASS={not blocking_items}")
    if blocking_items:
        print("BLOCKING_PROBLEMS=" + json.dumps(blocking_items, ensure_ascii=True))
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "raw")
    parser.add_argument("--processed-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed")
    args = parser.parse_args()
    raise SystemExit(validate(args.raw_dir, args.processed_dir))