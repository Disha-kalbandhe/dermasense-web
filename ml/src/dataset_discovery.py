"""Discover and validate a DermaCon-IN dataset without assuming a folder layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IGNORED_NAMES = {"__pycache__", ".git", ".ipynb_checkpoints", "node_modules"}
IGNORED_SUFFIXES = {".ipynb", ".py", ".pyc"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    split_dir: Path
    train_csv: Path
    test_csv: Path
    image_root: Path
    image_dirs: tuple[Path, ...]
    metadata_csv: Path | None


def _walk_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = set(path.relative_to(root).parts)
        if relative_parts & IGNORED_NAMES or path.suffix.lower() in IGNORED_SUFFIXES:
            continue
        yield path


def _casefold_name(path: Path) -> str:
    return path.name.casefold()


def _find_split_pair(root: Path) -> tuple[Path, Path]:
    train_files = [path for path in _walk_files(root) if _casefold_name(path) == "train_split.csv"]
    test_files = [path for path in _walk_files(root) if _casefold_name(path) == "test_split.csv"]
    pairs = [(train, test) for train in train_files for test in test_files if train.parent == test.parent]
    if not pairs:
        raise FileNotFoundError(
            f"Could not find train_split.csv and test_split.csv in the same directory below {root}."
        )
    return pairs[0]


def _image_directories(dataset_root: Path) -> tuple[Path, ...]:
    directories = []
    for directory in [dataset_root, *[path for path in dataset_root.rglob("*") if path.is_dir()]]:
        if set(directory.relative_to(dataset_root).parts) & IGNORED_NAMES:
            continue
        if any(path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES for path in directory.iterdir()):
            directories.append(directory)
    if not directories:
        raise FileNotFoundError(f"No image directory was found below {dataset_root}.")
    return tuple(sorted(set(directories), key=lambda path: str(path).casefold()))


def discover_dataset(root: Path) -> DatasetPaths:
    root = root.resolve()
    train_csv, test_csv = _find_split_pair(root)
    dataset_container = train_csv.parent
    dataset_candidates = [path for path in dataset_container.rglob("*") if path.is_dir() and path.name.casefold() == "dataset"]
    dataset_candidates = [path for path in dataset_candidates if not set(path.relative_to(dataset_container).parts) & IGNORED_NAMES]
    if not dataset_candidates:
        raise FileNotFoundError(f"Could not find a DATASET image directory below {root}.")
    image_root = max(dataset_candidates, key=lambda path: len(_image_directories(path)))
    image_dirs = _image_directories(image_root)
    metadata_candidates = [
        path for path in _walk_files(dataset_container)
        if path.suffix.casefold() == ".csv" and path.stem.replace("_", "").casefold() == "skinmetadata"
    ]
    metadata_csv = metadata_candidates[0] if metadata_candidates else None
    return DatasetPaths(dataset_container, dataset_container, train_csv, test_csv, image_root, image_dirs, metadata_csv)


def resolve_image(filename: str, image_dirs: tuple[Path, ...]) -> Path | None:
    for directory in image_dirs:
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def _column(frame: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    lookup = {name.casefold(): name for name in frame.columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    if required:
        raise ValueError(f"Missing required column; tried {candidates}, available columns: {list(frame.columns)}")
    return None


def validate_dataset(paths: DatasetPaths) -> dict:
    train = pd.read_csv(paths.train_csv, low_memory=False)
    test = pd.read_csv(paths.test_csv, low_memory=False)
    image_column = _column(train, ("Image_name", "ImageName", "filename", "file_name"))
    test_image_column = _column(test, (image_column, "Image_name", "ImageName", "filename", "file_name"))
    label_column = _column(train, ("Disease_label", "Diseaselabel", "DiseaseLabel"))
    test_label_column = _column(test, (label_column, "Disease_label", "Diseaselabel", "DiseaseLabel"))
    main_column = _column(train, ("Main_class", "Mainclass"), required=False)
    subclass_column = _column(train, ("Sub_class", "Subclass"), required=False)
    subject_column = _column(train, ("Subject_ID", "SubjectID", "SubjectId"), required=False)
    test_subject_column = _column(test, (subject_column or "Subject_ID", "SubjectID", "SubjectId"), required=False)

    all_images = [path for directory in paths.image_dirs for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    train_paths = [resolve_image(str(value), paths.image_dirs) for value in train[image_column].astype(str)]
    test_paths = [resolve_image(str(value), paths.image_dirs) for value in test[test_image_column].astype(str)]
    duplicate_filenames = int(pd.concat([train[image_column], test[test_image_column]], ignore_index=True).duplicated().sum())
    subject_overlap = None
    if subject_column and test_subject_column:
        subject_overlap = len(set(train[subject_column].dropna().astype(str)) & set(test[test_subject_column].dropna().astype(str)))

    print("RESOLVED_DATASET_PATHS")
    print(f"root={paths.root}")
    print(f"split_dir={paths.split_dir}")
    print(f"train_csv={paths.train_csv}")
    print(f"test_csv={paths.test_csv}")
    print(f"image_root={paths.image_root}")
    print(f"image_dirs={list(paths.image_dirs)}")
    print(f"metadata_csv={paths.metadata_csv}")
    print("DATASET_VALIDATION")
    print(f"train_rows={len(train)} test_rows={len(test)} image_count={len(all_images)}")
    print(f"image_column={image_column} label_column={label_column} mainclass_column={main_column} subclass_column={subclass_column}")
    print(f"disease_labels_train={train[label_column].nunique(dropna=True)} disease_labels_test={test[test_label_column].nunique(dropna=True)}")
    print(f"mainclass_values={sorted(train[main_column].dropna().unique().tolist()) if main_column else []}")
    print(f"subclass_values={sorted(train[subclass_column].dropna().unique().tolist()) if subclass_column else []}")
    print(f"missing_metadata_values={train.isna().sum().loc[lambda series: series > 0].to_dict()}")
    print(f"duplicate_image_filenames={duplicate_filenames}")
    print(f"subject_overlap_train_test={subject_overlap}")
    print(f"missing_train_image_paths={sum(path is None for path in train_paths)} missing_test_image_paths={sum(path is None for path in test_paths)}")
    print(f"sample_train_paths={[str(path) for path in train_paths[:5]]}")
    print(f"sample_test_paths={[str(path) for path in test_paths[:5]]}")

    if any(path is None for path in train_paths + test_paths):
        raise FileNotFoundError("Dataset validation failed: one or more CSV image filenames could not be resolved.")
    if not all_images:
        raise FileNotFoundError("Dataset validation failed: no image files were found.")
    return {
        "train": train,
        "test": test,
        "image_column": image_column,
        "test_image_column": test_image_column,
        "label_column": label_column,
        "test_label_column": test_label_column,
        "mainclass_column": main_column,
        "subclass_column": subclass_column,
        "subject_column": subject_column,
        "test_subject_column": test_subject_column,
    }


def print_tree(root: Path) -> None:
    print(f"DATASET_TREE root={root}")
    for path in sorted(_walk_files(root), key=lambda item: str(item).casefold()):
        print(path.relative_to(root))
