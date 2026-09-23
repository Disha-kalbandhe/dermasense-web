"""Validate hierarchy coverage, native taxonomy membership, and provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_PROVENANCE = {
    "source",
    "vote_share_before_override",
    "mapping_status",
    "manually_reviewed",
    "review_reason",
}


def validate_hierarchy(map_path: Path, labels_path: Path, metadata_path: Path) -> None:
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    metadata = pd.read_csv(metadata_path, low_memory=False)
    rows = payload["rows"]
    by_disease = {row["disease"]: row for row in rows}
    if len(rows) != len(labels) or set(by_disease) != set(labels):
        raise AssertionError("Hierarchy does not cover exactly the labels in label2idx.json.")
    main_values = set(metadata["Main_class"].dropna())
    subclass_values = set(metadata["Sub_class"].dropna())
    for row in rows:
        if labels[row["disease"]] != row["disease_idx"]:
            raise AssertionError(f"Disease index mismatch: {row['disease']}")
        if row["mainclass"] not in main_values or row["subclass"] not in subclass_values:
            raise AssertionError(f"Non-native hierarchy value: {row['disease']}")
        if not REQUIRED_PROVENANCE.issubset(row):
            raise AssertionError(f"Missing provenance: {row['disease']}")
    parasitic = "Infectious skin conditions -Parasitic"
    for disease in ("Pustular Scabies", "Scabies Infected"):
        row = by_disease[disease]
        if row["subclass"] != parasitic or " + " in row["subclass"]:
            raise AssertionError(f"Scabies mapping is not single native parasitic: {disease}")
    print(f"HIERARCHY_VALIDATION_PASS rows={len(rows)} labels={len(labels)} native_taxonomy=true scabies_single_parasitic=true provenance=true")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", dest="map_path", type=Path, default=root / "data/processed/hierarchy_map.json")
    parser.add_argument("--labels", type=Path, default=root / "data/processed/label2idx.json")
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()
    metadata = args.metadata
    if metadata is None:
        from dataset_discovery import discover_dataset

        metadata = discover_dataset(root / "data" / "raw").metadata_csv
    if metadata is None:
        raise FileNotFoundError("SkinMetadata.csv was not found during dataset discovery.")
    validate_hierarchy(args.map_path, args.labels, metadata)


if __name__ == "__main__":
    main()