"""Reconcile production labels and derived artifacts from the raw splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
LABEL_COLUMN = "Disease_label"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def majority_mapping(frame: pd.DataFrame, label: str, main_to_idx: dict[str, int], sub_to_idx: dict[str, int]) -> dict:
    rows = frame[frame[LABEL_COLUMN] == label]
    pairs = rows[["Main_class", "Sub_class"]].value_counts()
    (main, subclass), votes = pairs.index[0], int(pairs.iloc[0])
    return {
        "disease": label,
        "mainclass": main,
        "subclass": subclass,
        "mainclass_idx": main_to_idx[main],
        "subclass_idx": sub_to_idx[subclass],
        "majority_fraction": round(votes / len(rows), 6),
        "mapping_variants": int(len(pairs)),
        "vote_share_before_override": round(votes / len(rows), 6),
        "source": "DermaCon-IN native Main_class/Sub_class annotation plus reviewed normalization",
        "mapping_status": "native_majority_vote",
        "manually_reviewed": False,
        "review_reason": None,
    }


def main() -> None:
    train = pd.read_csv(RAW / "train_split.csv", low_memory=False)
    test = pd.read_csv(RAW / "test_split.csv", low_memory=False)
    counts_train = train[LABEL_COLUMN].value_counts()
    counts_test = test[LABEL_COLUMN].value_counts()
    all_labels = sorted(set(counts_train.index) | set(counts_test.index))
    kept = [label for label in all_labels if counts_train.get(label, 0) >= 3 and counts_test.get(label, 0) >= 1]
    excluded = [label for label in all_labels if label not in kept]

    label2idx = {label: index for index, label in enumerate(kept)}
    idx2label = {str(index): label for label, index in label2idx.items()}
    (PROCESSED / "label2idx.json").write_text(json.dumps(label2idx, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (PROCESSED / "idx2label.json").write_text(json.dumps(idx2label, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    previous_rows = {row["disease"]: row for row in json.loads((PROCESSED / "hierarchy_map.json").read_text(encoding="utf-8")).get("rows", [])}
    main_values = sorted(train["Main_class"].dropna().unique())
    subclass_values = sorted(train["Sub_class"].dropna().unique())
    main_to_idx = {name: index for index, name in enumerate(main_values)}
    sub_to_idx = {name: index for index, name in enumerate(subclass_values)}
    hierarchy_rows = []
    manual_applied = []
    manual_excluded = []
    for label in kept:
        row = majority_mapping(train, label, main_to_idx, sub_to_idx)
        previous = previous_rows.get(label)
        if previous and previous.get("manually_reviewed"):
            row.update({
                "mainclass": previous["mainclass"],
                "subclass": previous["subclass"],
                "mainclass_idx": main_to_idx[previous["mainclass"]],
                "subclass_idx": sub_to_idx[previous["subclass"]],
                "mapping_status": previous.get("mapping_status", "manually_reviewed"),
                "manually_reviewed": True,
                "review_reason": previous.get("review_reason"),
            })
            manual_applied.append(label)
        hierarchy_rows.append({"disease_idx": label2idx[label], **row})
    for label, previous in previous_rows.items():
        if previous.get("manually_reviewed") and label not in kept:
            manual_excluded.append(label)

    hierarchy = {
        "schema_version": 2,
        "source": "DermaCon-IN Skin_Metadata.csv and raw train_split.csv",
        "class_mapping_sha256": sha256(PROCESSED / "label2idx.json"),
        "production_class_count": len(kept),
        "mainclass_count": len(main_values),
        "subclass_count": len(subclass_values),
        "hierarchy_version": hashlib.sha256(json.dumps(hierarchy_rows, sort_keys=True).encode()).hexdigest(),
        "rows": hierarchy_rows,
    }
    hierarchy_path = PROCESSED / "hierarchy_map.json"
    hierarchy_path.write_text(json.dumps(hierarchy, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    old_rare_path = PROCESSED / "rare_classes.json"
    try:
        old_rare = json.loads(old_rare_path.read_text(encoding="utf-8"))
        old_rare_labels = list(old_rare) if isinstance(old_rare, list) else list(old_rare)
    except (FileNotFoundError, json.JSONDecodeError):
        old_rare_labels = []
    rare = {}
    for label in excluded:
        reasons = []
        if counts_train.get(label, 0) < 3:
            reasons.append("insufficient_train_samples")
        if counts_test.get(label, 0) < 1:
            reasons.append("missing_from_test")
        rare[label] = {
            "train_count": int(counts_train.get(label, 0)),
            "test_count": int(counts_test.get(label, 0)),
            "reasons": reasons,
            "previously_known_rare": label in old_rare_labels,
        }
    (PROCESSED / "rare_classes.json").write_text(json.dumps(rare, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    kept_train = train[train[LABEL_COLUMN].isin(kept)].copy()
    kept_test = test[test[LABEL_COLUMN].isin(kept)].copy()
    train_part, val_part = train_test_split(
        kept_train,
        test_size=0.15,
        random_state=42,
        stratify=kept_train[LABEL_COLUMN],
    )
    train_part = train_part.sort_values("Image_name").reset_index(drop=True)
    val_part = val_part.sort_values("Image_name").reset_index(drop=True)
    kept_test = kept_test.sort_values("Image_name").reset_index(drop=True)
    train_part.to_csv(PROCESSED / "train.csv", index=False)
    val_part.to_csv(PROCESSED / "val.csv", index=False)
    kept_test.to_csv(PROCESSED / "test.csv", index=False)

    counts = train_part[LABEL_COLUMN].map(label2idx).value_counts().reindex(range(len(kept)), fill_value=0).to_numpy(dtype=np.float64)
    beta = 0.999
    class_weights = np.zeros(len(kept), dtype=np.float32)
    present = counts > 0
    class_weights[present] = (1 - beta) / (1 - np.power(beta, counts[present]))
    class_weights[present] /= class_weights[present].mean()
    class_weights = np.clip(class_weights, 0.2, 5.0).astype(np.float32)
    sample_weights = np.asarray([class_weights[label2idx[label]] for label in train_part[LABEL_COLUMN]], dtype=np.float32)
    np.save(PROCESSED / "class_weights.npy", class_weights)
    np.save(PROCESSED / "sample_weights.npy", sample_weights)

    REPORTS.mkdir(parents=True, exist_ok=True)
    distribution = pd.DataFrame({"label": all_labels, "train_count": [int(counts_train.get(x, 0)) for x in all_labels], "test_count": [int(counts_test.get(x, 0)) for x in all_labels]})
    distribution.to_csv(REPORTS / "label_distribution.csv", index=False)
    audit_lines = [
        "# Reconciled Label and Hierarchy Audit", "", f"- Raw train labels: {len(set(train[LABEL_COLUMN]))}",
        f"- Raw union labels: {len(all_labels)}", f"- Production classes: {len(kept)}",
        "- Rule: keep labels with at least 3 train samples and at least 1 test sample.",
        f"- Excluded labels: {len(excluded)}", f"- Mainclass count: {len(main_values)}", f"- Subclass count: {len(subclass_values)}", "",
        "## Manual reviews", "", f"- Reapplied: {', '.join(manual_applied) or 'none'}", f"- Previously reviewed but excluded: {', '.join(manual_excluded) or 'none'}", "",
        "## Exclusions", "", "| Label | Train | Test | Reasons |", "|---|---:|---:|---|",
    ]
    audit_lines.extend(f"| {label} | {int(counts_train.get(label, 0))} | {int(counts_test.get(label, 0))} | {', '.join(rare[label]['reasons'])} |" for label in excluded)
    (REPORTS / "hierarchy_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")

    print(f"raw_train_labels={len(set(train[LABEL_COLUMN]))} raw_union_labels={len(all_labels)}")
    print(f"labels_in_both={sum(counts_train.get(x, 0) > 0 and counts_test.get(x, 0) > 0 for x in all_labels)} train_only={sum(counts_train.get(x, 0) > 0 and counts_test.get(x, 0) == 0 for x in all_labels)} test_only={sum(counts_train.get(x, 0) == 0 and counts_test.get(x, 0) > 0 for x in all_labels)}")
    print(f"train_lt3={int((counts_train < 3).sum())} train_lt5={int((counts_train < 5).sum())} train_lt10={int((counts_train < 10).sum())}")
    print(f"production_classes={len(kept)} excluded_classes={len(excluded)} kept_train_rows={len(train_part)} val_rows={len(val_part)} kept_test_rows={len(kept_test)}")
    print(f"train_class_min={int(counts.min())} train_class_median={float(np.median(counts))} train_class_max={int(counts.max())} imbalance_ratio={float(counts.max() / counts[counts > 0].min()):.3f}")
    print(f"manual_reviews_applied={len(manual_applied)} manual_reviews_excluded={len(manual_excluded)}")


if __name__ == "__main__":
    main()