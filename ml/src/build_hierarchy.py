"""Build the production disease hierarchy from the DermaCon-IN annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd


REVIEW_DECISIONS = {
    "Fissure Foot": {
        "mapping_status": "accepted_majority_vote",
        "manually_reviewed": True,
        "review_reason": "Existing majority mapping is medically coherent and was retained after audit.",
    },
    "Keratoderma": {
        "mapping_status": "accepted_majority_vote",
        "manually_reviewed": True,
        "review_reason": "Existing majority mapping is medically coherent and was retained after audit.",
    },
    "Lichen Planus pigmentosus": {
        "mapping_status": "accepted_majority_vote",
        "manually_reviewed": True,
        "review_reason": "Existing majority mapping is medically coherent and was retained after audit.",
    },
    "Lichen Simplex Chronicus": {
        "mapping_status": "accepted_majority_vote",
        "manually_reviewed": True,
        "review_reason": "Existing majority mapping is medically coherent and was retained after audit.",
    },
    "Palmoplantar Keratoderma": {
        "mapping_status": "accepted_majority_vote",
        "manually_reviewed": True,
        "review_reason": "Existing majority mapping is medically coherent and was retained after audit.",
    },
    "Paronychia": {
        "mapping_status": "accepted_majority_vote",
        "manually_reviewed": True,
        "review_reason": "Existing majority mapping is medically coherent and was retained after audit.",
    },
    "Sebaceous Cyst": {
        "mapping_status": "accepted_majority_vote",
        "manually_reviewed": True,
        "review_reason": "Existing majority mapping is medically coherent and was retained after audit.",
    },
    "Tinea Infected with secondary bacterial Infection": {
        "mapping_status": "accepted_majority_vote",
        "manually_reviewed": True,
        "review_reason": "Existing majority mapping is medically coherent and was retained after audit.",
    },
    "Pustular Scabies": {
        "mainclass": "Infectious Disorders",
        "subclass": "Infectious skin conditions -Parasitic",
        "mapping_status": "manually_normalized",
        "manually_reviewed": True,
        "review_reason": "Scabies is primarily a parasitic disease; secondary bacterial infection must not create a compound hierarchy class.",
    },
    "Scabies Infected": {
        "mainclass": "Infectious Disorders",
        "subclass": "Infectious skin conditions -Parasitic",
        "mapping_status": "manually_normalized",
        "manually_reviewed": True,
        "review_reason": "Scabies is primarily a parasitic disease; secondary bacterial infection must not create a compound hierarchy class.",
    },
    "Molluscum Contagiosum": {
        "mainclass": "Infectious Disorders",
        "subclass": "Infectious skin conditions -Viral",
        "mapping_status": "manually_corrected",
        "manually_reviewed": True,
        "review_reason": "Molluscum contagiosum is viral; the majority annotation placed it in a parasitic bucket, so the mapping was corrected to the existing native viral subclass.",
    },
    "Diabetic Ulcer": {
        "mapping_status": "low_confidence_majority_retained",
        "manually_reviewed": True,
        "review_reason": "No suitable native non-infectious wound category was available, so the dataset majority mapping was retained.",
    },
    "Pseudomonas": {
        "mapping_status": "low_confidence_majority_retained",
        "manually_reviewed": True,
        "review_reason": "The label is organism-associated and the existing native bacterial infectious mapping was retained despite low vote margin.",
    },
    "Zoon's Balanitis": {
        "mapping_status": "low_confidence_majority_retained",
        "manually_reviewed": True,
        "review_reason": "The existing native mapping was retained after reviewing the low-confidence split; no new category was introduced.",
    },
}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _majority_mapping(rows: pd.DataFrame) -> tuple[str, str, float, int]:
    pairs = Counter(zip(rows["Main_class"], rows["Sub_class"]))
    if not pairs:
        raise ValueError("No hierarchy rows were available for a production disease.")
    (mainclass, subclass), count = pairs.most_common(1)[0]
    return mainclass, subclass, count / len(rows), len(pairs)


def build_hierarchy(
    metadata_path: Path,
    labels_path: Path,
    output_path: Path,
    audit_path: Path,
    processed_train_path: Path | None = None,
) -> dict:
    metadata = pd.read_csv(metadata_path, low_memory=False)
    labels = _read_json(labels_path)
    required = {"Disease_label", "Main_class", "Sub_class", "Image_name"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")

    evidence_rows = metadata
    split_warning = ""
    if processed_train_path and processed_train_path.exists():
        processed_train = pd.read_csv(processed_train_path, low_memory=False)
        if "Image_name" in processed_train.columns:
            train_names = set(processed_train["Image_name"].astype(str))
            evidence_rows = metadata[metadata["Image_name"].astype(str).isin(train_names)]
            split_warning = (
                f"Hierarchy majority votes use {len(evidence_rows)} rows from "
                f"{processed_train_path.name}."
            )

    rows = []
    ambiguous = []
    mainclass_values = sorted(metadata["Main_class"].dropna().unique())
    subclass_values = sorted(metadata["Sub_class"].dropna().unique())
    for disease, disease_idx in sorted(labels.items(), key=lambda item: item[1]):
        disease_rows = evidence_rows[evidence_rows["Disease_label"] == disease]
        if disease_rows.empty:
            raise ValueError(f"Production disease has no metadata evidence: {disease}")
        mainclass, subclass, fraction, variants = _majority_mapping(disease_rows)
        decision = REVIEW_DECISIONS.get(disease, {})
        mainclass = decision.get("mainclass", mainclass)
        subclass = decision.get("subclass", subclass)
        record = {
            "disease": disease,
            "subclass": subclass,
            "mainclass": mainclass,
            "disease_idx": int(disease_idx),
            "mainclass_idx": mainclass_values.index(mainclass),
            "subclass_idx": subclass_values.index(subclass),
            "source": "DermaCon-IN Mainclass/Subclass annotation plus reviewed normalization" if decision else "DermaCon-IN Mainclass/Subclass annotation",
            "majority_fraction": round(fraction, 6),
            "mapping_variants": variants,
            "vote_share_before_override": round(fraction, 6),
            "mapping_status": decision.get("mapping_status", "native_majority_vote"),
            "manually_reviewed": decision.get("manually_reviewed", False),
            "review_reason": decision.get("review_reason", "Derived from the native dataset majority mapping."),
        }
        rows.append(record)
        if fraction < 0.9 or decision:
            ambiguous.append(record)

    label_hash = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "source": "DermaCon-IN Skin_Metadata.csv",
        "class_mapping_sha256": label_hash,
        "production_class_count": len(rows),
        "rows": rows,
    }
    payload["hierarchy_version"] = hashlib.sha256(
        json.dumps(rows, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    audit_lines = [
        "# Hierarchy Audit",
        "",
        "- Source: `ml/data/raw/Skin_Metadata.csv`.",
        f"- Production classes: {len(rows)}.",
        f"- Classes with majority fraction below 90%: {len(ambiguous)}.",
        f"- Label mapping SHA-256: `{label_hash}`.",
    ]
    if split_warning:
        audit_lines.extend(["", f"- {split_warning}"])
    audit_lines.extend(
        [
            "",
            "## Hierarchy-ambiguous classes",
            "",
            "| Disease | Samples | Old mapping | Vote share | Final mapping | Status | Changed |",
            "| --- | ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for record in ambiguous:
        disease_rows = evidence_rows[evidence_rows["Disease_label"] == record["disease"]]
        old_mainclass, old_subclass, _, _ = _majority_mapping(disease_rows)
        old_mapping = f"{old_mainclass} / {old_subclass}"
        final_mapping = f"{record['mainclass']} / {record['subclass']}"
        audit_lines.append(
            f"| {record['disease']} | {len(disease_rows)} | {old_mapping} | "
            f"{record['vote_share_before_override']:.3f} | {final_mapping} | "
            f"{record['mapping_status']} | {'yes' if old_mapping != final_mapping else 'no'} |"
        )
    audit_lines.extend(
        [
            "",
            "## Integrity notes",
            "",
            "The existing processed train/validation CSVs share subjects; validation results from that split must not be presented as leakage-free.",
            "The hierarchy is majority-voted for deterministic model targets, with the reviewed exceptions above. The test split was not modified.",
        ]
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=root / "data/processed/label2idx.json")
    parser.add_argument("--processed-train", type=Path, default=root / "data/processed/train.csv")
    parser.add_argument("--output", type=Path, default=root / "data/processed/hierarchy_map.json")
    parser.add_argument("--audit", type=Path, default=root / "outputs/reports/hierarchy_audit.md")
    args = parser.parse_args()
    metadata = args.metadata
    processed_train = args.processed_train
    if metadata is None:
        from dataset_discovery import discover_dataset

        discovered = discover_dataset(root / "data" / "raw")
        metadata = discovered.metadata_csv
        if metadata is None:
            raise FileNotFoundError(
                "SkinMetadata.csv was not found while discovering the dataset; "
                "the hierarchy cannot be built without native annotations."
            )
        processed_train = discovered.train_csv
    payload = build_hierarchy(
        metadata,
        args.labels,
        args.output,
        args.audit,
        processed_train,
    )
    print(f"Wrote {payload['production_class_count']} hierarchy rows to {args.output}")
    print(f"Wrote hierarchy audit to {args.audit}")


if __name__ == "__main__":
    main()