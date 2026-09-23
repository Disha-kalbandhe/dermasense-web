## Reproducible local run

Place the extracted dataset under `ml/data/raw/` with `DATASET/`, `train_split.csv`, `test_split.csv`, and `Skin_Metadata.csv`. Reconcile labels and validate before training:

```powershell
python ml/src/reconcile_labels.py
python ml/src/validate_data.py
python ml/src/validate_hierarchy.py
python ml/src/train.py --smoke-only --profile cpu_fast --no-pretrained
```

The raw union has 245 labels; 106 meet the production rule of at least 3 train samples and at least 1 test sample. The reconciled train distribution is min 3, median 14.5, max 416, ratio 138.667:1. On this CPU-only machine, the reduced ablation winner was metadata cross-attention (validation macro-F1 0.023281). The final two-epoch CPU run took 707.17s and 797.01s per epoch; final validation macro-F1 was 0.086752.

The final untouched eligible test evaluation was top-1 0.216700, top-3 0.359841, top-5 0.430417, macro-F1 0.069680, weighted-F1 0.184110, balanced accuracy 0.151364, and ECE 0.057924. The checkpoint is `ml/checkpoints/best_model.pth`; reports and eight Grad-CAM++ samples are under `ml/outputs/`. This was an honest reduced CPU run, not a claim of convergence or 92% accuracy.

## DermaSense ML pipeline

The pipeline supports five configurations through `ml/src/model.py`: image-only, metadata concatenation, cross-attention, concatenation with hierarchy, and cross-attention with hierarchy. The production hierarchy is generated from `Skin_Metadata.csv` and `label2idx.json`; see `data/processed/hierarchy_map.json` and `outputs/reports/hierarchy_audit.md`.

```bash
python src/build_hierarchy.py
python src/train.py --fusion-mode cross_attention --hierarchical --use-metadata
python src/evaluate.py --checkpoint checkpoints/best_model.pth
python -m pytest tests -q
```

The evaluator writes top-1/top-3/top-5 accuracy, macro and weighted metrics, balanced accuracy, calibration error, per-class metrics, a confusion matrix, and hierarchy consistency artifacts. A checkpoint is not generated automatically during repository setup, so no achieved accuracy or macro-F1 is reported here.
