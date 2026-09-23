## Local checkpoint verification

Start from `backend/` with `python -m uvicorn app.main:app --reload`. The backend validates the checkpoint class-map and hierarchy hashes before loading it. With the reconciled CPU checkpoint, `/health` reports `best_model.pth:57afc7d27f71`; `/predict` returns a real disease prediction, hierarchy path, differentials, Grad-CAM++ heatmap, model version, and medical disclaimer. The selected ablation winner did not enable auxiliary hierarchy heads, so `hierarchical_consistency` is returned as `null` rather than fabricated.

# DermaSense Backend

This directory contains the FastAPI application backend for DermaSense.

## Setup & Running

1. Open your terminal in this directory (`cd backend`).
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server (make sure you're in the `backend/` directory):
   ```bash
   uvicorn app.main:app --reload
   ```

The application will be accessible at `http://127.0.0.1:8000`.

_Note: The backend depends on some model structures exported in `../ml/src` for inference. Ensure path dependencies remain intact._

The `/predict` endpoint accepts the image, age, gender, skin tone, body location, duration, and JSON symptom list. It returns disease, native hierarchy, model-derived Grad-CAM++, checkpoint version, and the medical disclaimer. `/health` reports whether `ml/checkpoints/best_model.pth` is available. The backend does not fabricate a response when that checkpoint is missing.
